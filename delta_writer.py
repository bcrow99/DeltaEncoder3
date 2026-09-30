#!/usr/bin/env python3
"""
delta_writer.py version 1.0 -- DeltaWriter.java for Python (PySide6).

Quantizes an image, codes it as deltas and saves it (to the file "foo", as
the Java version does) in Delta format 'D' version 1: the same file layout
as DeltaWriter.java, so either program's files open in either reader.

    python3 delta_writer.py [image]

The coding (class DeltaCoder) is separate from the window (DeltaWriter), so
it can also be used without a display:

    coder = DeltaCoder(viewer_support.read_image("photo.png"))
    coder.survey(); coder.apply(); coder.save("photo.dlt")

The pixel pyramid ("Average" in the Quantization menu): after every other
quantizing step, the 3 selected channels are averaged down 1 or 2 times and
the deltas are coded at the top level; one sign bit per pixel per level
restores the detail when the reader expands back (see shrink_pyramid).

The first run compiles the Numba code (about a minute); later runs load it
from the cache.
"""

import os
import struct
import sys

import numpy as np

import arithmetic_mapper as am
import code_mapper as cm
import delta_mapper as dm
import delta_reader as dr
import image_mapper as im
import resize_mapper as rm
import string_mapper as sm
import viewer_support as vs
from java_io import DataOutput

FORMAT_ID = ord('D')
FORMAT_VERSION = 1

ENTROPY_NAMES = ["LZ77", "Huffman", "Arithmetic", "Adaptive", "Context"]
DELTA_MENU_NAMES = ["H", "V", "Average", "Med", "Directional", "Adaptive", "Scanline 1", "Scanline 2",
                    "Scanline 3", "Scanline 4", "Scanline 5", "Map 1", "Map 2", "Block Map"]


# =============================================================================
# The coding, without the window
# =============================================================================

class DeltaCoder:
    """Holds an image and the compression settings; survey() picks the
    channel set, delta type and datatype, apply() codes (and decodes, for
    the preview), save() writes the file."""

    def __init__(self, rgb, file_length=0):
        self.image_ydim, self.image_xdim = rgb.shape[:2]
        # Channel 0 is bits 16-23 of the Java pixel (red), as DeltaWriter.java
        # reads it; the names in DeltaMapper.SET_NAMES follow the Java.
        self.source = [rgb[:, :, c].reshape(-1).astype(np.int64) for c in range(3)]
        self.file_length = file_length

        self.pixel_quant = 4
        self.pixel_shift = 3
        self.pixel_segment = 10     # Arithmetic blocks: 10 = one block per channel; lower = blocks of 500+500*pixel_segment bytes
        self.correction = 0         # preview only: blends back toward the original by correction/10
        self.min_set_id = 0
        self.delta_type = 5
        self.compress_type = 1      # 0 Integer, 1 String, 2 String*
        self.entropy_type = 0
        self.smooth_level = 0
        self.smooth2_level = 0
        self.scanline5_variant = 0
        self.block_size = dm.BLOCK_DEFAULT
        self.block_set = 0
        # Pixel pyramid: number of shrink/expand levels (0 = none), capped at
        # 2 (deeper levels produced block artifacts even with sign-bit
        # correction). use_saddle: expand with the cross-derivative term too.
        self.pixel_pyramid = 0
        self.use_saddle = False
        self.sign_bit = [None] * 3

        self.channel_sum = [0] * 6
        self.set_sum = [0] * 10
        self.channel_min = [0] * 6
        self.channel_init = [0] * 6
        self.channel_delta_min = [0] * 6
        self.channel_length = [0] * 6
        self.channel_compressed_length = [0] * 6
        self.channel_iterations = [0] * 3
        self.int_allowed = True
        self.applied = False

    # ---- Channels -------------------------------------------------------------

    def quantized_channels(self, size, smooth):
        """The six candidate channels after smoothing (if asked), resizing and
        quantizing; sets channel_min and channel_init."""
        q = [None] * 3

        def one(i):
            ch = self.source[i]
            if smooth and self.smooth_level > 0:
                ch = dm.bilateral_smooth(ch, self.image_xdim, self.image_ydim, self.smooth_level)
            if smooth and self.smooth2_level > 0:
                ch = dm.anisotropic_smooth(ch, self.image_xdim, self.image_ydim, self.smooth2_level)
            if self.pixel_quant != 0:
                ch = rm.resize(ch, self.image_xdim, size[0], size[1])
            q[i] = dm.quantize_channel(ch, self.pixel_shift)

        vs.parallel(3, one)
        qc, mins = dm.get_candidate_channels(q[0], q[1], q[2])
        self.channel_min = mins
        self.channel_init = [int(c[0]) for c in qc]
        return qc

    def compute_set_sums(self, qc, size):
        def one(i):
            self.channel_sum[i] = int(np.floor(cm.get_shannon_limit(dm.get_ideal_frequency2(qc[i], size[0], size[1]))))
        vs.parallel(6, one)
        for s in range(10):
            c = dm.get_channels(s)
            self.set_sum[s] = self.channel_sum[c[0]] + self.channel_sum[c[1]] + self.channel_sum[c[2]]
        self.min_set_id = min(range(10), key=lambda s: (self.set_sum[s], s))

    # ---- Survey (Java: init) -----------------------------------------------------

    def survey(self):
        """Picks the channel set, ranks the 14 delta types by the compressed
        size of their deltas plus their map, then picks String or String*."""
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        w, h = size
        qc = self.quantized_channels(size, False)
        self.compute_set_sums(qc, size)
        self.print_channel_set_ranking()
        ids = dm.get_channels(self.min_set_id)

        T = dm.DELTA_TYPES
        delta_bits = [[0] * T for _ in range(3)]
        maps = [[None] * T for _ in range(3)]

        def channel(i):
            for t in range(T):
                d, m, _ = dm.get_deltas(qc[ids[i]], w, h, t, self.scanline5_variant, self.block_size, self.block_set)
                delta_bits[i][t] = sm.get_bitlength(pack_and_compress(d))
                maps[i][t] = m

        vs.parallel(3, channel)

        # Maps are ranked on what Save writes; the context form uses the
        # previous channel's map, so they are sized once all three exist.
        map_bits = [[0] * T for _ in range(3)]

        def map_type(t):
            if dm.has_map(t):
                for i in range(3):
                    map_bits[i][t] = 8 * dm.map_bytes(t, maps[i][t], maps[i - 1][t] if i > 0 else None, w)

        vs.parallel(T, map_type)

        dbits = [sum(delta_bits[i][t] for i in range(3)) for t in range(T)]
        mbits = [sum(map_bits[i][t] for i in range(3)) for t in range(T)]
        total = [dbits[t] + mbits[t] for t in range(T)]
        self.delta_type = min(range(T), key=lambda t: (total[t], t))
        self.print_delta_type_ranking(dbits, mbits, total)

        # String or String*, whichever is smaller for the selected type.
        s_bits = [0] * 3
        star_bits = [0] * 3

        def strings(i):
            d, _, _ = dm.get_deltas(qc[ids[i]], w, h, self.delta_type, self.scanline5_variant, self.block_size, self.block_set)
            s = sm.get_string_list(d, False)[3]
            s_bits[i] = sm.get_bitlength(s)
            star_bits[i] = sm.get_bitlength(sm.compress_strings(s))

        vs.parallel(3, strings)
        self.compress_type = 2 if sum(star_bits) < sum(s_bits) else 1
        if self.delta_type == 13:
            self.search_block_settings(qc, size, ids)

    def search_block_settings(self, qc=None, size=None, ids=None):
        """Picks block_size and block_set by coding with each candidate."""
        timer = vs.Timer()
        if qc is None:
            size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
            qc = self.quantized_channels(size, True)
            ids = dm.get_channels(self.min_set_id)
        # With a pixel pyramid the block map codes the top level, so the search does too.
        top = dr.get_pyramid_size(size[0], size[1], self.pixel_pyramid)
        ch = [qc[i] if self.pixel_pyramid == 0 else shrink_pyramid(qc[i], size[0], size[1], self.pixel_pyramid)[0]
              for i in ids]
        best, table = dm.find_best_block(ch, top[0], top[1])
        self.block_size, self.block_set = best
        print(dm.get_block_table(table, best), end="")
        print("Block search took " + timer.elapsed())
        print()

    def print_channel_set_ranking(self):
        order = sorted(range(10), key=lambda s: (self.set_sum[s], s))
        print("Channel sets, smallest first: estimated bytes for each channel's deltas")
        print("(entropy estimate, before real coding), and the set's total. <= marks the set used.")
        print("      %-32s %10s %10s %10s %12s" % ("channel set", "1st", "2nd", "3rd", "total"))
        for r, s in enumerate(order):
            c = dm.get_channels(s)
            print("  %2d. %-32s %10d %10d %10d %12d%s" % (r + 1, dm.SET_NAMES[s], self.channel_sum[c[0]] // 8,
                  self.channel_sum[c[1]] // 8, self.channel_sum[c[2]] // 8, self.set_sum[s] // 8,
                  "  <=" if s == self.min_set_id else ""))
        print()

    def print_delta_type_ranking(self, dbits, mbits, total):
        order = sorted(range(dm.DELTA_TYPES), key=lambda t: (total[t], t))
        print("Delta types, smallest first: bytes for the deltas and, for types that have one, the")
        print("predictor map, all 3 channels. <= marks the type chosen.")
        print("      %-16s %12s %12s %12s" % ("delta type", "deltas", "map", "total"))
        for r, t in enumerate(order):
            print("  %2d. %-16s %12d %12s %12d%s" % (r + 1, dm.DELTA_TYPE_NAMES[t], dbits[t] // 8,
                  str(mbits[t] // 8) if dm.has_map(t) else "", total[t] // 8, "  <=" if t == self.delta_type else ""))
        print()

    # ---- Apply --------------------------------------------------------------------

    def apply(self):
        """Quantizes, picks the channel set, codes the deltas (what Save
        writes), then decodes them the way DeltaReader does. Returns the
        preview as an (ydim, xdim, 3) uint8 array."""
        self.applied = False
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        w, h = size
        qc = self.quantized_channels(size, True)
        self.compute_set_sums(qc, size)
        ids = dm.get_channels(self.min_set_id)

        # Integer needs every delta - delta_min to fit in a byte.
        self.int_allowed = all((int(qc[j].max()) - int(qc[j].min())) * 2 <= 255 for j in ids)
        if not self.int_allowed and self.compress_type == 0:
            self.compress_type = 1

        tw, th = dr.get_pyramid_size(w, h, self.pixel_pyramid)
        table, payload, maps, deltas, decoded, sign_bit = ([None] * 3 for _ in range(6))

        def channel(i):
            j = ids[i]
            c = qc[j]
            if self.pixel_pyramid != 0:
                c, sign_bit[i] = shrink_pyramid(c, w, h, self.pixel_pyramid)
            d, m, _ = dm.get_deltas(c, tw, th, self.delta_type, self.scanline5_variant, self.block_size, self.block_set)
            deltas[i] = d
            maps[i] = m
            if self.compress_type == 0:
                # One byte per delta, delta - delta_min. The string fields
                # aren't used and keep their last values, as in the Java.
                self.channel_delta_min[j] = int(d.min())
                b = ((d - self.channel_delta_min[j]) & 0xFF).astype(np.uint8)
                b[0] = 0
                payload[i] = b
                d2 = b.astype(np.int64) + self.channel_delta_min[j]
                d2[0] = 0
            else:
                lo, bits, tbl, s = sm.get_string_list(d, self.compress_type == 2)
                self.channel_delta_min[j], self.channel_length[j] = lo, bits
                table[i], payload[i] = tbl, s
                self.channel_compressed_length[j] = sm.get_bitlength(s)
                self.channel_iterations[i] = sm.get_iterations(s)
                d2 = sm.unpack_strings(sm.decompress_strings(s), tbl, tw * th, bits)
                d2[0] = 0
                d2[1:] += lo
            ch = dm.get_values_from_deltas(d2, tw, th, self.channel_init[j], self.delta_type, m, self.scanline5_variant)
            if self.pixel_pyramid != 0:
                ch = dr.expand_pyramid(ch, w, h, sign_bit[i], j > 2, self.use_saddle)
            if j > 2:
                ch = ch + self.channel_min[j]
            decoded[i] = ch

        vs.parallel(3, channel)
        self.table, self.payload, self.map, self.delta_list, self.delta_xdim = table, payload, maps, deltas, tw
        self.sign_bit = sign_bit

        # As the reader: recombine the channel set first, then resize, then shift.
        bgr = dm.get_blue_green_red(self.min_set_id, *decoded)

        def restore(c):
            v = bgr[c]
            if self.pixel_quant != 0:
                v = rm.resize(v, w, self.image_xdim, self.image_ydim)
            if self.pixel_shift != 0:
                v = dm.shift(v, self.pixel_shift)
            if self.correction != 0:
                f = self.correction / 10.0
                v = v + ((self.source[c] - v) * f).astype(np.int64)     # (int) truncates toward zero
            bgr[c] = v

        vs.parallel(3, restore)
        rgb = np.stack([np.clip(v, 0, 255).reshape(self.image_ydim, self.image_xdim) for v in bgr], axis=2).astype(np.uint8)
        self.applied = True
        return rgb

    # ---- Save -------------------------------------------------------------------------

    def save(self, filename="foo"):
        """Header, then per channel: min, init, delta min, bit lengths,
        iterations, map (types 6-13), sign bits (pyramid), string table
        (String and String*, not Context), then the entropy-coded payload."""
        ids = dm.get_channels(self.min_set_id)
        out = DataOutput()
        out.write_byte(FORMAT_ID); out.write_byte(FORMAT_VERSION)
        out.write_short(self.image_xdim); out.write_short(self.image_ydim)
        for v in (self.pixel_shift, self.pixel_quant, self.min_set_id, self.delta_type, self.compress_type,
                  self.entropy_type, self.scanline5_variant, self.pixel_pyramid, 1 if self.use_saddle else 0):
            out.write_byte(v)

        timer = vs.Timer()
        if self.entropy_type == 4:
            coded = context_code(self.delta_list, self.delta_xdim)
        else:
            coded = entropy_code(self.payload, self.entropy_type, self.pixel_segment)
        print("Entropy coding [%s] took %s" % (ENTROPY_NAMES[self.entropy_type], timer.elapsed()))

        for i in range(3):
            j = ids[i]
            for v in (self.channel_min[j], self.channel_init[j], self.channel_delta_min[j],
                      self.channel_length[j], self.channel_compressed_length[j]):
                out.write_int(v)
            out.write_byte(self.channel_iterations[i])
            if dm.has_map(self.delta_type):
                dm.write_map(out, self.delta_type, self.map[i], self.map[i - 1] if i > 0 else None, self.delta_xdim)
            if self.pixel_pyramid != 0:
                write_sign_bits(out, self.sign_bit[i])
            if self.compress_type > 0 and self.entropy_type != 4:
                dm.write_table(out, self.table[i])
            out.write(coded[i])

        with open(filename, "wb") as f:
            f.write(out.to_bytes())
        raw = self.image_xdim * self.image_ydim * 3
        print("Original compression rate: %.4f" % (self.file_length / raw))
        print("Output  compression rate:  %.4f" % (out.size() / raw))
        return out.size()


def shrink_pyramid(c, xdim, ydim, levels):
    """Pads c to a multiple of 2^levels, then shrinks it levels times.
    Returns (top level, sign bits), where sign_bits[lvl] compares level lvl
    with level lvl+1 (delta_reader.expand_pyramid undoes it)."""
    mult = 1 << levels
    padded_xdim, padded_ydim = im.pad_to(xdim, mult), im.pad_to(ydim, mult)
    level = im.pad_edge_replicate_flat(np.asarray(c, dtype=np.int64), xdim, ydim, padded_xdim, padded_ydim)
    level_xdim = padded_xdim
    sign_bits = []
    for _ in range(levels):
        nxt = im.shrink_avg_flat(level, level_xdim)
        sign_bits.append(im.build_geq_bits_flat(level, nxt, level_xdim))
        level = nxt
        level_xdim //= 2
    return level, sign_bits


def write_sign_bits(out, sign_bits):
    """One bitmap per pyramid level: int length, then (length+7)/8 bytes,
    bit q in byte q>>3, bit q&7. The reader takes the number of levels from
    the header."""
    for bits in sign_bits:
        out.write_int(len(bits))
        out.write(np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="little").tobytes())


def pack_and_compress(values):
    """A StringMapper bit string of values, compressed."""
    return sm.compress_strings(sm.get_string_list(values, False)[3])


def context_code(delta, xdim):
    """Context entropy type: each channel's deltas context coded, channel i's
    contexts using channels 0..i-1. Checks each decodes back."""
    from java_io import DataInput
    coded = [None] * 3

    def one(i):
        coded[i] = dm.pack_context_deltas(delta[i], delta[:i], xdim)
        back = dm.read_context_deltas(DataInput(coded[i]), len(delta[i]), delta[:i], xdim)
        if not np.array_equal(back, delta[i]):
            print("WARNING: channel %d context-coded deltas do not decode back." % i)

    vs.parallel(3, one)
    return coded


def entropy_code(payload, entropy_type, pixel_segment):
    """Each result is what follows the channel's table in the file:
      LZ77:       int payload length, int Deflated length, Deflated payload;
      Huffman:    int length + the 256 code lengths (Deflated), int length + the code;
      Arithmetic: the frequency tables, then each block's int length and coded bytes;
      Adaptive:   int length + the coded bytes."""
    coded = [None] * 3
    if entropy_type == 0:
        def one(i):
            zipped = cm.deflate(payload[i], 9)
            coded[i] = struct.pack(">ii", len(payload[i]), len(zipped)) + zipped
        vs.parallel(3, one)
    elif entropy_type == 1:
        def one(i):
            lengths = cm.get_regular_huffman_length(am.get_frequency(payload[i]))
            code = cm.pack_regular_code(payload[i], lengths)
            if not np.array_equal(cm.unpack_regular_code(code, lengths, len(payload[i])), payload[i]):
                print("WARNING: channel %d Huffman payload does not decode back." % i)
            tables = cm.pack_regular_tables([lengths], 9)
            coded[i] = struct.pack(">i", len(tables)) + tables + struct.pack(">i", len(code)) + code.tobytes()
        vs.parallel(3, one)
    elif entropy_type == 2:
        blocks, freqs = [], []
        for i in range(3):
            n = 1 if pixel_segment >= 10 else max(1, len(payload[i]) // (500 + pixel_segment * 500))
            blocks.append(am.get_blocks(payload[i], n))
            freqs.append([am.get_frequency(b) for b in blocks[i]])
        jobs = [(i, m) for i in range(3) for m in range(len(blocks[i]))]
        enc = {}

        def block(k):
            i, m = jobs[k]
            enc[(i, m)] = am.get_interval_value_fast_fenwick(blocks[i][m], freqs[i][m])
        vs.parallel(len(jobs), block)
        for i in range(3):
            parts = [am.pack_frequencies(freqs[i], 9)]
            for m in range(len(blocks[i])):
                e = enc[(i, m)]
                parts.append(struct.pack(">i", len(e)) + e.tobytes())
            coded[i] = b"".join(parts)
    else:
        def one(i):
            code = am.get_interval_value_adaptive(payload[i])
            if not np.array_equal(am.get_arithmetic_values_adaptive(code, len(payload[i])), payload[i]):
                print("WARNING: channel %d adaptive payload does not decode back." % i)
            coded[i] = struct.pack(">i", len(code)) + code.tobytes()
        vs.parallel(3, one)
    return coded


# =============================================================================
# The window
# =============================================================================

def open_image(parent=None):
    from PySide6.QtWidgets import QFileDialog
    path, _ = QFileDialog.getOpenFileName(parent, "Open Image", "",
                                          "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pgm *.ppm);;All files (*)")
    if path:
        DeltaWriter(path)


_open_writers = []


class DeltaWriter:
    def __init__(self, filename):
        from PySide6.QtGui import QAction, QActionGroup, QKeySequence
        from PySide6.QtWidgets import QRadioButton, QButtonGroup, QCheckBox, QHBoxLayout, QLabel, QSpinBox, QWidget

        self.filename = filename
        try:
            rgb = vs.read_image(filename)
        except IOError as e:
            vs.show_error(None, str(e))
            return
        self.coder = c = DeltaCoder(rgb, os.path.getsize(filename))
        self.original = rgb
        self.updating = False
        print("Loaded %s, %d x %d" % (filename, c.image_xdim, c.image_ydim))

        self.view = view = vs.ImageWindow("Delta Writer  " + filename, c.image_xdim, c.image_ydim)
        bar = view.menuBar()
        _open_writers.append(self)
        view.destroyed.connect(lambda *_: _open_writers.remove(self) if self in _open_writers else None)

        # ---- File
        file_menu = bar.addMenu("File")
        a = QAction("Open...", view, shortcut=QKeySequence("Ctrl+O"))
        a.triggered.connect(lambda: open_image(view))
        file_menu.addAction(a)
        file_menu.addSeparator()
        a = QAction("Reset", view)
        a.triggered.connect(self.reset)
        file_menu.addAction(a)
        a = QAction("Save", view)
        a.triggered.connect(self.save)
        file_menu.addAction(a)

        view.make_view_menu()

        # ---- Quantization
        quant = bar.addMenu("Quantization")
        self.sliders = {}
        for key, title, lo, hi in (("smooth_level", "Smooth", 0, 10), ("smooth2_level", "Smooth2", 0, 10),
                                   ("pixel_quant", "Pixel Resolution", 0, 10), ("pixel_shift", "Color Resolution", 0, 7)):
            act, slider = vs.make_slider_dialog(view, title, lo, hi, getattr(c, key), self._setter(key))
            quant.addAction(act)
            self.sliders[key] = slider
        # Average: pyramid levels 0-2, plus the Use Saddle checkbox.
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self.pyramid_spin = QSpinBox()
        self.pyramid_spin.setRange(0, 2)
        self.pyramid_spin.setValue(c.pixel_pyramid)
        self.pyramid_spin.valueChanged.connect(self._setter("pixel_pyramid"))
        row_layout.addWidget(QLabel("Levels:"))
        row_layout.addWidget(self.pyramid_spin)
        self.saddle_checkbox = QCheckBox("Use Saddle")
        self.saddle_checkbox.setChecked(c.use_saddle)
        self.saddle_checkbox.toggled.connect(self._setter("use_saddle"))
        quant.addAction(vs.make_button_dialog(view, "Average", [row, self.saddle_checkbox]))
        # Error Correction is not a quantizing step: it blends the preview back
        # toward the original by correction/10. Preview only.
        quant.addSeparator()
        act, self.sliders["correction"] = vs.make_slider_dialog(view, "Error Correction", 0, 10, c.correction, self._setter("correction"))
        quant.addAction(act)

        # ---- Delta
        delta_menu = bar.addMenu("Delta")
        group = QActionGroup(view)
        self.delta_actions = []
        for t, name in enumerate(DELTA_MENU_NAMES):
            a = QAction(name, view, checkable=True)
            a.setChecked(t == c.delta_type)
            group.addAction(a)
            delta_menu.addAction(a)
            a.triggered.connect(lambda _=False, t=t: self.set_delta_type(t))
            self.delta_actions.append(a)
        delta_menu.addSeparator()
        act, self.block_spinner = vs.make_spinner_dialog(view, "Block Size", dm.BLOCK_MIN, dm.BLOCK_MAX, c.block_size, self.set_block_size)
        delta_menu.addAction(act)
        act, self.block_buttons = vs.make_radio_dialog(view, "Block Predictors", dm.BLOCK_SET_NAMES, c.block_set, self.set_block_set)
        delta_menu.addAction(act)
        a = QAction("Find Best Block Settings", view)
        a.triggered.connect(self.find_block_settings)
        delta_menu.addAction(a)

        # ---- Datatype: two dialogs, each with an Integer and a String button.
        self.datatype_menu = bar.addMenu("Datatype")
        int_a, str_a, int_b, str_b = (QRadioButton("Integer"), QRadioButton("String"),
                                      QRadioButton("Integer"), QRadioButton("String"))
        self.int_buttons = [int_a, int_b]
        self.str_buttons = [str_a, str_b]
        for pair in ((int_a, str_a), (int_b, str_b)):
            g = QButtonGroup(view)
            g.addButton(pair[0]); g.addButton(pair[1])
        for b in self.int_buttons:
            b.clicked.connect(lambda: self.set_compress_type(0))
        for b in self.str_buttons:
            b.clicked.connect(lambda: self.set_compress_type(1))
        self.datatype_menu.addAction(vs.make_button_dialog(view, "Integer", [int_a, str_a]))
        self.datatype_menu.addAction(vs.make_button_dialog(view, "String", [int_b, str_b], vertical=False))
        self.show_compress_type()

        # ---- Entropy
        entropy_menu = bar.addMenu("Entropy")
        group = QActionGroup(view)
        for et, name in enumerate(ENTROPY_NAMES):
            a = QAction(name, view, checkable=True)
            a.setChecked(et == c.entropy_type)
            group.addAction(a)
            entropy_menu.addAction(a)
            a.triggered.connect(lambda _=False, et=et: self.set_entropy_type(et))
        entropy_menu.addSeparator()
        act, _ = vs.make_slider_dialog(view, "Segment Size", 0, 10, c.pixel_segment, lambda v: setattr(c, "pixel_segment", v))
        entropy_menu.addAction(act)

        view.set_image(rgb)
        view.show_window()
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self.show_initial_image)

    # ---- Settings -----------------------------------------------------------------

    def _setter(self, key):
        def set_value(v):
            setattr(self.coder, key, v)
            if not self.updating:
                self.apply()
        return set_value

    def set_delta_type(self, t):
        if self.coder.delta_type == t:
            return
        self.coder.delta_type = t
        if t == 13:
            self.find_block_settings()
        else:
            self.apply()

    def set_block_size(self, v):
        self.coder.block_size = v
        if self.coder.delta_type == 13 and not self.updating:
            self.apply()

    def set_block_set(self, v):
        self.coder.block_set = v
        if self.coder.delta_type == 13 and not self.updating:
            self.apply()

    def set_compress_type(self, t):
        if (t == 0) != (self.coder.compress_type == 0):
            self.coder.compress_type = t
            self.show_compress_type()
            self.apply()

    def set_entropy_type(self, et):
        self.coder.entropy_type = et
        self.datatype_menu.menuAction().setEnabled(et != 4)   # Context codes the deltas directly

    def show_compress_type(self):
        for b in self.int_buttons:
            b.setChecked(self.coder.compress_type == 0)
            b.setEnabled(self.coder.int_allowed)
        for b in self.str_buttons:
            b.setChecked(self.coder.compress_type != 0)

    def show_settings(self):
        """Moves the controls to the coder's settings without applying."""
        self.updating = True
        c = self.coder
        self.delta_actions[c.delta_type].setChecked(True)
        self.block_spinner.setValue(c.block_size)
        self.block_buttons[c.block_set].setChecked(True)
        for key, slider in self.sliders.items():
            slider.setValue(getattr(c, key))
        self.pyramid_spin.setValue(c.pixel_pyramid)
        self.saddle_checkbox.setChecked(c.use_saddle)
        self.show_compress_type()
        self.updating = False

    def reset(self):
        c = self.coder
        c.smooth_level = c.smooth2_level = c.pixel_quant = c.pixel_shift = c.correction = c.pixel_pyramid = 0
        self.show_settings()
        self.apply()

    # ---- Work -----------------------------------------------------------------------

    def apply(self):
        try:
            rgb = self.coder.apply()
            self.view.set_image(rgb)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Apply: %r" % e)
        self.show_compress_type()

    def _background(self, status, job):
        """Runs job with the menus disabled, then shows the settings and applies."""
        self.view.set_menus_enabled(False)
        self.view.set_status(status)

        def done(result, error):
            if error is not None:
                print("%s: %r" % (status, error))
            self.view.set_menus_enabled(True)
            self.datatype_menu.menuAction().setEnabled(self.coder.entropy_type != 4)
            self.view.set_status(None)
            self.show_settings()
            self.apply()

        vs.run_in_background(job, done)

    def show_initial_image(self):
        """Applies the default quantization, then runs the survey in the
        background so Apply and Save can't overlap it."""
        self.apply()
        self._background("analyzing…", self.coder.survey)

    def find_block_settings(self):
        self._background("finding block settings…", self.coder.search_block_settings)

    def save(self):
        if not self.coder.applied:
            self.apply()
        if not self.coder.applied:
            vs.show_error(self.view, "Nothing saved: the last Apply failed.")
            return
        try:
            self.coder.save("foo")
        except Exception as e:
            import traceback
            traceback.print_exc()
            if os.path.exists("foo"):
                os.remove("foo")
            vs.show_error(self.view, "Save failed: %r" % e)


def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    if len(sys.argv) > 1:
        DeltaWriter(sys.argv[1])
    else:
        open_image()
    if not _open_writers:
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
