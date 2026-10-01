#!/usr/bin/env python3
"""
delta_writer2.py version 1.0 -- DeltaWriter2.java for Python (PySide6).

A streamlined Delta Writer. Its files are read by delta_reader.py (and
DeltaReader.java): the header and per-channel layout are the same, with
compress_type 3 (see COMPRESS_TYPE) and the pixel pyramid bytes 0.

    python3 delta_writer2.py [image]      # saves to the file "foo"

Compared with delta_writer.py:
  - no Datatype menu and no Integer path: deltas are always unary strings
    from string_mapper.get_string_list(delta, True), which keeps the string
    compressed only if that beats StringMapper's threshold;
  - the Quantization menu has just Pixel Resolution, Color Resolution and
    Error Correction (no Smooth or Smooth2);
  - only the scanline, frame-map and block-map delta types (6-13), and the
    startup survey only ranks those, followed by a map-coding report.

The coding (class DeltaCoder2) is separate from the window:

    coder = DeltaCoder2(viewer_support.read_image("photo.png"))
    coder.survey(); coder.apply(); coder.save("photo.dlt")
"""

import math
import os
import sys

import numpy as np

import arithmetic_mapper as am
import code_mapper as cm
import delta_mapper as dm
import resize_mapper as rm
import string_mapper as sm
import viewer_support as vs
from delta_writer import ENTROPY_NAMES, context_code, entropy_code, pack_and_compress
from java_io import DataOutput

FORMAT_ID = ord('D')
FORMAT_VERSION = 1

# compress_type written to the header: String* (strings that don't compress
# enough come back uncompressed from get_string_list). DeltaReader treats it like 2.
COMPRESS_TYPE = 3

# The delta types this program uses, in menu order (the same delta_type
# values as delta_writer / delta_reader).
FIRST_DELTA_TYPE = 6
N_DELTA_TYPES = 8
DELTA_MENU_NAMES = ["Scanline 1", "Scanline 2", "Scanline 3", "Scanline 4", "Scanline 5", "Map 1", "Map 2", "Block Map"]

MAP_REPORT_COLUMNS = 9


# =============================================================================
# Map-coding report helpers
# =============================================================================

def widen(b):
    """Bytes as unsigned ints (DeltaWriter.widen)."""
    return np.asarray(b, dtype=np.uint8).astype(np.int64)


def arithmetic_map_bytes(m):
    """Size of write_map's arithmetic-coded form (without its flag byte): int
    length, short K, K x (byte value, int count), int coded length, then the
    coded bytes (none when K <= 1)."""
    m = np.asarray(m, dtype=np.uint8)
    freq = am.get_frequency(m)
    K = int(np.count_nonzero(freq))
    coded = 0 if K <= 1 else len(am.get_interval_value_fast_fenwick(m, freq))
    return 4 + 2 + 5 * K + 4 + coded


def conditional_entropy_bits(data, width):
    """Empirical conditional entropy (bits) of each entry given its context:
    the previous entry when width == 0, else the left and upper entries of a
    width-wide grid. Entries without a full context get their own context."""
    data = np.asarray(data, dtype=np.uint8)
    index = [-1] * 256
    K = 0
    for b in data.tolist():
        if index[b] < 0:
            index[b] = K
            K += 1
    if K <= 1:
        return 0.0
    idx = np.array(index, dtype=np.int64)[data]
    n = len(data)
    pos = np.arange(n)
    left = np.empty(n, dtype=np.int64)
    left[1:] = idx[:-1]
    left[0] = K
    if width > 0:
        left[pos % width == 0] = K
        up = np.full(n, K, dtype=np.int64)
        up[width:] = idx[:-width]
        ctx = left * (K + 1) + up
        C = (K + 1) * (K + 1)
    else:
        ctx = left
        C = K + 1
    count = np.bincount(ctx * K + idx, minlength=C * K).reshape(C, K)
    bits = 0.0
    log2 = math.log(2)
    for row in count.tolist():          # same order of summation as the Java
        total = sum(row)
        for v in row:
            if v > 0:
                bits -= v * (math.log(v / total) / log2)
    return bits


def entropy_bits(data):
    bits = 0.0
    n = float(len(data))
    log2 = math.log(2)
    for v in am.get_frequency(data).tolist():
        if v > 0:
            bits -= v * (math.log(v / n) / log2)
    return bits


# =============================================================================
# The coding, without the window
# =============================================================================

class DeltaCoder2:
    """Holds an image and the settings; survey() picks the channel set and
    delta type, apply() codes (and decodes, for the preview), save() writes."""

    def __init__(self, rgb, file_length=0):
        self.image_ydim, self.image_xdim = rgb.shape[:2]
        self.source = [rgb[:, :, c].reshape(-1).astype(np.int64) for c in range(3)]
        self.file_length = file_length

        self.pixel_quant = 4
        self.pixel_shift = 3
        self.pixel_segment = 10     # Arithmetic blocks: 10 = one block per channel; lower = blocks of 500+500*pixel_segment bytes
        self.correction = 0         # preview only
        self.min_set_id = 0
        self.delta_type = 6         # 6-13 only (Scanline 1-5, Map 1-2, Block Map)
        self.entropy_type = 0
        self.scanline5_variant = 0
        self.block_size = dm.BLOCK_DEFAULT
        self.block_set = 0

        self.channel_sum = [0] * 6
        self.set_sum = [0] * 10
        self.channel_min = [0] * 6
        self.channel_init = [0] * 6
        self.channel_delta_min = [0] * 6
        self.channel_length = [0] * 6
        self.channel_compressed_length = [0] * 6
        self.channel_iterations = [0] * 3
        self.applied = False

    # ---- Channels -------------------------------------------------------------

    def quantized_channels(self, size):
        """The six candidate channels after resizing and quantizing; sets
        channel_min and channel_init."""
        q = [None] * 3

        def one(i):
            ch = self.source[i]
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
        """Picks the channel set, then ranks delta types 6-13 by the compressed
        size of their delta string plus their map, and selects the smallest."""
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        w, h = size
        qc = self.quantized_channels(size)
        self.compute_set_sums(qc, size)
        self.print_channel_set_ranking()
        ids = dm.get_channels(self.min_set_id)

        N = N_DELTA_TYPES
        delta_bits = [[0] * N for _ in range(3)]
        map_bits = [[0] * N for _ in range(3)]
        report = [[[0] * MAP_REPORT_COLUMNS for _ in range(N)] for _ in range(3)]
        maps = [[None] * N for _ in range(3)]

        def channel(i):
            for t in range(N):
                type_ = FIRST_DELTA_TYPE + t
                d, m, _ = dm.get_deltas(qc[ids[i]], w, h, type_, self.scanline5_variant, self.block_size, self.block_set)
                m = np.asarray(m, dtype=np.uint8)
                delta_bytes = pack_and_compress(d)
                delta_bits[i][t] = sm.get_bitlength(delta_bytes)
                maps[i][t] = m
                map_string = pack_and_compress(widen(m))
                mb = sm.get_bitlength(map_string)
                map_bits[i][t] = mb
                mr = report[i][t]
                mr[0] = len(m)
                mr[2] = (mb + 7) // 8
                mr[3] = len(cm.deflate(m, 9))
                mr[4] = arithmetic_map_bytes(m)
                mr[5] = math.ceil(entropy_bits(m) / 8)
                mr[6] = (sm.get_bitlength(delta_bytes) + 7) // 8
                mr[7] = math.ceil(conditional_entropy_bits(m, 0) / 8)
                # Left+up estimate for the 2-D maps: frame maps (interior
                # pixels, xdim - 2 across) and block maps (one entry per block,
                # after the 2 header bytes).
                if type_ in (11, 12):
                    mr[8] = math.ceil(conditional_entropy_bits(m, w - 2) / 8)
                elif type_ == 13:
                    mr[8] = math.ceil(conditional_entropy_bits(m[2:], (w - 2 + int(m[0]) - 1) // int(m[0])) / 8)
                else:
                    mr[8] = -1
        vs.parallel(3, channel)

        # Maps for types 9-13 are stored in the smallest of the string,
        # arithmetic and context-coded forms, and the context form uses the
        # previous channel's map, so they are sized once all three exist.
        def map_type(t):
            type_ = FIRST_DELTA_TYPE + t
            for i in range(3):
                written = dm.map_bytes(type_, maps[i][t], maps[i - 1][t] if i > 0 else None, w)
                report[i][t][1] = written
                map_bits[i][t] = 8 * written
        vs.parallel(N, map_type)

        dbits = [sum(delta_bits[i][t] for i in range(3)) for t in range(N)]
        mbits = [sum(map_bits[i][t] for i in range(3)) for t in range(N)]
        total = [dbits[t] + mbits[t] for t in range(N)]
        best = min(range(N), key=lambda t: (total[t], t))
        self.delta_type = FIRST_DELTA_TYPE + best
        self.print_delta_type_ranking(dbits, mbits, total)
        self.print_map_report(report)
        if self.delta_type == 13:
            self.search_block_settings(qc, size, ids)

    def search_block_settings(self, qc=None, size=None, ids=None):
        """Picks block_size and block_set by coding with each candidate."""
        timer = vs.Timer()
        if qc is None:
            size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
            qc = self.quantized_channels(size)
            ids = dm.get_channels(self.min_set_id)
        best, table = dm.find_best_block([qc[i] for i in ids], size[0], size[1])
        self.block_size, self.block_set = best
        print(dm.get_block_table(table, best), end="")
        print("Block search took " + timer.elapsed())
        print()

    def print_channel_set_ranking(self):
        order = sorted(range(10), key=lambda s: self.set_sum[s])
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
        order = sorted(range(N_DELTA_TYPES), key=lambda t: total[t])
        print("Delta types, smallest first: bytes for the deltas (unary strings) and the predictor map")
        print("(as Save writes it; scanline maps are tiny), all 3 channels. <= marks the type chosen.")
        print("      %-16s %12s %12s %12s" % ("delta type", "deltas", "map", "total"))
        for r, t in enumerate(order):
            print("  %2d. %-16s %12d %12d %12d%s" % (r + 1, dm.DELTA_TYPE_NAMES[FIRST_DELTA_TYPE + t], dbits[t] // 8,
                  mbits[t] // 8, total[t] // 8, "  <=" if FIRST_DELTA_TYPE + t == self.delta_type else ""))
        print()

    def print_map_report(self, report):
        print("Map coding: size of each delta type's predictor map, in bytes, all 3 channels together")
        print("  map entries    how many predictor choices the map holds")
        print("  Actual coders (the smallest is marked <):")
        print("    as saved     what Save writes: the smallest of unary strings, arithmetic, and context coding")
        print("    strings      unary strings (StringMapper)")
        print("    deflate      one byte per entry, zip-style Deflate")
        print("    arithmetic   arithmetic coding with fixed frequencies")
        print("  Estimates (theoretical minimum if each entry is coded knowing only...):")
        print("    alone        ...how often each value occurs (arithmetic lands just above this;")
        print("                 Deflate can go below it on long runs)")
        print("    given left   ...plus the entry to its left")
        print("    given l+up   ...plus the entries to its left and above (frame and block maps only; \"-\" otherwise).")
        print("                 \"as saved\" can beat this: the context coder also uses the previous channel.")
        print("  deltas         the deltas' size (unary strings), for scale")
        print("  %-16s %11s |%10s %10s %10s %11s  |%10s %11s %11s  |%10s"
              % ("delta type", "map entries", "as saved", "strings", "deflate", "arithmetic", "alone", "given left", "given l+up", "deltas"))
        for t in range(N_DELTA_TYPES):
            s = [sum(report[i][t][c] for i in range(3)) for c in range(MAP_REPORT_COLUMNS)]
            best = 1
            for c in range(2, 5):
                if s[c] < s[best]:
                    best = c
            cell = {c: str(s[c]) + ("<" if c == best else " ") for c in range(1, 5)}
            hlu = "-" if s[8] < 0 else str(s[8])
            print("  %-16s %11d |%11s%11s%11s%11s  |%10d %11d %11s  |%10d"
                  % (dm.DELTA_TYPE_NAMES[FIRST_DELTA_TYPE + t], s[0], cell[1], cell[2], cell[3], cell[4], s[5], s[7], hlu, s[6]))
        print()

    # ---- Apply --------------------------------------------------------------------

    def apply(self):
        """Quantizes, picks the channel set, codes the deltas as String* (what
        Save writes), then decodes them the way delta_reader does. Returns the
        preview as an (ydim, xdim, 3) uint8 array."""
        self.applied = False
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        w, h = size
        qc = self.quantized_channels(size)
        self.compute_set_sums(qc, size)
        ids = dm.get_channels(self.min_set_id)
        table, string, maps, deltas, decoded = ([None] * 3 for _ in range(5))

        def channel(i):
            j = ids[i]
            d, m, _ = dm.get_deltas(qc[j], w, h, self.delta_type, self.scanline5_variant, self.block_size, self.block_set)
            maps[i] = m
            deltas[i] = d
            lo, bits, tbl, s = sm.get_string_list(d, True)
            self.channel_delta_min[j], self.channel_length[j] = lo, bits
            table[i], string[i] = tbl, s
            self.channel_compressed_length[j] = sm.get_bitlength(s)
            self.channel_iterations[i] = sm.get_iterations(s)
            d2 = sm.unpack_strings(sm.decompress_strings(s), tbl, w * h, bits)
            d2[0] = 0
            d2[1:] += lo
            ch = dm.get_values_from_deltas(d2, w, h, self.channel_init[j], self.delta_type, m, self.scanline5_variant)
            if j > 2:
                ch = ch + self.channel_min[j]
            decoded[i] = ch
        vs.parallel(3, channel)
        self.table, self.string, self.map, self.delta_list, self.delta_xdim = table, string, maps, deltas, w

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
        """delta_writer's layout with compress_type COMPRESS_TYPE and the
        pyramid bytes 0. Per channel: min, init, delta min, bit lengths,
        iterations, map, string table (not for Context), then the
        entropy-coded payload. Returns the file size."""
        ids = dm.get_channels(self.min_set_id)
        out = DataOutput()
        out.write_byte(FORMAT_ID); out.write_byte(FORMAT_VERSION)
        out.write_short(self.image_xdim); out.write_short(self.image_ydim)
        for v in (self.pixel_shift, self.pixel_quant, self.min_set_id, self.delta_type, COMPRESS_TYPE,
                  self.entropy_type, self.scanline5_variant, 0, 0):
            out.write_byte(v)

        timer = vs.Timer()
        if self.entropy_type == 4:
            coded = context_code(self.delta_list, self.delta_xdim)
        else:
            coded = entropy_code(self.string, self.entropy_type, self.pixel_segment)
        print("Entropy coding [%s] took %s" % (ENTROPY_NAMES[self.entropy_type], timer.elapsed()))

        for i in range(3):
            j = ids[i]
            for v in (self.channel_min[j], self.channel_init[j], self.channel_delta_min[j],
                      self.channel_length[j], self.channel_compressed_length[j]):
                out.write_int(v)
            out.write_byte(self.channel_iterations[i])
            dm.write_map(out, self.delta_type, self.map[i], self.map[i - 1] if i > 0 else None, self.delta_xdim)
            if self.entropy_type != 4:
                dm.write_table(out, self.table[i])
            out.write(coded[i])

        with open(filename, "wb") as f:
            f.write(out.to_bytes())
        raw = self.image_xdim * self.image_ydim * 3
        print("Original compression rate: %.4f" % (self.file_length / raw))
        print("Output  compression rate:  %.4f" % (out.size() / raw))
        return out.size()


# =============================================================================
# The window
# =============================================================================

def open_image(parent=None):
    from PySide6.QtWidgets import QFileDialog
    path, _ = QFileDialog.getOpenFileName(parent, "Open Image", "",
                                          "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pgm *.ppm);;All files (*)")
    if path:
        DeltaWriter2(path)


_open_writers = []


class DeltaWriter2:
    def __init__(self, filename):
        from PySide6.QtGui import QAction, QActionGroup, QKeySequence

        self.filename = filename
        try:
            rgb = vs.read_image(filename)
        except IOError as e:
            vs.show_error(None, str(e))
            return
        self.coder = c = DeltaCoder2(rgb, os.path.getsize(filename))
        self.updating = False
        print("Loaded %s, %d x %d" % (filename, c.image_xdim, c.image_ydim))

        self.view = view = vs.ImageWindow("Delta Writer 2  " + filename, c.image_xdim, c.image_ydim)
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
        for key, title, lo, hi in (("pixel_quant", "Pixel Resolution", 0, 10), ("pixel_shift", "Color Resolution", 0, 7)):
            act, self.sliders[key] = vs.make_slider_dialog(view, title, lo, hi, getattr(c, key), self._setter(key))
            quant.addAction(act)
        # Error Correction blends the preview back toward the original by
        # correction/10. Preview only; the saved file is unaffected.
        quant.addSeparator()
        act, self.sliders["correction"] = vs.make_slider_dialog(view, "Error Correction", 0, 10, c.correction, self._setter("correction"))
        quant.addAction(act)

        # ---- Delta (types 6-13)
        delta_menu = bar.addMenu("Delta")
        group = QActionGroup(view)
        self.delta_actions = []
        for k, name in enumerate(DELTA_MENU_NAMES):
            t = FIRST_DELTA_TYPE + k
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

        # ---- Entropy (Save only)
        entropy_menu = bar.addMenu("Entropy")
        group = QActionGroup(view)
        for et, name in enumerate(ENTROPY_NAMES):
            a = QAction(name, view, checkable=True)
            a.setChecked(et == c.entropy_type)
            group.addAction(a)
            entropy_menu.addAction(a)
            a.triggered.connect(lambda _=False, et=et: setattr(self.coder, "entropy_type", et))
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

    def show_settings(self):
        """Moves the controls to the coder's settings without applying."""
        self.updating = True
        c = self.coder
        self.delta_actions[c.delta_type - FIRST_DELTA_TYPE].setChecked(True)
        self.block_spinner.setValue(c.block_size)
        self.block_buttons[c.block_set].setChecked(True)
        for key, slider in self.sliders.items():
            slider.setValue(getattr(c, key))
        self.updating = False

    def reset(self):
        c = self.coder
        c.pixel_quant = c.pixel_shift = c.correction = 0
        self.show_settings()
        self.apply()

    # ---- Work -----------------------------------------------------------------------

    def apply(self):
        try:
            self.view.set_image(self.coder.apply())
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Apply: %r" % e)

    def _background(self, status, job):
        """Runs job with the menus disabled, then shows the settings and applies."""
        self.view.set_menus_enabled(False)
        self.view.set_status(status)

        def done(result, error):
            if error is not None:
                print("%s: %r" % (status, error))
            self.view.set_menus_enabled(True)
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
        DeltaWriter2(sys.argv[1])
    else:
        open_image()
    if not _open_writers:
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
