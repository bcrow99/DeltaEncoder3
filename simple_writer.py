#!/usr/bin/env python3
"""
simple_writer.py version 1.0 -- SimpleWriter.java for Python (PySide6).

Quantizes an image, codes it as deltas packed into unary strings, and saves
(to the file "foo", as the Java version does) in Simple format 'S' version 1:
the same file layout as SimpleWriter.java, so either program's files open
in either reader.

    python3 simple_writer.py [image]

Deltas are always unary strings (get_string_list(delta, True)); StringMapper
skips its own compression when it doesn't pay off. The delta type is the
Delta menu's (default 2, average), not surveyed.

The coding (class SimpleCoder) is separate from the window:

    coder = SimpleCoder(viewer_support.read_image("photo.png"))
    coder.survey(); coder.apply(); coder.save("photo.smp")
"""

import os
import sys

import numpy as np

import code_mapper as cm
import delta_mapper as dm
import resize_mapper as rm
import string_mapper as sm
import viewer_support as vs
from delta_writer import context_code, entropy_code
from java_io import DataOutput

FORMAT_ID = ord('S')
FORMAT_VERSION = 1

ENTROPY_NAMES = ["LZ77", "Huffman", "Arithmetic", "Adaptive", "Context"]
DELTA_MENU_NAMES = ["H", "V", "Average", "Med", "Directional", "Adaptive", "Scanline 1", "Scanline 2",
                    "Scanline 3", "Scanline 4", "Scanline 5", "Map 1", "Map 2", "Block Map"]


# =============================================================================
# The coding, without the window
# =============================================================================

class SimpleCoder:
    """Holds an image and the settings; survey() picks the channel set,
    apply() codes (and decodes, for the preview), save() writes the file."""

    def __init__(self, rgb, file_length=0):
        self.image_ydim, self.image_xdim = rgb.shape[:2]
        self.source = [rgb[:, :, c].reshape(-1).astype(np.int64) for c in range(3)]
        self.file_length = file_length

        self.pixel_quant = 4
        self.pixel_shift = 3
        self.pixel_segment = 10     # Arithmetic blocks: 10 = one block per channel; lower = blocks of 500+500*pixel_segment bytes
        self.correction = 0         # preview only
        self.min_set_id = 0
        self.delta_type = 2         # average; the Delta menu selection
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
        # Java's getIdealFrequency; get_ideal_frequency2 gives the identical histogram.
        def one(i):
            self.channel_sum[i] = int(np.floor(cm.get_shannon_limit(dm.get_ideal_frequency2(qc[i], size[0], size[1]))))
        vs.parallel(6, one)
        for s in range(10):
            c = dm.get_channels(s)
            self.set_sum[s] = self.channel_sum[c[0]] + self.channel_sum[c[1]] + self.channel_sum[c[2]]
        self.min_set_id = min(range(10), key=lambda s: (self.set_sum[s], s))

    def survey(self):
        """Picks the channel set with the smallest entropy estimate (Java: init)."""
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        self.compute_set_sums(self.quantized_channels(size), size)
        self.print_channel_set_ranking()

    def print_channel_set_ranking(self):
        order = sorted(range(10), key=lambda s: self.set_sum[s])
        print("Channel sets (ranked):")
        for r, s in enumerate(order):
            c = dm.get_channels(s)
            print("  %2d. %-32s %10d %10d %10d %12d%s" % (r + 1, dm.SET_NAMES[s], self.channel_sum[c[0]],
                  self.channel_sum[c[1]], self.channel_sum[c[2]], self.set_sum[s], " **" if s == self.min_set_id else ""))
        print()

    def apply(self):
        """Quantizes, picks the channel set, codes the deltas as unary strings
        (what Save writes), then decodes them the way simple_reader does.
        Returns the preview as an (ydim, xdim, 3) uint8 array."""
        self.applied = False
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        w, h = size
        qc = self.quantized_channels(size)
        self.compute_set_sums(qc, size)
        ids = dm.get_channels(self.min_set_id)
        self.delta_xdim = w
        table, strings, maps, deltas, decoded = ([None] * 3 for _ in range(5))

        def channel(i):
            j = ids[i]
            d, m, _ = dm.get_deltas(qc[j], w, h, self.delta_type, self.scanline5_variant, self.block_size, self.block_set)
            maps[i] = m
            deltas[i] = d
            lo, bits, tbl, s = sm.get_string_list(d, True)
            self.channel_delta_min[j], self.channel_length[j] = lo, bits
            table[i], strings[i] = tbl, s
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
        self.table_list, self.string_list, self.map_list, self.delta_list = table, strings, maps, deltas

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

    def save(self, filename="foo"):
        """Header, then per channel: min, init, delta min, bit lengths,
        iterations, map (types 6-13), string table (not for Context), then the
        entropy-coded payload (the same layouts as delta_writer). Returns the
        file size."""
        ids = dm.get_channels(self.min_set_id)
        out = DataOutput()
        out.write_byte(FORMAT_ID); out.write_byte(FORMAT_VERSION)
        out.write_short(self.image_xdim); out.write_short(self.image_ydim)
        for v in (self.pixel_shift, self.pixel_quant, self.min_set_id, self.delta_type, self.entropy_type, self.scanline5_variant):
            out.write_byte(v)

        timer = vs.Timer()
        if self.entropy_type == 4:
            coded = context_code(self.delta_list, self.delta_xdim)
        else:
            coded = entropy_code(self.string_list, self.entropy_type, self.pixel_segment)
        print("Entropy coding [%s] took %s" % (ENTROPY_NAMES[self.entropy_type], timer.elapsed()))

        for i in range(3):
            j = ids[i]
            for v in (self.channel_min[j], self.channel_init[j], self.channel_delta_min[j],
                      self.channel_length[j], self.channel_compressed_length[j]):
                out.write_int(v)
            out.write_byte(self.channel_iterations[i])
            if dm.has_map(self.delta_type):
                dm.write_map(out, self.delta_type, self.map_list[i], self.map_list[i - 1] if i > 0 else None, self.delta_xdim)
            if self.entropy_type != 4:
                dm.write_table(out, self.table_list[i])
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
        SimpleWriter(path)


_open_writers = []


class SimpleWriter:
    def __init__(self, filename):
        from PySide6.QtGui import QAction, QActionGroup, QKeySequence

        self.filename = filename
        try:
            rgb = vs.read_image(filename)
        except IOError as e:
            vs.show_error(None, str(e))
            return
        self.coder = c = SimpleCoder(rgb, os.path.getsize(filename))
        self.updating = False
        print("Loaded %s, %d x %d" % (filename, c.image_xdim, c.image_ydim))

        self.view = view = vs.ImageWindow("Simple Writer  " + filename, c.image_xdim, c.image_ydim)
        bar = view.menuBar()
        _open_writers.append(self)
        view.destroyed.connect(lambda *_: _open_writers.remove(self) if self in _open_writers else None)

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
        # Block map settings (delta type 13); a change re-applies only when
        # the block map is selected.
        delta_menu.addSeparator()
        act, _ = vs.make_spinner_dialog(view, "Block Size", dm.BLOCK_MIN, dm.BLOCK_MAX, c.block_size, self.set_block_size)
        delta_menu.addAction(act)
        act, _ = vs.make_radio_dialog(view, "Block Predictors", dm.BLOCK_SET_NAMES, c.block_set, self.set_block_set)
        delta_menu.addAction(act)

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

    def _setter(self, key):
        def set_value(v):
            setattr(self.coder, key, v)
            if not self.updating:
                self.apply()
        return set_value

    def set_delta_type(self, t):
        if self.coder.delta_type != t:
            self.coder.delta_type = t
            self.apply()

    def set_block_size(self, v):
        self.coder.block_size = v
        if self.coder.delta_type == 13:
            self.apply()

    def set_block_set(self, v):
        self.coder.block_set = v
        if self.coder.delta_type == 13:
            self.apply()

    def reset(self):
        c = self.coder
        c.pixel_quant = c.pixel_shift = c.correction = 0
        self.updating = True
        for key, slider in self.sliders.items():
            slider.setValue(getattr(c, key))
        self.updating = False
        self.apply()

    def apply(self):
        try:
            self.view.set_image(self.coder.apply())
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Apply: %r" % e)

    def show_initial_image(self):
        """Applies the default quantization, then runs the survey in the
        background with the menus disabled so Apply and Save can't overlap it."""
        self.apply()
        self.view.set_menus_enabled(False)
        self.view.set_status("analyzing…")

        def done(result, error):
            if error is not None:
                print("init: %r" % error)
            self.view.set_menus_enabled(True)
            self.view.set_status(None)
            self.delta_actions[self.coder.delta_type].setChecked(True)
            self.apply()
        vs.run_in_background(self.coder.survey, done)

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
        SimpleWriter(sys.argv[1])
    else:
        open_image()
    if not _open_writers:
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
