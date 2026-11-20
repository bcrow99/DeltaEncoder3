#!/usr/bin/env python3
"""
generic_writer.py version 1.0 -- GenericWriter.java for Python (PySide6).

simple_writer2.py and huffman_writer.py in one, with a choice of deltas: the
same image, the same Apply and the same channels (green, blue-green and
red-green, SET_ID), with the deltas chosen by the Delta menu and the coding by
the Entropy menu. Nothing is coded until Save. Pixel Resolution resizes with
resize_mapper.resize2 (centred). The Bands menu splits the image into up to
MAX_BANDS bands of rows that a reader can decode at the same time, for a small
cost per band boundary.

The Delta menu chooses the deltas:
  LS:  every interior pixel predicted by the least-squares fit from the rows
       above (delta_mapper's _rf_*, DeltaMapper.RowFit: the window is the
       RADIUS rows above, solved at every EVERY-th pixel along a row; a
       channel's fit uses the channel before it). The default.
  Avg: the classic average filter (delta_mapper.get_deltas' type 2: each
       pixel from (left + above) / 2), no fit.

The Entropy menu chooses the coding:
  Context aware: the deltas context-aware coded with the second opinions
       (delta_mapper.pack_mixed_deltas; a channel's contexts use the deltas
       of the channels before it). Simple 2 format 'F': version 2 with LS,
       the same files as simple_writer2.py; 3 with Avg (the delta type after
       the Pixel Resolution). The default.
  Huffman: each channel's deltas as huffman_writer.py codes them
       (huffman_writer.huffman_code); no contexts, so a channel's coding uses
       no other channel. Huffman format 'H': version 1 with LS, the same files
       as huffman_writer.py; 2 with Avg. Bigger files, faster to decode.

The files are GenericWriter.java's (and NativeEncoder's NativeWriter's),
byte for byte, read by generic_reader.py and GenericReader.java (versions 2
and 1 by simple_reader2.py and huffman_reader.py too). Saves to the output
file given, or "foo", as the Java version does.

    python3 generic_writer.py [image [output]]

The coding (class GenericCoder) is separate from the window:

    coder = GenericCoder(viewer_support.read_image("photo.png"))
    coder.delta_type = DELTA_AVERAGE; coder.apply(); coder.save("photo.gen")
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import delta_mapper as dm
import huffman_writer as hw
import viewer_support as vs
from java_io import DataInput, DataOutput
from simple_writer import band_start
from simple_writer2 import SET_ID, CHANNELS, RADIUS, EVERY, SimpleCoder2  # noqa: F401  (SET_ID and CHANNELS: generic_reader reads them from here)

SIMPLE_ID = ord('F')
SIMPLE_VERSION = 2              # the Simple 2 format with the LS deltas: simple_writer2.py's
SIMPLE_VERSION_DELTA_TYPE = 3   # version 2 with a byte, the delta type, after the Pixel Resolution: written for the average deltas
HUFFMAN_ID = ord('H')
HUFFMAN_VERSION = 1             # the Huffman format with the LS deltas: huffman_writer.py's
HUFFMAN_VERSION_DELTA_TYPE = 2  # version 1 with the delta type, as above
ENTROPY_MIXED, ENTROPY_HUFFMAN = 0, 1
ENTROPY_NAMES = ("Context aware", "Huffman")
DELTA_LS, DELTA_AVERAGE = 0, 1
DELTA_NAMES = ("LS", "Avg")
MAX_BANDS = 64                  # the most bands Save offers (GenericWriter.MAX_BANDS)


def encode(value, mins, xdim, image_xdim, image_ydim, pixel_shift, pixel_quant, n, delta_type, entropy_type, check=False):
    """Codes the channels (as quantized, in coding order, difference
    channels less their minimums mins) and returns (the file's bytes, the
    seconds the decode check took) (GenericWriter.encode). Header: the
    format's id and version (SIMPLE_ID or HUFFMAN_ID; the version for LS, or
    the one with the delta type), width, height (unsigned shorts),
    pixel_shift, pixel_quant, the delta type (an unsigned byte, only in the
    versions with it); per channel (coding order) int min; the number of
    bands n (an unsigned byte), n int lengths, and the n bands. Band b is rows
    band_start(b) to band_start(b + 1) of the quantized image, coded as an
    image of its own (the fit and the contexts start again at its top): per
    channel, int init, then its deltas coded: context aware
    (pack_mixed_deltas, with the second opinions) or Huffman
    (huffman_writer.huffman_code).

    With LS, each channel is fitted once over the whole image; the bands'
    deltas come from it, with only each band's first rows fitted again
    (delta_mapper.get_row_fit_band_deltas). The average deltas are found a
    band at a time. A channel's coding needs only the deltas of the channels
    before it (context aware) or its own (Huffman), so its bands are coded
    while the next channel's deltas are found. Once everything is coded, if
    check is set, each band's channels are checked by decoding them back, at
    the same time; the time that took is returned (0 without it), so the
    coding can be timed without it."""
    huffman = entropy_type == ENTROPY_HUFFMAN
    value = [np.asarray(v, dtype=np.int64) for v in value]
    rows = len(value[0]) // xdim
    d = [[None] * 3 for _ in range(n)]          # [band][channel]
    coded = [[None] * 3 for _ in range(n)]

    def band(b, i):
        y0, y1 = band_start(b, n, rows), band_start(b + 1, n, rows)
        return value[i][y0 * xdim:y1 * xdim]

    def code(b, i):
        if huffman:
            coded[b][i] = hw.huffman_code(d[b][i])
        else:
            coded[b][i] = dm.pack_mixed_deltas(d[b][i], band(b, i), d[b][:i], xdim, True)

    def check_band(b, i):
        dd = np.asarray(d[b][i], dtype=np.int64)
        if huffman:
            back = hw.huffman_decode(DataInput(coded[b][i]), len(dd))
            # (Delta 0 isn't coded: the band's first value, init, is written instead.)
            same = len(back) == len(dd) and np.array_equal(back[1:], dd[1:])
        else:
            ch = band(b, i)
            back = dm.unpack_mixed_deltas(coded[b][i], ch, d[b][:i], xdim, ch - dd)
            same = np.array_equal(back, dd)
        if not same:
            print("WARNING: band %d channel %d deltas do not decode back." % (b, i))

    jobs = []
    with ThreadPoolExecutor(max(1, min(3 * n, os.cpu_count() or 1))) as coder:
        try:
            for i in range(3):
                if delta_type == DELTA_LS:
                    ref = value[i - 1] if i > 0 else None
                    whole = dm.get_row_fit_deltas(value[i], ref, xdim, rows, RADIUS, EVERY)
                    for b in range(n):
                        y0, y1 = band_start(b, n, rows), band_start(b + 1, n, rows)
                        d[b][i] = whole if n == 1 else dm.get_row_fit_band_deltas(whole, value[i], ref, xdim, y0, y1, RADIUS, EVERY)
                else:
                    for b in range(n):
                        y0, y1 = band_start(b, n, rows), band_start(b + 1, n, rows)
                        d[b][i] = dm.get_deltas(band(b, i), xdim, y1 - y0, 2)[0]
                jobs += [coder.submit(code, b, i) for b in range(n)]
        finally:
            for j in jobs:
                j.result()
        # The check (if asked for), once everything is coded (so its time can
        # be told apart).
        check_time = 0.0
        if check:
            start = time.perf_counter()
            for j in [coder.submit(check_band, b, i) for b in range(n) for i in range(3)]:
                j.result()
            check_time = time.perf_counter() - start

    out = DataOutput()
    if huffman:
        out.write_byte(HUFFMAN_ID); out.write_byte(HUFFMAN_VERSION if delta_type == DELTA_LS else HUFFMAN_VERSION_DELTA_TYPE)
    else:
        out.write_byte(SIMPLE_ID); out.write_byte(SIMPLE_VERSION if delta_type == DELTA_LS else SIMPLE_VERSION_DELTA_TYPE)
    out.write_short(image_xdim); out.write_short(image_ydim)
    out.write_byte(pixel_shift); out.write_byte(pixel_quant)
    if delta_type != DELTA_LS:
        out.write_byte(delta_type)
    for i in range(3):
        out.write_int(int(mins[i]))
    out.write_byte(n)
    bands = []
    for b in range(n):
        y0 = band_start(b, n, rows)
        bo = DataOutput()
        for i in range(3):
            bo.write_int(int(value[i][y0 * xdim]))
            bo.write(coded[b][i])
        bands.append(bo.to_bytes())
    for bb in bands:
        out.write_int(len(bb))
    for bb in bands:
        out.write(bb)
    return out.to_bytes(), check_time


# =============================================================================
# The coding, without the window
# =============================================================================

class GenericCoder(SimpleCoder2):
    """SimpleCoder2 with a choice of deltas and coding: apply() quantizes the
    channels (and makes the preview), as SimpleCoder2's does; save() codes
    (this module's encode) and writes the file."""

    def __init__(self, rgb, file_length=0):
        super().__init__(rgb, file_length)
        self.delta_type = DELTA_LS          # the Delta menu
        self.entropy_type = ENTROPY_MIXED   # the Entropy menu

    def encode(self):
        """(The file's bytes, the seconds the decode check took) (encode)."""
        return encode(self.value_list, self.channel_min, self.delta_xdim, self.image_xdim, self.image_ydim, self.pixel_shift, self.pixel_quant,
                      self.bands_used(), self.delta_type, self.entropy_type, self.decode_check)

    def timing_key(self):
        """The coding's name, for print_run_times: each choice of deltas and
        coding is code of its own, compiled the first time it runs."""
        return "GenericCoder %s %s" % (DELTA_NAMES[self.delta_type], ENTROPY_NAMES[self.entropy_type])

    def save(self, filename="foo"):
        """As SimpleCoder2.save, saying the deltas and coding first."""
        print("%s deltas, %s" % (DELTA_NAMES[self.delta_type], ENTROPY_NAMES[self.entropy_type]))
        return super().save(filename)


# =============================================================================
# The window
# =============================================================================

def open_image(parent=None):
    from PySide6.QtWidgets import QFileDialog
    path, _ = QFileDialog.getOpenFileName(parent, "Open Image", "",
                                          "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pgm *.ppm);;All files (*)")
    if path:
        GenericWriter(path)


_open_writers = []


class GenericWriter:
    def __init__(self, filename, output_file=vs.DEFAULT_OUTPUT):
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QAction, QActionGroup, QKeySequence

        self.filename = filename
        self.output_file = output_file   # the file Save writes
        try:
            rgb = vs.read_image(filename)
        except IOError as e:
            vs.show_error(None, str(e))
            return
        self.coder = c = GenericCoder(rgb, os.path.getsize(filename))
        self.updating = False
        print("Loaded %s, %d x %d (%s pixels)" % (filename, c.image_xdim, c.image_ydim, format(c.image_xdim * c.image_ydim, ",")))

        self.view = view = vs.ImageWindow("Generic Writer  " + filename, c.image_xdim, c.image_ydim)
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
        # Decode Check: Save also decodes what it coded and says whether it
        # comes back the same (off unless switched on: Save is faster without
        # it; if a file doesn't decode right, it tells whether the writer or
        # the reader is at fault).
        a = QAction("Decode Check", view, checkable=True)
        a.setChecked(c.decode_check)
        a.toggled.connect(lambda on: setattr(c, "decode_check", on))
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

        # Delta and Entropy menus: the deltas and the coding. Like Save itself,
        # they never change the preview.
        for title, names, key in (("Delta", DELTA_NAMES, "delta_type"), ("Entropy", ENTROPY_NAMES, "entropy_type")):
            menu = bar.addMenu(title)
            group = QActionGroup(view)
            group.setExclusive(True)
            for t, name in enumerate(names):
                a = QAction(name, view, checkable=True)
                a.setChecked(t == getattr(c, key))
                a.triggered.connect(lambda _=False, key=key, t=t: setattr(c, key, t))
                group.addAction(a)
                menu.addAction(a)

        # Bands menu: Save codes the image as this many bands of rows, each
        # decodable on its own (so a reader can decode them at the same time),
        # for a small cost per band boundary. Like Save itself, it never
        # changes the preview.
        bands_menu = bar.addMenu("Bands")
        act, self.bands_slider = vs.make_slider_dialog(view, "Number of Bands", 1, MAX_BANDS, c.bands, self._set_bands)
        bands_menu.addAction(act)

        view.set_image(rgb)
        view.show_window()
        QTimer.singleShot(0, self.apply)

    def _setter(self, key):
        """A Quantization slider's change: applies again."""
        def set_value(v):
            setattr(self.coder, key, v)
            if not self.updating:
                self.apply()
        return set_value

    def _set_bands(self, v):
        self.coder.bands = v
        self.view.set_status(self.coder.band_text())

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

    def save(self):
        if not self.coder.applied:
            self.apply()
        if not self.coder.applied:
            vs.show_error(self.view, "Nothing saved: the last Apply failed.")
            return
        try:
            self.coder.save(self.output_file)
        except Exception as e:
            import traceback
            traceback.print_exc()
            if os.path.exists(self.output_file):
                os.remove(self.output_file)
            vs.show_error(self.view, "Save failed: %r" % e)


def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    if len(sys.argv) > 1:
        # The image, and the file Save writes ("foo" if none is given).
        GenericWriter(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else vs.DEFAULT_OUTPUT)
    else:
        open_image()
    if not _open_writers:
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
