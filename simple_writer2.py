#!/usr/bin/env python3
"""
simple_writer2.py version 1.0 -- SimpleWriter2.java for Python (PySide6).

simple_writer.py without the analysis, for speed: one channel set, green,
blue-green and red-green (SET_ID; the set the analysis picks most often, and
within about 1% of its choice on the test images), every interior pixel
predicted by the least-squares fit (the wide clamp) with one window radius,
RADIUS, for every channel and image, solved at every EVERY-th pixel along a
row, and the deltas mixed-context coded with the second opinions
(delta_mapper.pack_mixed_deltas). As in simple_writer.py, a channel's fit
uses the channel before it and its contexts the deltas of the channels
before it. Nothing is fitted or coded until Save. Pixel Resolution resizes
with resize_mapper.resize2 (centred). The Bands menu splits the image into up
to MAX_BANDS bands of rows that a reader can decode at the same time, for a
small cost per band boundary. Saves (to the file "foo", as the Java version
does) in Simple 2 format 'F' version 1, read by simple_reader2.py and
SimpleReader2.java.

    python3 simple_writer2.py [image]

The coding (class SimpleCoder2) is separate from the window:

    coder = SimpleCoder2(viewer_support.read_image("photo.png"))
    coder.apply(); coder.save("photo.sm2")
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import delta_mapper as dm
import resize_mapper as rm
import viewer_support as vs
from java_io import DataOutput
from simple_writer import band_start, describe_bands

FORMAT_ID = ord('F')
FORMAT_VERSION = 1
SET_ID = 7             # the channel set coded: green, blue-green, red-green (delta_mapper.get_channels)
CHANNELS = tuple(dm.get_channels(SET_ID))   # its channels, in coding order (delta_mapper's channel ids)
RADIUS = 10            # the least-squares window radius, for every channel
EVERY = 2              # how often along a row the fit is solved: at every EVERY-th pixel, the pixels between using the last solution
MAX_BANDS = 16         # the most bands Save offers
BAND_MIN_ROWS = 64     # the fewest rows (of the quantized image) a band may have


def encode(value, mins, xdim, image_xdim, image_ydim, pixel_shift, pixel_quant, n):
    """Codes the channels (as quantized, in coding order, difference
    channels less their minimums mins) and returns the file's bytes
    (SimpleWriter2.encode). Format 1. Header: FORMAT_ID, FORMAT_VERSION,
    width, height (unsigned shorts), pixel_shift, pixel_quant; per channel
    (coding order) int min; the number of bands n (an unsigned byte), n int
    lengths, and the n bands. Band b is rows band_start(b) to
    band_start(b + 1) of the quantized image, coded as an image of its own
    (the fit and the contexts start again at its top): per channel (green,
    blue-green, red-green), int init, then the least-squares deltas (radius
    RADIUS, solved at every EVERY-th pixel along a row) mixed-context coded
    with the second opinions (pack_mixed_deltas). Channels 1 and 2 use the ones before
    them (their values in the fit, their deltas as context), so a band's
    channels are decoded in order, each a few rows behind the one before; the
    bands can be decoded at the same time.

    Each channel is fitted once over the whole image (the fit splits itself
    across the cores); the bands' deltas come from it, with only each band's
    first rows fitted again (delta_mapper.get_least_squares_band_deltas). A
    channel's coding needs only the deltas of the channels before it, so its
    bands are coded (and checked by decoding them back) while the next
    channel is fitted."""
    value = [np.asarray(v, dtype=np.int64) for v in value]
    rows = len(value[0]) // xdim
    d = [[None] * 3 for _ in range(n)]          # [band][channel]
    coded = [[None] * 3 for _ in range(n)]

    def code(b, i):
        y0, y1 = band_start(b, n, rows), band_start(b + 1, n, rows)
        ch = value[i][y0 * xdim:y1 * xdim]
        dd = d[b][i]
        previous = d[b][:i]
        coded[b][i] = dm.pack_mixed_deltas(dd, ch, previous, xdim, True)
        back = dm.unpack_mixed_deltas(coded[b][i], ch, previous, xdim, ch - np.asarray(dd, dtype=np.int64))
        if not np.array_equal(back, dd):
            print("WARNING: band %d channel %d deltas do not decode back." % (b, i))

    jobs = []
    with ThreadPoolExecutor(max(1, min(3 * n, os.cpu_count() or 1))) as coder:
        try:
            for i in range(3):
                ref = value[i - 1] if i > 0 else None
                whole = dm.get_least_squares_deltas(value[i], ref, xdim, rows, RADIUS, EVERY)
                for b in range(n):
                    y0, y1 = band_start(b, n, rows), band_start(b + 1, n, rows)
                    d[b][i] = whole if n == 1 else dm.get_least_squares_band_deltas(whole, value[i], ref, xdim, y0, y1, RADIUS, EVERY)
                jobs += [coder.submit(code, b, i) for b in range(n)]
        finally:
            for j in jobs:
                j.result()

    out = DataOutput()
    out.write_byte(FORMAT_ID); out.write_byte(FORMAT_VERSION)
    out.write_short(image_xdim); out.write_short(image_ydim)
    out.write_byte(pixel_shift); out.write_byte(pixel_quant)
    for i in range(3):
        out.write_int(int(mins[i]))
    out.write_byte(n)
    band = []
    for b in range(n):
        y0 = band_start(b, n, rows)
        bo = DataOutput()
        for i in range(3):
            bo.write_int(int(value[i][y0 * xdim]))
            bo.write(coded[b][i])
        band.append(bo.to_bytes())
    for bb in band:
        out.write_int(len(bb))
    for bb in band:
        out.write(bb)
    return out.to_bytes()


# =============================================================================
# The coding, without the window
# =============================================================================

class SimpleCoder2:
    """Holds an image and the quantization; apply() quantizes the channels
    (and makes the preview), save() codes and writes the file."""

    def __init__(self, rgb, file_length=0):
        self.image_ydim, self.image_xdim = rgb.shape[:2]
        self.source = [rgb[:, :, c].reshape(-1).astype(np.int64) for c in range(3)]
        self.file_length = file_length

        self.pixel_quant = 4
        self.pixel_shift = 3
        self.correction = 0         # preview only
        self.bands = 1              # independently decodable bands Save writes
        self.channel_min = [0] * 3  # each channel's minimum, in coding order (added back to the difference channels by the reader)
        self.applied = False

    def quantized_channels(self, size):
        """Blue, green and red, resized and quantized."""
        q = [None] * 3

        def one(i):
            ch = self.source[i]
            if self.pixel_quant != 0:
                ch = rm.resize2(ch, self.image_xdim, size[0], size[1])
            q[i] = np.asarray(dm.quantize_channel(ch, self.pixel_shift), dtype=np.int64)
        vs.parallel(3, one)
        return q

    def bands_used(self):
        """The bands Save will write: bands, but no more than leave
        BAND_MIN_ROWS rows in each."""
        rows = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)[1]
        return max(1, min(self.bands, rows // BAND_MIN_ROWS))

    def band_text(self):
        """The Bands dialog's status line."""
        n = self.bands_used()
        if n == 1:
            return "1 band: the image is too small for more" if self.bands > 1 else "1 band"
        return describe_bands(n, dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)[1])

    def apply(self):
        """Quantizes (the channels Save codes), then rebuilds the image for
        the preview. Returns the preview as an (ydim, xdim, 3) uint8 array."""
        self.applied = False
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        w = size[0]
        q = self.quantized_channels(size)
        qc, mins = dm.get_candidate_channels(q[0], q[1], q[2])
        self.value_list = [qc[j] for j in CHANNELS]
        self.channel_min = [mins[j] for j in CHANNELS]
        self.delta_xdim = w

        # As the reader: resize (centred: resize2), then shift. The reader
        # rebuilds blue, green and red exactly as quantized, so the preview
        # uses them as they are: no need to decode.
        bgr = list(q)

        def restore(c):
            v = bgr[c]
            if self.pixel_quant != 0:
                v = rm.resize2(v, w, self.image_xdim, self.image_ydim)
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

    def encode(self):
        """The file's bytes (encode)."""
        return encode(self.value_list, self.channel_min, self.delta_xdim, self.image_xdim, self.image_ydim, self.pixel_shift, self.pixel_quant, self.bands_used())

    def save(self, filename="foo"):
        """Codes the image (encode), twice, and writes it; prints both times
        (the first includes compiling the code: Numba's, the first time it
        runs, or loading it from the cache). Returns the file size."""
        took = []
        data = None
        for _ in range(2):
            start = time.perf_counter()
            got = self.encode()
            took.append(time.perf_counter() - start)
            if data is not None and got != data:
                print("WARNING: the second coding differs from the first.")
            data = got
        with open(filename, "wb") as f:
            f.write(data)
        n = self.bands_used()
        raw = self.image_xdim * self.image_ydim * 3
        w = self.delta_xdim
        rows = len(self.value_list[0]) // w
        if n > 1:
            print("Saved %s: %s bytes (%s)" % (os.path.basename(filename), format(len(data), ","), describe_bands(n, rows)))
        else:
            print("Saved %s: %s bytes" % (os.path.basename(filename), format(len(data), ",")))
        print("Coded %d x %d (%s pixels)" % (w, rows, format(w * rows, ",")))
        print("Run time plus compile time: " + vs.format_duration(took[0]))
        print("Run time only:              " + vs.format_duration(took[1]))
        print("Original compression rate: %.4f" % (self.file_length / raw))
        print("Output  compression rate:  %.4f" % (len(data) / raw))
        print()
        return len(data)


# =============================================================================
# The window
# =============================================================================

def open_image(parent=None):
    from PySide6.QtWidgets import QFileDialog
    path, _ = QFileDialog.getOpenFileName(parent, "Open Image", "",
                                          "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pgm *.ppm);;All files (*)")
    if path:
        SimpleWriter2(path)


_open_writers = []


class SimpleWriter2:
    def __init__(self, filename):
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QAction, QKeySequence

        self.filename = filename
        try:
            rgb = vs.read_image(filename)
        except IOError as e:
            vs.show_error(None, str(e))
            return
        self.coder = c = SimpleCoder2(rgb, os.path.getsize(filename))
        self.updating = False
        print("Loaded %s, %d x %d (%s pixels)" % (filename, c.image_xdim, c.image_ydim, format(c.image_xdim * c.image_ydim, ",")))

        self.view = view = vs.ImageWindow("Simple Writer 2  " + filename, c.image_xdim, c.image_ydim)
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
        SimpleWriter2(sys.argv[1])
    else:
        open_image()
    if not _open_writers:
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
