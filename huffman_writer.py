#!/usr/bin/env python3
"""
huffman_writer.py version 1.0 -- HuffmanWriter.java for Python (PySide6).

simple_writer2.py up to the entropy coding: the same channels (green,
blue-green and red-green, SET_ID), the same least-squares fit from the rows
above (delta_mapper's _rf_*, DeltaMapper.RowFit: the window is the RADIUS
rows above, solved at every EVERY-th pixel along a row), the same bands,
Apply and Save. Then, in place of the mixed context coding, each band's
channel's deltas are coded as delta_writer2.py codes them: unary strings by
frequency rank (string_mapper.get_string_list), with string_mapper's
bit-pair transform kept only if it saves enough (its rule), then a canonical
Huffman code over the string's bytes, one per channel and band (code_mapper's
regular Huffman code, delta_writer's entropy type 1). No contexts, so a
channel's coding uses no other channel. Less compression than Simple 2 (on
the test photos the files are 9-29% bigger), faster to decode. Saves (to
the file "foo", as the Java version does) in Huffman format 'H' version 1,
read by huffman_reader.py and HuffmanReader.java.

    python3 huffman_writer.py [image]

The coding (class HuffmanCoder) is separate from the window:

    coder = HuffmanCoder(viewer_support.read_image("photo.png"))
    coder.apply(); coder.save("photo.huf")
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import arithmetic_mapper as am
import code_mapper as cm
import delta_mapper as dm
import string_mapper as sm
import viewer_support as vs
from java_io import DataInput, DataOutput
from simple_writer import band_start
# Simple 2's channels, fit and bands (huffman_reader reads SET_ID, CHANNELS, RADIUS and EVERY from here).
from simple_writer2 import SET_ID, CHANNELS, RADIUS, EVERY, MAX_BANDS, SimpleCoder2  # noqa: F401

FORMAT_ID = ord('H')
FORMAT_VERSION = 1     # 1: the first


def encode(value, mins, xdim, image_xdim, image_ydim, pixel_shift, pixel_quant, n):
    """Codes the channels (as quantized, in coding order, difference
    channels less their minimums mins) and returns (the file's bytes, the
    seconds the decode check took) (HuffmanWriter.encode). Format 1. Header: FORMAT_ID, FORMAT_VERSION,
    width, height (unsigned shorts), pixel_shift, pixel_quant; per channel
    (coding order) int min; the number of bands n (an unsigned byte), n int
    lengths, and the n bands. Band b is rows band_start(b) to
    band_start(b + 1) of the quantized image, coded as an image of its own
    (the fit starts again at its top): per channel (green, blue-green,
    red-green), int init, then the deltas of the least-squares fit from the
    rows above (radius RADIUS, solved at every EVERY-th pixel along a row)
    Huffman coded (huffman_code, see there for the layout). Channels 1 and 2
    use the ones before them in the fit (their values), so a band's channels
    are rebuilt in order, each a row behind the one before; the bands can be
    decoded at the same time.

    Each channel is fitted once over the whole image (the fit splits itself
    across the cores); the bands' deltas come from it, with only each band's
    first rows fitted again (delta_mapper.get_row_fit_band_deltas). A
    channel's coding needs only its own deltas, so its bands are coded while
    the next channel is fitted. Once everything is
    coded, each band's channels are checked by decoding them back, at the
    same time; the time that took is returned, so the coding can be timed
    without it."""
    value = [np.asarray(v, dtype=np.int64) for v in value]
    rows = len(value[0]) // xdim
    d = [[None] * 3 for _ in range(n)]          # [band][channel]
    coded = [[None] * 3 for _ in range(n)]

    def code(b, i):
        coded[b][i] = huffman_code(d[b][i])

    def check(b, i):
        dd = np.asarray(d[b][i], dtype=np.int64)
        back = huffman_decode(DataInput(coded[b][i]), len(dd))
        # (Delta 0 isn't coded: the band's first value, init, is written instead.)
        if len(back) != len(dd) or not np.array_equal(back[1:], dd[1:]):
            print("WARNING: band %d channel %d deltas do not decode back." % (b, i))

    jobs = []
    with ThreadPoolExecutor(max(1, min(3 * n, os.cpu_count() or 1))) as coder:
        try:
            for i in range(3):
                ref = value[i - 1] if i > 0 else None
                whole = dm.get_row_fit_deltas(value[i], ref, xdim, rows, RADIUS, EVERY)
                for b in range(n):
                    y0, y1 = band_start(b, n, rows), band_start(b + 1, n, rows)
                    d[b][i] = whole if n == 1 else dm.get_row_fit_band_deltas(whole, value[i], ref, xdim, y0, y1, RADIUS, EVERY)
                jobs += [coder.submit(code, b, i) for b in range(n)]
        finally:
            for j in jobs:
                j.result()
        # The check, once everything is coded (so its time can be told apart).
        start = time.perf_counter()
        for j in [coder.submit(check, b, i) for b in range(n) for i in range(3)]:
            j.result()
        check_time = time.perf_counter() - start

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
    return out.to_bytes(), check_time


def huffman_code(delta):
    """A channel's deltas Huffman coded, as delta_writer2.py codes them:
    unary strings by frequency rank (string_mapper.get_string_list, the
    bit-pair transform kept only if it saves enough), then a canonical
    Huffman code over the string's bytes (code_mapper's regular Huffman
    code) (HuffmanWriter.huffmanCode). Layout: int the deltas' minimum, int
    the unary string's bit length, its rank table (delta_mapper.write_table),
    int the (transformed) string's byte length, int and the Huffman code
    lengths (code_mapper.pack_regular_tables), int and the code. Delta 0
    isn't used (the band's first value is written instead)."""
    lo, bits, table, string = sm.get_string_list(delta, True)
    lengths = cm.get_regular_huffman_length(am.get_frequency(string))
    code = cm.pack_regular_code(string, lengths)
    tables = cm.pack_regular_tables([lengths], 9)
    out = DataOutput()
    out.write_int(int(lo))
    out.write_int(int(bits))
    dm.write_table(out, table)
    out.write_int(len(string))
    out.write_int(len(tables)); out.write(tables)
    out.write_int(len(code));   out.write(bytes(code))
    return out.to_bytes()


class HuffmanChannel:
    """A channel's Huffman coding as huffman_code wrote it, read by
    huffman_read, its code lengths inflated (HuffmanWriter.HuffmanChannel)."""
    __slots__ = ("min", "bits", "length", "table", "lengths", "code")


def huffman_read(inp):
    """Reads a channel's coding huffman_code wrote (HuffmanWriter.huffmanRead)."""
    h = HuffmanChannel()
    h.min = inp.read_int()
    h.bits = inp.read_int()
    h.table = dm.read_table(inp)
    h.length = inp.read_int()
    tables = inp.read_fully(inp.read_int())
    h.code = inp.read_fully(inp.read_int())
    h.lengths = cm.unpack_regular_tables(tables, 1)[0]
    return h


def huffman_decode(inp, n):
    """Reads a channel's deltas huffman_code wrote: n deltas, delta 0 being
    0 (HuffmanWriter.huffmanDecode). inp is a DataInput at their start, or
    a HuffmanChannel already read."""
    h = inp if isinstance(inp, HuffmanChannel) else huffman_read(inp)
    string = cm.unpack_regular_code(h.code, h.lengths, h.length)
    delta = np.asarray(sm.unpack_strings(sm.decompress_strings(string), h.table, n, h.bits), dtype=np.int64)
    delta[0] = 0
    delta[1:] += h.min
    return delta


# =============================================================================
# The coding, without the window
# =============================================================================

class HuffmanCoder(SimpleCoder2):
    """SimpleCoder2 with the Huffman coding: apply() quantizes the channels
    (and makes the preview), as SimpleCoder2's does; save() codes (this
    module's encode) and writes the file."""

    def encode(self):
        """(The file's bytes, the seconds the decode check took) (encode)."""
        return encode(self.value_list, self.channel_min, self.delta_xdim, self.image_xdim, self.image_ydim, self.pixel_shift, self.pixel_quant, self.bands_used())


# =============================================================================
# The window
# =============================================================================

def open_image(parent=None):
    from PySide6.QtWidgets import QFileDialog
    path, _ = QFileDialog.getOpenFileName(parent, "Open Image", "",
                                          "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pgm *.ppm);;All files (*)")
    if path:
        HuffmanWriter(path)


_open_writers = []


class HuffmanWriter:
    def __init__(self, filename):
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QAction, QKeySequence

        self.filename = filename
        try:
            rgb = vs.read_image(filename)
        except IOError as e:
            vs.show_error(None, str(e))
            return
        self.coder = c = HuffmanCoder(rgb, os.path.getsize(filename))
        self.updating = False
        print("Loaded %s, %d x %d (%s pixels)" % (filename, c.image_xdim, c.image_ydim, format(c.image_xdim * c.image_ydim, ",")))

        self.view = view = vs.ImageWindow("Huffman Writer  " + filename, c.image_xdim, c.image_ydim)
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
        HuffmanWriter(sys.argv[1])
    else:
        open_image()
    if not _open_writers:
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
