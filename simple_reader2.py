#!/usr/bin/env python3
"""
simple_reader2.py version 1.0 -- SimpleReader2.java for Python (PySide6).

Reads Simple 2 format 'F' version 1, from simple_writer2.py or
SimpleWriter2.java: the channels green, blue-green and red-green
(simple_writer2.SET_ID), every interior pixel
predicted by the least-squares fit (the wide clamp, window radius
simple_writer2.RADIUS), the deltas mixed-context coded with the second
opinions, in bands of rows decoded at the same time, resized back with
resize_mapper.resize2 (centred). See simple_writer2.encode for the layout.

    python3 simple_reader2.py file

The decoding (class SimpleDecoder2) is separate from the window:

    rgb = SimpleDecoder2("foo").decode()      # (ydim, xdim, 3) uint8, R, G, B
"""

import sys

import numpy as np

import delta_mapper as dm
import resize_mapper as rm
import simple_writer as sw
import simple_writer2 as sw2
import viewer_support as vs
from java_io import DataInput

FORMAT_ID = ord('F')
FORMAT_VERSION = 1


class SimpleDecoder2:
    def __init__(self, filename):
        self.filename = filename
        timer = vs.Timer()
        with open(filename, "rb") as f:
            self._read(DataInput(f))
        print("File read in " + timer.elapsed() + ".")

    def _read(self, inp):
        ident, version = inp.read_unsigned_byte(), inp.read_unsigned_byte()
        if ident != FORMAT_ID:
            raise IOError(self.filename + " is not a Simple Writer 2 file.")
        if version != FORMAT_VERSION:
            raise IOError("%s is Simple 2 format version %d; this reader reads version %d only." % (self.filename, version, FORMAT_VERSION))
        self.xdim = inp.read_unsigned_short()
        self.ydim = inp.read_unsigned_short()
        self.pixel_shift, self.pixel_quant = inp.read_byte(), inp.read_byte()
        self.min = [inp.read_int() for _ in range(3)]
        self.size = dm.get_quantized_size(self.xdim, self.ydim, self.pixel_quant)
        w, h = self.size
        n = self.bands = inp.read_unsigned_byte()
        if n < 1 or h // n < 2:
            raise IOError("%s: %d bands of a %d-row image." % (self.filename, n, h))
        length = [inp.read_int() for _ in range(n)]
        self.band_init, self.band_delta = [], []
        for b in range(n):
            bi = DataInput(inp.read_fully(length[b]))
            init, deltas = [], []
            for i in range(3):
                init.append(bi.read_int())
                deltas.append(dm.read_mixed_deltas(bi))
            self.band_init.append(init)
            self.band_delta.append(deltas)
        print("Image:        %d x %d" % (self.xdim, self.ydim))
        if n > 1:
            print(sw.describe_bands(n, h))
        print()

    def decode(self):
        """The image as an (ydim, xdim, 3) uint8 array, R, G, B: the bands
        decoded at the same time (each band's channels at the same time too,
        each a few rows behind the one before: a channel's fit and contexts
        use the ones before it), then blue, green and red rebuilt from them,
        resized and shifted."""
        timer = vs.Timer()
        w, h = self.size
        n = self.bands
        radius = [sw2.RADIUS] * 3
        raw = [np.zeros(w * h, dtype=np.int64) for _ in range(3)]

        def one(b):
            y0, y1 = sw.band_start(b, n, h), sw.band_start(b + 1, n, h)
            v = dm.get_values_from_least_squares_channels(self.band_delta[b], w, y1 - y0, self.band_init[b], radius)[1]
            for i in range(3):
                raw[i][y0 * w:y1 * w] = v[i]
        vs.parallel(n, one)
        channel = [raw[i] + self.min[i] if sw2.CHANNELS[i] > 2 else raw[i] for i in range(3)]
        bgr = dm.get_blue_green_red(sw2.SET_ID, *channel)
        if self.pixel_quant != 0:
            def resize(c):
                bgr[c] = rm.resize2(bgr[c], w, self.xdim, self.ydim)
            vs.parallel(3, resize)
        rgb = dm.pack_image(bgr, self.xdim, self.ydim, self.pixel_shift)
        print("Image rebuilt in " + timer.elapsed() + ".")
        return rgb


class SimpleReader2:
    def __init__(self, filename):
        try:
            self.decoder = SimpleDecoder2(filename)
        except Exception as e:
            message = str(e) if isinstance(e, (IOError, EOFError)) and str(e) else "Can't decode %s: %r" % (filename, e)
            vs.show_error(None, message)
            self.view = None
            return
        d = self.decoder
        self.view = vs.ImageWindow("Simple Reader 2  " + filename, d.xdim, d.ydim)
        self.view.make_view_menu()
        self.view.set_status("decoding…")
        self.view.show_window()
        vs.run_in_background(d.decode, self.done)

    def done(self, rgb, error):
        if error is not None:
            self.view.set_status("decode failed")
            vs.show_error(self.view, "Can't decode %s: %r" % (self.decoder.filename, error))
            return
        self.view.set_image(rgb)
        self.view.set_status(None)
        self.view.fit_and_shrink()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 simple_reader2.py <filename>")
        sys.exit(0)
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    reader = SimpleReader2(sys.argv[1])
    if reader.view is None:
        sys.exit(1)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
