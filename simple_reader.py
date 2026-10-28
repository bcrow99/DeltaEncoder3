#!/usr/bin/env python3
"""
simple_reader.py version 3.2 -- SimpleReader.java for Python (PySide6).

Reads Simple format 'S' version 8, from simple_writer.py or
SimpleWriter.java: every interior pixel predicted by the least-squares fit
(the wide clamp) with each channel's window radius, the deltas mixed-context
coded with the second opinions, in bands of rows decoded at the same time,
resized back with resize_mapper.resize2 (centred). See
simple_writer.SimpleCoder.encode for the layout. Older Simple files (formats 3 to 7) are not read.

    python3 simple_reader.py file

The decoding (class SimpleDecoder) is separate from the window:

    rgb = SimpleDecoder("foo").decode()      # (ydim, xdim, 3) uint8, R, G, B
"""

import sys

import numpy as np

import delta_mapper as dm
import resize_mapper as rm
import simple_writer as sw
import viewer_support as vs
from java_io import DataInput

FORMAT_ID = ord('S')
FORMAT_VERSION = 8   # 6: least squares alone, a window radius and clamp per channel, in bands; 7: 6 with centred resizing (resize_mapper.resize2); 8: 7 with the wide clamp only (no clamp byte)


class SimpleDecoder:
    def __init__(self, filename):
        self.filename = filename
        timer = vs.Timer()
        with open(filename, "rb") as f:
            self._read(DataInput(f))
        print("File read in " + timer.elapsed() + ".")

    def _read(self, inp):
        ident, version = inp.read_unsigned_byte(), inp.read_unsigned_byte()
        if ident != FORMAT_ID:
            raise IOError(self.filename + " is not a Simple Writer file.")
        if version != FORMAT_VERSION:
            raise IOError("%s is Simple format version %d; this reader reads version %d only." % (self.filename, version, FORMAT_VERSION))
        self.xdim = inp.read_unsigned_short()
        self.ydim = inp.read_unsigned_short()
        self.pixel_shift, self.pixel_quant, self.set_id = [inp.read_byte() for _ in range(3)]
        self.size = dm.get_quantized_size(self.xdim, self.ydim, self.pixel_quant)
        w, h = self.size
        self.min, self.radius = [0] * 3, [0] * 3
        for i in range(3):
            self.min[i] = inp.read_int()
            self.radius[i] = inp.read_unsigned_byte()
            if self.radius[i] < 1:
                raise IOError("%s: channel %d has window radius %d." % (self.filename, i, self.radius[i]))
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
        print("Channel set:  " + dm.SET_NAMES[self.set_id])
        for i in range(3):
            print("Channel %d - %s: radius %d" % (i, dm.channel_name(self.set_id, i), self.radius[i]))
        if n > 1:
            print(sw.describe_bands(n, h))
        print()

    def decode(self):
        """The image as an (ydim, xdim, 3) uint8 array, R, G, B: the bands
        decoded at the same time (each band's channels at the same time too,
        each a few rows behind the one before: a channel's fit and contexts
        use the ones before it), then the channel set recombined, resized and
        shifted."""
        timer = vs.Timer()
        w, h = self.size
        n = self.bands
        raw = [np.zeros(w * h, dtype=np.int64) for _ in range(3)]

        def one(b):
            y0, y1 = sw.band_start(b, n, h), sw.band_start(b + 1, n, h)
            v = dm.get_values_from_least_squares_channels(self.band_delta[b], w, y1 - y0, self.band_init[b], self.radius)[1]
            for i in range(3):
                raw[i][y0 * w:y1 * w] = v[i]
        vs.parallel(n, one)
        ids = dm.get_channels(self.set_id)
        channel = [raw[i] + self.min[i] if ids[i] > 2 else raw[i] for i in range(3)]
        bgr = dm.get_blue_green_red(self.set_id, *channel)
        if self.pixel_quant != 0:
            def resize(c):
                bgr[c] = rm.resize2(bgr[c], w, self.xdim, self.ydim)
            vs.parallel(3, resize)
        rgb = np.stack([np.clip(dm.shift(v, self.pixel_shift), 0, 255).reshape(self.ydim, self.xdim) for v in bgr], axis=2)
        print("Image rebuilt in " + timer.elapsed() + ".")
        return rgb.astype(np.uint8)


class SimpleReader:
    def __init__(self, filename):
        try:
            self.decoder = SimpleDecoder(filename)
        except Exception as e:
            message = str(e) if isinstance(e, (IOError, EOFError)) and str(e) else "Can't decode %s: %r" % (filename, e)
            vs.show_error(None, message)
            self.view = None
            return
        d = self.decoder
        self.view = vs.ImageWindow("Simple Reader  " + filename, d.xdim, d.ydim)
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
        print("Usage: python3 simple_reader.py <filename>")
        sys.exit(0)
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    reader = SimpleReader(sys.argv[1])
    if reader.view is None:
        sys.exit(1)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
