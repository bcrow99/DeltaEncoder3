#!/usr/bin/env python3
"""
simple_reader2.py version 1.0 -- SimpleReader2.java for Python (PySide6).

Reads Simple 2 format 'F' version 1, from simple_writer2.py or
SimpleWriter2.java: the channels green, blue-green and red-green
(simple_writer2.SET_ID), every interior pixel predicted by the least-squares
fit from the rows above (the wide clamp, window radius simple_writer2.RADIUS,
solved at every simple_writer2.EVERY-th pixel along a row; a row's fits are
solved at the same time, before the row is decoded), the deltas
mixed-context coded with the second opinions, in bands of rows decoded at
the same time, resized back with resize_mapper.resize2 (centred). The file
is read in stages: each band's channels start decoding as soon as the start
of them has been read, and decode their coded bytes as they arrive (a row at
a time, once the most the row can read has arrived), while the rest of the
file is read. See simple_writer2.encode for the layout.

    python3 simple_reader2.py file

The decoding (class SimpleDecoder2) is separate from the window:

    rgb = SimpleDecoder2("foo").decode()      # (ydim, xdim, 3) uint8, R, G, B
"""

import sys
import threading
import time
from concurrent.futures import Future

import numpy as np

import delta_mapper as dm
import resize_mapper as rm
import simple_writer as sw
import simple_writer2 as sw2
import viewer_support as vs
from java_io import DataInput

FORMAT_ID = ord('F')
FORMAT_VERSION = 1
CHUNK = 1 << 16   # how many coded bytes are read at a time, saying each time that they have arrived (SimpleReader2.CHUNK)


class SimpleDecoder2:
    """Reads a Simple 2 file's header; decode() reads the rest and decodes
    it, in stages (SimpleReader2.decodeFile)."""

    def __init__(self, filename):
        self.filename = filename
        with open(filename, "rb") as f:
            self._read_header(DataInput(f))
        w, h = self.size
        print("Image:        %d x %d" % (self.xdim, self.ydim))
        if self.bands > 1:
            print(sw.describe_bands(self.bands, h))
        print()

    def _read_header(self, inp):
        """The image's size, the quantization, the channels' minimums, the
        number of bands and their lengths (SimpleReader2.readHeader)."""
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
        self.length = [inp.read_int() for _ in range(n)]

    def decode(self):
        """The image as an (ydim, xdim, 3) uint8 array, R, G, B, read and
        decoded in stages (SimpleReader2.decodeFile): the bands start
        decoding at once, each in its own thread (each band's channels in
        threads of their own, each a few rows behind the one before: a
        channel's fit and contexts use the ones before it), and each channel
        starts as soon as this thread has read the start of it (its rank
        table), decoding its coded bytes as they arrive, a chunk at a time,
        while the rest are read (delta_mapper.Arriving).
        Then blue, green and red are rebuilt, resized and shifted. Its time
        (seconds) is left in took, and the reading's in read_time."""
        start = time.perf_counter()
        w, h = self.size
        n = self.bands
        radius = [sw2.RADIUS] * 3
        raw = [np.zeros(w * h, dtype=np.int64) for _ in range(3)]
        init = [[0] * 3 for _ in range(n)]
        coded = [[Future() for _ in range(3)] for _ in range(n)]
        failed = []

        def one(b):
            try:
                y0, y1 = sw.band_start(b, n, h), sw.band_start(b + 1, n, h)
                v = dm.get_values_from_row_fit_channels(coded[b], w, y1 - y0, init[b], radius, sw2.EVERY)[1]
                for i in range(3):
                    raw[i][y0 * w:y1 * w] = v[i]
            except BaseException as e:
                failed.append(e)

        threads = [threading.Thread(target=one, args=(b,), daemon=True) for b in range(n)]
        for t in threads:
            t.start()
        read = time.perf_counter()
        arriving = []
        try:
            with open(self.filename, "rb") as f:
                inp = DataInput(f)
                self._read_header(inp)
                for b in range(n):
                    end = f.tell() + self.length[b]
                    for i in range(3):
                        init[b][i] = inp.read_int()
                        value, arr = dm.read_arriving_mixed_deltas(inp)
                        arriving.append(arr)
                        coded[b][i].set_result((value, arr))
                        buf = memoryview(arr.data)
                        got = 0
                        while got < len(buf):
                            k = f.readinto(buf[got:got + min(CHUNK, len(buf) - got)])
                            if not k:
                                raise EOFError("%s ends in band %d, channel %d." % (self.filename, b, i))
                            got += k
                            arr.arrived(got)
                    if f.tell() != end:
                        raise IOError("%s: band %d is not the length its header says." % (self.filename, b))
        except BaseException as e:
            for row in coded:
                for c in row:
                    if not c.done():
                        c.set_exception(e)
            for arr in arriving:
                arr.fail(e)
            for t in threads:
                t.join()
            raise
        self.read_time = time.perf_counter() - read
        for t in threads:
            t.join()
        if failed:
            raise failed[0]
        channel = [raw[i] + self.min[i] if sw2.CHANNELS[i] > 2 else raw[i] for i in range(3)]
        bgr = dm.get_blue_green_red(sw2.SET_ID, *channel)
        if self.pixel_quant != 0:
            def resize(c):
                bgr[c] = rm.resize2(bgr[c], w, self.xdim, self.ydim)
            vs.parallel(3, resize)
        rgb = dm.pack_image(bgr, self.xdim, self.ydim, self.pixel_shift)
        self.took = time.perf_counter() - start
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
        # Decoded twice, the times of both printed: the first includes
        # compiling the code (Numba's, the first time it runs, or loading it
        # from the cache), the second is the run time only. The image shows
        # after the first.
        vs.run_in_background(d.decode, self.done)

    def done(self, rgb, error):
        if error is not None:
            self.view.set_status("decode failed")
            vs.show_error(self.view, "Can't decode %s: %r" % (self.decoder.filename, error))
            return
        self.first, self.first_time, self.first_read = rgb, self.decoder.took, self.decoder.read_time
        self.view.set_image(rgb)
        self.view.set_status(None)
        self.view.fit_and_shrink()
        vs.run_in_background(self.decoder.decode, self.done_again)

    def done_again(self, rgb, error):
        if error is not None:
            print("Second decoding failed: %r" % error)
            return
        if not np.array_equal(rgb, self.first):
            print("WARNING: the second decoding differs from the first.")
        d = self.decoder
        print("Image rebuilt (%d x %d); the file read in %s, then %s, while decoding." % (
            d.xdim, d.ydim, vs.format_duration(self.first_read), vs.format_duration(d.read_time)))
        print("Run time plus compile time: " + vs.format_duration(self.first_time))
        print("Run time only:              " + vs.format_duration(d.took))


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
