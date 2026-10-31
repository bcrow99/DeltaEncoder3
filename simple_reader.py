#!/usr/bin/env python3
"""
simple_reader.py version 3.3 -- SimpleReader.java for Python (PySide6).

Reads Simple format 'S' version 8, from simple_writer.py or
SimpleWriter.java: every interior pixel predicted by the least-squares fit
(the wide clamp) with each channel's window radius, the deltas mixed-context
coded with the second opinions, in bands of rows decoded at the same time,
resized back with resize_mapper.resize2 (centred). The file is read in
stages: each band's channels start decoding as soon as the start of them has
been read, and decode their coded bytes as they arrive (a row at a time,
once the most the row can read has arrived), while the rest of the file is
read. See simple_writer.encode for the layout. Older Simple files (formats 3
to 7) are not read.

    python3 simple_reader.py file

The decoding (class SimpleDecoder) is separate from the window:

    rgb = SimpleDecoder("foo").decode()      # (ydim, xdim, 3) uint8, R, G, B
"""

import sys
import threading
import time
from concurrent.futures import Future

import numpy as np

import delta_mapper as dm
import resize_mapper as rm
import simple_writer as sw
import viewer_support as vs
from java_io import DataInput

FORMAT_ID = ord('S')
FORMAT_VERSION = 8   # 6: least squares alone, a window radius and clamp per channel, in bands; 7: 6 with centred resizing (resize_mapper.resize2); 8: 7 with the wide clamp only (no clamp byte)
CHUNK = 1 << 16      # how many coded bytes are read at a time, saying each time that they have arrived (SimpleReader.CHUNK)


class SimpleDecoder:
    """Reads a Simple file's header; decode() reads the rest and decodes it,
    in stages (SimpleReader.decodeFile)."""

    def __init__(self, filename):
        self.filename = filename
        with open(filename, "rb") as f:
            self._read_header(DataInput(f))
        w, h = self.size
        print("Image:        %d x %d" % (self.xdim, self.ydim))
        print("Channel set:  " + dm.SET_NAMES[self.set_id])
        for i in range(3):
            print("Channel %d - %s: radius %d" % (i, dm.channel_name(self.set_id, i), self.radius[i]))
        if self.bands > 1:
            print(sw.describe_bands(self.bands, h))
        print()

    def _read_header(self, inp):
        """The image's size, the quantization, the channel set, each
        channel's minimum and window radius, the number of bands and their
        lengths (SimpleReader.readHeader)."""
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
        self.length = [inp.read_int() for _ in range(n)]

    def decode(self):
        """The image as an (ydim, xdim, 3) uint8 array, R, G, B, read and
        decoded in stages (SimpleReader.decodeFile): the bands start decoding
        at once, each in its own thread (each band's channels in threads of
        their own, each a few rows behind the one before: a channel's fit and
        contexts use the ones before it), and each channel starts as soon as
        this thread has read the start of it (its rank table), decoding its
        coded bytes as they arrive, a chunk at a time, while the rest are read
        (delta_mapper.Arriving). Then the channel set is recombined, resized
        and shifted. Its time (seconds) is left in took, and the reading's in
        read_time."""
        start = time.perf_counter()
        w, h = self.size
        n = self.bands
        radius = list(self.radius)
        raw = [np.zeros(w * h, dtype=np.int64) for _ in range(3)]
        init = [[0] * 3 for _ in range(n)]
        coded = [[Future() for _ in range(3)] for _ in range(n)]
        failed = []

        def one(b):
            try:
                y0, y1 = sw.band_start(b, n, h), sw.band_start(b + 1, n, h)
                v = dm.get_values_from_least_squares_channels(coded[b], w, y1 - y0, init[b], radius)[1]
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
                header = (self.xdim, self.ydim, self.bands, self.set_id, list(self.length), list(self.min), list(self.radius))
                self._read_header(inp)
                if header != (self.xdim, self.ydim, self.bands, self.set_id, list(self.length), list(self.min), list(self.radius)):
                    raise IOError(self.filename + " changed while it was read.")
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
        ids = dm.get_channels(self.set_id)
        channel = [raw[i] + self.min[i] if ids[i] > 2 else raw[i] for i in range(3)]
        bgr = dm.get_blue_green_red(self.set_id, *channel)
        if self.pixel_quant != 0:
            def resize(c):
                bgr[c] = rm.resize2(bgr[c], w, self.xdim, self.ydim)
            vs.parallel(3, resize)
        rgb = dm.pack_image(bgr, self.xdim, self.ydim, self.pixel_shift)
        self.took = time.perf_counter() - start
        return rgb


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
