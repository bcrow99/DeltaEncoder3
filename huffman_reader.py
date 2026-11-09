#!/usr/bin/env python3
"""
huffman_reader.py version 1.0 -- HuffmanReader.java for Python (PySide6).

Reads Huffman format 'H' version 1 only, from huffman_writer.py or
HuffmanWriter.java: simple_reader2.py with the Huffman decoding in place of
the mixed context decoding. The file is read whole; then each band's
channels' deltas are decoded (code_mapper's regular Huffman code, then
string_mapper's unary strings), all bands and channels at the same time;
then each band's channels are rebuilt from them with the least-squares fit
from the rows above (delta_mapper.get_values_from_row_fit_deltas: the
channels at the same time, each a row behind the one before; the bands at
the same time), and blue, green and red rebuilt, resized back and packed as
simple_reader2.py does. See huffman_writer.encode and huffman_code for the
layout.

    python3 huffman_reader.py file

The decoding (class HuffmanDecoder) is separate from the window:

    rgb = HuffmanDecoder("foo").decode()      # (ydim, xdim, 3) uint8, R, G, B
"""

import sys
import time

import numpy as np

import delta_mapper as dm
import resize_mapper as rm
import simple_writer as sw
import huffman_writer as hw
import viewer_support as vs
from java_io import DataInput

FORMAT_ID = hw.FORMAT_ID
FORMAT_VERSION = hw.FORMAT_VERSION


class HuffmanDecoder:
    """Reads a Huffman file's header; decode() reads the rest and decodes
    it (HuffmanReader.decodeFile)."""

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
        number of bands and their lengths (HuffmanReader.readHeader)."""
        ident, version = inp.read_unsigned_byte(), inp.read_unsigned_byte()
        if ident != FORMAT_ID:
            raise IOError(self.filename + " is not a Huffman file.")
        if version != FORMAT_VERSION:
            raise IOError("%s is Huffman format version %d; this reader reads version %d only." % (self.filename, version, FORMAT_VERSION))
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
        """The image as an (ydim, xdim, 3) uint8 array, R, G, B
        (HuffmanReader.decodeFile): reads the file whole, then decodes every
        band's channels' deltas at the same time (huffman_writer.huffman_decode),
        then rebuilds each band's channels from them
        (delta_mapper.get_values_from_row_fit_deltas), the bands at the same
        time; then blue, green and red are rebuilt, resized and shifted. Its
        time (seconds) is left in took, and the reading's in read_time."""
        start = time.perf_counter()
        w, h = self.size
        n = self.bands
        radius = [hw.RADIUS] * 3
        raw = [np.zeros(w * h, dtype=np.int64) for _ in range(3)]
        init = [[0] * 3 for _ in range(n)]
        coded = [[None] * 3 for _ in range(n)]
        with open(self.filename, "rb") as f:
            inp = DataInput(f)
            self._read_header(inp)
            for b in range(n):
                band = inp.read_fully(self.length[b])
                bi = DataInput(band)
                # Each channel: init, then huffman_code's bytes, whose lengths
                # say where they end.
                for i in range(3):
                    init[b][i] = bi.read_int()
                    begin = bi.f.tell()
                    bi.read_int(); bi.read_int()
                    dm.read_table(bi)
                    bi.read_int()
                    for _ in range(2):
                        bi.read_fully(bi.read_int())
                    coded[b][i] = band[begin:bi.f.tell()]
                if bi.f.tell() != len(band):
                    raise IOError("%s: band %d is not the length its header says." % (self.filename, b))
        self.read_time = time.perf_counter() - start

        # Every band's channels' deltas at the same time, then each band's
        # channels rebuilt, the bands at the same time.
        delta = [[None] * 3 for _ in range(n)]

        def decode_one(j):
            b, i = j // 3, j % 3
            y0, y1 = sw.band_start(b, n, h), sw.band_start(b + 1, n, h)
            delta[b][i] = hw.huffman_decode(DataInput(coded[b][i]), w * (y1 - y0))
        vs.parallel(3 * n, decode_one)

        def rebuild(b):
            y0, y1 = sw.band_start(b, n, h), sw.band_start(b + 1, n, h)
            v = dm.get_values_from_row_fit_deltas(delta[b], w, y1 - y0, init[b], radius, hw.EVERY)
            for i in range(3):
                raw[i][y0 * w:y1 * w] = v[i]
        vs.parallel(n, rebuild)

        channel = [raw[i] + self.min[i] if hw.CHANNELS[i] > 2 else raw[i] for i in range(3)]
        bgr = dm.get_blue_green_red(hw.SET_ID, *channel)
        if self.pixel_quant != 0:
            def resize(c):
                bgr[c] = rm.resize2(bgr[c], w, self.xdim, self.ydim)
            vs.parallel(3, resize)
        rgb = dm.pack_image(bgr, self.xdim, self.ydim, self.pixel_shift)
        self.took = time.perf_counter() - start
        return rgb


class HuffmanReader:
    def __init__(self, filename):
        try:
            self.decoder = HuffmanDecoder(filename)
        except Exception as e:
            message = str(e) if isinstance(e, (IOError, EOFError)) and str(e) else "Can't decode %s: %r" % (filename, e)
            vs.show_error(None, message)
            self.view = None
            return
        d = self.decoder
        self.view = vs.ImageWindow("Huffman Reader  " + filename, d.xdim, d.ydim)
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
        print("Image rebuilt (%d x %d); the file read in %s, then %s." % (
            d.xdim, d.ydim, vs.format_duration(self.first_read), vs.format_duration(d.read_time)))
        print("Run time plus compile time: " + vs.format_duration(self.first_time))
        print("Run time only:              " + vs.format_duration(d.took))


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 huffman_reader.py <filename>")
        sys.exit(0)
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    reader = HuffmanReader(sys.argv[1])
    if reader.view is None:
        sys.exit(1)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
