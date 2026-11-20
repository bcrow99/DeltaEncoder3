#!/usr/bin/env python3
"""
generic_reader.py version 1.0 -- GenericReader.java for Python (PySide6).

Reads generic_writer.py's files (and GenericWriter.java's and NativeEncoder's
NativeWriter's, the same files, and simple_writer2.py's and huffman_writer.py's:
Simple 2 format 2 and Huffman format 1): the channels green, blue-green and
red-green (generic_writer.SET_ID), in bands of rows decoded at the same time
(with LS deltas, viewer_support.ls_bands_at_once at a time), resized back with
resize_mapper.resize2 (centred). The file's first byte says how its deltas
were coded:

  Simple 2 ('F', context aware): the file is read in stages: each band's
  channels start decoding as soon as the start of them has been read, and
  decode their coded bytes as they arrive, while the rest of the file is
  read; a band's channels are decoded at the same time, each a few rows
  behind the one before (its contexts use their deltas).

  Huffman ('H'): the file is read, then every band's channels' deltas are
  decoded (huffman_writer.huffman_decode) and the channels rebuilt from them.

The deltas: LS, every interior pixel predicted by the least-squares fit from
the rows above (window radius generic_writer.RADIUS, solved at every
generic_writer.EVERY-th pixel along a row), or, when the version says so, the
delta type in the header: Avg, each pixel predicted from (left + above) / 2.
Then blue, green and red are rebuilt, resized back and packed. See
generic_writer.encode for the layout.

    python3 generic_reader.py file

The decoding (class GenericDecoder) is separate from the window:

    rgb = GenericDecoder("foo").decode()      # (ydim, xdim, 3) uint8, R, G, B
"""

import sys
import time
from concurrent.futures import Future

import numpy as np

import delta_mapper as dm
import generic_writer as gw
import huffman_writer as hw
import resize_mapper as rm
import simple_writer as sw
import viewer_support as vs
from java_io import DataInput

CHUNK = 1 << 16   # how many coded bytes are read at a time, saying each time that they have arrived (GenericReader.CHUNK)


class GenericDecoder:
    """Reads a Simple 2 or Huffman file's header; decode() reads the rest and
    decodes it (GenericReader.decodeFile)."""

    def __init__(self, filename):
        self.filename = filename
        with open(filename, "rb") as f:
            self._read_header(DataInput(f))
        w, h = self.size
        print("Image:        %d x %d" % (self.xdim, self.ydim))
        print("Deltas:       " + gw.DELTA_NAMES[self.delta_type])
        print("Entropy:      " + gw.ENTROPY_NAMES[self.entropy_type])
        if self.bands > 1:
            print(sw.describe_bands(self.bands, h))
        print()

    def _read_header(self, inp):
        """The format (so the coding), the image's size, the quantization,
        the delta type, the channels' minimums, the number of bands and their
        lengths (GenericReader.readHeader)."""
        ident, version = inp.read_unsigned_byte(), inp.read_unsigned_byte()
        if ident == gw.SIMPLE_ID:
            self.entropy_type, ls_version, type_version, fmt = gw.ENTROPY_MIXED, gw.SIMPLE_VERSION, gw.SIMPLE_VERSION_DELTA_TYPE, "Simple 2"
        elif ident == gw.HUFFMAN_ID:
            self.entropy_type, ls_version, type_version, fmt = gw.ENTROPY_HUFFMAN, gw.HUFFMAN_VERSION, gw.HUFFMAN_VERSION_DELTA_TYPE, "Huffman"
        else:
            raise IOError(self.filename + " is neither a Simple 2 nor a Huffman file.")
        if version != ls_version and version != type_version:
            raise IOError("%s is %s format version %d; this reader reads versions %d and %d only." % (self.filename, fmt, version, ls_version, type_version))
        self.xdim = inp.read_unsigned_short()
        self.ydim = inp.read_unsigned_short()
        self.pixel_shift, self.pixel_quant = inp.read_byte(), inp.read_byte()
        self.delta_type = gw.DELTA_LS if version == ls_version else inp.read_unsigned_byte()
        if self.delta_type not in (gw.DELTA_LS, gw.DELTA_AVERAGE):
            raise IOError("%s: unknown delta type %d." % (self.filename, self.delta_type))
        self.min = [inp.read_int() for _ in range(3)]
        self.size = dm.get_quantized_size(self.xdim, self.ydim, self.pixel_quant)
        w, h = self.size
        n = self.bands = inp.read_unsigned_byte()
        if n < 1 or h // n < 2:
            raise IOError("%s: %d bands of a %d-row image." % (self.filename, n, h))
        self.length = [inp.read_int() for _ in range(n)]

    def _check_header(self, inp):
        """Reads the header again, checking it's the one the constructor read."""
        before = (self.xdim, self.ydim, self.bands, self.delta_type, self.entropy_type, list(self.length))
        self._read_header(inp)
        if (self.xdim, self.ydim, self.bands, self.delta_type, self.entropy_type, list(self.length)) != before:
            raise IOError(self.filename + " changed while it was read.")

    def decode(self):
        """The image as an (ydim, xdim, 3) uint8 array, R, G, B
        (GenericReader.decodeFile): the channels decoded (_decode_mixed or
        _decode_huffman, by the format), then blue, green and red rebuilt,
        resized and shifted. Its time (seconds) is left in took, and the
        reading's in read_time."""
        start = time.perf_counter()
        w, h = self.size
        raw = [np.zeros(w * h, dtype=np.int64) for _ in range(3)]
        if self.entropy_type == gw.ENTROPY_MIXED:
            self._decode_mixed(w, h, raw)
        else:
            self._decode_huffman(w, h, raw)
        channel = [raw[i] + self.min[i] if gw.CHANNELS[i] > 2 else raw[i] for i in range(3)]
        bgr = dm.get_blue_green_red(gw.SET_ID, *channel)
        if self.pixel_quant != 0:
            def resize(c):
                bgr[c] = rm.resize2(bgr[c], w, self.xdim, self.ydim)
            vs.parallel(3, resize)
        rgb = dm.pack_image(bgr, self.xdim, self.ydim, self.pixel_shift)
        self.took = time.perf_counter() - start
        return rgb

    def _decode_mixed(self, w, h, raw):
        """A Simple 2 file, read and decoded in stages
        (GenericReader.decodeMixed): the bands decode in order, in threads of
        their own (all at once with the average deltas,
        viewer_support.ls_bands_at_once at a time with LS; each band's
        channels in threads of their own, each a few rows behind the one
        before), and each channel starts as soon as this thread has read the
        start of it, decoding its coded bytes as they arrive while the rest
        are read (delta_mapper.Arriving)."""
        n = self.bands
        radius = [gw.RADIUS] * 3
        init = [[0] * 3 for _ in range(n)]
        coded = [[Future() for _ in range(3)] for _ in range(n)]
        failed = []

        def one(b):
            try:
                y0, y1 = sw.band_start(b, n, h), sw.band_start(b + 1, n, h)
                if self.delta_type == gw.DELTA_AVERAGE:
                    v = dm.get_values_from_average_channels(coded[b], w, y1 - y0, init[b])[1]
                else:
                    v = dm.get_values_from_row_fit_channels(coded[b], w, y1 - y0, init[b], radius, gw.EVERY)[1]
                for i in range(3):
                    raw[i][y0 * w:y1 * w] = v[i]
            except BaseException as e:
                failed.append(e)

        at_once = vs.ls_bands_at_once(n) if self.delta_type == gw.DELTA_LS else n
        threads = vs.start_band_threads(n, at_once, one)
        read = time.perf_counter()
        arriving = []
        try:
            with open(self.filename, "rb") as f:
                inp = DataInput(f)
                self._check_header(inp)
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

    def _decode_huffman(self, w, h, raw):
        """A Huffman file (GenericReader.decodeHuffman): read whole, then
        every band's channels' deltas decoded at the same time, then the
        channels rebuilt: with the average deltas each on its own, all at
        once; with LS, viewer_support.ls_bands_at_once bands at a time."""
        n = self.bands
        radius = [gw.RADIUS] * 3
        init = [[0] * 3 for _ in range(n)]
        coded = [[None] * 3 for _ in range(n)]
        read = time.perf_counter()
        with open(self.filename, "rb") as f:
            inp = DataInput(f)
            self._check_header(inp)
            for b in range(n):
                band = inp.read_fully(self.length[b])
                bi = DataInput(band)
                # Each channel: init, then huffman_code's bytes.
                for i in range(3):
                    init[b][i] = bi.read_int()
                    coded[b][i] = hw.huffman_read(bi)
                if bi.f.tell() != len(band):
                    raise IOError("%s: band %d is not the length its header says." % (self.filename, b))
        self.read_time = time.perf_counter() - read

        delta = [[None] * 3 for _ in range(n)]

        def decode_one(j):
            b, i = j // 3, j % 3
            y0, y1 = sw.band_start(b, n, h), sw.band_start(b + 1, n, h)
            delta[b][i] = hw.huffman_decode(coded[b][i], w * (y1 - y0))
            if self.delta_type == gw.DELTA_AVERAGE:
                raw[i][y0 * w:y1 * w] = dm.get_values_from_deltas(delta[b][i], w, y1 - y0, init[b][i], 2)
        vs.parallel(3 * n, decode_one)

        def rebuild(b):
            y0, y1 = sw.band_start(b, n, h), sw.band_start(b + 1, n, h)
            v = dm.get_values_from_row_fit_deltas(delta[b], w, y1 - y0, init[b], radius, gw.EVERY)
            for i in range(3):
                raw[i][y0 * w:y1 * w] = v[i]
        if self.delta_type == gw.DELTA_LS:
            vs.for_bands(n, vs.ls_bands_at_once(n), rebuild)


class GenericReader:
    def __init__(self, filename):
        try:
            self.decoder = GenericDecoder(filename)
        except Exception as e:
            message = str(e) if isinstance(e, (IOError, EOFError)) and str(e) else "Can't decode %s: %r" % (filename, e)
            vs.show_error(None, message)
            self.view = None
            return
        d = self.decoder
        self.view = vs.ImageWindow("Generic Reader  " + filename, d.xdim, d.ydim)
        self.view.make_view_menu()
        self.view.set_status("decoding…")
        self.view.show_window()
        # Decoded twice (viewer_support.print_run_times: the first decoding in
        # this process includes compiling the code, so the times are printed
        # apart, the second as the run time). The image shows after the first.
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
        how = ", while decoding." if d.entropy_type == gw.ENTROPY_MIXED else "."
        print("Image rebuilt (%d x %d); the file read in %s, then %s%s" % (
            d.xdim, d.ydim, vs.format_duration(self.first_read), vs.format_duration(d.read_time), how))
        # (The image is shown between the two decodings, so they aren't timed
        # together: their sum stands in for it. A reader decodes one file, so
        # its first decoding is always the first in the process anyway.)
        vs.print_run_times("GenericDecoder %s %s" % (gw.DELTA_NAMES[d.delta_type], gw.ENTROPY_NAMES[d.entropy_type]), self.first_time, d.took, self.first_time + d.took)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 generic_reader.py <filename>")
        sys.exit(0)
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    reader = GenericReader(sys.argv[1])
    if reader.view is None:
        sys.exit(1)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
