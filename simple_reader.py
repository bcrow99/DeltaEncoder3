#!/usr/bin/env python3
"""
simple_reader.py version 1.0 -- SimpleReader.java for Python (PySide6).

Reads Simple format 'S' version 1 files, from simple_writer.py or
SimpleWriter.java, and shows the image.

    python3 simple_reader.py file

The decoding (class SimpleDecoder) is separate from the window:

    rgb = SimpleDecoder("foo").decode()      # (ydim, xdim, 3) uint8, R, G, B
"""

import sys

import numpy as np

import arithmetic_mapper as am
import code_mapper as cm
import delta_mapper as dm
import resize_mapper as rm
import string_mapper as sm
import viewer_support as vs
from java_io import DataInput

FORMAT_ID = ord('S')
FORMAT_VERSION = 1

ENTROPY_NAMES = ["LZ77", "Huffman", "Arithmetic", "Adaptive", "Context"]


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
            raise IOError("%s is Simple format version %d; this reader reads version %d." % (self.filename, version, FORMAT_VERSION))
        self.xdim = inp.read_unsigned_short()
        self.ydim = inp.read_unsigned_short()
        # There is no compress_type: the Simple writers always write unary strings.
        (self.pixel_shift, self.pixel_quant, self.set_id, self.delta_type,
         self.entropy_type, self.scanline5_variant) = [inp.read_byte() for _ in range(6)]

        print("Image:        %d x %d" % (self.xdim, self.ydim))
        print("Channel set:  " + dm.SET_NAMES[self.set_id])
        print("Delta type:   " + dm.DELTA_TYPE_NAMES[self.delta_type])
        print("Entropy type: " + ENTROPY_NAMES[self.entropy_type])
        print()

        self.size = dm.get_quantized_size(self.xdim, self.ydim, self.pixel_quant)
        w, h = self.size
        self.min, self.init, self.delta_min, self.length, self.compressed_length = ([0] * 3 for _ in range(5))
        self.table, self.map, self.coded, self.delta, self.lengths, self.freqs, self.blocks = ([None] * 3 for _ in range(7))

        for i in range(3):
            self.min[i], self.init[i], self.delta_min[i], self.length[i], self.compressed_length[i] = [inp.read_int() for _ in range(5)]
            inp.read_byte()                          # iterations (the string carries them too)
            if dm.has_map(self.delta_type):
                self.map[i] = dm.read_map(inp, self.delta_type, self.map[i - 1] if i > 0 else None, w)
            if self.entropy_type != 4:
                self.table[i] = dm.read_table(inp)
            if self.entropy_type == 0:               # LZ77: payload length, Deflated length, Deflated bytes
                n = inp.read_int()
                self.coded[i] = np.frombuffer(cm.inflate(inp.read_fully(inp.read_int()), n)[:n], dtype=np.uint8)
            elif self.entropy_type == 1:             # Huffman: code lengths (Deflated), then the code
                self.lengths[i] = cm.unpack_regular_tables(inp.read_fully(inp.read_int()), 1)[0]
                self.coded[i] = np.frombuffer(inp.read_fully(inp.read_int()), dtype=np.uint8)
            elif self.entropy_type == 2:             # Arithmetic: frequency tables, then the blocks
                self.freqs[i] = am.read_frequencies(inp)
                self.blocks[i] = [np.frombuffer(inp.read_fully(inp.read_int()), dtype=np.uint8) for _ in range(len(self.freqs[i]))]
            elif self.entropy_type == 4:             # Context: decoded here, in channel order
                self.delta[i] = dm.read_context_deltas(inp, w * h, self.delta[:i], w)
            else:                                    # Adaptive
                self.coded[i] = np.frombuffer(inp.read_fully(inp.read_int()), dtype=np.uint8)

    def decode_channel(self, i):
        """Entropy decode -> unary strings -> deltas -> channel values."""
        w, h = self.size
        if self.entropy_type == 4:
            delta = self.delta[i]
        else:
            payload_length = sm.get_bytelength(self.compressed_length[i])
            if self.entropy_type == 0:
                payload = self.coded[i]
            elif self.entropy_type == 1:
                payload = cm.unpack_regular_code(self.coded[i], self.lengths[i], payload_length)
            elif self.entropy_type == 3:
                payload = am.get_arithmetic_values_adaptive(self.coded[i], payload_length)
            else:
                lengths = am.get_block_lengths(payload_length, len(self.freqs[i]))
                payload = am.join_blocks([am.get_arithmetic_values_fast_fenwick(self.blocks[i][k], self.freqs[i][k], lengths[k])
                                          for k in range(len(lengths))])
            delta = sm.unpack_strings(sm.decompress_strings(payload), self.table[i], w * h, self.length[i])
            delta[0] = 0
            delta[1:] += self.delta_min[i]
        channel = dm.get_values_from_deltas(delta, w, h, self.init[i], self.delta_type, self.map[i], self.scanline5_variant)
        if dm.get_channels(self.set_id)[i] > 2:
            channel = channel + self.min[i]
        return channel

    def decode(self):
        """The image as an (ydim, xdim, 3) uint8 array, R, G, B."""
        timer = vs.Timer()
        channel = [None] * 3

        def one(i):
            channel[i] = self.decode_channel(i)
        vs.parallel(3, one)
        print("Channels processed in " + timer.elapsed() + ".")

        # Recombine the channel set, then resize, then shift.
        timer = vs.Timer()
        bgr = dm.get_blue_green_red(self.set_id, *channel)
        if self.pixel_quant != 0:
            def resize(c):
                bgr[c] = rm.resize(bgr[c], self.size[0], self.xdim, self.ydim)
            vs.parallel(3, resize)
        rgb = np.stack([np.clip(dm.shift(v, self.pixel_shift), 0, 255).reshape(self.ydim, self.xdim) for v in bgr], axis=2)
        print("RGB assembled in " + timer.elapsed() + ".")
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
