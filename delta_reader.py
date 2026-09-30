#!/usr/bin/env python3
"""
delta_reader.py version 1.0 -- DeltaReader.java for Python (PySide6).

Reads Delta format 'D' version 1 files, from delta_writer.py or
DeltaWriter.java (or DeltaWriter2.java), and shows the image.

    python3 delta_reader.py file

The decoding (class DeltaDecoder) is separate from the window, so it can
be used without a display:

    rgb = DeltaDecoder("foo").decode()      # (ydim, xdim, 3) uint8, R, G, B

Not ported yet: files made with the pixel pyramid ("Average" in the Java
DeltaWriter's Quantization menu) are refused with a message.
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

FORMAT_ID = ord('D')
FORMAT_VERSION = 1

ENTROPY_NAMES = ["LZ77", "Huffman", "Arithmetic", "Adaptive", "Context"]
COMPRESS_NAMES = ["Integer", "String", "String*", "String* (DeltaWriter2)"]


class DeltaDecoder:
    def __init__(self, filename):
        self.filename = filename
        timer = vs.Timer()
        with open(filename, "rb") as f:
            self._read(DataInput(f))
        print("File read in " + timer.elapsed() + ".")

    def _read(self, inp):
        ident, version = inp.read_unsigned_byte(), inp.read_unsigned_byte()
        if ident != FORMAT_ID:
            raise IOError(self.filename + " is not a Delta Writer file.")
        if version != FORMAT_VERSION:
            raise IOError("%s is Delta format version %d; this reader reads version %d." % (self.filename, version, FORMAT_VERSION))
        self.xdim = inp.read_unsigned_short()
        self.ydim = inp.read_unsigned_short()
        (self.pixel_shift, self.pixel_quant, self.set_id, self.delta_type, self.compress_type,
         self.entropy_type, self.scanline5_variant, self.pixel_pyramid, saddle) = [inp.read_byte() for _ in range(9)]
        if self.pixel_pyramid != 0:
            raise IOError(self.filename + " was saved with the pixel pyramid (Average), which the Python reader doesn't support yet.")

        print("Image:        %d x %d" % (self.xdim, self.ydim))
        print("Channel set:  " + dm.SET_NAMES[self.set_id])
        print("Delta type:   " + dm.DELTA_TYPE_NAMES[self.delta_type])
        print("Datatype:     " + COMPRESS_NAMES[self.compress_type])
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
            if self.entropy_type == 4:               # Context: decoded here, in channel order
                self.delta[i] = dm.read_context_deltas(inp, w * h, self.delta[:i], w)
                continue
            if self.compress_type > 0:
                self.table[i] = dm.read_table(inp)
            if self.entropy_type == 0:               # LZ77
                n = inp.read_int()
                self.coded[i] = np.frombuffer(cm.inflate(inp.read_fully(inp.read_int()), n)[:n], dtype=np.uint8)
            elif self.entropy_type == 1:             # Huffman: code lengths (Deflated), then the code
                self.lengths[i] = cm.unpack_regular_tables(inp.read_fully(inp.read_int()), 1)[0]
                self.coded[i] = np.frombuffer(inp.read_fully(inp.read_int()), dtype=np.uint8)
            elif self.entropy_type == 2:             # Arithmetic: frequency tables, then the blocks
                self.freqs[i] = am.read_frequencies(inp)
                self.blocks[i] = [np.frombuffer(inp.read_fully(inp.read_int()), dtype=np.uint8) for _ in range(len(self.freqs[i]))]
            else:                                    # Adaptive
                self.coded[i] = np.frombuffer(inp.read_fully(inp.read_int()), dtype=np.uint8)

    def decode_channel(self, i):
        """Entropy decode -> deltas -> channel values."""
        w, h = self.size
        n = w * h
        if self.entropy_type == 4:
            delta = self.delta[i]
        else:
            payload_length = n if self.compress_type == 0 else sm.get_bytelength(self.compressed_length[i])
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
            if self.compress_type == 0:
                delta = payload[:n].astype(np.int64) + self.delta_min[i]      # one unsigned byte per delta
                delta[0] = 0
            else:
                delta = sm.unpack_strings(sm.decompress_strings(payload), self.table[i], n, self.length[i])
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


class DeltaReader:
    def __init__(self, filename):
        try:
            self.decoder = DeltaDecoder(filename)
        except Exception as e:
            message = str(e) if isinstance(e, (IOError, EOFError)) and str(e) else "Can't decode %s: %r" % (filename, e)
            vs.show_error(None, message)
            self.view = None
            return
        d = self.decoder
        self.view = vs.ImageWindow("Delta Reader  " + filename, d.xdim, d.ydim)
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
        print("Usage: python3 delta_reader.py <filename>")
        sys.exit(0)
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    reader = DeltaReader(sys.argv[1])
    if reader.view is None:
        sys.exit(1)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
