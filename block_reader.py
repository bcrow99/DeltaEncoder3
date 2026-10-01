#!/usr/bin/env python3
"""
block_reader.py version 1.0 -- BlockReader.java for Python (PySide6).

Reads Block format 'B' version 1 files, from block_writer.py or
BlockWriter.java: block map deltas (delta type 13), Context coded. See
block_writer.BlockCoder.save for the layout.

    python3 block_reader.py file

The decoding (class BlockDecoder) is separate from the window:

    rgb = BlockDecoder("foo").decode()      # (ydim, xdim, 3) uint8, R, G, B
"""

import sys

import numpy as np

import delta_mapper as dm
import resize_mapper as rm
import viewer_support as vs
from java_io import DataInput

FORMAT_ID = ord('B')
FORMAT_VERSION = 1


class BlockDecoder:
    def __init__(self, filename):
        self.filename = filename
        timer = vs.Timer()
        with open(filename, "rb") as f:
            self._read(DataInput(f))
        print("File read and entropy decoded in " + timer.elapsed() + ".")

    def _read(self, inp):
        ident, version = inp.read_unsigned_byte(), inp.read_unsigned_byte()
        if ident != FORMAT_ID:
            raise IOError(self.filename + " is not a Block Writer file.")
        if version != FORMAT_VERSION:
            raise IOError("%s is Block format version %d; this reader reads version %d." % (self.filename, version, FORMAT_VERSION))
        self.xdim = inp.read_unsigned_short()
        self.ydim = inp.read_unsigned_short()
        self.pixel_shift, self.pixel_quant, self.set_id = [inp.read_byte() for _ in range(3)]
        self.size = dm.get_quantized_size(self.xdim, self.ydim, self.pixel_quant)
        w, h = self.size

        self.min, self.init = [0] * 3, [0] * 3
        self.map, self.delta = [None] * 3, [None] * 3
        for i in range(3):
            self.min[i] = inp.read_int()
            self.init[i] = inp.read_int()
            self.map[i] = dm.read_map(inp, 13, self.map[i - 1] if i > 0 else None, w)
            self.delta[i] = dm.read_context_deltas(inp, w * h, self.delta[:i], w)
        print("Image:        %d x %d" % (self.xdim, self.ydim))
        print("Channel set:  " + dm.SET_NAMES[self.set_id])
        print("Block map:    size %d, %s" % (int(self.map[0][0]), dm.BLOCK_SET_NAMES[int(self.map[0][1])]))
        print()

    def decode(self):
        """The image as an (ydim, xdim, 3) uint8 array, R, G, B."""
        timer = vs.Timer()
        w, h = self.size
        ids = dm.get_channels(self.set_id)
        channel = [None] * 3

        def one(i):
            ch = dm.get_values_from_deltas(self.delta[i], w, h, self.init[i], 13, self.map[i], 0)
            if ids[i] > 2:
                ch = ch + self.min[i]
            channel[i] = ch
        vs.parallel(3, one)
        bgr = dm.get_blue_green_red(self.set_id, *channel)
        if self.pixel_quant != 0:
            def resize(c):
                bgr[c] = rm.resize(bgr[c], w, self.xdim, self.ydim)
            vs.parallel(3, resize)
        rgb = np.stack([np.clip(dm.shift(v, self.pixel_shift), 0, 255).reshape(self.ydim, self.xdim) for v in bgr], axis=2)
        print("Image rebuilt in " + timer.elapsed() + ".")
        return rgb.astype(np.uint8)


class BlockReader:
    def __init__(self, filename):
        try:
            self.decoder = BlockDecoder(filename)
        except Exception as e:
            message = str(e) if isinstance(e, (IOError, EOFError)) and str(e) else "Can't decode %s: %r" % (filename, e)
            vs.show_error(None, message)
            self.view = None
            return
        d = self.decoder
        self.view = vs.ImageWindow("Block Reader  " + filename, d.xdim, d.ydim)
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
        print("Usage: python3 block_reader.py <filename>")
        sys.exit(0)
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    reader = BlockReader(sys.argv[1])
    if reader.view is None:
        sys.exit(1)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
