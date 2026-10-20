#!/usr/bin/env python3
"""
pixel_reader.py version 1.3 -- PixelReader.java for Python (PySide6).

Reads the Pixel format files pixel_writer.py and PixelWriter.java save
(versions 1 to 3) and shows the image.

    python3 pixel_reader.py file

The decoding (class PixelDecoder) is separate from the window:

    rgb = PixelDecoder("foo").decode()      # (ydim, xdim, 3) uint8, R, G, B
"""

import sys

import numpy as np

import delta_mapper as dm
import pixel_mapper as px
import viewer_support as vs
from java_io import DataInput

FORMAT_ID = ord('X')   # it was 'P' until that turned out to be the Packet format's id too
FORMAT_VERSION = 3      # 2 added the coding byte, 3 the channels byte; older files are read with the defaults


class PixelDecoder:
    def __init__(self, filename):
        self.filename = filename
        timer = vs.Timer()
        with open(filename, "rb") as f:
            self._read(DataInput(f))
        print("File read in " + timer.elapsed() + ".")

    def _read(self, inp):
        ident, version = inp.read_unsigned_byte(), inp.read_unsigned_byte()
        if ident == ord('P'):
            raise IOError(self.filename + " starts with 'P': a Packet Writer file (open it with packet_reader.py), or a Pixel Writer"
                          " file saved before the Pixel format's id changed from 'P' to 'X' (save the image again with pixel_writer.py).")
        if ident != FORMAT_ID:
            raise IOError(self.filename + " is not a Pixel Writer file.")
        if version < 1 or version > FORMAT_VERSION:
            raise IOError("%s is Pixel format version %d; this reader reads versions 1 to %d."
                          % (self.filename, version, FORMAT_VERSION))
        self.xdim = inp.read_unsigned_short()
        self.ydim = inp.read_unsigned_short()
        self.pixel_shift = inp.read_byte()
        self.pixel_quant = inp.read_byte()
        self.block_size = inp.read_unsigned_short()
        self.coding = inp.read_byte() if version >= 2 else px.VALUES
        self.channels = inp.read_byte() if version >= 3 else px.PRIMARY
        if (not 0 <= self.pixel_shift <= 7 or not 0 <= self.pixel_quant <= 10 or self.block_size < 1
                or not 0 <= self.coding < len(px.CODING_NAMES) or not 0 <= self.channels < len(px.CHANNEL_SET_NAMES)):
            raise IOError("%s: bad header (Color Resolution %d, Pixel Resolution %d, block size %d, coding %d, channels %d)."
                          % (self.filename, self.pixel_shift, self.pixel_quant, self.block_size, self.coding, self.channels))
        self.coded = [np.frombuffer(inp.read_fully(inp.read_int()), dtype=np.uint8) for _ in range(3)]

        print("Image:            %d x %d" % (self.xdim, self.ydim))
        print("Pixel Resolution: %d" % self.pixel_quant)
        print("Color Resolution: %d" % self.pixel_shift)
        print("Block size:       %d" % self.block_size)
        print("Coding:           " + px.CODING_NAMES[self.coding])
        print("Channels:         " + px.CHANNEL_SET_NAMES[self.channels])
        print()

    def decode(self):
        """The image as an (ydim, xdim, 3) uint8 array, R, G, B."""
        timer = vs.Timer()
        w, h = dm.get_quantized_size(self.xdim, self.ydim, self.pixel_quant)
        channel = px.decode(self.coded, w, h, 255 >> self.pixel_shift, self.pixel_shift, self.block_size, self.coding, self.channels)
        print("Channels decoded in " + timer.elapsed() + ".")
        timer = vs.Timer()
        rgb = px.get_rgb(channel, w, self.xdim, self.ydim, self.pixel_quant, self.pixel_shift)
        print("RGB assembled in " + timer.elapsed() + ".")
        return rgb


class PixelReader:
    def __init__(self, filename):
        try:
            self.decoder = PixelDecoder(filename)
        except Exception as e:
            message = str(e) if isinstance(e, (IOError, EOFError)) and str(e) else "Can't decode %s: %r" % (filename, e)
            vs.show_error(None, message)
            self.view = None
            return
        d = self.decoder
        self.view = vs.ImageWindow("Pixel Reader  " + filename, d.xdim, d.ydim)
        self.view.make_view_menu()
        self.view.set_status("decoding...")
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
        print("Usage: python3 pixel_reader.py <filename>")
        sys.exit(0)
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    reader = PixelReader(sys.argv[1])
    if reader.view is None:
        sys.exit(1)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
