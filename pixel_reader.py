#!/usr/bin/env python3
"""
pixel_reader.py version 1.5 -- PixelReader.java for Python (PySide6).

Reads the Pixel format files pixel_writer.py and PixelWriter.java save
(versions 1 to 5) and shows the image. Version 5 adds the prediction (MED or
least squares), each coded channel's offset and largest value (the channel
set green, blue - green, red - green), and resizes back centred
(resize_mapper.resize2).

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
FORMAT_VERSION = 5      # 2 added the coding byte, 3 the channels byte, 4 the averaging pyramid, 5 the prediction, offsets and largest values, and centred resizing; older files are read with the defaults


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
        self.version = version
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
        self.prediction, self.radius = px.MED, px.LS_RADIUS
        if version >= 5:
            self.prediction = inp.read_byte()
            self.radius = inp.read_unsigned_byte()
        self.pixel_pyramid = inp.read_byte() if version >= 4 else 0
        self.use_saddle = version >= 4 and inp.read_byte() != 0
        if (not 0 <= self.pixel_shift <= 7 or not 0 <= self.pixel_quant <= 10 or self.block_size < 1
                or not 0 <= self.coding < len(px.CODING_NAMES) or not 0 <= self.channels < len(px.CHANNEL_SET_NAMES)
                or (version < 5 and self.channels == px.GREEN_DIFFERENCES)
                or not 0 <= self.prediction < len(px.PREDICTION_NAMES) or self.radius < 1
                or not 0 <= self.pixel_pyramid <= px.PYRAMID_MAX):
            raise IOError("%s: bad header (Color Resolution %d, Pixel Resolution %d, block size %d, coding %d, channels %d, prediction %d)."
                          % (self.filename, self.pixel_shift, self.pixel_quant, self.block_size, self.coding, self.channels, self.prediction))
        max_value = 255 >> self.pixel_shift
        if version >= 5:
            self.offset, self.max = [0] * 3, [0] * 3
            for i in range(3):
                self.offset[i] = inp.read_int()
                self.max[i] = inp.read_int()
                if not 0 <= self.max[i] <= 2 * max_value:
                    raise IOError("%s: bad header (channel %d's largest value %d)." % (self.filename, i, self.max[i]))
        else:
            self.offset = [0, -max_value, -max_value] if self.channels == px.DIFFERENCE else [0, 0, 0]
            self.max = px.channel_max(max_value, self.channels)
        self.coded = [np.frombuffer(inp.read_fully(inp.read_int()), dtype=np.uint8) for _ in range(3)]
        self.sign_bits = [px.read_sign_bits(inp, self.pixel_pyramid) for _ in range(3)] if self.pixel_pyramid > 0 else None

        print("Image:            %d x %d" % (self.xdim, self.ydim))
        print("Pixel Resolution: %d" % self.pixel_quant)
        print("Color Resolution: %d" % self.pixel_shift)
        print("Block size:       %d" % self.block_size)
        print("Coding:           " + px.CODING_NAMES[self.coding])
        print("Channels:         " + px.CHANNEL_SET_NAMES[self.channels])
        print("Prediction:       " + px.PREDICTION_NAMES[self.prediction] + (" (window radius %d)" % self.radius if self.prediction == px.LS else ""))
        print("Average:          %d%s" % (self.pixel_pyramid, " (saddle)" if self.pixel_pyramid > 0 and self.use_saddle else ""))
        print()

    def decode(self):
        """The image as an (ydim, xdim, 3) uint8 array, R, G, B."""
        timer = vs.Timer()
        w, h = dm.get_quantized_size(self.xdim, self.ydim, self.pixel_quant)
        tw, th = px.pyramid_size(w, h, self.pixel_pyramid)
        channel = px.decode(self.coded, tw, th, self.pixel_shift, self.block_size, self.coding, self.channels,
                            self.prediction, self.radius, self.offset, self.max)
        if self.pixel_pyramid > 0:
            channel = [px.expand_pyramid(channel[i], w, h, self.sign_bits[i], 255 >> self.pixel_shift, self.use_saddle) for i in range(3)]
        print("Channels decoded in " + timer.elapsed() + ".")
        timer = vs.Timer()
        rgb = px.get_rgb(channel, w, self.xdim, self.ydim, self.pixel_quant, self.pixel_shift, self.version >= 5)
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
