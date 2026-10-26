#!/usr/bin/env python3
"""
block_reader.py version 2.5 -- BlockReader.java for Python (PySide6).

Reads Block format 'B' versions 3 to 10, from block_writer.py or
BlockWriter.java: block map deltas (delta type 13), Context coded (version
3) or mixed-context coded (versions 4 to 8; from 5 each channel has its own
block size; 6, 8 and 10 are in bands of rows, decoded at the same time; 7 to
10 code with the second opinions; 9 and 10 use the 225-entry table). See
block_writer.BlockCoder.save for the layout.

    python3 block_reader.py file

The decoding (class BlockDecoder) is separate from the window:

    rgb = BlockDecoder("foo").decode()      # (ydim, xdim, 3) uint8, R, G, B
"""

import sys

import numpy as np

import block_writer as bw
import delta_mapper as dm
import resize_mapper as rm
import viewer_support as vs
from java_io import DataInput

FORMAT_ID = ord('B')
FORMAT_VERSION = 10  # 3: one predictor table (delta_mapper.TABLE_256) and cross-channel block maps; 4: mixed context coding; 5: a block size per channel; 6: independently decodable bands; 7 and 8: 5 and 6 with the mixed coder's second opinions; 9 and 10: 7 and 8 with the 225-entry predictor table (delta_mapper.TABLE)
OLDEST_VERSION = 3   # older files are not read (the block predictors changed in version 3)


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
        if not OLDEST_VERSION <= version <= FORMAT_VERSION:
            raise IOError("%s is Block format version %d; this reader reads versions %d to %d (the block predictors changed in version %d)."
                          % (self.filename, version, OLDEST_VERSION, FORMAT_VERSION, OLDEST_VERSION))
        self.version = version
        self.xdim = inp.read_unsigned_short()
        self.ydim = inp.read_unsigned_short()
        self.pixel_shift, self.pixel_quant, self.set_id = [inp.read_byte() for _ in range(3)]
        self.size = dm.get_quantized_size(self.xdim, self.ydim, self.pixel_quant)
        w, h = self.size

        self.min, self.init = [0] * 3, [0] * 3
        self.map, self.delta = [None] * 3, [None] * 3
        self.bands = 1
        self.opinions = version >= 7
        self.predictor_table = dm.TABLE if version >= 9 else dm.TABLE_256   # what the block maps' ids index
        if version in (6, 8, 10):
            self._read_bands(inp, w, h)
            return
        for i in range(3):
            self.min[i] = inp.read_int()
            self.init[i] = inp.read_int()
            self.map[i] = dm.read_map(inp, 13, self.map[i - 1] if i > 0 else None, w)
            if version >= 4:
                self.delta[i] = dm.read_mixed_deltas(inp)   # decoded with the values (decode)
            else:
                self.delta[i] = dm.read_context_deltas(inp, w * h, self.delta[:i], w)
        print("Image:        %d x %d" % (self.xdim, self.ydim))
        print(dm.describe_block_maps(self.set_id, self.map, w, self.predictor_table), end="")
        print()

    def _read_bands(self, inp, w, h):
        """Formats 6, 8 and 10 (see block_writer.BlockCoder.in_bands): the bands, read
        (maps decoded, deltas not yet)."""
        n = self.bands = inp.read_unsigned_byte()
        if n < 1 or h // n < 2:
            raise IOError("A Block file with %d bands of a %d-row image: not one this version reads." % (n, h))
        self.min = [inp.read_int() for _ in range(3)]
        length = [inp.read_int() for _ in range(n)]
        self.band_init, self.band_map, self.band_delta = [], [], []
        for b in range(n):
            bi = DataInput(inp.read_fully(length[b]))
            init, maps, deltas = [], [], []
            for i in range(3):
                init.append(bi.read_int())
                maps.append(dm.read_map(bi, 13, maps[i - 1] if i > 0 else None, w))
                deltas.append(dm.read_mixed_deltas(bi))
            self.band_init.append(init)
            self.band_map.append(maps)
            self.band_delta.append(deltas)
        # The listing: each channel's maps over all the bands (the same block
        # size and predictor set in every band).
        for i in range(3):
            head = dm.map_header_length(self.band_map[0][i])
            self.map[i] = np.concatenate([np.asarray(self.band_map[0][i], dtype=np.uint8)]
                                         + [np.asarray(self.band_map[b][i], dtype=np.uint8)[head:] for b in range(1, n)])
        print("Image:        %d x %d" % (self.xdim, self.ydim))
        print(dm.describe_block_maps(self.set_id, self.map, w, self.predictor_table), end="")
        print(bw.describe_bands(n, h))
        print()

    def _decode_bands(self, w, h):
        """Formats 6, 8 and 10: the bands decoded at the same time; returns the channels."""
        n = self.bands
        channel = [np.zeros(w * h, dtype=np.int64) for _ in range(3)]

        def one(b):
            y0, y1 = bw.band_start(b, n, h), bw.band_start(b + 1, n, h)
            d, v = [None] * 3, [None] * 3
            for i in range(3):
                value, coded = self.band_delta[b][i]
                d[i], v[i] = dm.get_values_from_mixed_deltas(value, coded, d[:i], w, y1 - y0, self.band_init[b][i], self.band_map[b][i],
                                                             v[i - 1] if i > 0 else None, self.opinions, self.predictor_table)
                channel[i][y0 * w:y1 * w] = v[i]
        vs.parallel(n, one)
        return channel

    def decode(self):
        """The image as an (ydim, xdim, 3) uint8 array, R, G, B."""
        timer = vs.Timer()
        w, h = self.size
        ids = dm.get_channels(self.set_id)
        channel = [None] * 3

        if self.version in (6, 8, 10):
            raw = self._decode_bands(w, h)
        else:
            raw, deltas = [None] * 3, [None] * 3
            # In order: each channel's predictors may use the one before it.
            for i in range(3):
                ref = raw[i - 1] if i > 0 else None
                if self.version >= 4:
                    value, coded = self.delta[i]
                    deltas[i], raw[i] = dm.get_values_from_mixed_deltas(value, coded, deltas[:i], w, h, self.init[i], self.map[i], ref, self.opinions, self.predictor_table)
                else:
                    raw[i] = dm.get_values_from_deltas(self.delta[i], w, h, self.init[i], 13, self.map[i], 0, ref=ref, table=self.predictor_table)
        for i in range(3):
            channel[i] = raw[i] + self.min[i] if ids[i] > 2 else raw[i]
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
