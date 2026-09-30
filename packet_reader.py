#!/usr/bin/env python3
"""
packet_reader.py version 1.0 -- PacketReader.java for Python (PySide6).

Reads Packet format 'P' version 1 files, from packet_writer.py or
PacketWriter.java, and shows the image.

    python3 packet_reader.py file

The decoding (class PacketDecoder) is separate from the window, so it can
be used without a display:

    rgb = PacketDecoder("foo").decode()      # (ydim, xdim, 3) uint8, R, G, B
"""

import sys

import numpy as np

import arithmetic_mapper as am
import code_mapper as cm
import delta_mapper as dm
import resize_mapper as rm
import segment_mapper as sg
import string_mapper as sm
import viewer_support as vs
from java_io import DataInput
from packet_writer import read_segment_table

FORMAT_ID = ord('P')
FORMAT_VERSION = 1

ENTROPY_NAMES = ["LZ77", "Huffman", "Arithmetic", "Arithmetic (per packet)", "Huffman (per packet)",
                 "Adaptive", "Adaptive (per packet)", "Context"]


def _read_bytes(inp):
    """int length, then that many bytes."""
    return inp.read_fully(inp.read_int())


class PacketDecoder:
    def __init__(self, filename):
        self.filename = filename
        timer = vs.Timer()
        with open(filename, "rb") as f:
            self._read(DataInput(f))
        print("File read in " + timer.elapsed() + ".")

    def _read(self, inp):
        ident, version = inp.read_unsigned_byte(), inp.read_unsigned_byte()
        if ident != FORMAT_ID:
            raise IOError(self.filename + " is not a Packet Writer file.")
        if version != FORMAT_VERSION:
            raise IOError("%s is Packet format version %d; this reader reads version %d." % (self.filename, version, FORMAT_VERSION))
        self.xdim = inp.read_unsigned_short()
        self.ydim = inp.read_unsigned_short()
        # There is no compress_type: the Packet writers always write unary
        # strings. packet_level is informational; segmentation is self-describing.
        (self.pixel_shift, self.pixel_quant, self.set_id, self.delta_type, self.entropy_type,
         self.scanline5_variant, self.packet_level) = [inp.read_byte() for _ in range(7)]

        print("Image:        %d x %d" % (self.xdim, self.ydim))
        print("Channel set:  " + dm.SET_NAMES[self.set_id])
        print("Delta type:   " + dm.DELTA_TYPE_NAMES[self.delta_type])
        print("Entropy type: " + ENTROPY_NAMES[self.entropy_type])
        print("Packet level: %d" % self.packet_level)
        print()

        self.size = dm.get_quantized_size(self.xdim, self.ydim, self.pixel_quant)
        w, h = self.size
        t = self.entropy_type
        self.min, self.init, self.delta_min, self.length = ([0] * 3 for _ in range(4))
        (self.table, self.map, self.string_data, self.segment_bytelength, self.segment_data,
         self.payload_length, self.coded, self.blocks, self.freqs, self.huffman_lengths,
         self.delta) = ([None] * 3 for _ in range(11))

        for i in range(3):
            self.min[i], self.init[i], self.delta_min[i], self.length[i] = [inp.read_int() for _ in range(4)]
            if dm.has_map(self.delta_type):
                self.map[i] = dm.read_map(inp, self.delta_type, self.map[i - 1] if i > 0 else None, w)
            if t == 7:                                   # Context: decoded here, in channel order
                self.delta[i] = dm.read_context_deltas(inp, w * h, self.delta[:i], w)
                continue
            self.table[i] = dm.read_table(inp)
            self._read_segment_table(inp, i)

            n = len(self.segment_bytelength[i])
            if t == 0:                                   # LZ77: payload length, Deflated length, Deflated bytes
                payload_bytes = inp.read_int()
                self.coded[i] = np.frombuffer(cm.inflate(_read_bytes(inp), payload_bytes)[:payload_bytes], dtype=np.uint8)
            elif t == 1:                                 # Huffman: code lengths (Deflated), coded payload
                self.huffman_lengths[i] = cm.unpack_regular_tables(_read_bytes(inp), 1)
                self.coded[i] = np.frombuffer(_read_bytes(inp), dtype=np.uint8)
            elif t == 2:                                 # Arithmetic: frequency tables, then the coded blocks
                self.freqs[i] = am.read_frequencies(inp)
                self.blocks[i] = [np.frombuffer(_read_bytes(inp), dtype=np.uint8) for _ in range(len(self.freqs[i]))]
            elif t == 5:                                 # Adaptive: the coded payload
                self.coded[i] = np.frombuffer(_read_bytes(inp), dtype=np.uint8)
            else:                                        # per packet: tables (3, 4), coded lengths, coded packets
                if t == 3:
                    type_ = inp.read_int()
                    self.freqs[i] = am.inflate_frequencies(_read_bytes(inp), n, type_)
                elif t == 4:
                    self.huffman_lengths[i] = cm.unpack_regular_tables(_read_bytes(inp), n)
                coded_length = cm.unpack_regular_lengths(_read_bytes(inp), n)
                self.blocks[i] = [np.frombuffer(inp.read_fully(L), dtype=np.uint8) for L in coded_length]

    def _read_segment_table(self, inp, i):
        """The string's data byte, then the Deflated segment table: count,
        length-field width, all byte lengths, then all data bytes."""
        self.string_data[i] = inp.read_unsigned_byte()
        raw_length = inp.read_int()
        lengths, data = read_segment_table(cm.inflate(_read_bytes(inp), raw_length))
        self.segment_bytelength[i], self.segment_data[i] = lengths, data
        bits = sum(L * 8 - ((int(d) >> 5) & 7) for L, d in zip(lengths, data))
        self.payload_length[i] = (bits + 7) // 8
        print("Channel %d: %d segments, %d bits, %d payload bytes" % (i, len(lengths), bits, self.payload_length[i]))

    def decode_channel(self, i):
        """Entropy decode -> segments -> unary string -> deltas -> channel values."""
        w, h = self.size
        t = self.entropy_type
        if t == 7:
            return self._to_channel(i, self.delta[i])
        blen = self.segment_bytelength[i]
        if t in (3, 4, 6):
            # Each packet on its own, with its data byte re-attached.
            segments = [None] * len(blen)

            def one(k):
                if t == 3:
                    body = np.zeros(0, dtype=np.uint8) if blen[k] == 0 else \
                        am.get_arithmetic_values_fast_fenwick(self.blocks[i][k], self.freqs[i][k], blen[k])
                elif t == 4:
                    body = cm.unpack_regular_code(self.blocks[i][k], self.huffman_lengths[i][k], blen[k])
                else:
                    body = am.get_arithmetic_values_adaptive(self.blocks[i][k], blen[k])
                seg = np.zeros(blen[k] + 1, dtype=np.uint8)
                seg[:blen[k]] = body[:blen[k]]
                seg[blen[k]] = self.segment_data[i][k]
                segments[k] = seg
            vs.parallel(len(blen), one)
        else:
            n = self.payload_length[i]
            if t == 0:
                payload = self.coded[i]
            elif t == 1:
                payload = cm.unpack_regular_code(self.coded[i], self.huffman_lengths[i][0], n)
            elif t == 5:
                payload = am.get_arithmetic_values_adaptive(self.coded[i], n)
            else:
                lengths = am.get_block_lengths(n, len(self.freqs[i]))
                payload = am.join_blocks([am.get_arithmetic_values_fast_fenwick(self.blocks[i][k], self.freqs[i][k], lengths[k])
                                          for k in range(len(lengths))])
            segments = sg.unpack_segments3(payload, blen, self.segment_data[i])
        # restore2 decompresses whichever segments were compressed and joins
        # them at their bit offsets.
        string = sg.restore2(segments, self.string_data[i])
        delta = sm.unpack_strings(string, self.table[i], w * h, self.length[i])
        delta[0] = 0
        delta[1:] += self.delta_min[i]
        return self._to_channel(i, delta)

    def _to_channel(self, i, delta):
        w, h = self.size
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


class PacketReader:
    def __init__(self, filename):
        try:
            self.decoder = PacketDecoder(filename)
        except Exception as e:
            message = str(e) if isinstance(e, (IOError, EOFError)) and str(e) else "Can't decode %s: %r" % (filename, e)
            vs.show_error(None, message)
            self.view = None
            return
        d = self.decoder
        self.view = vs.ImageWindow("Packet Reader  " + filename, d.xdim, d.ydim)
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
        print("Usage: python3 packet_reader.py <filename>")
        sys.exit(0)
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    reader = PacketReader(sys.argv[1])
    if reader.view is None:
        sys.exit(1)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
