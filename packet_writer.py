#!/usr/bin/env python3
"""
packet_writer.py version 1.0 -- PacketWriter.java for Python (PySide6).

Quantizes an image, codes it as deltas packed into unary strings, splits
each string into separately compressed segments ("packets") and entropy
codes them, then saves (to the file "foo", as the Java version does) in
Packet format 'P' version 1: the same file layout as PacketWriter.java, so
either program's files open in either reader.

    python3 packet_writer.py [image]

The coding (class PacketCoder) is separate from the window (PacketWriter),
so it can also be used without a display:

    coder = PacketCoder(viewer_support.read_image("photo.png"))
    coder.survey(); coder.apply(); coder.save("photo.pkt")

Segmentation (the Packet menu's Segment Length, 0-10) and the entropy
settings only take effect at Save; they never change the preview.
"""

import math
import os
import struct
import sys

import numpy as np

import arithmetic_mapper as am
import code_mapper as cm
import delta_mapper as dm
import resize_mapper as rm
import segment_mapper as sg
import string_mapper as sm
import viewer_support as vs
from delta_writer import context_code
from java_io import DataOutput

FORMAT_ID = ord('P')
FORMAT_VERSION = 1

# Entropy menu, in entropy_type order. 3, 4 and 6 code every packet on its
# own instead of all of a channel's packets as one stream (see code_packets).
# 5 and 6 use the adaptive coder, which stores no frequency table.
ENTROPY_NAMES = ["LZ77", "Huffman", "Arithmetic", "Arithmetic (per packet)", "Huffman (per packet)",
                 "Adaptive", "Adaptive (per packet)", "Context"]
ARITHMETIC_PER_PACKET = 3
HUFFMAN_PER_PACKET = 4
ADAPTIVE = 5
ADAPTIVE_PER_PACKET = 6
CONTEXT = 7            # deltas context-coded directly; no packets

CHANNEL_NAMES = ["blue", "green", "red", "blue-green", "red-green", "red-blue"]
DELTA_MENU_NAMES = ["H", "V", "Average", "Med", "Directional", "Adaptive", "Scanline 1", "Scanline 2",
                    "Scanline 3", "Scanline 4", "Scanline 5", "Map 1", "Map 2", "Block Map"]

# ---- Packet (string segmentation) parameters ----------------------------------
# packet_level 0..10 sets the minimum segment length passed to
# segment_mapper.get_segmented_data3: 0 = MIN_SEGMENT_BITS (32 bytes), 10 = no
# segmentation (the whole string compressed as one piece). Levels in between
# are geometric steps. Merging is capped at MAX_PACKET_FACTOR times the
# minimum segment length.
MIN_SEGMENT_BITS = 256
MAX_PACKET_FACTOR = 4
SEGMENT_TYPE = 2       # segment -> merge -> combine (no splice)
MERGE_TYPE = 2
BINS = 20


def bin_width(number_of_bins):
    """Bin width for a bin count; merge recovers the count as int(1/bin), so
    nudge the width down if rounding would make that come out one short."""
    w = 1.0 / number_of_bins
    while int(1.0 / w) < number_of_bins:
        w = np.nextafter(w, 0.0)
    return float(w)


BIN = bin_width(BINS)

AUTO_STOP_PERCENT = 3.0
DEFAULT_COMPRESSION = -1   # Deflater.DEFAULT_COMPRESSION


# =============================================================================
# Segmentation and packets
# =============================================================================

def get_minimum_segment_bits(total_bits, level):
    """Minimum segment length in bits for a packet_level, or 0 meaning
    "don't segment". Level 0 is MIN_SEGMENT_BITS; each level multiplies it
    by the same factor."""
    if level >= 10 or total_bits <= 2 * MIN_SEGMENT_BITS:
        return 0
    bits = MIN_SEGMENT_BITS * math.pow(total_bits / MIN_SEGMENT_BITS, level / 10.0)
    b = int(bits) // 8 * 8
    if b < MIN_SEGMENT_BITS:
        b = MIN_SEGMENT_BITS
    if b >= total_bits:
        return 0
    return b


def segment_string(string, level):
    """Splits an uncompressed unary string into segments, each with its own
    data byte recording whether and how it was compressed."""
    min_bits = get_minimum_segment_bits(sm.get_bitlength(string), level)
    if min_bits != 0:
        result = sg.get_segmented_data3(string, min_bits, SEGMENT_TYPE, MERGE_TYPE, BIN, MAX_PACKET_FACTOR * min_bits)
        if result:
            return result[0]
    return [sm.compress_strings(string)]


def same_bits(a, b):
    """True if two strings (each with a data byte) hold the same bits."""
    bl = sm.get_bitlength(a)
    if bl != sm.get_bitlength(b):
        return False
    full = bl // 8
    if not np.array_equal(a[:full], b[:full]):
        return False
    odd = bl % 8
    if odd:
        m = (1 << odd) - 1
        if (int(a[full]) & m) != (int(b[full]) & m):
            return False
    return True


class Packet:
    """A channel's segments stored in two parts:
      payload: every segment's bits packed back to back (pack_segments3), no
        padding and no data bytes in between -- what the entropy coder sees;
      table: segment count, length-field width (1, 2 or 4 bytes), all
        segment byte lengths (without the data byte), then all data bytes;
        stored as columns so Deflate finds the repetition, and written
        Deflated (ztable)."""

    def __init__(self, segments):
        self.segments = segments
        n = len(segments)
        mx = max((len(s) - 1 for s in segments), default=0)
        width = 1 if mx <= 255 else 2 if mx <= 65535 else 4
        table = bytearray(struct.pack(">ib", n, width))
        fmt = {1: ">B", 2: ">H", 4: ">i"}[width]
        for s in segments:
            table += struct.pack(fmt, len(s) - 1)
        self.compressed = 0
        self.bits = 0
        for s in segments:
            table.append(int(s[-1]))
            it = sm.get_iterations(s)
            if it != 0 and it != 16:
                self.compressed += 1
            self.bits += sm.get_bitlength(s)
        self.table = bytes(table)
        self.ztable = cm.deflate(self.table, 9)
        self.payload = sg.pack_segments3(segments)[0]


def read_segment_table(raw):
    """(byte lengths, data bytes) from an inflated segment table."""
    n, width = struct.unpack(">ib", raw[:5])
    fmt = {1: "B", 2: "H", 4: "i"}[width]
    lengths = list(struct.unpack(">%d%s" % (n, fmt), raw[5:5 + n * width]))
    data = np.frombuffer(raw[5 + n * width:5 + n * width + n], dtype=np.uint8).copy()
    return lengths, data


def unpack_packet(p, string_data):
    """Rebuilds the uncompressed string from a packet exactly the way
    packet_reader does, so Save can verify the round trip."""
    lengths, data = read_segment_table(cm.inflate(p.ztable, len(p.table)))
    return sg.restore2(sg.unpack_segments3(p.payload, lengths, data), string_data)


# =============================================================================
# Entropy coding
# =============================================================================
# code_payload and code_packets return the bytes written for a channel after
# its segment table. With a warning list they also decode the result back (as
# packet_reader will) and set warning[i] on a mismatch.

def is_per_packet(t):
    return t in (ARITHMETIC_PER_PACKET, HUFFMAN_PER_PACKET, ADAPTIVE_PER_PACKET)


def arithmetic_blocks(length, pixel_segment):
    """Arithmetic blocks for a payload (pixel_segment 10 = one block)."""
    return 1 if pixel_segment >= 10 else max(1, length // (500 + pixel_segment * 500))


def code_payload(payload, t, pixel_segment, warning=None, i=0):
    """Types 0, 1, 2 and 5 code a channel's whole payload:
      LZ77:       int payload length, int Deflated length, Deflated payload;
      Huffman:    int table length, the 256 code lengths Deflated, int coded
                  length, the coded payload;
      Arithmetic: the block frequency tables (pack_frequencies), then each
                  block's int length and coded bytes;
      Adaptive:   int coded length, the coded payload.
    The per-packet types 3, 4 and 6 give their whole-payload counterpart."""
    if t == 0:
        zipped = cm.deflate(payload, 9)
        return struct.pack(">ii", len(payload), len(zipped)) + zipped
    if t in (1, HUFFMAN_PER_PACKET):
        lengths = cm.get_regular_huffman_length(am.get_frequency(payload))
        table = cm.pack_regular_tables([lengths], 9)
        code = cm.pack_regular_code(payload, lengths)
        if warning is not None and not np.array_equal(payload, cm.unpack_regular_code(code, lengths, len(payload))):
            warning[i] = "Huffman payload does not decode back."
        return struct.pack(">i", len(table)) + table + struct.pack(">i", len(code)) + code.tobytes()
    if t in (ADAPTIVE, ADAPTIVE_PER_PACKET):
        code = am.get_interval_value_adaptive(payload)
        if warning is not None and not np.array_equal(payload, am.get_arithmetic_values_adaptive(code, len(payload))):
            warning[i] = "adaptive payload does not decode back."
        return struct.pack(">i", len(code)) + code.tobytes()
    blocks = am.get_blocks(payload, arithmetic_blocks(len(payload), pixel_segment))
    freq = [None] * len(blocks)
    enc = [None] * len(blocks)

    def one(m):
        freq[m] = am.get_frequency(blocks[m])
        enc[m] = am.get_interval_value_fast_fenwick(blocks[m], freq[m])
    vs.parallel(len(blocks), one)
    parts = [am.pack_frequencies(freq, 9)]
    for e in enc:
        parts.append(struct.pack(">i", len(e)) + e.tobytes())
    return b"".join(parts)


def code_packets(segments, t, warning=None, i=0):
    """Types 3, 4 and 6 code each packet (a segment's bytes without its data
    byte, which is in the segment table) on its own:
      Arithmetic per packet: int type, int zipped length, every packet's
          frequency table Deflated together;
      Huffman per packet: int tables length, every packet's 256 code
          lengths Deflated together;
      Adaptive per packet: no tables;
    then int lengths length, each packet's coded length as a varint, and the
    coded packets back to back."""
    n = len(segments)
    body = [np.ascontiguousarray(s[:-1]) for s in segments]
    coded = [None] * n
    ok = [True] * n
    if t == ARITHMETIC_PER_PACKET:
        freq = [None] * n

        def one(k):
            freq[k] = am.get_frequency(body[k])
            coded[k] = np.zeros(0, dtype=np.uint8) if len(body[k]) == 0 else am.get_interval_value_fast_fenwick(body[k], freq[k])
        vs.parallel(n, one)
        freq_type = am.get_frequency_type(freq)
        zipped = am.deflate_frequencies(freq, freq_type, 9)
        head = struct.pack(">ii", freq_type, len(zipped)) + zipped
        if warning is not None:
            def check(k):
                ok[k] = len(body[k]) == 0 or np.array_equal(
                    body[k], am.get_arithmetic_values_fast_fenwick(coded[k], freq[k], len(body[k])))
            vs.parallel(n, check)
    elif t == HUFFMAN_PER_PACKET:
        table = [None] * n

        def one(k):
            table[k] = cm.get_regular_huffman_length(am.get_frequency(body[k]))
            coded[k] = cm.pack_regular_code(body[k], table[k])
        vs.parallel(n, one)
        tables = cm.pack_regular_tables(table, 9)
        head = struct.pack(">i", len(tables)) + tables
        if warning is not None:
            try:
                unpacked = cm.unpack_regular_tables(tables, n)

                def check(k):
                    ok[k] = np.array_equal(body[k], cm.unpack_regular_code(coded[k], unpacked[k], len(body[k])))
                vs.parallel(n, check)
            except Exception:
                ok[0] = False
    else:
        def one(k):
            coded[k] = am.get_interval_value_adaptive(body[k])
        vs.parallel(n, one)
        head = b""
        if warning is not None:
            def check(k):
                ok[k] = np.array_equal(body[k], am.get_arithmetic_values_adaptive(coded[k], len(body[k])))
            vs.parallel(n, check)
    if not all(ok):
        warning[i] = "packets do not decode back to their segments."
    lengths = cm.pack_regular_lengths(coded)
    return head + struct.pack(">i", len(lengths)) + lengths + b"".join(np.asarray(c, dtype=np.uint8).tobytes() for c in coded)


# ---- Size estimates for Auto -------------------------------------------------
# Auto only compares levels, so Arithmetic isn't coded: a block's size is
# predicted from its byte counts, and frequency tables use Deflate's default
# setting. Save codes the chosen level for real.

def _log_factorial(k):
    """ln(k!) by Stirling's series."""
    if k < 2:
        return 0.0
    x = float(k)
    return x * math.log(x) - x + 0.5 * math.log(2 * math.pi * x) + 1 / (12 * x) - 1 / (360 * x * x * x)


def _block_estimate(f, n):
    """About log2(n! / (c0! c1! ...)) bits, plus the range coder's leading
    byte and its flush."""
    if n == 0:
        return 0
    ln = _log_factorial(n)
    for v in f:
        if v > 0:
            ln -= _log_factorial(int(v))
    return int(math.ceil(ln / math.log(2) / 8)) + 6


def _table_estimate(freq):
    return len(am.deflate_frequencies(freq, am.get_frequency_type(freq), DEFAULT_COMPRESSION))


def estimate_arithmetic(payload, pixel_segment):
    if len(payload) == 0:
        return 0
    blocks = am.get_blocks(payload, arithmetic_blocks(len(payload), pixel_segment))
    freq = [am.get_frequency(b) for b in blocks]
    size = 12 + _table_estimate(freq)
    for f, b in zip(freq, blocks):
        size += 4 + _block_estimate(f, len(b))
    return size


def estimate_per_packet(segments):
    freq = []
    size = 12
    for s in segments:
        f = am.get_frequency(s[:-1])
        freq.append(f)
        b = _block_estimate(f, len(s) - 1)
        size += b + cm.get_varint_bytes(b)
    return size + _table_estimate(freq)


def estimate_huffman_per_packet(segments):
    """Exact apart from the tables, which use Deflate's default setting."""
    n = len(segments)
    table = [None] * n
    coded = [0] * n

    def one(k):
        freq = am.get_frequency(segments[k][:-1])
        table[k] = cm.get_regular_huffman_length(freq)
        coded[k] = cm.get_regular_code_bytes(freq, table[k])
    vs.parallel(n, one)
    size = 8 + len(cm.pack_regular_tables(table, DEFAULT_COMPRESSION))
    for c in coded:
        size += c + cm.get_varint_bytes(c)
    return size


def _ddiv(a, b):
    """Java double division: x/0 gives Infinity or NaN rather than an error."""
    if b != 0:
        return a / b
    return math.nan if a == 0 else math.copysign(math.inf, a)


def _f(spec, v):
    """Formats a float like Java's String.format, including NaN, Infinity."""
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return ("+" if "+" in spec and v > 0 else "-" if v < 0 else "") + "Infinity"
    return spec % v


# =============================================================================
# The coding, without the window
# =============================================================================

class PacketCoder:
    """Holds an image and the compression settings; survey() picks the
    channel set, apply() codes (and decodes, for the preview), save()
    segments, entropy codes and writes the file."""

    def __init__(self, rgb, file_length=0):
        self.image_ydim, self.image_xdim = rgb.shape[:2]
        # Channel 0 is bits 16-23 of the Java pixel (red), as the Java reads it.
        self.source = [rgb[:, :, c].reshape(-1).astype(np.int64) for c in range(3)]
        self.file_length = file_length

        self.pixel_quant = 4
        self.pixel_shift = 3
        self.pixel_segment = 10     # Arithmetic blocks: 10 = one block per channel; lower = blocks of 500+500*pixel_segment bytes
        self.correction = 0         # preview only
        self.min_set_id = 0
        self.delta_type = 2         # average; the Delta menu selection (not surveyed)
        self.entropy_type = 0
        self.scanline5_variant = 0
        self.block_size = dm.BLOCK_DEFAULT
        self.block_set = 0
        self.packet_level = 0
        self.packet_auto = False    # Save picks the level (choose_level)

        self.channel_sum = [0] * 6
        self.set_sum = [0] * 10
        self.channel_min = [0] * 6
        self.channel_init = [0] * 6
        self.channel_delta_min = [0] * 6
        self.channel_length = [0] * 6
        self.applied = False

    # ---- Channels -------------------------------------------------------------

    def quantized_channels(self, size):
        """The six candidate channels after resizing and quantizing; sets
        channel_min and channel_init."""
        q = [None] * 3

        def one(i):
            ch = self.source[i]
            if self.pixel_quant != 0:
                ch = rm.resize(ch, self.image_xdim, size[0], size[1])
            q[i] = dm.quantize_channel(ch, self.pixel_shift)
        vs.parallel(3, one)
        qc, mins = dm.get_candidate_channels(q[0], q[1], q[2])
        self.channel_min = mins
        self.channel_init = [int(c[0]) for c in qc]
        return qc

    def compute_set_sums(self, qc, size):
        def one(i):
            self.channel_sum[i] = int(np.floor(cm.get_shannon_limit(dm.get_ideal_frequency2(qc[i], size[0], size[1]))))
        vs.parallel(6, one)
        for s in range(10):
            c = dm.get_channels(s)
            self.set_sum[s] = self.channel_sum[c[0]] + self.channel_sum[c[1]] + self.channel_sum[c[2]]
        self.min_set_id = min(range(10), key=lambda s: (self.set_sum[s], s))

    def survey(self):
        """Picks the channel set with the smallest entropy estimate (Java:
        init). The delta type is the Delta menu's, not surveyed."""
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        self.compute_set_sums(self.quantized_channels(size), size)
        self.print_channel_set_ranking()

    def print_channel_set_ranking(self):
        order = sorted(range(10), key=lambda s: (self.set_sum[s], s))
        print("Channel sets (ranked):")
        for r, s in enumerate(order):
            c = dm.get_channels(s)
            print("  %2d. %-32s %10d %10d %10d %12d%s" % (r + 1, dm.SET_NAMES[s], self.channel_sum[c[0]],
                  self.channel_sum[c[1]], self.channel_sum[c[2]], self.set_sum[s], " **" if s == self.min_set_id else ""))
        print()

    # ---- Apply --------------------------------------------------------------------

    def apply(self):
        """Quantizes, picks the channel set and packs the deltas as
        uncompressed unary strings (Save segments and compresses them), then
        decodes them the way packet_reader does. Returns the preview as an
        (ydim, xdim, 3) uint8 array."""
        self.applied = False
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        w, h = size
        qc = self.quantized_channels(size)
        self.compute_set_sums(qc, size)
        ids = dm.get_channels(self.min_set_id)
        self.delta_xdim = w
        tables, strings, maps, deltas, decoded = ([None] * 3 for _ in range(5))

        def channel(i):
            j = ids[i]
            d, m, _ = dm.get_deltas(qc[j], w, h, self.delta_type, self.scanline5_variant, self.block_size, self.block_set)
            maps[i] = m
            deltas[i] = d
            lo, bits, tbl, s = sm.get_string_list(d, False)
            self.channel_delta_min[j], self.channel_length[j] = lo, bits
            tables[i], strings[i] = tbl, s
            # Decode back, as packet_reader will.
            d2 = sm.unpack_strings(sm.decompress_strings(s), tbl, w * h, bits)
            d2[0] = 0
            d2[1:] += lo
            ch = dm.get_values_from_deltas(d2, w, h, self.channel_init[j], self.delta_type, m, self.scanline5_variant)
            if j > 2:
                ch = ch + self.channel_min[j]
            decoded[i] = ch
        vs.parallel(3, channel)
        self.table_list, self.string_list, self.map_list, self.delta_list = tables, strings, maps, deltas

        # As the reader: recombine the channel set first, then resize, then shift.
        bgr = dm.get_blue_green_red(self.min_set_id, *decoded)

        def restore(c):
            v = bgr[c]
            if self.pixel_quant != 0:
                v = rm.resize(v, w, self.image_xdim, self.image_ydim)
            if self.pixel_shift != 0:
                v = dm.shift(v, self.pixel_shift)
            if self.correction != 0:
                f = self.correction / 10.0
                v = v + ((self.source[c] - v) * f).astype(np.int64)     # (int) truncates toward zero
            bgr[c] = v
        vs.parallel(3, restore)
        rgb = np.stack([np.clip(v, 0, 255).reshape(self.image_ydim, self.image_xdim) for v in bgr], axis=2).astype(np.uint8)
        self.applied = True
        return rgb

    # ---- Save -------------------------------------------------------------------------

    def save(self, filename="foo"):
        """Header, then per channel: min, init, delta min, bit length, map
        (types 6-13), string table, the string's data byte, the segment table
        (raw length, Deflated length, Deflated bytes), then the entropy-coded
        payload. Returns the file size."""
        out = DataOutput()
        if self.entropy_type == CONTEXT:
            self._save_context(out)
        else:
            packets, coded = self._save_packets(out)
        with open(filename, "wb") as f:
            f.write(out.to_bytes())
        if self.entropy_type != CONTEXT:
            self.print_breakdown(packets, coded)
        raw = self.image_xdim * self.image_ydim * 3
        print("Original compression rate: %.4f" % (self.file_length / raw))
        print("Output  compression rate:  %.4f" % (out.size() / raw))
        print()
        return out.size()

    def _header(self, out):
        out.write_byte(FORMAT_ID); out.write_byte(FORMAT_VERSION)
        out.write_short(self.image_xdim); out.write_short(self.image_ydim)
        for v in (self.pixel_shift, self.pixel_quant, self.min_set_id, self.delta_type,
                  self.entropy_type, self.scanline5_variant, self.packet_level):
            out.write_byte(v)

    def _channel_start(self, out, i, j):
        for v in (self.channel_min[j], self.channel_init[j], self.channel_delta_min[j], self.channel_length[j]):
            out.write_int(v)
        if dm.has_map(self.delta_type):
            dm.write_map(out, self.delta_type, self.map_list[i], self.map_list[i - 1] if i > 0 else None, self.delta_xdim)

    def _save_packets(self, out):
        if self.packet_auto:
            self.packet_level = self.choose_level()

        # Segment each channel's string and check the round trip packet_reader will perform.
        timer = vs.Timer()
        string_data = [0] * 3
        packets = [None] * 3
        restore_ok = [False] * 3

        def segment(i):
            string = self.string_list[i]
            string_data[i] = int(string[-1])
            packets[i] = Packet(segment_string(string, self.packet_level))
            try:
                restore_ok[i] = same_bits(string, unpack_packet(packets[i], string_data[i]))
            except Exception:
                restore_ok[i] = False
        vs.parallel(3, segment)
        for i in range(3):
            if not restore_ok[i]:
                print("WARNING: channel %d packet does not restore to the original string." % i)
        print("Segmentation (level %d) took %s" % (self.packet_level, timer.elapsed()))

        timer = vs.Timer()
        warning = [None] * 3
        coded = [None] * 3

        def code(i):
            if is_per_packet(self.entropy_type):
                coded[i] = code_packets(packets[i].segments, self.entropy_type, warning, i)
            else:
                coded[i] = code_payload(packets[i].payload, self.entropy_type, self.pixel_segment, warning, i)
        vs.parallel(3, code)
        print("Entropy coding [%s] took %s" % (ENTROPY_NAMES[self.entropy_type], timer.elapsed()))
        for i in range(3):
            if warning[i] is not None:
                print("WARNING: channel %d %s" % (i, warning[i]))

        ids = dm.get_channels(self.min_set_id)
        self._header(out)
        for i in range(3):
            self._channel_start(out, i, ids[i])
            dm.write_table(out, self.table_list[i])
            out.write_byte(string_data[i])
            out.write_int(len(packets[i].table)); out.write_int(len(packets[i].ztable)); out.write(packets[i].ztable)
            out.write(coded[i])
        return packets, coded

    def _save_context(self, out):
        """Context: per channel min, init, delta min, bit length, map (types
        6-13), then the context-coded deltas. Packets don't apply."""
        timer = vs.Timer()
        coded = context_code(self.delta_list, self.delta_xdim)
        print("Entropy coding [%s] took %s" % (ENTROPY_NAMES[self.entropy_type], timer.elapsed()))
        ids = dm.get_channels(self.min_set_id)
        self._header(out)
        for i in range(3):
            self._channel_start(out, i, ids[i])
            out.write(coded[i])

    def entropy_size(self, payload):
        """Bytes code_payload writes for a payload with the selected coder."""
        return 0 if len(payload) == 0 else len(code_payload(payload, self.entropy_type, self.pixel_segment))

    def print_breakdown(self, packets, coded):
        """Per channel: segmented vs whole string, before and after entropy coding."""
        ids = dm.get_channels(self.min_set_id)
        whole = [None] * 3
        ew = [0] * 3

        def one(i):
            whole[i] = Packet([sm.compress_strings(self.string_list[i])])
            ew[i] = self.entropy_size(whole[i].payload) + len(whole[i].ztable)
        vs.parallel(3, one)
        name = ENTROPY_NAMES[self.entropy_type]
        print("Packet level %d%s, entropy %s" % (self.packet_level, " (Auto)" if self.packet_auto else "", name))
        for i in range(3):
            sp, wp = packets[i], whole[i]
            U, W, S = sm.get_bitlength(self.string_list[i]), wp.bits, sp.bits
            Tw, Ts = len(wp.ztable) * 8, len(sp.ztable) * 8
            d_payload, d_over = S - W, Ts - Tw
            es = len(coded[i]) + len(sp.ztable)
            print("Channel %d (%s):" % (i, CHANNEL_NAMES[ids[i]]))
            print("  uncompressed string          %10d bits" % U)
            print("  whole string, compressed     %10d bits   ratio %s   table %6d bits" % (W, _f("%.4f", _ddiv(W, U)), Tw))
            print("  segmented (%5d segs, %5d compr) %4s%10d bits   ratio %s   table %6d bits (raw %d)"
                  % (len(sp.segments), sp.compressed, "", S, _f("%.4f", _ddiv(S, U)), Ts, len(sp.table) * 8))
            print("  payload difference  (S - W)  %+10d bits" % d_payload)
            print("  overhead difference (Ts - Tw)%+10d bits" % d_over)
            print("  total difference             %+10d bits   (%s%% of whole)" % (d_payload + d_over, _f("%.2f", _ddiv(100.0 * (d_payload + d_over), W + Tw))))
            print("  after %-10s whole %8d B   segmented %8d B   difference %+d B (%s%%)   [entropy output + table]"
                  % (name, ew[i], es, es - ew[i], _f("%+.2f", _ddiv(100.0 * (es - ew[i]), ew[i]))))

    def choose_level(self):
        """The Segment Length level with the smallest output for the selected
        entropy coder: segment table plus coded payload over the 3 channels.
        Levels run from 10 down; one with the same minimum segment lengths as
        the previous reuses its result. Stops once a level is more than
        AUTO_STOP_PERCENT above the best so far. Ties go to the higher level."""
        timer = vs.Timer()
        cost = [-1] * 11
        packets_at = [0] * 11
        previous = None
        best_cost = sys.maxsize
        t = self.entropy_type
        for level in range(10, -1, -1):
            signature = "".join("%d," % get_minimum_segment_bits(sm.get_bitlength(s), level) for s in self.string_list)
            if signature == previous:
                cost[level] = cost[level + 1]
                packets_at[level] = packets_at[level + 1]
                continue
            previous = signature
            c = [0] * 3
            n = [0] * 3

            def one(i):
                segs = segment_string(self.string_list[i], level)
                p = Packet(segs)
                n[i] = len(segs)
                if t == ARITHMETIC_PER_PACKET:
                    size = estimate_per_packet(segs)
                elif t == HUFFMAN_PER_PACKET:
                    size = estimate_huffman_per_packet(segs)
                elif t == ADAPTIVE_PER_PACKET:
                    size = len(code_packets(segs, t))
                elif t == 2:
                    size = estimate_arithmetic(p.payload, self.pixel_segment)
                else:
                    size = self.entropy_size(p.payload)
                c[i] = len(p.ztable) + size
            vs.parallel(3, one)
            cost[level] = sum(c)
            packets_at[level] = sum(n)
            if cost[level] < best_cost:
                best_cost = cost[level]
            elif cost[level] > best_cost * (1 + AUTO_STOP_PERCENT / 100):
                break
        best = 10
        for level in range(9, -1, -1):
            if 0 <= cost[level] < cost[best]:
                best = level
        estimated = t in (2, ARITHMETIC_PER_PACKET, HUFFMAN_PER_PACKET)
        print("Auto Segment Length (%s), bytes by level%s:" % (ENTROPY_NAMES[t], " (estimated)" if estimated else ""))
        for level in range(11):
            if cost[level] < 0:
                print("  %2d  (skipped)" % level)
            else:
                print("  %2d  %8d packets  %10d B  %7s%%%s" % (level, packets_at[level], cost[level],
                      _f("%+.2f", _ddiv(100.0 * (cost[level] - cost[10]), cost[10])), "  <" if level == best else ""))
        print("  (%% vs level 10, the whole string); chose level %d in %s" % (best, timer.elapsed()))
        return best


# =============================================================================
# The window
# =============================================================================

def open_image(parent=None):
    from PySide6.QtWidgets import QFileDialog
    path, _ = QFileDialog.getOpenFileName(parent, "Open Image", "",
                                          "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pgm *.ppm);;All files (*)")
    if path:
        PacketWriter(path)


_open_writers = []


class PacketWriter:
    def __init__(self, filename):
        from PySide6.QtGui import QAction, QActionGroup, QKeySequence

        self.filename = filename
        try:
            rgb = vs.read_image(filename)
        except IOError as e:
            vs.show_error(None, str(e))
            return
        self.coder = c = PacketCoder(rgb, os.path.getsize(filename))
        self.updating = False
        print("Loaded %s, %d x %d" % (filename, c.image_xdim, c.image_ydim))

        self.view = view = vs.ImageWindow("Packet Writer  " + filename, c.image_xdim, c.image_ydim)
        bar = view.menuBar()
        _open_writers.append(self)
        view.destroyed.connect(lambda *_: _open_writers.remove(self) if self in _open_writers else None)

        # ---- File
        file_menu = bar.addMenu("File")
        a = QAction("Open...", view, shortcut=QKeySequence("Ctrl+O"))
        a.triggered.connect(lambda: open_image(view))
        file_menu.addAction(a)
        file_menu.addSeparator()
        a = QAction("Reset", view)
        a.triggered.connect(self.reset)
        file_menu.addAction(a)
        a = QAction("Save", view)
        a.triggered.connect(self.save)
        file_menu.addAction(a)

        view.make_view_menu()

        # ---- Quantization
        quant = bar.addMenu("Quantization")
        self.sliders = {}
        for key, title, lo, hi in (("pixel_quant", "Pixel Resolution", 0, 10), ("pixel_shift", "Color Resolution", 0, 7)):
            act, self.sliders[key] = vs.make_slider_dialog(view, title, lo, hi, getattr(c, key), self._setter(key))
            quant.addAction(act)
        # Error Correction blends the preview back toward the original by
        # correction/10. Preview only; the saved file is unaffected.
        quant.addSeparator()
        act, self.sliders["correction"] = vs.make_slider_dialog(view, "Error Correction", 0, 10, c.correction, self._setter("correction"))
        quant.addAction(act)

        # ---- Delta
        delta_menu = bar.addMenu("Delta")
        group = QActionGroup(view)
        self.delta_actions = []
        for t, name in enumerate(DELTA_MENU_NAMES):
            a = QAction(name, view, checkable=True)
            a.setChecked(t == c.delta_type)
            group.addAction(a)
            delta_menu.addAction(a)
            a.triggered.connect(lambda _=False, t=t: self.set_delta_type(t))
            self.delta_actions.append(a)
        # Block map settings (delta type 13); a change re-applies only when
        # the block map is selected.
        delta_menu.addSeparator()
        act, _ = vs.make_spinner_dialog(view, "Block Size", dm.BLOCK_MIN, dm.BLOCK_MAX, c.block_size, self.set_block_size)
        delta_menu.addAction(act)
        act, _ = vs.make_radio_dialog(view, "Block Predictors", dm.BLOCK_SET_NAMES, c.block_set, self.set_block_set)
        delta_menu.addAction(act)

        # ---- Packet: minimum segment length for string segmentation (Save only).
        packet_menu = bar.addMenu("Packet")
        packet_menu.addAction(self._make_segment_length_dialog())

        # ---- Entropy (Save only)
        entropy_menu = bar.addMenu("Entropy")
        group = QActionGroup(view)
        for et, name in enumerate(ENTROPY_NAMES):
            a = QAction(name, view, checkable=True)
            a.setChecked(et == c.entropy_type)
            group.addAction(a)
            entropy_menu.addAction(a)
            a.triggered.connect(lambda _=False, et=et: setattr(self.coder, "entropy_type", et))
        entropy_menu.addSeparator()
        act, _ = vs.make_slider_dialog(view, "Segment Size", 0, 10, c.pixel_segment, lambda v: setattr(c, "pixel_segment", v))
        entropy_menu.addAction(act)

        view.set_image(rgb)
        view.show_window()
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self.show_initial_image)

    def _make_segment_length_dialog(self):
        """Segment Length: a 0-10 slider plus an Auto toggle. With Auto on,
        the slider is disabled and Save picks the level (then shows it here)."""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QAction
        from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLineEdit, QSlider, QVBoxLayout
        c = self.coder
        dialog = QDialog(self.view)
        dialog.setWindowTitle("Segment Length")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 10)
        slider.setValue(c.packet_level)
        slider.setTickInterval(1)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setMinimumWidth(220)
        field = QLineEdit(str(c.packet_level))
        field.setReadOnly(True)
        field.setFixedWidth(field.fontMetrics().horizontalAdvance("0000") + 12)
        auto = QCheckBox("Auto")
        auto.setChecked(c.packet_auto)
        slider.setEnabled(not c.packet_auto)

        def level_changed(v):
            field.setText(str(v))
            c.packet_level = v

        def auto_changed(on):
            c.packet_auto = on
            slider.setEnabled(not on)
        slider.valueChanged.connect(level_changed)
        auto.toggled.connect(auto_changed)
        row = QHBoxLayout()
        row.addWidget(slider)
        row.addWidget(field)
        layout = QVBoxLayout(dialog)
        layout.addLayout(row)
        layout.addWidget(auto)
        self.packet_slider = slider
        action = QAction("Segment Length", self.view)
        action.triggered.connect(lambda: vs._open_dialog(self.view, dialog))
        return action

    # ---- Settings -----------------------------------------------------------------

    def _setter(self, key):
        def set_value(v):
            setattr(self.coder, key, v)
            if not self.updating:
                self.apply()
        return set_value

    def set_delta_type(self, t):
        if self.coder.delta_type != t:
            self.coder.delta_type = t
            self.apply()

    def set_block_size(self, v):
        self.coder.block_size = v
        if self.coder.delta_type == 13:
            self.apply()

    def set_block_set(self, v):
        self.coder.block_set = v
        if self.coder.delta_type == 13:
            self.apply()

    def reset(self):
        c = self.coder
        c.pixel_quant = c.pixel_shift = c.correction = 0
        self.updating = True
        for key, slider in self.sliders.items():
            slider.setValue(getattr(c, key))
        self.updating = False
        self.apply()

    # ---- Work -----------------------------------------------------------------------

    def apply(self):
        try:
            self.view.set_image(self.coder.apply())
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Apply: %r" % e)

    def show_initial_image(self):
        """Applies the default quantization, then runs the survey in the
        background with the menus disabled so Apply and Save can't overlap it."""
        self.apply()
        self.view.set_menus_enabled(False)
        self.view.set_status("analyzing…")

        def done(result, error):
            if error is not None:
                print("init: %r" % error)
            self.view.set_menus_enabled(True)
            self.view.set_status(None)
            self.delta_actions[self.coder.delta_type].setChecked(True)
            self.apply()
        vs.run_in_background(self.coder.survey, done)

    def save(self):
        if not self.coder.applied:
            self.apply()
        if not self.coder.applied:
            vs.show_error(self.view, "Nothing saved: the last Apply failed.")
            return
        try:
            self.coder.save("foo")
        except Exception as e:
            import traceback
            traceback.print_exc()
            if os.path.exists("foo"):
                os.remove("foo")
            vs.show_error(self.view, "Save failed: %r" % e)
            return
        if self.coder.packet_auto:
            self.packet_slider.setValue(self.coder.packet_level)


def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    if len(sys.argv) > 1:
        PacketWriter(sys.argv[1])
    else:
        open_image()
    if not _open_writers:
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
