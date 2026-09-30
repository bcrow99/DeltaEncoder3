"""
segment_mapper.py -- the parts of SegmentMapper.java (version 1.1) the Packet
programs use: splitting a unary string into segments that are compressed
separately (get_segmented_data3: segment -> merge3 -> combine3), packing
the segments' bits back to back (pack_segments3 / unpack_segments3), and
rejoining them (restore2).

A string or segment is a numpy uint8 array whose last byte is its data byte,
as in string_mapper.py: bits 0-4 the compression iterations (+16 for the
one-bit transform, or on an uncompressed segment "mostly ones"), bits 5-7
the number of unused bits in the last data byte. Bits are used least
significant first within a byte.

This replaces the older, broader segment_mapper.py translation, which
predated the current string_mapper.py (it called functions that no longer
exist) and SegmentMapper's later fixes.
"""

import numpy as np

import string_mapper as sm

# Zero bits in each byte value (StringMapper.getBitTable).
_ZEROS = np.array([8 - bin(v).count("1") for v in range(256)], dtype=np.int64)


def _u8(a):
    return sm._u8(a)


def get_zero_ratio(string, bit_length):
    """Fraction of zero bits among the first bit_length bits (NaN for none,
    like Java's 0.0/0)."""
    byte_length = bit_length // 8
    zero_sum = int(_ZEROS[string[:byte_length]].sum())
    one_sum = byte_length * 8 - zero_sum
    remainder = bit_length % 8
    if remainder:
        last = int(string[byte_length])
        ones = bin(last & ((1 << remainder) - 1)).count("1")
        one_sum += ones
        zero_sum += remainder - ones
    total = zero_sum + one_sum
    return zero_sum / total if total else float("nan")


def get_bin_number(ratio, bin_width):
    """Which bin of width bin_width a ratio falls in (the Java's running sum,
    so floating-point edges land where they do in Java)."""
    number = 0
    total = bin_width
    while ratio > total:
        number += 1
        total += bin_width
    return number


def _is_similar_bin(merge_type, current_bin, next_bin, bin_divider, difference):
    if merge_type == 0:
        return (current_bin < bin_divider and next_bin < bin_divider) or \
               (current_bin >= bin_divider and next_bin >= bin_divider)
    if merge_type == 1:
        return (current_bin < bin_divider - 1 and next_bin < bin_divider - 1) or \
               (current_bin > bin_divider + 1 and next_bin > bin_divider + 1)
    if merge_type == 2:
        return (current_bin < bin_divider and next_bin < bin_divider and abs(current_bin - next_bin) < difference) or \
               (current_bin >= bin_divider and next_bin >= bin_divider and abs(current_bin - next_bin) < difference)
    if merge_type == 3:
        return current_bin == next_bin
    return False


def _uncompressed(iterations):
    return iterations == 0 or iterations == 16


# ---- Segmenting -------------------------------------------------------------

def segment(string, bitlength, bin_width):
    """Splits a string into segments of bitlength bits (a multiple of 8), the
    last one taking the remainder. Returns [segments, min_segment_bytelength,
    max_segment_bytelength, extra_bits, string_data, bin_number], or [] if
    bitlength isn't a multiple of 8."""
    if bitlength % 8 != 0:
        print("Segment bitlength must be a multiple of 8.")
        return []
    string = _u8(string)
    string_bitlength = sm.get_bitlength(string)
    number_of_segments = string_bitlength // bitlength
    string_data = int(string[-1])

    if number_of_segments == 0:
        # The whole string is shorter than one segment: one segment of it.
        last_bitlength = string_bitlength
        last_bytelength = last_bitlength // 8
        extra_bits = 0
        if last_bitlength % 8 != 0:
            extra_bits = ((8 - last_bitlength % 8) << 5) & 0xFF
            last_bytelength += 1
        last_bytelength += 1
        seg = np.zeros(last_bytelength, dtype=np.uint8)
        seg[:-1] = string[:last_bytelength - 1]
        seg[-1] = extra_bits
        zero_ratio = get_zero_ratio(seg, last_bitlength)
        bin_num = get_bin_number(zero_ratio, bin_width)
        if zero_ratio < .5:
            seg[-1] |= 16
        return [[seg], last_bytelength, last_bytelength, extra_bits, string_data, [bin_num]]

    segment_bytelength = bitlength // 8 + 1
    odd_bits = string_bitlength % bitlength
    last_bitlength = bitlength + odd_bits
    last_bytelength = last_bitlength // 8
    extra_bits = 0
    if odd_bits % 8 != 0:
        extra_bits = ((8 - odd_bits % 8) << 5) & 0xFF
        last_bytelength += 1
    last_bytelength += 1

    bin_number = [0] * number_of_segments
    segments = []
    step = segment_bytelength - 1
    for i in range(number_of_segments):
        start = i * step
        if i < number_of_segments - 1:
            seg = np.zeros(segment_bytelength, dtype=np.uint8)
            seg[:-1] = string[start:start + segment_bytelength - 1]
            zero_ratio = get_zero_ratio(seg, bitlength)
            bin_number[i] = get_bin_number(zero_ratio, bin_width)
            if zero_ratio < .5:
                seg[-1] = 16
        else:
            seg = np.zeros(last_bytelength, dtype=np.uint8)
            seg[:-1] = string[start:start + last_bytelength - 1]
            seg[-1] = extra_bits
            zero_ratio = get_zero_ratio(seg, last_bitlength)
            bin_number[i] = get_bin_number(zero_ratio, bin_width)
            if zero_ratio < .5:
                seg[-1] |= 16
        segments.append(seg)
    return [segments, segment_bytelength, last_bytelength, extra_bits, string_data, bin_number]


def _join(parts, extra=0):
    """The parts' data bytes (each without its data byte) back to back, plus
    a data byte of `extra`."""
    body = np.concatenate([p[:-1] for p in parts]) if parts else np.zeros(0, dtype=np.uint8)
    out = np.zeros(len(body) + 1, dtype=np.uint8)
    out[:-1] = body
    out[-1] = extra
    return out


def merge3(segments, bin_number, bin_width, min_segment_bytelength, max_segment_bytelength,
           extra_bits, string_data, merge_type, max_run):
    """Joins runs of up to max_run neighbouring segments with similar
    zero-bit ratios, then compresses every resulting segment. Returns
    [segments] if one is left, else [segments, min_bytelength, max_bytelength,
    extra_bits, string_data, uncompressed, uncompressed_adjacent,
    max_iterations]."""
    merged = []
    n = len(segments)
    number_of_bins = int(1. / bin_width)
    bin_divider = number_of_bins // 2
    difference = bin_divider // 2

    i = 0
    while i < n - 1:
        current_bin = bin_number[i]
        j = 1
        next_bin = bin_number[i + j]
        similar = max_run > 1 and _is_similar_bin(merge_type, current_bin, next_bin, bin_divider, difference)
        if similar:
            while similar and i + j < n - 1 and j + 1 < max_run:
                next_bin = bin_number[i + j + 1]
                similar = _is_similar_bin(merge_type, current_bin, next_bin, bin_divider, difference)
                if similar:
                    j += 1
            seg = _join(segments[i:i + j + 1])
            if i + j == n - 1:
                seg[-1] |= extra_bits
            ratio = get_zero_ratio(seg, sm.get_bitlength(seg))
            if ratio < .5:
                seg[-1] |= 16
            merged.append(seg)
            i += j
        else:
            merged.append(segments[i])
        i += 1
    if i == n - 1:
        merged.append(segments[i])

    compressed = [sm.compress_strings(s) for s in merged]

    max_segment_bytelength = 0
    min_segment_bytelength = 2147483647
    uncompressed = 0
    uncompressed_adjacent = 0
    previous_iterations = 1
    max_iterations = 0
    for c in compressed:
        max_segment_bytelength = max(max_segment_bytelength, len(c) - 1)
        min_segment_bytelength = min(min_segment_bytelength, len(c) - 1)
        it = sm.get_iterations(c)
        if _uncompressed(it):
            uncompressed += 1
        elif it > 16:
            max_iterations = max(max_iterations, it - 16)
        else:
            max_iterations = max(max_iterations, it)
        if _uncompressed(previous_iterations) and _uncompressed(it):
            uncompressed_adjacent += 1
        previous_iterations = it

    if len(compressed) == 1:
        return [compressed]
    return [compressed, min_segment_bytelength, max_segment_bytelength, extra_bits, string_data,
            uncompressed, uncompressed_adjacent, max_iterations]


def combine3(segments, min_segment_bytelength, max_segment_bytelength, extra_bits, string_data, max_packet_bytes):
    """Joins runs of neighbouring uncompressed segments while the result stays
    within max_packet_bytes. Returns [segments] if one is left, else
    [segments, min_bytelength, max_bytelength, extra_bits, string_data,
    uncompressed]."""
    combined = []
    n = len(segments)
    i = 0
    while i < n - 1:
        current = segments[i]
        current_it = sm.get_iterations(current)
        nxt = segments[i + 1]
        next_it = sm.get_iterations(nxt)
        if _uncompressed(current_it) and _uncompressed(next_it) and len(current) + len(nxt) - 2 <= max_packet_bytes:
            j = 1
            run_bytes = len(current) + len(nxt) - 2
            while _uncompressed(next_it) and i + j + 1 < n:
                nxt = segments[i + j + 1]
                next_it = sm.get_iterations(nxt)
                if _uncompressed(next_it) and run_bytes + len(nxt) - 1 <= max_packet_bytes:
                    j += 1
                    run_bytes += len(nxt) - 1
                else:
                    break
            seg = _join(segments[i:i + j + 1])
            max_segment_bytelength = max(max_segment_bytelength, len(seg) - 1)
            if i + j == n - 1:
                last_bitlength = (len(seg) - 1) * 8 - ((extra_bits >> 5) & 7)
                zero_ratio = get_zero_ratio(seg, last_bitlength)
                seg[-1] = extra_bits
            else:
                zero_ratio = get_zero_ratio(seg, (len(seg) - 1) * 8)
            if zero_ratio < .5:
                seg[-1] |= 16
            combined.append(seg)
            i += j
        else:
            combined.append(current)
        i += 1
    if i == n - 1:
        combined.append(segments[i])

    if len(combined) == 1:
        return [combined]
    uncompressed = sum(1 for s in combined if _uncompressed(sm.get_iterations(s)))
    return [combined, min_segment_bytelength, max_segment_bytelength, extra_bits, string_data, uncompressed]


def get_segmented_data3(string, minimum_bitlength, segment_type, merge_type, bin_width, maximum_bitlength):
    """Segments a string (segment_type 0: split only; 1: then merge; 2: then
    also combine), with no merged or combined packet longer than
    maximum_bitlength (<= minimum_bitlength: no merging). Returns
    [segments, max_segment_bytelength, string_data], or [] for an invalid
    minimum_bitlength or segment_type."""
    if minimum_bitlength % 8 != 0 or segment_type < 0 or segment_type > 2:
        return []
    segments, min_len, max_len, extra_bits, string_data, bin_number = segment(string, minimum_bitlength, bin_width)
    if segment_type == 0:
        return [segments, max_len, string_data]

    merged_list = merge3(segments, bin_number, bin_width, min_len, max_len, extra_bits, string_data,
                         merge_type, max(1, maximum_bitlength // minimum_bitlength))
    merged = merged_list[0]
    if len(merged) == 1:
        return [merged, min_len, string_data]
    min_len, max_len, extra_bits, string_data = merged_list[1:5]
    uncompressed_adjacent = merged_list[6]
    if segment_type == 1 or uncompressed_adjacent == 0:
        return [merged, max_len, string_data]

    combined_list = combine3(merged, min_len, max_len, extra_bits, string_data, maximum_bitlength // 8)
    combined = combined_list[0]
    if len(combined) > 1:
        max_len = combined_list[2]
    return [combined, max_len, string_data]


# ---- Packing ----------------------------------------------------------------

def _bits(seg, bitlength):
    return np.unpackbits(seg[:-1], bitorder="little")[:bitlength]


def pack_segments3(segments):
    """[every segment's bits back to back (no padding, no data bytes),
    bitlengths]."""
    bitlength = [sm.get_bitlength(s) for s in segments]
    if not segments:
        return [np.zeros(0, dtype=np.uint8), bitlength]
    bits = np.concatenate([_bits(s, b) for s, b in zip(segments, bitlength)])
    return [np.packbits(bits, bitorder="little"), bitlength]


def unpack_segments3(string, bytelength, data):
    """Inverse of pack_segments3: bytelength[i] is segment i's byte length
    without its data byte, data[i] its data byte."""
    string = _u8(string)
    stream = np.unpackbits(string, bitorder="little")
    segments = []
    offset = 0
    for blen, d in zip(bytelength, data):
        d = int(d) & 0xFF
        bitlength = blen * 8 - ((d >> 5) & 7)
        bits = np.zeros(blen * 8, dtype=np.uint8)
        piece = stream[offset:offset + bitlength]
        bits[:len(piece)] = piece
        seg = np.zeros(blen + 1, dtype=np.uint8)
        seg[:-1] = np.packbits(bits, bitorder="little")
        seg[-1] = d
        segments.append(seg)
        offset += bitlength
    return segments


# ---- Restoring --------------------------------------------------------------

def _shift_left(src, bit_shift):
    """SegmentMapper.shiftLeft for a shift of 1-7 bits: one byte longer."""
    low = (src.astype(np.int64) << bit_shift) & 0xFF
    high = src.astype(np.int64) >> (8 - bit_shift)
    dst = np.zeros(len(src) + 1, dtype=np.int64)
    dst[:-1] |= low
    dst[1:] |= high
    return dst.astype(np.uint8)


def restore2(segments, string_data):
    """Decompresses whichever segments were compressed and joins them at their
    bit offsets; the result ends with string_data as its data byte."""
    decompressed = [s if _uncompressed(sm.get_iterations(s)) else sm.decompress_strings(s) for s in segments]
    total = sum(sm.get_bitlength(s) for s in decompressed)
    dst = np.zeros((total + 7) // 8 + 1, dtype=np.uint8)
    bit_offset = 0
    byte_offset = 0
    for seg in decompressed:
        bitlength = sm.get_bitlength(seg)
        bit_shift = bit_offset % 8
        if bit_shift == 0:
            n = len(seg) - 1
            if byte_offset + n > len(dst):
                raise IndexError("segment runs past the end of the string")
            dst[byte_offset:byte_offset + n] = seg[:-1]
        else:
            shifted = _shift_left(seg[:-1], bit_shift)
            if byte_offset + len(shifted) > len(dst):
                raise IndexError("segment runs past the end of the string")
            dst[byte_offset] |= shifted[0]
            dst[byte_offset + 1:byte_offset + len(shifted)] = shifted[1:]
        bit_offset += bitlength
        byte_offset = bit_offset // 8
    dst[-1] = int(string_data) & 0xFF
    return dst
