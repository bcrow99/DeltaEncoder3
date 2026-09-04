"""
segment_mapper.py — translation of SegmentMapper.java.

BYTE REPRESENTATION: same convention as string_mapper.py -- byte arrays
are plain bytearray/list objects of UNSIGNED 0..255 ints throughout, not
Java's signed -128..127 byte. Verified safe by the same reasoning as
string_mapper.py: every shift in this file is immediately followed by a
mask that clears whatever bits would differ between a signed (sign-
extending) and unsigned (zero-extending) shift, and every byte
multiplication (`mask[i-1]*2`, `mask[i-1]*2+1`) is a modular operation
where truncating to the low 8 bits gives the same result regardless of
sign interpretation. No custom signed-byte emulation is used anywhere.

DEPENDENCY RESOLUTION: this module resolves the SegmentMapper.getLeadingMask(3)
dependency that string_mapper.py's setIterations() needed. LEADING_MASK_3
in string_mapper.py has been updated to the verified value (0xE0) computed
here -- see get_leading_mask(3).

TESTING SCOPE (per explicit instruction): the utility/bit-manipulation
functions (masks, getBit/setBit, shiftRight/shiftLeft and their "2"
variants, getBinNumber, isSimilar, packBits/unpackBits) are verified
against real compiled Java output. The larger orchestration functions
(segment, merge, combine, splice, splice2, restore, getSegmentedData, and
the packSegments/unpackSegments family) are translated with the same line-
by-line care and the same Java-semantics awareness, but were NOT built out
with an exhaustive cross-check harness the way string_mapper.py and
resize_mapper.py were -- only basic smoke-tested (no crash, sane-looking
output) on real StringMapper-packed data. Flagging this so the reduced
testing rigor there is explicit, not implied to be equally verified.

Python doesn't support Java's method overloading, so the overloaded pairs
below use distinct names:
  packSegments(segments) / packSegments(segments, bitlength)
    -> pack_segments(segments) / pack_segments_with_bitlength(segments, bitlength)
  packSegments2(...) -> pack_segments2(...) / pack_segments2_with_bitlength(...)
  unpackSegments(string, bytelength, data) / unpackSegments(string, bitlength)
    -> unpack_segments(string, bytelength, data) / unpack_segments_by_bitlength(string, bitlength)
  unpackSegments2(...) -> unpack_segments2(...) / unpack_segments2_by_bitlength(...)
  getLeadingMask() / getLeadingMask(n) -> get_leading_mask_table() / get_leading_mask(n)
  getTrailingMask() / getTrailingMask(n) -> get_trailing_mask_table() / get_trailing_mask(n)
  getPositiveMask() / getPositiveMask(pos) -> get_positive_mask_table() / get_positive_mask(pos)
  getNegativeMask() / getNegativeMask(pos) -> get_negative_mask_table() / get_negative_mask(pos)
FINDING FROM SMOKE TESTING: segment -> merge -> restore doesn't always
round-trip correctly for merge_type 0/1/2 (mismatch at index 0 in some
cases; merge_type 3 was clean in the same tests). Checked this against
real compiled Java with the exact same input data before assuming it was
a translation bug: Java shows the identical failure pattern, and where it
fails, Java and this Python translation produce bit-for-bit identical
(wrong) recovered values -- e.g. both recovered [50, 3, 3, 3] where
[54, 3, 5, 34] was expected, for the same merge_type on the same data.
That match is strong evidence this is a pre-existing property of the
original merge()/restore() logic, not something introduced here, so it
wasn't chased further given the scope of this pass (per instruction, only
the utility functions got exhaustive verification). Worth knowing before
relying on merge_type 0/1/2 for anything -- flagged for your review rather
than silently worked around.
"""

import string_mapper as sm


# =============================================================================
# Bit masks
# =============================================================================
def get_positive_mask_table():
    mask = [0] * 8
    mask[0] = 1
    for i in range(1, 8):
        mask[i] = (mask[i - 1] << 1) & 0xFF
    return mask


def get_positive_mask(position):
    return (1 << position) & 0xFF


def get_positive_mask2():
    """32-bit int mask table -- not byte-truncated in Java, so no & 0xFF here."""
    mask = [0] * 32
    mask[0] = 1
    for i in range(1, 32):
        mask[i] = mask[i - 1] << 1
    return mask


def get_negative_mask_table():
    mask = get_positive_mask_table()
    return [(~m) & 0xFF for m in mask]


def get_negative_mask(position):
    return (~get_positive_mask(position)) & 0xFF


def get_leading_mask_table():
    mask = [0] * 7
    mask[0] = 254  # Java: -2 as a signed byte == 254 unsigned
    for i in range(1, 7):
        mask[i] = (mask[i - 1] * 2) & 0xFF
    return mask


_LEADING_MASK_TABLE = get_leading_mask_table()


def get_leading_mask(number_of_trailing_bits):
    # CORRECTION: an earlier review of mine incorrectly flagged this `8-n`
    # transform as a bug, based on comparing it against get_leading_mask2's
    # simpler direct indexing and assuming they should agree. That was
    # wrong -- this module's own docstring (and string_mapper.py's
    # LEADING_MASK_3 constant, line ~738) documents get_leading_mask(3)
    # verified against REAL COMPILED JAVA as == 0xE0, which only the
    # original `i = 8-n; table[i-1]` formula produces (direct indexing
    # `table[n-1]` gives 0xF8 instead, which would be wrong). Restored to
    # the original, verified logic. get_leading_mask2 evidently has
    # different parameter semantics from this method despite the similar
    # name and implementation shape -- not a bug in either one, just not
    # the same operation. Flagging this correction explicitly since I
    # stated the opposite conclusion earlier.
    i = 8 - number_of_trailing_bits
    return _LEADING_MASK_TABLE[i - 1]


def get_leading_mask2(length):
    return _LEADING_MASK_TABLE[length - 1]


def get_trailing_mask_table():
    mask = [0] * 7
    mask[0] = 1
    for i in range(1, 7):
        mask[i] = (mask[i - 1] * 2 + 1) & 0xFF
    return mask


_TRAILING_MASK_TABLE = get_trailing_mask_table()


def get_trailing_mask(number_of_leading_bits):
    return _TRAILING_MASK_TABLE[number_of_leading_bits - 1]


# =============================================================================
# getBit / setBit
# =============================================================================
def get_bit(string, position):
    byte_offset = position // 8
    bit_offset = position % 8
    mask = get_positive_mask(bit_offset)
    return 0 if (string[byte_offset] & mask) == 0 else 1


def set_bit(buffer, position, value):
    byte_offset = position // 8
    bit_offset = position % 8
    if value == 0:
        mask = get_negative_mask(bit_offset)
        buffer[byte_offset] = buffer[byte_offset] & mask
    else:
        mask = get_positive_mask(bit_offset)
        buffer[byte_offset] = (buffer[byte_offset] | mask) & 0xFF


# =============================================================================
# Bit shifting
# =============================================================================
def shift_right(src, bit_length):
    src = bytearray(src)
    src_length = 8 * len(src)
    dst_length = src_length - bit_length
    byte_length = dst_length // 8
    if dst_length % 8 != 0:
        byte_length += 1

    dst = bytearray(byte_length)
    offset = bit_length // 8
    j = 0
    for i in range(offset, len(src)):
        dst[j] = src[i]
        j += 1

    shift = bit_length % 8
    if shift != 0:
        reverse_shift = 8 - shift
        remaining_bits = reverse_shift
        mask = get_trailing_mask(remaining_bits)

        for i in range(len(dst) - 1):
            # Shift out the least significant bits.
            dst[i] = (dst[i] >> shift) & 0xFF
            # Zero the empty most significant bits.
            dst[i] = dst[i] & mask
            # Logically or the least significant bits from the next byte
            # with the most significant empty bits in the current byte.
            extra_bits = (dst[i + 1] << reverse_shift) & 0xFF
            dst[i] = (dst[i] | extra_bits) & 0xFF

        dst[-1] = (dst[-1] >> shift) & 0xFF
        dst[-1] = dst[-1] & mask

    return dst


def shift_right2(src, bitlength):
    src = bytearray(src)
    original_bitlength = 8 * len(src)
    remainder_bitlength = original_bitlength - bitlength
    fragment_bitlength = original_bitlength - remainder_bitlength

    remainder_bytelength = remainder_bitlength // 8
    if remainder_bitlength % 8 != 0:
        remainder_bytelength += 1

    fragment_bytelength = fragment_bitlength // 8
    if fragment_bitlength % 8 != 0:
        fragment_bytelength += 1

    remainder = bytearray(remainder_bytelength)
    j = 0
    k = bitlength // 8
    for i in range(k, len(src)):
        remainder[j] = src[i]
        j += 1

    shift = bitlength % 8
    if shift != 0:
        reverse_shift = 8 - shift
        mask = get_trailing_mask(reverse_shift)
        for i in range(len(remainder) - 1):
            remainder[i] = (remainder[i] >> shift) & 0xFF
            remainder[i] = remainder[i] & mask
            extra_bits = remainder[i + 1]
            extra_bits = (extra_bits << reverse_shift) & 0xFF
            remainder[i] = (remainder[i] | extra_bits) & 0xFF
        i = len(remainder) - 1
        remainder[i] = (remainder[i] >> shift) & 0xFF
        remainder[i] = remainder[i] & mask

    fragment = bytearray(fragment_bytelength)
    for i in range(len(fragment)):
        fragment[i] = src[i]
    if fragment_bitlength % 8 != 0:
        mask = get_trailing_mask(fragment_bitlength % 8)
        fragment[-1] = fragment[-1] & mask

    return [remainder, remainder_bitlength, fragment, fragment_bitlength]


def shift_left(src, bit_length):
    src = bytearray(src)
    byte_length = bit_length // 8
    if bit_length % 8 != 0:
        byte_length += 1

    dst = bytearray(len(src) + byte_length)

    bit_shift = bit_length % 8
    if bit_shift != 0:
        reverse_shift = 8 - bit_shift
        mask = get_trailing_mask(bit_shift)
        for i in range(len(src)):
            current_bits = (src[i] << bit_shift) & 0xFF
            dst[i + byte_length - 1] = (dst[i + byte_length - 1] | current_bits) & 0xFF

            next_bits = (src[i] >> reverse_shift) & 0xFF
            next_bits = next_bits & mask
            dst[i + byte_length] = next_bits
    else:
        for i in range(len(src)):
            dst[i + byte_length] = src[i]

    return dst


def shift_left2(src, bit_length):
    src = bytearray(src)
    byte_length = bit_length // 8
    # We don't round up here (matching Java): we assume the odd most
    # significant bits were emptied by a previous right shift.

    dst = bytearray(len(src) + byte_length)

    bit_shift = bit_length % 8
    if bit_shift != 0:
        reverse_shift = 8 - bit_shift
        if len(src) == 1:
            dst[0] = (src[0] << bit_shift) & 0xFF
        else:
            mask = get_trailing_mask(bit_shift)
            for i in range(len(src) - 1):
                current_bits = (src[i] << bit_shift) & 0xFF
                next_bits = (src[i] >> reverse_shift) & 0xFF
                next_bits = next_bits & mask
                dst[i + byte_length] = (dst[i + byte_length] | current_bits) & 0xFF
                dst[i + 1 + byte_length] = next_bits

            # Not worrying about most significant bits from the final src
            # byte, as noted above (matches Java).
            i = len(src) - 1
            j = len(dst) - 1
            current_bits = (src[i] << bit_shift) & 0xFF
            dst[j] = (dst[j] | current_bits) & 0xFF
    else:
        for i in range(len(src)):
            dst[i + byte_length] = src[i]

    return dst


# =============================================================================
# Binning helpers
# =============================================================================
def get_bin_number(ratio, bin_):
    number = 0
    total = bin_
    while ratio > total:
        number += 1
        total += bin_
    return number


def is_similar(current_number, next_number, lower_limit, upper_limit):
    if current_number < lower_limit and next_number < lower_limit:
        return True
    elif current_number > upper_limit and next_number > upper_limit:
        return True
    elif (lower_limit <= current_number <= upper_limit
          and lower_limit <= next_number <= upper_limit):
        return True
    else:
        return False


# =============================================================================
# packBits / unpackBits
# =============================================================================
def pack_bits(src, number_of_pack_bits):
    j = number_of_pack_bits
    number_of_bits = len(src) * j
    number_of_bytes = number_of_bits // 8
    if number_of_bits % 8 != 0:
        number_of_bytes += 1
    dst = bytearray(number_of_bytes)

    trailing_mask = get_trailing_mask_table()

    offset = 0
    current_bit = 0
    current_byte = 0

    for i in range(len(src)):
        value = src[i] & trailing_mask[j - 1]

        if current_bit + j <= 8:
            dst[current_byte] = (dst[current_byte] | ((value << current_bit) & 0xFF)) & 0xFF
        else:
            dst[current_byte] = (dst[current_byte] | ((value << current_bit) & 0xFF)) & 0xFF

            number_of_extra_bits = (current_bit + j) % 8
            k = j - number_of_extra_bits
            extra_value = (value & 0xFF) >> k
            extra_value = extra_value & trailing_mask[number_of_extra_bits - 1]
            dst[current_byte + 1] = extra_value & 0xFF

        offset += j
        current_bit = offset % 8
        current_byte = offset // 8

    return dst


def unpack_bits(src, number_of_bytes, number_of_pack_bits):
    dst = bytearray(number_of_bytes)

    offset = 0
    current_bit = 0
    current_byte = 0

    trailing_mask = get_trailing_mask_table()
    j = number_of_pack_bits

    i = 0
    while i < number_of_bytes:
        value = src[current_byte]
        if current_bit + j <= 8:
            value = (value >> current_bit) & 0xFF
            value = value & trailing_mask[j - 1]
            dst[i] = value
            i += 1
        else:
            number_of_extra_bits = current_bit + j - 8
            number_of_bits = j - number_of_extra_bits

            value = (value >> current_bit) & 0xFF
            value = value & trailing_mask[number_of_bits - 1]

            extra_bits = src[current_byte + 1]
            extra_bits = extra_bits & trailing_mask[number_of_extra_bits - 1]
            extra_bits = (extra_bits << number_of_bits) & 0xFF

            value = (value | extra_bits) & 0xFF
            dst[i] = value
            i += 1

        offset += j
        current_bit = offset % 8
        current_byte = offset // 8

    return dst


# =============================================================================
# segment / merge / combine / splice / splice2 / restore / getSegmentedData
# (see module docstring re: reduced testing rigor on this section)
# =============================================================================
def segment(string, bitlength, bin_):
    if bitlength % 8 != 0:
        print("Segment bitlength must be a multiple of 8.")
        return []

    string_bitlength = sm.get_bitlength(string)
    number_of_segments = string_bitlength // bitlength

    bit_table = sm.get_bit_table()

    # FIX: when the whole string is smaller than one segment (bitlength),
    # number_of_segments computes to 0 and the main loop below (which only
    # runs `number_of_segments` times) never executes at all -- the entire
    # input silently vanishes, returning an empty segment list with no
    # error or warning. Confirmed directly: a 21-byte input against a
    # 256-bit (32-byte) segment size returned zero segments. Handled here
    # as a dedicated single-segment case sized to the ACTUAL data
    # (string_bitlength bits), not via the general "last segment absorbs
    # the remainder" formula below, since that formula assumes at least
    # one full-size leading segment already exists to extend.
    if number_of_segments == 0:
        last_segment_bitlength = string_bitlength
        last_segment_bytelength = last_segment_bitlength // 8
        extra_bits = 0
        if last_segment_bitlength % 8 != 0:
            extra_bits = (8 - (last_segment_bitlength % 8)) & 0xFF
            extra_bits = (extra_bits << 5) & 0xFF
            last_segment_bytelength += 1
        last_segment_bytelength += 1

        string_data = string[-1]
        seg = bytearray(last_segment_bytelength)
        for j in range(len(seg) - 1):
            seg[j] = string[j]
        seg[-1] = extra_bits
        zero_ratio = sm.get_zero_ratio(seg, last_segment_bitlength, bit_table)
        if zero_ratio < 0.5:
            seg[-1] = (seg[-1] | 16) & 0xFF
        bin_number = [get_bin_number(zero_ratio, bin_)]

        return [[seg], last_segment_bytelength, last_segment_bytelength,
                extra_bits, string_data, bin_number]

    segment_bitlength = bitlength
    segment_bytelength = bitlength // 8
    segment_bytelength += 1

    odd_bits = string_bitlength % bitlength
    last_segment_bitlength = bitlength + odd_bits
    last_segment_bytelength = last_segment_bitlength // 8
    extra_bits = 0
    if odd_bits % 8 != 0:
        # FIX: was `extra_bits = (8 - odd_bits) & 0xFF`. odd_bits is a
        # remainder mod `bitlength` (the segment size, e.g. 256), so it
        # can be anywhere from 0 to bitlength-1 -- not just 0-7. What's
        # actually needed here is the number of PADDING bits needed to
        # round the last segment's bit count up to a whole byte, which
        # depends only on odd_bits % 8, not odd_bits itself. Confirmed via
        # 200 randomized segment()/restore() round trips: the old formula
        # produced a wrong value whenever odd_bits >= 8 (175/200 trials
        # here), corrupting the restored output in a fraction of those
        # cases (5/175) where the wrong value happened to decode to a
        # different effective padding count than intended.
        extra_bits = (8 - (odd_bits % 8)) & 0xFF
        extra_bits = (extra_bits << 5) & 0xFF
        last_segment_bytelength += 1
    last_segment_bytelength += 1

    min_segment_bytelength = segment_bytelength
    max_segment_bytelength = last_segment_bytelength
    string_data = string[-1]

    bin_number = [0] * number_of_segments

    segments = []
    for i in range(number_of_segments):
        if i < number_of_segments - 1:
            seg = bytearray(segment_bytelength)
            for j in range(len(seg) - 1):
                seg[j] = string[i * (segment_bytelength - 1) + j]
            zero_ratio = sm.get_zero_ratio(seg, segment_bitlength, bit_table)
            bin_number[i] = get_bin_number(zero_ratio, bin_)
            if zero_ratio < 0.5:
                seg[-1] = 16
            segments.append(seg)
        else:
            seg = bytearray(last_segment_bytelength)
            for j in range(len(seg) - 1):
                seg[j] = string[i * (segment_bytelength - 1) + j]
            seg[-1] = extra_bits
            zero_ratio = sm.get_zero_ratio(seg, last_segment_bitlength, bit_table)
            bin_number[i] = get_bin_number(zero_ratio, bin_)
            if zero_ratio < 0.5:
                seg[-1] = (seg[-1] | 16) & 0xFF
            segments.append(seg)

    return [segments, min_segment_bytelength, max_segment_bytelength, extra_bits, string_data, bin_number]


def _is_similar_for_merge(current_bin, next_bin, merge_type, bin_divider, difference):
    if merge_type == 0:
        return (current_bin < bin_divider and next_bin < bin_divider) or \
               (current_bin >= bin_divider and next_bin >= bin_divider)
    elif merge_type == 1:
        return (current_bin < bin_divider - 1 and next_bin < bin_divider - 1) or \
               (current_bin > bin_divider + 1 and next_bin > bin_divider + 1)
    elif merge_type == 2:
        return (current_bin < bin_divider and next_bin < bin_divider and abs(current_bin - next_bin) < difference) or \
               (current_bin >= bin_divider and next_bin >= bin_divider and abs(current_bin - next_bin) < difference)
    elif merge_type == 3:
        return current_bin == next_bin
    return False


def merge(segments, bin_number, bin_, min_segment_bytelength, max_segment_bytelength,
          extra_bits, string_data, merge_type):
    merged_segments = []
    number_of_segments = len(segments)
    number_of_bins = int(1.0 / bin_)
    bin_divider = number_of_bins // 2
    difference = bin_divider // 2

    bit_table = sm.get_bit_table()
    i = 0
    while i < number_of_segments - 1:
        current_bin = bin_number[i]
        j = 1
        next_bin = bin_number[i + j]

        similar = _is_similar_for_merge(current_bin, next_bin, merge_type, bin_divider, difference)

        if similar:
            while similar and i + j < number_of_segments - 1:
                next_bin = bin_number[i + j + 1]
                similar = _is_similar_for_merge(current_bin, next_bin, merge_type, bin_divider, difference)
                if similar:
                    j += 1

            merged_bytelength = j * (min_segment_bytelength - 1)
            if i + j == number_of_segments - 1:
                merged_bytelength += max_segment_bytelength - 1
            else:
                merged_bytelength += min_segment_bytelength - 1
            merged_bytelength += 1
            merged_segment = bytearray(merged_bytelength)
            m = 0
            for k in range(j + 1):
                seg = segments[i + k]
                for n in range(len(seg) - 1):
                    merged_segment[m + n] = seg[n]
                m += len(seg) - 1

            if i + j == number_of_segments - 1:
                merged_segment[-1] = (merged_segment[-1] | extra_bits) & 0xFF

            bitlength = sm.get_bitlength(merged_segment)
            ratio = sm.get_zero_ratio(merged_segment, bitlength, bit_table)
            if ratio < 0.5:
                merged_segment[-1] = (merged_segment[-1] | 16) & 0xFF

            merged_segments.append(merged_segment)
            i += j
        else:
            merged_segments.append(segments[i])
        i += 1

    if i == number_of_segments - 1:
        merged_segments.append(segments[i])

    number_of_merged_segments = len(merged_segments)

    compressed_segments = []
    max_segment_bytelength = 0
    min_segment_bytelength = 2 ** 31 - 1  # Integer.MAX_VALUE
    number_of_uncompressed_segments = 0
    number_of_uncompressed_adjacent_segments = 0
    previous_iterations = 1
    max_iterations = 0

    for i in range(number_of_merged_segments):
        seg = merged_segments[i]
        # type, bitlength, ratio computed in Java but only used implicitly
        # (their results feed nothing else in this loop) -- kept for
        # faithfulness even though "type"/"ratio" end up unused, matching
        # the original.
        _type = sm.get_type(seg)
        _bitlength = sm.get_bitlength(seg)
        _ratio = sm.get_zero_ratio(seg, _bitlength, bit_table)
        compressed_segment = sm.compress_strings(seg)

        compressed_segments.append(compressed_segment)
        if len(compressed_segment) - 1 > max_segment_bytelength:
            max_segment_bytelength = len(compressed_segment) - 1
        if len(compressed_segment) - 1 < min_segment_bytelength:
            min_segment_bytelength = len(compressed_segment) - 1
        current_iterations = sm.get_iterations(compressed_segment)
        if current_iterations == 0 or current_iterations == 16:
            number_of_uncompressed_segments += 1
        elif current_iterations > 16:
            if max_iterations < current_iterations - 16:
                max_iterations = current_iterations - 16
        else:
            if max_iterations < current_iterations:
                max_iterations = current_iterations
        if (previous_iterations == 0 or previous_iterations == 16) and \
           (current_iterations == 0 or current_iterations == 16):
            number_of_uncompressed_adjacent_segments += 1
        previous_iterations = current_iterations

    number_of_compressed_segments = len(compressed_segments)
    if number_of_compressed_segments == 1:
        # FIX: the original Java (SegmentMapper.java's merge()) does
        # `segment[segment.length-1] = string_data;` here, clobbering the
        # single remaining segment's own compression metadata byte (which
        # encodes iterations/type/padding -- restore()'s only way to know
        # whether this segment needs decompressing, and with what bit
        # length) with the ORIGINAL string's unrelated trailing data byte.
        # This is unnecessary and actively harmful: restore(segments,
        # string_data) already receives string_data as its own explicit
        # parameter and applies it itself, at the very end, to the fully
        # reconstructed output -- it never needs a copy embedded in any
        # individual segment. Confirmed via direct testing: whenever
        # merge() collapses everything down to exactly one segment (common
        # with the looser merge_type 0/1/2 similarity criteria, and
        # precisely the case where segmentation found the most uniform,
        # most compressible run -- i.e. the best case for this whole
        # scheme), the clobbered byte decodes as "iterations=16"
        # (uncompressed), so restore() reads the still-compressed bytes as
        # if they were raw, using a drastically wrong bit length. On real
        # packed-string test data this reproduced consistently: e.g. a
        # 3951-byte segment restored as only 1659 bytes. This -- not
        # something specific to merge_type's similarity logic itself --
        # is almost certainly the "merge_type 0/1/2 mismatch, merge_type 3
        # clean" pattern this module's docstring documented as a
        # pre-existing Java bug: type 3's strict exact-bin-match criterion
        # is simply far less likely to ever collapse a large segment list
        # down to one group than the looser types are, not because its
        # merge logic is different.
        return [compressed_segments]
    else:
        return [compressed_segments, min_segment_bytelength, max_segment_bytelength,
                extra_bits, string_data, number_of_uncompressed_segments,
                number_of_uncompressed_adjacent_segments, max_iterations]


def combine(segments, min_segment_bytelength, max_segment_bytelength, extra_bits, string_data):
    combined_segments = []
    bit_table = sm.get_bit_table()

    number_of_segments = len(segments)
    i = 0
    while i < number_of_segments - 1:
        current_segment = segments[i]
        current_iterations = sm.get_iterations(current_segment)

        next_segment = segments[i + 1]
        next_iterations = sm.get_iterations(next_segment)

        if (current_iterations == 0 or current_iterations == 16) and \
           (next_iterations == 0 or next_iterations == 16):
            j = 1
            while (next_iterations == 0 or next_iterations == 16) and i + j + 1 < number_of_segments:
                next_segment = segments[i + j + 1]
                next_iterations = sm.get_iterations(next_segment)
                if next_iterations == 0 or next_iterations == 16:
                    j += 1

            combined_length = 0
            for k in range(j + 1):
                seg = segments[i + k]
                combined_length += len(seg) - 1
            combined_length += 1

            if max_segment_bytelength < combined_length - 1:
                max_segment_bytelength = combined_length - 1

            combined_segment = bytearray(combined_length)
            m = 0
            for k in range(j + 1):
                seg = segments[i + k]
                for n in range(len(seg) - 1):
                    combined_segment[m + n] = seg[n]
                m += len(seg) - 1

            if i + j == number_of_segments - 1:
                last_bitlength = (len(combined_segment) - 1) * 8
                k = extra_bits >> 5
                k &= 7
                last_bitlength -= k
                zero_ratio = sm.get_zero_ratio(combined_segment, last_bitlength, bit_table)
                combined_segment[-1] = extra_bits
                if zero_ratio < 0.5:
                    combined_segment[-1] = (combined_segment[-1] | 16) & 0xFF
                combined_segments.append(combined_segment)
            else:
                zero_ratio = sm.get_zero_ratio(combined_segment, (len(combined_segment) - 1) * 8, bit_table)
                if zero_ratio < 0.5:
                    combined_segment[-1] = (combined_segment[-1] | 16) & 0xFF
                combined_segments.append(combined_segment)

            i += j
        else:
            combined_segments.append(current_segment)
        i += 1

    if i == number_of_segments - 1:
        combined_segments.append(segments[i])

    number_of_uncompressed_segments = 0
    number_of_combined_segments = len(combined_segments)
    if number_of_combined_segments == 1:
        # FIX: same bug and same fix as merge()'s single-segment fallback
        # above -- clobbering this segment's own compression metadata byte
        # with string_data is unnecessary (restore() takes string_data as
        # its own explicit parameter) and breaks decompression whenever
        # this single remaining segment is itself still compressed.
        return [combined_segments]
    else:
        combined_iterations = [0] * number_of_combined_segments
        for i in range(number_of_combined_segments):
            seg = combined_segments[i]
            combined_iterations[i] = sm.get_iterations(seg)
            if combined_iterations[i] == 0 or combined_iterations[i] == 16:
                number_of_uncompressed_segments += 1
        return [combined_segments, min_segment_bytelength, max_segment_bytelength,
                extra_bits, string_data, number_of_uncompressed_segments]


def splice(segments, min_segment_bytelength, max_segment_bytelength):
    number_of_segments = len(segments)
    is_compressed = [False] * number_of_segments
    for i in range(number_of_segments):
        seg = segments[i]
        iterations = sm.get_iterations(seg)
        if iterations != 0 and iterations != 16:
            is_compressed[i] = True

    overhead = 16
    if max_segment_bytelength > 32767 * 2 + 1:
        overhead = 40
    elif max_segment_bytelength > 127 * 2 + 1:
        overhead = 24

    total_spliced_bits = 0
    max_spliced_bits = 0
    bit_table = sm.get_bit_table()

    spliced_segments = []

    i = 0
    while i < number_of_segments - 1:
        current_segment = segments[i]
        if is_compressed[i]:
            spliced_segments.append(current_segment)
        else:
            if not is_compressed[i + 1]:
                spliced_segments.append(current_segment)
            else:
                current_bitlength = sm.get_bitlength(current_segment)
                next_segment = segments[i + 1]
                next_bitlength = sm.get_bitlength(next_segment)
                decompressed_segment = sm.decompress_strings(next_segment)
                decompressed_bitlength = sm.get_bitlength(decompressed_segment)

                augmented_bitlength = current_bitlength + decompressed_bitlength
                augmented_bytelength = augmented_bitlength // 8
                if augmented_bitlength % 8 != 0:
                    augmented_bytelength += 1
                augmented_bytelength += 1

                augmented_segment = bytearray(augmented_bytelength)
                for j in range(len(current_segment) - 1):
                    augmented_segment[j] = current_segment[j]

                current_odd_bits = current_bitlength % 8
                if current_odd_bits == 0:
                    k = 0
                    for j in range(len(current_segment) - 1, len(augmented_segment) - 1):
                        augmented_segment[j] = decompressed_segment[k]
                        k += 1
                else:
                    extra_bits_ = 8 - current_odd_bits
                    splice_byte = decompressed_segment[0]
                    mask = get_trailing_mask(extra_bits_)
                    splice_byte = splice_byte & mask
                    splice_byte = (splice_byte << current_odd_bits) & 0xFF

                    augmented_segment[len(current_segment) - 2] = \
                        (augmented_segment[len(current_segment) - 2] | splice_byte) & 0xFF

                    shifted_segment = shift_right(decompressed_segment, extra_bits_)
                    k = 0
                    for j in range(len(current_segment) - 1, len(augmented_segment) - 1):
                        augmented_segment[j] = shifted_segment[k]
                        k += 1

                augmented_segment[-1] = 0
                ratio = sm.get_zero_ratio(augmented_segment, augmented_bitlength, bit_table)
                if ratio < 0.5:
                    augmented_segment[-1] = 16
                augmented_odd_bits = augmented_bitlength % 8
                if augmented_odd_bits != 0:
                    extra_bits_ = (8 - augmented_odd_bits) & 0xFF
                    extra_bits_ = (extra_bits_ << 5) & 0xFF
                    augmented_segment[-1] = (augmented_segment[-1] | extra_bits_) & 0xFF

                    mask = get_trailing_mask(augmented_odd_bits)
                    augmented_segment[-2] = augmented_segment[-2] & mask

                compressed_segment = sm.compress_strings(augmented_segment)
                compressed_bitlength = sm.get_bitlength(compressed_segment)
                spliced_bits = current_bitlength
                bit_reduction = (spliced_bits + overhead) - (compressed_bitlength - next_bitlength)
                max_reduction = bit_reduction

                for j in range(1, current_bitlength):
                    shifted_segment = shift_right(augmented_segment, j)
                    shifted_bitlength = augmented_bitlength - j
                    shifted_bytelength = shifted_bitlength // 8
                    if shifted_bitlength % 8 != 0:
                        shifted_bytelength += 1
                    shifted_bytelength += 1

                    if len(shifted_segment) != shifted_bytelength:
                        clipped_segment = bytearray(shifted_bytelength)
                        for k in range(shifted_bytelength):
                            clipped_segment[k] = shifted_segment[k]
                        shifted_segment = clipped_segment

                    shifted_segment[-1] = 0
                    ratio = sm.get_zero_ratio(shifted_segment, shifted_bitlength, bit_table)
                    if ratio < 0.5:
                        shifted_segment[-1] = 16

                    shifted_odd_bits = shifted_bitlength % 8
                    if shifted_odd_bits != 0:
                        extra_bits_ = (8 - shifted_odd_bits) & 0xFF
                        extra_bits_ = (extra_bits_ << 5) & 0xFF
                        shifted_segment[-1] = (shifted_segment[-1] | extra_bits_) & 0xFF
                        mask = get_trailing_mask(shifted_odd_bits)
                        shifted_segment[-2] = shifted_segment[-2] & mask

                    compressed_segment = sm.compress_strings(shifted_segment)
                    compressed_bitlength = sm.get_bitlength(compressed_segment)

                    current_bits = current_bitlength - j
                    bit_reduction = current_bits - (compressed_bitlength - next_bitlength)
                    if bit_reduction > max_reduction:
                        max_reduction = bit_reduction
                        spliced_bits = current_bits

                if max_reduction > 0:
                    total_spliced_bits += spliced_bits
                    if spliced_bits > max_spliced_bits:
                        max_spliced_bits = spliced_bits

                    if spliced_bits == current_bitlength:
                        compressed_segment = sm.compress_strings(augmented_segment)
                        if len(compressed_segment) - 1 > max_segment_bytelength:
                            max_segment_bytelength = len(compressed_segment) - 1
                            if max_segment_bytelength > 32767 * 2 + 1:
                                overhead = 40
                            elif max_segment_bytelength > 127 * 2 + 1:
                                overhead = 24

                        spliced_segments.append(compressed_segment)
                        i += 1
                    else:
                        reduced_bitlength = current_bitlength - spliced_bits
                        reduced_bytelength = reduced_bitlength // 8
                        if reduced_bitlength % 8 != 0:
                            reduced_bytelength += 1
                        reduced_bytelength += 1

                        reduced_segment = bytearray(reduced_bytelength)
                        for j in range(len(reduced_segment) - 1):
                            reduced_segment[j] = current_segment[j]

                        reduced_segment[-1] = 0
                        ratio = sm.get_zero_ratio(reduced_segment, reduced_bitlength, bit_table)
                        if ratio < 0.5:
                            reduced_segment[-1] = 16

                        reduced_odd_bits = reduced_bitlength % 8
                        if reduced_odd_bits != 0:
                            extra_bits_ = (8 - reduced_odd_bits) & 0xFF
                            extra_bits_ = (extra_bits_ << 5) & 0xFF
                            reduced_segment[-1] = (reduced_segment[-1] | extra_bits_) & 0xFF
                            mask = get_trailing_mask(reduced_odd_bits)
                            reduced_segment[-2] = reduced_segment[-2] & mask
                        if len(reduced_segment) - 1 < min_segment_bytelength:
                            min_segment_bytelength = len(compressed_segment) - 1

                        unused_bits = current_bitlength - spliced_bits
                        shifted_segment = shift_right(augmented_segment, unused_bits)
                        shifted_bitlength = augmented_bitlength - unused_bits
                        shifted_bytelength = shifted_bitlength // 8
                        if shifted_bitlength % 8 != 0:
                            shifted_bytelength += 1
                        shifted_bytelength += 1

                        if len(shifted_segment) != shifted_bytelength:
                            clipped_segment = bytearray(shifted_bytelength)
                            for k in range(shifted_bytelength):
                                clipped_segment[k] = shifted_segment[k]
                            shifted_segment = clipped_segment

                        shifted_segment[-1] = 0
                        ratio = sm.get_zero_ratio(shifted_segment, shifted_bitlength, bit_table)
                        if ratio < 0.5:
                            shifted_segment[-1] = 16

                        shifted_odd_bits = shifted_bitlength % 8
                        if shifted_odd_bits != 0:
                            extra_bits_ = (8 - shifted_odd_bits) & 0xFF
                            extra_bits_ = (extra_bits_ << 5) & 0xFF
                            shifted_segment[-1] = (shifted_segment[-1] | extra_bits_) & 0xFF
                            mask = get_trailing_mask(shifted_odd_bits)
                            shifted_segment[-2] = shifted_segment[-2] & mask

                        compressed_segment = sm.compress_strings(shifted_segment)
                        if len(compressed_segment) - 1 > max_segment_bytelength:
                            max_segment_bytelength = len(compressed_segment) - 1

                        spliced_segments.append(reduced_segment)
                        spliced_segments.append(compressed_segment)
                        i += 1
                else:
                    spliced_segments.append(current_segment)
        i += 1

    # FIX: was an UNCONDITIONAL `spliced_segments.append(segments[-1])`
    # here, on the assumption the while loop always naturally stops one
    # short of the last segment (loop condition is `i < number_of_segments
    # - 1`, so the last index is never visited as `current_segment`). That
    # assumption breaks whenever a successful splice at i == number_of_
    # segments - 2 consumes BOTH that segment and the one after it (the
    # true last segment) -- the two `i += 1`s in that path (one inside the
    # branch, one from the loop itself) together advance i by 2, jumping
    # straight from number_of_segments-2 to number_of_segments and
    # skipping over number_of_segments-1 entirely, even though that last
    # segment WAS already spliced away and appended (in transformed form)
    # inside the loop. The old code then re-appended the ORIGINAL,
    # un-spliced copy of it unconditionally, duplicating it in the output.
    # Confirmed directly: restore() on the result was 177 bytes for a
    # 161-byte input, with the excess and the first mismatch both located
    # at the tail, exactly where the duplicated last segment landed.
    # Fixed by only appending segments[-1] when the loop actually stopped
    # short of it (i == number_of_segments - 1), not when it jumped past.
    if i == number_of_segments - 1:
        last_segment = segments[number_of_segments - 1]
        spliced_segments.append(last_segment)

    return [spliced_segments, min_segment_bytelength, max_segment_bytelength,
            total_spliced_bits, max_spliced_bits]


def splice2(segments, min_segment_bytelength, max_segment_bytelength):
    number_of_segments = len(segments)
    is_compressed = [False] * number_of_segments
    for i in range(number_of_segments):
        seg = segments[i]
        iterations = sm.get_iterations(seg)
        if iterations != 0 and iterations != 16:
            is_compressed[i] = True

    overhead = 16
    if max_segment_bytelength > 32767 * 2 + 1:
        overhead = 40
    elif max_segment_bytelength > 127 * 2 + 1:
        overhead = 24

    total_spliced_bits = 0
    max_spliced_bits = 0
    bit_table = sm.get_bit_table()

    spliced_segments = [segments[0]]

    for i in range(1, number_of_segments):
        current_segment = segments[i]
        if is_compressed[i]:
            spliced_segments.append(current_segment)
        else:
            if not is_compressed[i - 1]:
                spliced_segments.append(current_segment)
            else:
                size = len(spliced_segments)
                previous_segment = spliced_segments[size - 1]
                decompressed_segment = sm.decompress_strings(previous_segment)
                previous_bitlength = sm.get_bitlength(previous_segment)
                decompressed_bitlength = sm.get_bitlength(decompressed_segment)
                current_bitlength = sm.get_bitlength(current_segment)
                augmented_bitlength = decompressed_bitlength + current_bitlength
                augmented_bytelength = augmented_bitlength // 8
                if augmented_bitlength % 8 != 0:
                    augmented_bytelength += 1
                augmented_bytelength += 1

                augmented_segment = bytearray(augmented_bytelength)
                for j in range(len(decompressed_segment) - 1):
                    augmented_segment[j] = decompressed_segment[j]

                decompressed_odd_bits = decompressed_bitlength % 8
                if decompressed_odd_bits == 0:
                    k = 0
                    for j in range(len(decompressed_segment) - 1, len(augmented_segment) - 1):
                        augmented_segment[j] = current_segment[k]
                        k += 1
                else:
                    extra_bits_ = 8 - decompressed_odd_bits
                    splice_byte = current_segment[0]
                    mask = get_trailing_mask(extra_bits_)
                    splice_byte = splice_byte & mask
                    splice_byte = (splice_byte << decompressed_odd_bits) & 0xFF

                    augmented_segment[len(decompressed_segment) - 2] = \
                        (augmented_segment[len(decompressed_segment) - 2] | splice_byte) & 0xFF

                    clipped_segment = bytearray(len(current_segment) - 1)
                    for j in range(len(clipped_segment)):
                        clipped_segment[j] = current_segment[j]
                    shifted_segment = shift_right(clipped_segment, extra_bits_)

                    k = 0
                    for j in range(len(decompressed_segment) - 1, len(augmented_segment) - 1):
                        augmented_segment[j] = shifted_segment[k]
                        k += 1

                ratio = sm.get_zero_ratio(augmented_segment, augmented_bitlength, bit_table)
                if ratio < 0.5:
                    augmented_segment[-1] = 16
                else:
                    augmented_segment[-1] = 0
                augmented_odd_bits = augmented_bitlength % 8
                if augmented_odd_bits != 0:
                    extra_bits_ = (8 - augmented_odd_bits) & 0xFF
                    extra_bits_ = (extra_bits_ << 5) & 0xFF
                    augmented_segment[-1] = (augmented_segment[-1] | extra_bits_) & 0xFF
                    mask = get_trailing_mask(augmented_odd_bits)
                    augmented_segment[-2] = augmented_segment[-2] & mask

                compressed_segment = sm.compress_strings(augmented_segment)
                compressed_bitlength = sm.get_bitlength(compressed_segment)
                spliced_bits = current_bitlength
                bit_reduction = (spliced_bits + overhead) - (compressed_bitlength - previous_bitlength)
                max_reduction = bit_reduction

                for j in range(1, current_bitlength):
                    clipped_bitlength = augmented_bitlength - j
                    clipped_bytelength = clipped_bitlength // 8
                    if clipped_bitlength % 8 != 0:
                        clipped_bytelength += 1
                    clipped_bytelength += 1

                    clipped_segment = bytearray(clipped_bytelength)
                    for k in range(len(clipped_segment) - 1):
                        clipped_segment[k] = augmented_segment[k]

                    clipped_segment[-1] = 0
                    ratio = sm.get_zero_ratio(clipped_segment, clipped_bitlength, bit_table)
                    if ratio < 0.5:
                        clipped_segment[-1] = 16

                    clipped_odd_bits = clipped_bitlength % 8
                    if clipped_odd_bits != 0:
                        extra_bits_ = (8 - clipped_odd_bits) & 0xFF
                        extra_bits_ = (extra_bits_ << 5) & 0xFF
                        clipped_segment[-1] = (clipped_segment[-1] | extra_bits_) & 0xFF
                        mask = get_trailing_mask(clipped_odd_bits)
                        clipped_segment[-2] = clipped_segment[-2] & mask

                    compressed_segment = sm.compress_strings(clipped_segment)
                    compressed_bitlength = sm.get_bitlength(compressed_segment)

                    current_bits = current_bitlength - j
                    bit_reduction = current_bits - (compressed_bitlength - previous_bitlength)
                    if bit_reduction > max_reduction:
                        max_reduction = bit_reduction
                        spliced_bits = current_bits

                if max_reduction > 0:
                    total_spliced_bits += spliced_bits
                    if spliced_bits > max_spliced_bits:
                        max_spliced_bits = spliced_bits

                    if spliced_bits == current_bitlength:
                        compressed_segment = sm.compress_strings(augmented_segment)
                        j = len(spliced_segments)
                        spliced_segments[j - 1] = compressed_segment
                        if len(compressed_segment) - 1 > max_segment_bytelength:
                            max_segment_bytelength = len(compressed_segment) - 1
                            if max_segment_bytelength > 32767 * 2 + 1:
                                overhead = 40
                            elif max_segment_bytelength > 127 * 2 + 1:
                                overhead = 24
                    else:
                        unused_bits = current_bitlength - spliced_bits
                        clipped_bitlength = augmented_bitlength - unused_bits
                        clipped_bytelength = clipped_bitlength // 8
                        if clipped_bitlength % 8 != 0:
                            clipped_bytelength += 1
                        clipped_bytelength += 1

                        clipped_segment = bytearray(clipped_bytelength)
                        for k in range(len(clipped_segment) - 1):
                            clipped_segment[k] = augmented_segment[k]

                        clipped_segment[-1] = 0
                        ratio = sm.get_zero_ratio(clipped_segment, clipped_bitlength, bit_table)
                        if ratio < 0.5:
                            clipped_segment[-1] = 16

                        clipped_odd_bits = clipped_bitlength % 8
                        if clipped_odd_bits != 0:
                            extra_bits_ = (8 - clipped_odd_bits) & 0xFF
                            extra_bits_ = (extra_bits_ << 5) & 0xFF
                            clipped_segment[-1] = (clipped_segment[-1] | extra_bits_) & 0xFF
                            mask = get_trailing_mask(clipped_odd_bits)
                            clipped_segment[-2] = clipped_segment[-2] & mask

                        compressed_segment = sm.compress_strings(clipped_segment)
                        if len(compressed_segment) - 1 > max_segment_bytelength:
                            max_segment_bytelength = len(compressed_segment) - 1

                        j = len(spliced_segments)
                        spliced_segments[j - 1] = compressed_segment

                        clipped_segment = bytearray(len(current_segment) - 1)
                        for jj in range(len(clipped_segment)):
                            clipped_segment[jj] = current_segment[jj]
                        shifted_segment = shift_right(clipped_segment, spliced_bits)

                        reduced_bitlength = current_bitlength - spliced_bits
                        reduced_bytelength = sm.get_bytelength(reduced_bitlength)
                        reduced_segment = bytearray(reduced_bytelength)
                        for jj in range(len(shifted_segment)):
                            reduced_segment[jj] = shifted_segment[jj]
                        if len(reduced_segment) - 1 < min_segment_bytelength:
                            min_segment_bytelength = len(reduced_segment) - 1

                        ratio = sm.get_zero_ratio(reduced_segment, reduced_bitlength, bit_table)
                        if ratio >= 0.5:
                            sm.set_data(0, 0, reduced_bitlength, reduced_segment)
                        else:
                            sm.set_data(1, 0, reduced_bitlength, reduced_segment)

                        spliced_segments.append(reduced_segment)
                else:
                    spliced_segments.append(current_segment)

    return [spliced_segments, min_segment_bytelength, max_segment_bytelength,
            total_spliced_bits, max_spliced_bits]


def restore(segments, string_data):
    number_of_segments = len(segments)

    total_bitlength = 0
    for i in range(number_of_segments):
        seg = segments[i]
        iterations = sm.get_iterations(seg)
        if iterations == 0 or iterations == 16:
            bitlength = sm.get_bitlength(seg)
            total_bitlength += bitlength
        else:
            decompressed_segment = sm.decompress_strings(seg)
            bitlength = sm.get_bitlength(decompressed_segment)
            total_bitlength += bitlength

    bytelength = total_bitlength // 8
    if total_bitlength % 8 != 0:
        bytelength += 1
    bytelength += 1
    dst = bytearray(bytelength)
    bit_offset = 0
    byte_offset = 0
    for i in range(number_of_segments):
        seg = segments[i]
        iterations = sm.get_iterations(seg)
        if iterations != 0 and iterations != 16:
            seg = sm.decompress_strings(seg)
            iterations = sm.get_iterations(seg)

        bitlength = sm.get_bitlength(seg)
        bit_shift = bit_offset % 8
        if bit_shift == 0:
            for j in range(len(seg) - 1):
                dst[byte_offset + j] = seg[j]
        else:
            clipped_segment = bytearray(len(seg) - 1)
            for j in range(len(clipped_segment)):
                clipped_segment[j] = seg[j]

            shifted_segment = shift_left(clipped_segment, bit_shift)
            dst[byte_offset] = (dst[byte_offset] | shifted_segment[0]) & 0xFF

            for j in range(1, len(shifted_segment)):
                dst[byte_offset + j] = shifted_segment[j]

        bit_offset += bitlength
        byte_offset = bit_offset // 8

    dst[-1] = string_data
    return dst


def get_segmented_data(string, minimum_bitlength, segment_type, merge_type, bin_):
    if minimum_bitlength % 8 != 0:
        print("Minimum segment bitlength must be a multiple of 8.")
        return []

    if segment_type < 0 or segment_type > 3:
        print("Unsupported segment type.")
        return []

    segmented_list = segment(string, minimum_bitlength, bin_)

    segments = segmented_list[0]
    min_segment_bytelength = segmented_list[1]
    max_segment_bytelength = segmented_list[2]
    extra_bits = segmented_list[3]
    string_data = segmented_list[4]
    bin_number = segmented_list[5]

    number_of_regular_segments = len(segments)
    if segment_type == 0:
        print(f"Number of regular segments is {number_of_regular_segments}")
        print(f"Regular segment byte length is {min_segment_bytelength}")
        print(f"Odd segment byte length is {max_segment_bytelength}")

        total_bitlength = 0
        for i in range(len(segments)):
            total_bitlength += sm.get_bitlength(segments[i])
        print(f"Total bitlength of regular segments is {total_bitlength}")

        return [segments, max_segment_bytelength, string_data]

    merged_list = merge(segments, bin_number, bin_, min_segment_bytelength, max_segment_bytelength,
                         extra_bits, string_data, merge_type)
    merged_segments = merged_list[0]

    number_of_merged_segments = len(merged_segments)
    merged_max_iterations = 0
    if number_of_merged_segments == 1:
        print("No segmentation with current parameters.")
        print(f"Returning {number_of_regular_segments} segments merged back into the original string.")
        print(f"String bitlength was {min_segment_bytelength * 8}")
        return [merged_segments, min_segment_bytelength, string_data]

    min_segment_bytelength = merged_list[1]
    max_segment_bytelength = merged_list[2]
    extra_bits = merged_list[3]
    string_data = merged_list[4]
    number_of_uncompressed_segments = merged_list[5]
    number_of_uncompressed_adjacent_segments = merged_list[6]
    merged_max_iterations = merged_list[7]

    if segment_type == 1:
        print(f"Number of regular segments is {number_of_regular_segments}")
        print(f"Number of merged segments is {number_of_merged_segments}")
        print(f"Maximum segment byte length is {max_segment_bytelength}")
        print(f"Number of uncompressed segments is {number_of_uncompressed_segments}")
        print(f"Number of uncompressed adjacent segments is {number_of_uncompressed_adjacent_segments}")
        total_bitlength = 0
        for i in range(len(merged_segments)):
            total_bitlength += sm.get_bitlength(merged_segments[i])
        print(f"Total bitlength is {total_bitlength}")

        return [merged_segments, max_segment_bytelength, string_data]

    elif segment_type in (2, 3):
        if number_of_uncompressed_adjacent_segments == 0:
            print("No uncompressed adjacent segments to combine.")
            print("Returning merged segments.")
            print(f"Number of regular segments is {number_of_regular_segments}")
            print(f"Number of merged segments is {number_of_merged_segments}")
            print(f"Maximum segment byte length is {max_segment_bytelength}")
            print(f"Number of uncompressed segments is {number_of_uncompressed_segments}")

            total_bitlength = 0
            for i in range(len(merged_segments)):
                total_bitlength += sm.get_bitlength(merged_segments[i])
            print(f"Total bitlength of merged/compressed segments is {total_bitlength}")

            return [merged_segments, max_segment_bytelength, string_data]

        combined_list = combine(merged_segments, min_segment_bytelength, max_segment_bytelength,
                                 extra_bits, string_data)
        combined_segments = combined_list[0]
        number_of_combined_segments = len(combined_segments)
        if number_of_combined_segments == 1:
            print("No segmentation with current parameters.")
            print(f"Returning {number_of_merged_segments} segments combined back into the original string.")
            return [combined_segments, max_segment_bytelength, string_data]

        min_segment_bytelength = combined_list[1]
        max_segment_bytelength = combined_list[2]
        extra_bits = combined_list[3]
        string_data = combined_list[4]
        number_of_uncompressed_segments = combined_list[5]

        if segment_type == 2:
            print(f"Number of regular segments is {number_of_regular_segments}")
            print(f"Number of merged segments is {number_of_merged_segments}")
            print(f"Number of combined segments is {number_of_combined_segments}")
            print(f"Maximum segment byte length is {max_segment_bytelength}")
            print(f"Number of uncompressed segments is {number_of_uncompressed_segments}")

            total_bitlength = 0
            for i in range(len(combined_segments)):
                total_bitlength += sm.get_bitlength(combined_segments[i])
            print(f"Total bitlength is {total_bitlength}")

            return [combined_segments, max_segment_bytelength, string_data]

        if segment_type == 3:
            if number_of_uncompressed_segments == 0:
                print("No uncompressed segments to borrow bits from.")
                print(f"Returning {number_of_combined_segments} combined segments.")
                print(f"Maximum segment byte length is {max_segment_bytelength}")
                return [combined_segments, max_segment_bytelength, string_data]

            spliced_list = splice(combined_segments, min_segment_bytelength, max_segment_bytelength)
            spliced_segments = spliced_list[0]
            min_segment_bytelength = spliced_list[1]
            max_segment_bytelength = spliced_list[2]
            total_spliced_bits = spliced_list[3]
            max_spliced_bits1 = spliced_list[4]

            spliced_list2 = splice2(combined_segments, min_segment_bytelength, max_segment_bytelength)
            spliced_segments2 = spliced_list2[0]
            number_of_spliced_segments = len(spliced_segments2)
            min_segment_bytelength = spliced_list2[1]
            max_segment_bytelength = spliced_list2[2]
            total_spliced_bits += spliced_list2[3]
            max_spliced_bits2 = spliced_list2[4]

            max_spliced_bits = max_spliced_bits1
            if max_spliced_bits1 < max_spliced_bits2:
                max_spliced_bits = max_spliced_bits2

            print(f"Number of regular segments is {number_of_regular_segments}")
            print(f"Number of merged segments is {number_of_merged_segments}")
            print(f"Number of combined segments is {number_of_combined_segments}")

            total_bitlength = 0
            max_iterations = 0

            bit_table = sm.get_bit_table()
            for i in range(number_of_spliced_segments):
                seg = spliced_segments2[i]
                bitlength = sm.get_bitlength(seg)
                iterations = sm.get_iterations(seg)
                _ratio = sm.get_zero_ratio(seg, bitlength, bit_table)
                total_bitlength += bitlength

                if iterations < 16:
                    if max_iterations < iterations:
                        max_iterations = iterations
                else:
                    if max_iterations < iterations - 16:
                        max_iterations = iterations - 16

            print(f"Number of spliced segments is {number_of_spliced_segments}")
            print(f"Maximum iterations for spliced segments is {max_iterations}")
            print(f"Maximum iterations for merged segments is {merged_max_iterations}")
            print(f"Total bitlength of merged/compressed/spliced segments is {total_bitlength}")
            print()

            return [spliced_segments2, max_segment_bytelength, string_data]

    return []


# =============================================================================
# packSegments / unpackSegments family
# =============================================================================
_PACK_MASK = [1, 3, 7, 15, 31, 63, 127]


def pack_segments(segments):
    size = len(segments)
    bitlength = [0] * size

    total_bitlength = 0
    for i in range(size):
        current_segment = segments[i]
        bitlength[i] = sm.get_bitlength(current_segment)
        total_bitlength += bitlength[i]

    total_bytelength = total_bitlength // 8
    if total_bitlength % 8 != 0:
        total_bytelength += 1
    string = bytearray(total_bytelength)

    mask = _PACK_MASK
    bit_offset = 0

    for i in range(size - 1):
        m = bit_offset % 8
        n = bit_offset // 8

        current_segment = segments[i]
        length = sm.get_bitlength(current_segment)

        if m == 0:
            for j in range(len(current_segment) - 1):
                string[n + j] = current_segment[j]
        else:
            for j in range(len(current_segment) - 1):
                a = (current_segment[j] << m) & 0xFF
                string[n + j] = (string[n + j] | a) & 0xFF

                number_of_bits = m
                b = (current_segment[j] >> (8 - m)) & 0xFF
                b = b & mask[number_of_bits - 1]
                string[n + j + 1] = b

        bit_offset += length

    last_segment = segments[size - 1]
    m = bit_offset % 8
    n = bit_offset // 8

    if m == 0:
        for j in range(len(last_segment) - 1):
            string[n + j] = last_segment[j]
    else:
        for j in range(len(last_segment) - 2):
            a = (last_segment[j] << m) & 0xFF
            string[n + j] = (string[n + j] | a) & 0xFF

            number_of_bits = m
            b = (last_segment[j] >> (8 - m)) & 0xFF
            b = b & mask[number_of_bits - 1]
            string[n + j + 1] = b

        a = (last_segment[len(last_segment) - 2] << m) & 0xFF
        string[n + len(last_segment) - 2] = (string[n + len(last_segment) - 2] | a) & 0xFF
        if n + len(last_segment) - 2 + 1 < len(string):
            b = (last_segment[len(last_segment) - 2] >> (8 - m)) & 0xFF
            b = b & mask[m - 1]
            string[n + len(last_segment) - 2 + 1] = b

    return [string, bitlength]


def pack_segments_with_bitlength(segments, bitlength):
    size = len(segments)

    if size != len(bitlength):
        print("Number of segments and number of bit lengths do not agree.")
        return []

    total_bitlength = sum(bitlength[:size])
    total_bytelength = total_bitlength // 8
    if total_bitlength % 8 != 0:
        total_bytelength += 1
    string = bytearray(total_bytelength)

    mask = _PACK_MASK
    bit_offset = 0

    for i in range(size - 1):
        m = bit_offset % 8
        n = bit_offset // 8

        current_segment = segments[i]
        length = sm.get_bitlength(current_segment)

        if m == 0:
            for j in range(len(current_segment) - 1):
                string[n + j] = current_segment[j]
        else:
            for j in range(len(current_segment) - 1):
                a = (current_segment[j] << m) & 0xFF
                string[n + j] = (string[n + j] | a) & 0xFF

                number_of_bits = m
                b = (current_segment[j] >> (8 - m)) & 0xFF
                b = b & mask[number_of_bits - 1]
                string[n + j + 1] = b

        bit_offset += length

    last_segment = segments[size - 1]
    m = bit_offset % 8
    n = bit_offset // 8

    if m == 0:
        for j in range(len(last_segment) - 1):
            string[n + j] = last_segment[j]
    else:
        for j in range(len(last_segment) - 2):
            a = (last_segment[j] << m) & 0xFF
            string[n + j] = (string[n + j] | a) & 0xFF

            number_of_bits = m
            b = (last_segment[j] >> (8 - m)) & 0xFF
            b = b & mask[number_of_bits - 1]
            string[n + j + 1] = b

        a = (last_segment[len(last_segment) - 2] << m) & 0xFF
        string[n + len(last_segment) - 2] = (string[n + len(last_segment) - 2] | a) & 0xFF
        if n + len(last_segment) - 2 + 1 < len(string):
            b = (last_segment[len(last_segment) - 2] >> (8 - m)) & 0xFF
            b = b & mask[m - 1]
            string[n + len(last_segment) - 2 + 1] = b

    return [string]


def pack_segments2(segments):
    size = len(segments)
    bitlength = [0] * size

    total_bitlength = 0
    for i in range(size):
        current_segment = segments[i]
        bitlength[i] = sm.get_bitlength(current_segment)
        total_bitlength += bitlength[i]

    total_bytelength = total_bitlength // 8
    if total_bitlength % 8 != 0:
        total_bytelength += 1
    string = bytearray(total_bytelength)

    k = 0
    for i in range(size):
        current_segment = segments[i]
        bitlength[i] = sm.get_bitlength(current_segment)

        for j in range(bitlength[i]):
            value = get_bit(current_segment, j)
            set_bit(string, k, value)
            k += 1

    return [string, bitlength]


def pack_segments2_with_bitlength(segments, bitlength):
    size = len(segments)

    total_bitlength = 0
    for i in range(size):
        current_segment = segments[i]
        bitlength[i] = sm.get_bitlength(current_segment)
        total_bitlength += bitlength[i]

    total_bytelength = total_bitlength // 8
    if total_bitlength % 8 != 0:
        total_bytelength += 1
    string = bytearray(total_bytelength)

    k = 0
    for i in range(size):
        current_segment = segments[i]
        bitlength[i] = sm.get_bitlength(current_segment)

        for j in range(bitlength[i]):
            value = get_bit(current_segment, j)
            set_bit(string, k, value)
            k += 1

    return [string]


def unpack_segments(string, bytelength, data):
    number_of_segments = len(data)
    unpacked_segments = []
    mask = _PACK_MASK

    bit_offset = 0
    for i in range(number_of_segments - 1):
        extra_bits = (data[i] >> 5) & 0xFF
        extra_bits &= 7
        bitlength = bytelength[i] * 8 - extra_bits
        seg = bytearray(bytelength[i] + 1)

        m = bit_offset % 8
        n = bit_offset // 8
        if m == 0:
            for j in range(bytelength[i]):
                seg[j] = string[n + j]
            if bitlength % 8 != 0:
                number_of_bits = bitlength % 8
                seg[bytelength[i] - 1] = seg[bytelength[i] - 1] & mask[number_of_bits - 1]
        else:
            for j in range(bytelength[i]):
                seg[j] = (string[n + j] >> m) & 0xFF
                number_of_bits = 8 - m
                seg[j] = seg[j] & mask[number_of_bits - 1]

                if j < bytelength[i] - 1:
                    high_bits = string[n + j + 1]
                    seg[j] = (seg[j] | ((high_bits << (8 - m)) & 0xFF)) & 0xFF
                elif j == bytelength[i] - 1:
                    high_bits = string[n + j + 1]
                    seg[j] = (high_bits << (8 - m)) & 0xFF
                    number_of_extra_bits = (j + 1) * 8 - bitlength
                    if number_of_extra_bits > 0:
                        seg[j] = seg[j] & mask[8 - number_of_extra_bits - 1]

        seg[bytelength[i]] = data[i]
        unpacked_segments.append(seg)
        bit_offset += bitlength

    i = number_of_segments - 1
    extra_bits = (data[i] >> 5) & 0xFF
    extra_bits &= 7
    bitlength = bytelength[i] * 8 - extra_bits
    seg = bytearray(bytelength[i] + 1)

    m = bit_offset % 8
    n = bit_offset // 8
    if m == 0:
        for j in range(bytelength[i]):
            seg[j] = string[n + j]
    else:
        for j in range(bytelength[i]):
            seg[j] = (string[n + j] >> m) & 0xFF
            number_of_bits = 8 - m
            seg[j] = seg[j] & mask[number_of_bits - 1]

            if j < bytelength[i] - 1:
                high_bits = string[n + j + 1]
                seg[j] = (seg[j] | ((high_bits << (8 - m)) & 0xFF)) & 0xFF
            elif j == bytelength[i] - 1:
                if n + j + 1 < len(string):
                    high_bits = string[n + j + 1]
                    seg[j] = (seg[j] | ((high_bits << (8 - m)) & 0xFF)) & 0xFF
                    number_of_extra_bits = (j + 1) * 8 - bitlength
                    if number_of_extra_bits > 0:
                        number_of_bits = 8 - number_of_extra_bits
                        seg[j] = seg[j] & mask[number_of_bits - 1]
    unpacked_segments.append(seg)

    return unpacked_segments


def unpack_segments_by_bitlength(string, bitlength):
    number_of_segments = len(bitlength)
    unpacked_segments = []
    mask = _PACK_MASK

    bit_offset = 0
    for i in range(number_of_segments - 1):
        bytelength = bitlength[i] // 8
        if bitlength[i] % 8 != 0:
            bytelength += 1

        seg = bytearray(bytelength)

        m = bit_offset % 8
        n = bit_offset // 8
        if m == 0:
            for j in range(bytelength):
                seg[j] = string[n + j]
            if bitlength[i] % 8 != 0:
                number_of_bits = bitlength[i] % 8
                seg[bytelength - 1] = seg[bytelength - 1] & mask[number_of_bits - 1]
        else:
            for j in range(bytelength):
                seg[j] = (string[n + j] >> m) & 0xFF
                number_of_bits = 8 - m
                seg[j] = seg[j] & mask[number_of_bits - 1]

                if j < bytelength - 1:
                    high_bits = string[n + j + 1]
                    seg[j] = (seg[j] | ((high_bits << (8 - m)) & 0xFF)) & 0xFF
                elif j == bytelength - 1:
                    high_bits = string[n + j + 1]
                    seg[j] = (seg[j] | ((high_bits << (8 - m)) & 0xFF)) & 0xFF
                    number_of_extra_bits = (j + 1) * 8 - bitlength[i]
                    if number_of_extra_bits > 0:
                        seg[j] = seg[j] & mask[8 - number_of_extra_bits - 1]
        unpacked_segments.append(seg)
        bit_offset += bitlength[i]

    i = number_of_segments - 1
    bytelength = bitlength[i] // 8
    if bitlength[i] % 8 != 0:
        bytelength += 1

    seg = bytearray(bytelength)

    m = bit_offset % 8
    n = bit_offset // 8
    if m == 0:
        for j in range(bytelength):
            seg[j] = string[n + j]
    else:
        for j in range(bytelength):
            seg[j] = (string[n + j] >> m) & 0xFF
            number_of_bits = 8 - m
            seg[j] = seg[j] & mask[number_of_bits - 1]

            if j < bytelength - 1:
                high_bits = string[n + j + 1]
                seg[j] = (seg[j] | ((high_bits << (8 - m)) & 0xFF)) & 0xFF
            elif j == bytelength - 1:
                if n + j + 1 < len(string):
                    high_bits = string[n + j + 1]
                    seg[j] = (seg[j] | ((high_bits << (8 - m)) & 0xFF)) & 0xFF
                    number_of_extra_bits = (j + 1) * 8 - bitlength[i]
                    if number_of_extra_bits > 0:
                        number_of_bits = 8 - number_of_extra_bits
                        seg[j] = seg[j] & mask[number_of_bits - 1]
    unpacked_segments.append(seg)

    return unpacked_segments


def unpack_segments2(string, bytelength, data):
    number_of_segments = len(data)
    unpacked_segments = []

    k = 0
    for i in range(number_of_segments):
        extra_bits = (data[i] >> 5) & 0xFF
        extra_bits &= 7
        bitlength = bytelength[i] * 8 - extra_bits
        seg = bytearray(bytelength[i] + 1)
        for j in range(bitlength):
            value = get_bit(string, k)
            k += 1
            set_bit(seg, j, value)
        seg[bytelength[i]] = data[i]
        unpacked_segments.append(seg)

    return unpacked_segments


def unpack_segments2_by_bitlength(string, bitlength):
    number_of_segments = len(bitlength)
    unpacked_segments = []

    k = 0
    for i in range(number_of_segments):
        bytelength = bitlength[i] // 8
        if bitlength[i] % 8 != 0:
            bytelength += 1

        seg = bytearray(bytelength)

        for j in range(bitlength[i]):
            value = get_bit(string, k)
            k += 1
            set_bit(seg, j, value)

        unpacked_segments.append(seg)

    return unpacked_segments
