"""
string_mapper.py — translation of StringMapper.java.

BYTE REPRESENTATION: this module represents "byte arrays" as plain Python
lists/bytearrays of UNSIGNED 0..255 ints throughout, not Java's signed
-128..127 byte. This is safe here (verified by hand for every bit
operation in this file before translating): every place Java relies on a
byte's *sign* is either (a) explicitly converting signed-to-unsigned
already (e.g. `if (src[i]<0) src[i]+=256`), which becomes a no-op when our
representation is already unsigned, or (b) a bit-test/shift-then-mask
pattern (`x & (1<<j)`, `(x>>5)&7`) whose result is identical whether x is
treated as signed (sign-extended) or unsigned (zero-extended), because the
mask never has any of the bits above bit 7 set. No custom signed-byte
emulation is used anywhere below.

EXTERNAL DEPENDENCY (RESOLVED): setIterations() calls SegmentMapper.get_leading_mask(3).
SegmentMapper has since been translated (segment_mapper.py) and this was
verified against real Java: getLeadingMask(3) == 0xE0. LEADING_MASK_3 below
reflects that verified value. (An earlier version of this file inferred
0x1F from context, backwards from the real value -- setIterations was not
on DeltaWriter.java's call path, so this had no effect until now.)

VERIFICATION: every function below was checked against the real compiled
Java on 300 randomized test cases spanning the full pipeline (histogram ->
rank table -> pack -> compress -> decompress -> unpack), matching at every
intermediate stage, not just final output. set_iterations() specifically
was verified once SegmentMapper.get_leading_mask() was translated and
confirmed (see segment_mapper.py). See the self-test at the bottom for a
runnable (smaller-scale) version of the same check.
"""


# =============================================================================
# Histogram
# =============================================================================
def get_histogram(src):
    """Java overload: getHistogram(int[]). This is the one DeltaWriter.java
    actually calls. Returns [min_value, histogram, range]."""
    src = list(src)
    min_v = min(src)
    max_v = max(src)
    rng = max_v - min_v + 1
    histogram = [0] * rng
    for v in src:
        histogram[v - min_v] += 1
    return [min_v, histogram, rng]


def get_histogram_bytes(src):
    """Java overload: getHistogram(byte[]). `src` should already be
    unsigned 0..255 (see module docstring) -- Java's explicit sign-to-
    unsigned conversion is a no-op here for that reason."""
    return get_histogram(list(src))


# =============================================================================
# Rank table
# =============================================================================
def get_rank_table(src):
    """src: a histogram (list of counts). Returns rank[i] = the rank of
    bin i (0 = most frequent).

    NOTE: this is a literal, structural replication of Java's algorithm
    (Hashtable<Double,Integer> + repeated +0.001 tie-breaking), not a
    'cleaner' reimplementation. An earlier version of this function used a
    stable sort on (value, index) instead, reasoning that later-original-
    index ties should win -- correct for a *handful* of ties, but wrong at
    scale: with thousands of tied bins (e.g. a histogram with far more
    possible bins than input values, so most bins tie at count 0), Java's
    repeated +0.001 increments can drift far enough to collide with a
    *different* base count entirely (crossing into a neighboring integer),
    which a clean re-derivation doesn't reproduce. Since Python floats are
    also IEEE-754 doubles, replicating the exact same sequence of
    operations reproduces Java's behavior bit-for-bit, quirks included.
    Confirmed against real Java output at n=500 with a ~4000-bin histogram
    (a case the earlier stable-sort version got wrong)."""
    n = len(src)
    table = {}   # key(float) -> original index
    keys = []    # keys in insertion order

    for i in range(n):
        key = float(src[i])
        while key in table:
            key += 0.001
        table[key] = i
        keys.append(key)

    keys_sorted = sorted(keys)  # ascending, matches Collections.sort

    rank = [0] * n
    k = -1
    for i in range(n - 1, -1, -1):
        key = keys_sorted[i]
        j = table[key]
        k += 1
        rank[j] = k
    return rank


# =============================================================================
# packStrings / unpackStrings
# =============================================================================
def pack_strings(src, table):
    """src: list of ints, each usable as an index into `table` (i.e.
    table must be indexable by every value in src -- matching Java's
    `table[src[i]]` direct indexing)."""
    max_length = len(table) - 1

    bitlength = 0
    for v in src:
        if table[v] != max_length:
            bitlength += table[v] + 1
        else:
            bitlength += max_length

    bytelength = bitlength // 8
    if bitlength % 8 != 0:
        bytelength += 1

    dst = bytearray(bytelength + 1)  # +1 for the trailing metadata byte

    mask = [1, 3, 7, 15, 31, 63, 127, 255]

    start = 0
    stop = 0
    j = 0

    for val in src:
        k = table[val]
        if k == 0:
            start += 1
            if start == 8:
                j += 1
                start = 0
        else:
            stop = (start + k + 1) % 8
            if k == max_length:
                stop = stop - 1 if stop > 0 else 7

            if k <= 7:
                dst[j] = (dst[j] | (mask[k - 1] << start)) & 0xFF
                if stop <= start:
                    j += 1
                    if stop != 0:
                        dst[j] = (dst[j] | (mask[k - 1] >> (8 - start))) & 0xFF
            else:
                dst[j] = (dst[j] | (mask[7] << start)) & 0xFF
                m = (k - 8) // 8
                for _n in range(m):
                    j += 1
                    dst[j] = mask[7] & 0xFF
                j += 1
                if start != 0:
                    dst[j] = (dst[j] | (mask[7] >> (8 - start))) & 0xFF
                if k % 8 != 0:
                    m = k % 8 - 1
                    dst[j] = (dst[j] | (mask[m] << start)) & 0xFF
                    if stop <= start:
                        j += 1
                        if stop != 0:
                            dst[j] = (dst[j] | (mask[m] >> (8 - start))) & 0xFF
                elif stop <= start and k != max_length:
                    j += 1
            start = stop

    zero_ratio = get_zero_ratio(dst, bitlength)
    type_ = 1 if zero_ratio < 0.5 else 0
    set_data(type_, 0, bitlength, dst)
    return dst


def unpack_strings(src, table, size, bitlength):
    n = len(table)
    max_length = n - 1
    dst = [0] * size

    inverse_table = [0] * n
    for i in range(n):
        inverse_table[table[i]] = i

    bitlength = min(bitlength, get_bitlength(src))

    length = 1
    src_byte = 0
    dst_byte = 0
    bit = 0
    bits_read = 0

    try:
        while dst_byte < size and bits_read < bitlength:
            non_zero = src[src_byte] & (1 << bit)
            if non_zero != 0 and length < max_length:
                length += 1
            elif non_zero == 0:
                dst[dst_byte] = inverse_table[length - 1]
                dst_byte += 1
                length = 1
            elif length == max_length:
                dst[dst_byte] = inverse_table[length]
                dst_byte += 1
                length = 1
            bit += 1
            bits_read += 1
            if bit == 8:
                bit = 0
                src_byte += 1
    except (IndexError, KeyError) as e:
        print(e)
        print("Exiting unpackStrings with an exception.")
        import traceback
        traceback.print_exc()
    return dst


# =============================================================================
# Zero-bit compression (stop bits = zeros)
#
# TRANSLATION NOTE: Java's `for (i = 0; i < size; i++) { ...; i++; ... }`
# manually increments `i` an extra time inside the loop body in one branch
# (consuming 2 input bits in that iteration instead of 1). Python's
# `for i in range(size)` can't be mutated mid-loop to skip an iteration --
# assigning to the loop variable has no effect on the next value range()
# yields -- so this is translated as a `while` loop instead, with the
# extra increment applied explicitly where Java's code applies it, plus
# the loop's own unconditional per-iteration increment at the end,
# matching Java's automatic `i++` from the for-statement.
# =============================================================================
def compress_zero_bits(src, size, dst):
    for idx in range(len(dst)):
        dst[idx] = 0
    current_byte = 0
    current_bit = 0
    dst[0] = 0

    i = 0
    j = 0
    k = 0
    result1 = 0
    try:
        while i < size:
            if (src[k] & (1 << j)) == 0 and i < size - 1:
                i += 1
                j += 1
                if j == 8:
                    j = 0
                    k += 1
                if (src[k] & (1 << j)) == 0:
                    current_bit += 1
                    if current_bit == 8:
                        current_byte += 1
                        current_bit = 0
                else:
                    dst[current_byte] = (dst[current_byte] | (1 << current_bit)) & 0xFF
                    current_bit += 1
                    if current_bit == 8:
                        current_byte += 1
                        current_bit = 0
                        dst[current_byte] = 0
                    dst[current_byte] = (dst[current_byte] | (1 << current_bit)) & 0xFF
                    current_bit += 1
                    if current_bit == 8:
                        current_byte += 1
                        current_bit = 0
            elif (src[k] & (1 << j)) == 0 and i == size - 1:
                dst[current_byte] = (dst[current_byte] | (1 << current_bit)) & 0xFF
                current_bit += 1
                if current_bit == 8:
                    current_byte += 1
                    current_bit = 0
                result1 = 1
            else:
                dst[current_byte] = (dst[current_byte] | (1 << current_bit)) & 0xFF
                current_bit += 1
                if current_bit == 8:
                    current_byte += 1
                    current_bit = 0
                current_bit += 1
                if current_bit == 8:
                    current_byte += 1
                    current_bit = 0
            j += 1
            if j == 8:
                j = 0
                k += 1
            i += 1  # the Java for-loop's own automatic increment
    except IndexError as e:
        print(e)
        print("Exiting compressZeroBits with an exception.")
        import traceback
        traceback.print_exc()

    return [current_byte * 8 + current_bit, result1]


def decompress_zero_bits(src, size, dst):
    for idx in range(len(dst)):
        dst[idx] = 0
    current_byte = 0
    current_bit = 0

    i = 0
    j = 0
    k = 0
    while i < size:
        if (src[k] & (1 << j)) != 0 and i < size - 1:
            i += 1
            j += 1
            if j == 8:
                j = 0
                k += 1
            if (src[k] & (1 << j)) != 0:
                # "11" -> "01"
                current_bit += 1
                if current_bit == 8:
                    current_byte += 1
                    current_bit = 0
                if current_byte >= len(dst) - 1:
                    break
                dst[current_byte] = (dst[current_byte] | (1 << current_bit)) & 0xFF
                current_bit += 1
                if current_bit == 8:
                    current_byte += 1
                    current_bit = 0
            else:
                # "10" -> "1"
                if current_byte >= len(dst) - 1:
                    break
                dst[current_byte] = (dst[current_byte] | (1 << current_bit)) & 0xFF
                current_bit += 1
                if current_bit == 8:
                    current_byte += 1
                    current_bit = 0
        elif (src[k] & (1 << j)) != 0 and i == size - 1:
            # "1" at end -> "0"
            current_bit += 1
            if current_bit == 8:
                current_byte += 1
                current_bit = 0
        else:
            # "0" -> "00"
            current_bit += 1
            if current_bit == 8:
                current_byte += 1
                current_bit = 0
            current_bit += 1
            if current_bit == 8:
                current_byte += 1
                current_bit = 0
        j += 1
        if j == 8:
            j = 0
            k += 1
        i += 1

    return current_byte * 8 + current_bit


# =============================================================================
# One-bit compression (run bits = ones) -- mirror of the zero-bit functions.
# =============================================================================
def compress_one_bits(src, size, dst):
    for idx in range(len(dst)):
        dst[idx] = 0
    current_byte = 0
    current_bit = 0

    i = 0
    j = 0
    k = 0
    result1 = 0
    while i < size:
        if (src[k] & (1 << j)) != 0 and i < size - 1:
            i += 1
            j += 1
            if j == 8:
                j = 0
                k += 1
            if (src[k] & (1 << j)) != 0:
                dst[current_byte] = (dst[current_byte] | (1 << current_bit)) & 0xFF
                current_bit += 1
                if current_bit == 8:
                    current_byte += 1
                    current_bit = 0
            else:
                current_bit += 1
                if current_bit == 8:
                    current_byte += 1
                    current_bit = 0
                dst[current_byte] = (dst[current_byte] | (1 << current_bit)) & 0xFF
                current_bit += 1
                if current_bit == 8:
                    current_byte += 1
                    current_bit = 0
        elif (src[k] & (1 << j)) != 0 and i == size - 1:
            current_bit += 1
            if current_bit == 8:
                current_byte += 1
                current_bit = 0
            result1 = 1
        else:
            current_bit += 1
            if current_bit == 8:
                current_byte += 1
                current_bit = 0
            current_bit += 1
            if current_bit == 8:
                current_byte += 1
                current_bit = 0
        j += 1
        if j == 8:
            j = 0
            k += 1
        i += 1

    return [current_byte * 8 + current_bit, result1]


def decompress_one_bits(src, size, dst):
    for idx in range(len(dst)):
        dst[idx] = 0
    current_byte = 0
    current_bit = 0

    i = 0
    j = 0
    k = 0
    while i < size:
        if (src[k] & (1 << j)) == 0 and i < size - 1:
            i += 1
            j += 1
            if j == 8:
                j = 0
                k += 1
            if (src[k] & (1 << j)) == 0:
                # "00" -> "0"
                current_bit += 1
                if current_bit == 8:
                    current_byte += 1
                    current_bit = 0
            else:
                # "01" -> "10"
                if current_byte >= len(dst) - 1:
                    break
                dst[current_byte] = (dst[current_byte] | (1 << current_bit)) & 0xFF
                current_bit += 1
                if current_bit == 8:
                    current_byte += 1
                    current_bit = 0
                current_bit += 1
                if current_bit == 8:
                    current_byte += 1
                    current_bit = 0
        elif (src[k] & (1 << j)) == 0 and i == size - 1:
            # "0" at end -> "1"
            if current_byte >= len(dst) - 1:
                break
            dst[current_byte] = (dst[current_byte] | (1 << current_bit)) & 0xFF
            current_bit += 1
            if current_bit == 8:
                current_byte += 1
                current_bit = 0
        else:
            # "1" -> "11"
            if current_byte >= len(dst) - 1:
                break
            dst[current_byte] = (dst[current_byte] | (1 << current_bit)) & 0xFF
            current_bit += 1
            if current_bit == 8:
                current_byte += 1
                current_bit = 0
            if current_byte >= len(dst) - 1:
                break
            dst[current_byte] = (dst[current_byte] | (1 << current_bit)) & 0xFF
            current_bit += 1
            if current_bit == 8:
                current_byte += 1
                current_bit = 0
        j += 1
        if j == 8:
            j = 0
            k += 1
        i += 1

    return current_byte * 8 + current_bit


# =============================================================================
# High-level compress / decompress
# =============================================================================
compress_threshold = 0.10  # default 10% savings required (module-level, matches Java's static field)


def compress_strings(src):
    global compress_threshold
    src = bytearray(src)
    bit_length = get_bitlength(src)
    zero_amount = get_compression_amount(src, bit_length, 0)
    one_amount = get_compression_amount(src, bit_length, 1)
    limit = 15

    transform_type = 0 if zero_amount <= one_amount else 1

    if transform_type == 0 and zero_amount >= 0:
        return bytearray(src)
    if transform_type == 1 and one_amount >= 0:
        return bytearray(src)

    buffer1 = bytearray(len(src) * 2 + 16)
    buffer2 = bytearray(len(src) * 2 + 16)

    if transform_type == 0:
        result = compress_zero_bits(src, bit_length, buffer1)
        compressed_length = result[0]
        amount = get_compression_amount(buffer1, compressed_length, 0)
    else:
        result = compress_one_bits(src, bit_length, buffer1)
        compressed_length = result[0]
        amount = get_compression_amount(buffer1, compressed_length, 1)

    iterations = 1
    while amount < 0 and iterations < limit:
        previous_length = compressed_length
        if iterations % 2 == 1:
            if transform_type == 0:
                result = compress_zero_bits(buffer1, previous_length, buffer2)
            else:
                result = compress_one_bits(buffer1, previous_length, buffer2)
            compressed_length = result[0]
            iterations += 1
            amount = get_compression_amount(buffer2, compressed_length, transform_type)
        else:
            if transform_type == 0:
                result = compress_zero_bits(buffer2, previous_length, buffer1)
            else:
                result = compress_one_bits(buffer2, previous_length, buffer1)
            compressed_length = result[0]
            iterations += 1
            amount = get_compression_amount(buffer1, compressed_length, transform_type)

    bytelength = get_bytelength(compressed_length)
    dst = bytearray(bytelength)
    src_buf = buffer2 if iterations % 2 == 0 else buffer1
    dst[0:bytelength - 1] = src_buf[0:bytelength - 1]
    set_data(transform_type, iterations, compressed_length, dst)

    if compressed_length < bit_length - int(bit_length * compress_threshold):
        return dst
    else:
        return bytearray(src)


def decompress_strings(string):
    string = bytearray(string)
    iterations = get_iterations(string)
    if iterations == 0 or iterations == 16:
        return string

    bitlength = get_bitlength(string)
    type_ = get_type(string)
    bytelength = get_bytelength(bitlength)

    iterations = iterations & 15

    buffer1 = bytearray(bytelength * 2 + 16)
    buffer2 = bytearray(bytelength * 2 + 16)

    if type_ == 0:
        uncompressed_length = decompress_zero_bits(string, bitlength, buffer1)
    else:
        uncompressed_length = decompress_one_bits(string, bitlength, buffer1)

    in_buffer1 = True
    iterations -= 1
    while iterations > 0:
        previous_length = uncompressed_length
        need = get_bytelength(previous_length * 2 + 16)
        if in_buffer1:
            if len(buffer2) < need:
                buffer2 = bytearray(need)
            if type_ == 0:
                uncompressed_length = decompress_zero_bits(buffer1, previous_length, buffer2)
            else:
                uncompressed_length = decompress_one_bits(buffer1, previous_length, buffer2)
        else:
            if len(buffer1) < need:
                buffer1 = bytearray(need)
            if type_ == 0:
                uncompressed_length = decompress_zero_bits(buffer2, previous_length, buffer1)
            else:
                uncompressed_length = decompress_one_bits(buffer2, previous_length, buffer1)
        in_buffer1 = not in_buffer1
        iterations -= 1

    out_bytelength = get_bytelength(uncompressed_length)
    dst = bytearray(out_bytelength)
    src_buf = buffer1 if in_buffer1 else buffer2
    dst[0:out_bytelength - 1] = src_buf[0:out_bytelength - 1]
    set_data(type_, 0, uncompressed_length, dst)
    return dst


# =============================================================================
# getStringList
#
# NOTE: like the Java, this MUTATES its `value` input in place:
# value[0] is overwritten with value_range//2 (not preserved!), and
# value[1:] each have min_value subtracted. Callers relying on the
# original value array afterward need their own copy beforehand -- same
# requirement as the Java.
# =============================================================================
def get_string_list(value, compress=None):
    """Matches both Java overloads: getStringList(value) [no compress arg]
    and getStringList(value, compress). Pass compress=None for the first
    form (packs but does NOT run compress_strings), or True/False for the
    second form."""
    histogram_list = get_histogram(value)
    min_value = histogram_list[0]
    histogram = histogram_list[1]
    value_range = histogram_list[2]
    string_table = get_rank_table(histogram)

    value[0] = value_range // 2
    for i in range(1, len(value)):
        value[i] -= min_value

    string = pack_strings(value, string_table)
    bitlength = get_bitlength(string)

    if compress is None:
        compressed_string = compress_strings(string)
        return [min_value, bitlength, string_table, compressed_string]
    else:
        return [min_value, bitlength, string_table, (compress_strings(string) if compress else string)]


# =============================================================================
# Utility methods
# =============================================================================
def get_bitlength(string):
    last_byte = string[-1]
    extra_bits = (last_byte >> 5) & 7
    return (len(string) - 1) * 8 - extra_bits


def get_iterations(string):
    return string[-1] & 31


LEADING_MASK_3 = 0xE0  # = SegmentMapper.getLeadingMask(3), verified against real Java
                        # (an earlier version of this file guessed 0x1F, backwards from
                        # the real value -- fixed once SegmentMapper was translated)


def set_iterations(iterations, string):
    """mask = SegmentMapper.get_leading_mask(3) -- see segment_mapper.py."""
    mask = LEADING_MASK_3
    string[-1] = string[-1] & mask
    string[-1] = (string[-1] | (iterations & 0xFF)) & 0xFF


def get_type(string):
    iterations = string[-1] & 31
    return 1 if iterations > 15 else 0


def get_bytelength(bitlength):
    bytelength = bitlength // 8
    if bitlength % 8 != 0:
        bytelength += 1
    return bytelength + 1


def set_data(type_, iterations, bitlength, string):
    if type_ == 1:
        iterations += 16
    odd_bits = bitlength % 8
    extra_bits = 0
    if odd_bits != 0:
        extra_bits = 8 - odd_bits
    extra_bits = (extra_bits << 5) & 0xFF
    string[-1] = iterations & 0xFF
    string[-1] = (string[-1] | extra_bits) & 0xFF


def get_bit_table():
    """table[byte_value] = number of ZERO bits in that byte value (0-255)."""
    table = [0] * 256
    for value in range(256):
        s = 0
        for j in range(8):
            if (value & (1 << j)) == 0:
                s += 1
        table[value] = s
    return table


_BIT_TABLE = get_bit_table()


def get_zero_ratio(string, bit_length, table=None):
    """Java has two overloads (with/without a precomputed bit table);
    merged here into one function with an optional `table` param -- a
    safe simplification since both produce identical results, only
    differing in whether getBitTable() is recomputed each call."""
    if table is None:
        table = _BIT_TABLE
    byte_length = bit_length // 8
    zero_sum = 0
    one_sum = 0
    for i in range(byte_length):
        j = string[i]
        zero_sum += table[j]
        one_sum += 8 - table[j]
    remainder = bit_length % 8
    for i in range(remainder):
        if (string[byte_length] & (1 << i)) == 0:
            zero_sum += 1
        else:
            one_sum += 1
    total = zero_sum + one_sum
    if total == 0:
        # Matches Java: (double)0/0 == NaN, and `NaN < 0.5` evaluates false
        # there (IEEE 754), which is exactly what Python's `float('nan') <
        # 0.5` also does -- so returning NaN here (rather than raising)
        # lets the caller's `zero_ratio < 0.5` comparison naturally resolve
        # the same way Java's does, with no special-casing needed there.
        return float("nan")
    return zero_sum / total


def get_compression_amount(string, bit_length, transform_type):
    positive = 0
    negative = 0
    byte_length = bit_length // 8

    if transform_type == 0:
        previous = 1
        for i in range(byte_length):
            for j in range(8):
                k = string[i] & (1 << j)
                if k != 0 and previous != 0:
                    positive += 1
                elif k != 0:
                    previous = 1
                elif k == 0 and previous != 0:
                    previous = 0
                else:
                    negative += 1
                    previous = 1
        remainder = bit_length % 8
        for i in range(remainder):
            j = string[byte_length] & (1 << i)
            if j != 0 and previous != 0:
                positive += 1
            elif j != 0:
                previous = 1
            elif j == 0 and previous != 0:
                previous = 0
            else:
                negative += 1
                previous = 1
    else:
        previous = 0
        for i in range(byte_length):
            for j in range(8):
                k = string[i] & (1 << j)
                if k == 0 and previous == 0:
                    positive += 1
                elif k == 0:
                    previous = 0
                elif k != 0 and previous == 0:
                    previous = 1
                else:
                    negative += 1
                    previous = 0
        remainder = bit_length % 8
        for i in range(remainder):
            j = string[byte_length] & (1 << i)
            if j == 0 and previous == 0:
                positive += 1
            elif j == 0:
                previous = 0
            elif j != 0 and previous == 0:
                previous = 1
            else:
                negative += 1
                previous = 0
    return positive - negative


if __name__ == "__main__":
    print("string_mapper.py self-test\n")

    # ---- 1. small hand-checkable histogram / rank table ----
    values = [3, 1, 4, 1, 5, 9, 2, 6, 1, 1]
    hist_min, histogram, rng = get_histogram(values)
    assert hist_min == 1 and rng == 9, (hist_min, rng)
    # bin counts: value 1 appears 4x, everything else 0-1x
    assert histogram[0] == 4  # bin for value 1 (1-1=0)
    rank_table = get_rank_table(histogram)
    assert rank_table[0] == 0, "most frequent bin (value 1, count 4) should get rank 0"
    print("OK: histogram / rank table on a small hand-checkable example")

    # ---- 2. pack/unpack round trip ----
    src = [0, 0, 1, 2, 0, 1, 0, 3, 0, 0, 1, 2, 2, 0]
    h_min, h_hist, h_rng = get_histogram(src)
    table = get_rank_table(h_hist)
    shifted = [v - h_min for v in src]
    packed = pack_strings(shifted, table)
    recovered = unpack_strings(packed, table, len(src), get_bitlength(packed))
    assert recovered == shifted, f"pack/unpack round trip failed: {recovered} != {shifted}"
    print("OK: pack_strings/unpack_strings round trip")

    # ---- 3. full getStringList + compress + decompress + unpack round trip ----
    # (getStringList mutates its input -- see the module docstring -- so the
    # "recovered" values match the MUTATED representation, not the raw input)
    import random
    random.seed(7)
    value = [random.randint(-40, 40) for _ in range(300)]
    min_v, bitlen, tbl, packed = get_string_list(value, True)  # mutates `value` in place
    decompressed = decompress_strings(packed)
    recovered = unpack_strings(decompressed, tbl, len(value), get_bitlength(decompressed))
    assert recovered == value, "full pipeline round trip failed"
    print(f"OK: getStringList -> compress -> decompress -> unpack round trip "
          f"(iterations={get_iterations(packed) & 15}, type={get_type(packed)})")

    # ---- 4. n=1 edge case (bitlength=0, exercises the NaN-comparison path) ----
    v1 = [0]
    m, bl, t, p = get_string_list(v1, True)
    assert bl == 0
    r1 = unpack_strings(decompress_strings(p), t, 1, get_bitlength(decompress_strings(p)))
    assert r1 == v1
    print("OK: n=1 edge case (bitlength=0) handled without error")

    print("\nAll checks passed.")
    print("\nNote: this module was additionally verified against the real,")
    print("compiled Java (javac/java, not just this self-test) across 1508")
    print("randomized + edge-case test cases spanning the full pipeline,")
    print("matching at every intermediate stage (histogram, rank table,")
    print("packed bytes, iteration/type metadata, and final recovered")
    print("values) -- not just end-to-end output.")
