"""
string_mapper.py -- StringMapper.java: unary ("string") coding of integers,
plus the zero-bit / one-bit run compression applied to those strings.

Only what the Delta programs use. A "string" is a numpy uint8 array whose
last byte holds metadata: bits 0-4 the iteration count (+16 for the one-bit
transform), bits 5-7 the number of unused bits in the last data byte.

Byte-for-byte compatible with the Java; the bit loops are Numba-compiled.
As in the Java, the bit-pair transforms and their measure run a byte at a
time through tables made from their rules, and the unary strings are written
and read a run of bits at a time: the same bits as going a bit at a time.
"""

import numpy as np

from numba_support import njit

def _u8(a):
    if isinstance(a, np.ndarray) and a.dtype == np.uint8:
        return a
    if isinstance(a, (bytes, bytearray)):
        return np.frombuffer(bytes(a), dtype=np.uint8).copy()
    return np.asarray(a, dtype=np.uint8)


# ---- Metadata byte ----------------------------------------------------------

def get_bitlength(string):
    last = int(string[-1])
    return (len(string) - 1) * 8 - ((last >> 5) & 7)


def get_iterations(string):
    return int(string[-1]) & 31


def get_type(string):
    return 1 if (int(string[-1]) & 31) > 15 else 0


def get_bytelength(bitlength):
    n = bitlength // 8
    if bitlength % 8 != 0:
        n += 1
    return n + 1


@njit
def _set_data(type_, iterations, bitlength, string):
    if type_ == 1:
        iterations += 16
    odd = bitlength % 8
    extra = 0
    if odd != 0:
        extra = 8 - odd
    string[string.shape[0] - 1] = (iterations | (extra << 5)) & 0xFF


def set_data(type_, iterations, bitlength, string):
    _set_data(type_, iterations, bitlength, string)


# ---- Histogram and rank table -------------------------------------------------

def get_histogram(values):
    """[min, histogram, range] of an int array."""
    v = np.asarray(values, dtype=np.int64)
    lo, hi = int(v.min()), int(v.max())
    hist = np.bincount(v - lo, minlength=hi - lo + 1).astype(np.int64)
    return lo, hist, hi - lo + 1


def get_rank_table(histogram):
    """rank[i] of each histogram bin, 0 = most frequent (StringMapper.getRankTable).

    Among equal counts the later bin ranks first, as the Java does."""
    h = np.asarray(histogram, dtype=np.int64)
    n = len(h)
    idx = np.arange(n, dtype=np.int64)
    order = np.lexsort((-idx, -h))          # count descending, then index descending
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    return rank


# ---- Packing ----------------------------------------------------------------

@njit
def _zero_ratio(string, bit_length):
    byte_length = bit_length // 8
    zeros = 0
    ones = 0
    for i in range(byte_length):
        b = string[i]
        for j in range(8):
            if (b >> j) & 1:
                ones += 1
            else:
                zeros += 1
    for i in range(bit_length % 8):
        if (string[byte_length] >> i) & 1:
            ones += 1
        else:
            zeros += 1
    if zeros + ones == 0:
        return np.nan
    return zeros / (zeros + ones)


@njit
def _pack_strings(src, table):
    """StringMapper.packStrings: a value of rank k is k ones and then a zero,
    or just the ones for the last rank; the bits go in from the low bit of
    each byte up, gathered in acc (held of them) and written a byte at a
    time."""
    n = table.shape[0]
    max_length = n - 1
    bitlength = 0
    for i in range(src.shape[0]):
        t = table[src[i]]
        if t != max_length:
            bitlength += t + 1
        else:
            bitlength += max_length
    bytelength = bitlength // 8
    if bitlength % 8 != 0:
        bytelength += 1
    dst = np.zeros(bytelength + 1, dtype=np.uint8)
    acc = np.uint64(0)
    held = 0
    j = 0
    for i in range(src.shape[0]):
        k = table[src[i]]
        ones = k
        while ones >= 32:
            acc |= np.uint64(0xFFFFFFFF) << np.uint64(held)
            held += 32
            ones -= 32
            while held >= 8:
                dst[j] = acc & np.uint64(0xFF)
                j += 1
                acc >>= np.uint64(8)
                held -= 8
        acc |= ((np.uint64(1) << np.uint64(ones)) - np.uint64(1)) << np.uint64(held)
        held += ones + (1 if k != max_length else 0)
        while held >= 8:
            dst[j] = acc & np.uint64(0xFF)
            j += 1
            acc >>= np.uint64(8)
            held -= 8
    if held > 0:
        dst[j] = acc & np.uint64(0xFF)

    ratio = _zero_ratio(dst, bitlength)
    type_ = 1 if ratio < .5 else 0          # NaN (empty) compares false, as in Java
    _set_data(type_, 0, bitlength, dst)
    return dst


def pack_strings(values, table):
    return _pack_strings(np.asarray(values, dtype=np.int64), np.asarray(table, dtype=np.int64))


@njit
def _bits_from(src, i):
    """Up to 64 bits of src from bit i on, the first in the low bit (zeros
    past its end)."""
    start = i >> 3
    shift = i & 7
    x = np.uint64(0)
    for j in range(8):
        if start + j < src.shape[0]:
            x |= np.uint64(src[start + j]) << np.uint64(8 * j)
    x >>= np.uint64(shift)
    if shift != 0 and start + 8 < src.shape[0]:
        x |= np.uint64(src[start + 8]) << np.uint64(64 - shift)
    return x


def _numba_ctz():
    """Numba's count of trailing zero bits (one instruction), or None: it's
    in an internal module, and works only when Numba compiles."""
    try:
        from numba import config
        from numba.cpython.unsafe.numbers import trailing_zeros
        return None if config.DISABLE_JIT else trailing_zeros
    except ImportError:
        return None


_ctz = _numba_ctz()
if _ctz is not None:
    @njit
    def _trailing_zeros(x):
        """The zero bits below x's lowest one (64 if x is 0)."""
        if x == 0:
            return 64
        return np.int64(_ctz(x))
else:
    @njit
    def _trailing_zeros(x):
        """The zero bits below x's lowest one (64 if x is 0)."""
        if x == 0:
            return 64
        n = 0
        while (x & np.uint64(0xFFFF)) == 0:
            x >>= np.uint64(16)
            n += 16
        while (x & np.uint64(1)) == 0:
            x >>= np.uint64(1)
            n += 1
        return n


@njit
def _unpack_strings(src, table, size, bitlength):
    """StringMapper.unpackStrings. Bit by bit, "length" counts from 1: a one
    adds to it (up to max_length), a zero ends the value (rank length - 1),
    and a one when it's already max_length ends the last rank's value. Here
    the bits are taken up to 56 at a time, and each run of zeros or ones in
    them at once."""
    n = table.shape[0]
    max_length = n - 1
    dst = np.zeros(size, dtype=np.int64)
    inverse = np.zeros(n, dtype=np.int64)
    for i in range(n):
        inverse[table[i]] = i
    length = 1
    dst_byte = 0
    i = 0
    while dst_byte < size and i < bitlength:
        x = _bits_from(src, i)
        avail = min(56, bitlength - i)
        used = 0
        while used < avail and dst_byte < size:
            if (x & np.uint64(1)) == 0:
                # z zeros: the first ends the value under way, each other one
                # is a value of rank 0.
                z = min(_trailing_zeros(x), avail - used)
                dst[dst_byte] = inverse[length - 1]
                dst_byte += 1
                length = 1
                for _ in range(1, z):
                    if dst_byte >= size:
                        break
                    dst[dst_byte] = inverse[0]
                    dst_byte += 1
                x >>= np.uint64(z)
                used += z
            else:
                ones = min(_trailing_zeros(~x), avail - used)
                x >>= np.uint64(ones)
                used += ones
                if max_length < 1:
                    continue                    # one value: ones mean nothing
                while ones > 0 and dst_byte < size:
                    if length < max_length:
                        m = min(max_length - length, ones)
                        length += m
                        ones -= m
                    else:
                        dst[dst_byte] = inverse[length]
                        dst_byte += 1
                        length = 1
                        ones -= 1
        i += used
    return dst


def unpack_strings(src, table, size, bitlength):
    src = _u8(src)
    bitlength = min(bitlength, get_bitlength(src))
    return _unpack_strings(src, np.asarray(table, dtype=np.int64), size, bitlength)


def get_string_list(values, compress):
    """[min, bitlength, table, string] -- StringMapper.getStringList. Unlike
    the Java, the array passed in is not changed."""
    value = np.array(values, dtype=np.int64)
    lo, hist, value_range = get_histogram(value)
    table = get_rank_table(hist)
    value -= lo
    value[0] = value_range // 2
    string = pack_strings(value, table)
    bitlength = get_bitlength(string)
    return [lo, bitlength, table, compress_strings(string) if compress else string]


# ---- Zero-bit / one-bit compression ---------------------------------------------

# Each transform, and each undoing of one, is a machine with two states: 0
# at the start of a pair, 1 halfway through one (its first bit read).
# _step_rule says what one input bit does: the bits it writes and the next
# state; an input ending halfway through a pair finishes with its lone bit's
# (_END_BIT). The transforms run a byte at a time through tables made from
# those rules (_BYTE_STEP), a bit at a time only at the end of the input.
#
#   compressZeroBits    "00" -> "0", "01" -> "11", "1" -> "10", a lone "0" -> "1"
#   compressOneBits     "11" -> "1", "10" -> "01", "0" -> "00", a lone "1" -> "0"
#   decompressZeroBits  "0" -> "00", "11" -> "01", "10" -> "1", a lone "1" -> "0"
#   decompressOneBits   "1" -> "11", "00" -> "0", "01" -> "10", a lone "0" -> "1"

COMPRESS_ZERO, COMPRESS_ONE, DECOMPRESS_ZERO, DECOMPRESS_ONE = 0, 1, 2, 3


def _step_rule(transform, state, bit):
    """One input bit of a transform: (pattern, count, next state), the
    pattern's first bit in its low bit (StringMapper.stepRule)."""
    pair_start = 0 if transform in (COMPRESS_ZERO, DECOMPRESS_ONE) else 1
    if state == 0:
        if bit == pair_start:
            return 0, 0, 1
        return {COMPRESS_ZERO: 1, COMPRESS_ONE: 0, DECOMPRESS_ZERO: 0, DECOMPRESS_ONE: 3}[transform], 2, 0
    if transform == COMPRESS_ZERO:
        return (3, 2, 0) if bit else (0, 1, 0)      # "01" -> "11", "00" -> "0"
    if transform == COMPRESS_ONE:
        return (1, 1, 0) if bit else (2, 2, 0)      # "11" -> "1",  "10" -> "01"
    if transform == DECOMPRESS_ZERO:
        return (2, 2, 0) if bit else (1, 1, 0)      # "11" -> "01", "10" -> "1"
    return (1, 2, 0) if bit else (0, 1, 0)          # "01" -> "10", "00" -> "0"


def _amount_rule(type_, previous, bit):
    """One bit of getCompressionAmount's count: (amount, next previous)."""
    if type_ == 0:
        if bit and previous:
            return 1, previous
        if bit:
            return 0, 1
        if previous:
            return 0, 0
        return -1, 1
    if not bit and not previous:
        return 1, previous
    if not bit:
        return 0, 0
    if not previous:
        return 0, 1
    return -1, 0


def _make_tables():
    bit_step = np.zeros((4, 2, 2, 3), dtype=np.int64)       # [transform][state][bit] -> pattern, count, state
    byte_step = np.zeros((4, 2, 256, 3), dtype=np.int64)    # the same for a whole byte
    amount = np.zeros((2, 2, 256, 2), dtype=np.int64)       # [type][previous][byte] -> amount, previous
    for t in range(4):
        for state in range(2):
            for bit in range(2):
                bit_step[t, state, bit] = _step_rule(t, state, bit)
    for t in range(4):
        for state in range(2):
            for b in range(256):
                s, pattern, count = state, 0, 0
                for j in range(8):
                    p, c, s = _step_rule(t, s, (b >> j) & 1)
                    pattern |= p << count
                    count += c
                byte_step[t, state, b] = (pattern, count, s)
    for type_ in range(2):
        for previous in range(2):
            for b in range(256):
                p, total = previous, 0
                for j in range(8):
                    a, p = _amount_rule(type_, p, (b >> j) & 1)
                    total += a
                amount[type_, previous, b] = (total, p)
    return bit_step, byte_step, amount


_BIT_STEP, _BYTE_STEP, _AMOUNT = _make_tables()
_END_BIT = np.array([1, 0, 0, 1], dtype=np.int64)          # each transform's lone last bit's output


@njit
def _transform_bits(transform, src, size, dst, bit_step, byte_step, end_bit):
    """Runs a transform over size bits of src into dst (zeroed first). The
    undoing ones stop early rather than write a one into dst's last byte; as
    the bit-at-a-time code did, zeros past dst's end are only counted.
    Returns the number of output bits."""
    for i in range(dst.shape[0]):
        dst[i] = 0
    undoing = transform == DECOMPRESS_ZERO or transform == DECOMPRESS_ONE
    limit = dst.shape[0] - 1
    acc = np.uint64(0)
    held = 0
    out = 0
    state = 0
    i = 0
    # Whole bytes, while no stop could come in one (a byte writes at most 2).
    while i + 8 <= size and (not undoing or out + 3 < limit):
        step = byte_step[transform, state, src[i >> 3]]
        acc |= np.uint64(step[0]) << np.uint64(held)
        held += step[1]
        state = step[2]
        while held >= 8:
            if out < dst.shape[0]:
                dst[out] = acc & np.uint64(0xFF)
            out += 1
            acc >>= np.uint64(8)
            held -= 8
        i += 8
    # The rest a bit at a time, checking for the stop before each one.
    stopped = False
    while i < size and not stopped:
        step = bit_step[transform, state, (src[i >> 3] >> (i & 7)) & 1]
        state = step[2]
        for j in range(step[1]):
            bit = (step[0] >> j) & 1
            if undoing and bit == 1 and out >= limit:
                stopped = True
                break
            acc |= np.uint64(bit) << np.uint64(held)
            held += 1
            while held >= 8:
                if out < dst.shape[0]:
                    dst[out] = acc & np.uint64(0xFF)
                out += 1
                acc >>= np.uint64(8)
                held -= 8
        i += 1
    if not stopped and state == 1:
        bit = end_bit[transform]
        if not (undoing and bit == 1 and out >= limit):
            acc |= np.uint64(bit) << np.uint64(held)
            held += 1
            while held >= 8:
                if out < dst.shape[0]:
                    dst[out] = acc & np.uint64(0xFF)
                out += 1
                acc >>= np.uint64(8)
                held -= 8
    if held > 0 and out < dst.shape[0]:
        dst[out] = acc & np.uint64(0xFF)
    return out * 8 + held


def _compress_bits(src, size, dst, one):
    """compressZeroBits (one=0) / compressOneBits (one=1). Returns the
    compressed bit length."""
    return _transform_bits(COMPRESS_ONE if one else COMPRESS_ZERO, src, size, dst, _BIT_STEP, _BYTE_STEP, _END_BIT)


def _decompress_bits(src, size, dst, one):
    """decompressZeroBits (one=0) / decompressOneBits (one=1)."""
    return _transform_bits(DECOMPRESS_ONE if one else DECOMPRESS_ZERO, src, size, dst, _BIT_STEP, _BYTE_STEP, _END_BIT)


@njit
def _amount(string, bit_length, type_, amount):
    previous = 1 if type_ == 0 else 0
    total = 0
    byte_length = bit_length // 8
    for i in range(byte_length):
        step = amount[type_, previous, string[i]]
        total += step[0]
        previous = step[1]
    for i in range(byte_length * 8, bit_length):
        bit = (string[byte_length] >> (i & 7)) & 1
        # (the last bits, by the same rules: a one-byte table lookup of just them)
        if type_ == 0:
            if bit != 0 and previous != 0:
                total += 1
            elif bit != 0:
                previous = 1
            elif previous != 0:
                previous = 0
            else:
                total -= 1
                previous = 1
        else:
            if bit == 0 and previous == 0:
                total += 1
            elif bit == 0:
                previous = 0
            elif previous == 0:
                previous = 1
            else:
                total -= 1
                previous = 0
    return total


def _compression_amount(string, bit_length, transform_type):
    """getCompressionAmount: the bits one pass of the transform would save
    (the pairs it shortens less the bits it lengthens), a byte at a time
    through a table made from its rules (_amount_rule)."""
    return _amount(string, bit_length, 0 if transform_type == 0 else 1, _AMOUNT)


compress_threshold = 0.10


def compress_strings(src):
    src = _u8(src)
    bit_length = get_bitlength(src)
    zero_amount = _compression_amount(src, bit_length, 0)
    one_amount = _compression_amount(src, bit_length, 1)
    transform = 0 if zero_amount <= one_amount else 1
    if (transform == 0 and zero_amount >= 0) or (transform == 1 and one_amount >= 0):
        return src.copy()
    buffer1 = np.zeros(len(src) * 2 + 16, dtype=np.uint8)
    buffer2 = np.zeros(len(src) * 2 + 16, dtype=np.uint8)
    length = _compress_bits(src, bit_length, buffer1, transform)
    amount = _compression_amount(buffer1, length, transform)
    iterations = 1
    while amount < 0 and iterations < 15:
        if iterations % 2 == 1:
            length = _compress_bits(buffer1, length, buffer2, transform)
            amount = _compression_amount(buffer2, length, transform)
        else:
            length = _compress_bits(buffer2, length, buffer1, transform)
            amount = _compression_amount(buffer1, length, transform)
        iterations += 1
    bytelength = get_bytelength(length)
    dst = np.zeros(bytelength, dtype=np.uint8)
    dst[:bytelength - 1] = (buffer2 if iterations % 2 == 0 else buffer1)[:bytelength - 1]
    _set_data(transform, iterations, length, dst)
    if length < bit_length - int(bit_length * compress_threshold):
        return dst
    return src.copy()


def decompress_strings(string):
    string = _u8(string)
    iterations = get_iterations(string)
    if iterations == 0 or iterations == 16:
        return string
    bitlength = get_bitlength(string)
    type_ = get_type(string)
    bytelength = get_bytelength(bitlength)
    iterations &= 15
    buffer1 = np.zeros(bytelength * 2 + 16, dtype=np.uint8)
    buffer2 = np.zeros(bytelength * 2 + 16, dtype=np.uint8)
    length = _decompress_bits(string, bitlength, buffer1, type_)
    in_buffer1 = True
    iterations -= 1
    while iterations > 0:
        need = get_bytelength(length * 2 + 16)
        if in_buffer1:
            if len(buffer2) < need:
                buffer2 = np.zeros(need, dtype=np.uint8)
            length = _decompress_bits(buffer1, length, buffer2, type_)
        else:
            if len(buffer1) < need:
                buffer1 = np.zeros(need, dtype=np.uint8)
            length = _decompress_bits(buffer2, length, buffer1, type_)
        in_buffer1 = not in_buffer1
        iterations -= 1
    out_len = get_bytelength(length)
    dst = np.zeros(out_len, dtype=np.uint8)
    dst[:out_len - 1] = (buffer1 if in_buffer1 else buffer2)[:out_len - 1]
    _set_data(type_, 0, length, dst)
    return dst
