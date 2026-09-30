"""
string_mapper.py -- StringMapper.java: unary ("string") coding of integers,
plus the zero-bit / one-bit run compression applied to those strings.

Only what the Delta programs use. A "string" is a numpy uint8 array whose
last byte holds metadata: bits 0-4 the iteration count (+16 for the one-bit
transform), bits 5-7 the number of unused bits in the last data byte.

Byte-for-byte compatible with the Java; the bit loops are Numba-compiled.
"""

import numpy as np

from numba_support import njit

_MASK = np.array([1, 3, 7, 15, 31, 63, 127, 255], dtype=np.int64)


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
    """rank[i] of each histogram bin, 0 = most frequent.

    Replicates the Java exactly, including its tie-breaking: equal counts
    are made distinct by adding .001 repeatedly (Java Hashtable of Double
    keys), then sorted. With many ties the nudged keys can drift into the
    next integer, which changes the order -- so this is done literally, not
    with a stable sort. Python floats are the same IEEE doubles."""
    table = {}
    keys = []
    for i, c in enumerate(np.asarray(histogram).tolist()):
        key = float(c)
        while key in table:
            key += .001
        table[key] = i
        keys.append(key)
    keys.sort()
    n = len(keys)
    rank = np.zeros(n, dtype=np.int64)
    k = -1
    for i in range(n - 1, -1, -1):
        k += 1
        rank[table[keys[i]]] = k
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
def _pack_strings(src, table, mask):
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

    start = 0
    stop = 0
    j = 0
    for i in range(src.shape[0]):
        k = table[src[i]]
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
                for _ in range(m):
                    j += 1
                    dst[j] = 255
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

    ratio = _zero_ratio(dst, bitlength)
    type_ = 1 if ratio < .5 else 0          # NaN (empty) compares false, as in Java
    _set_data(type_, 0, bitlength, dst)
    return dst


def pack_strings(values, table):
    return _pack_strings(np.asarray(values, dtype=np.int64), np.asarray(table, dtype=np.int64), _MASK)


@njit
def _unpack_strings(src, table, size, bitlength):
    n = table.shape[0]
    max_length = n - 1
    dst = np.zeros(size, dtype=np.int64)
    inverse = np.zeros(n, dtype=np.int64)
    for i in range(n):
        inverse[table[i]] = i
    length = 1
    src_byte = 0
    dst_byte = 0
    bit = 0
    bits_read = 0
    while dst_byte < size and bits_read < bitlength and src_byte < src.shape[0]:
        non_zero = (src[src_byte] >> bit) & 1
        if non_zero != 0 and length < max_length:
            length += 1
        elif non_zero == 0:
            dst[dst_byte] = inverse[length - 1]
            dst_byte += 1
            length = 1
        elif length == max_length:
            dst[dst_byte] = inverse[length]
            dst_byte += 1
            length = 1
        bit += 1
        bits_read += 1
        if bit == 8:
            bit = 0
            src_byte += 1
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

@njit
def _compress_bits(src, size, dst, one):
    """compressZeroBits (one=0) / compressOneBits (one=1). Returns the
    compressed bit length."""
    for i in range(dst.shape[0]):
        dst[i] = 0
    cb = 0
    cbit = 0
    i = 0
    j = 0
    k = 0
    while i < size:
        bit = (src[k] >> j) & 1
        run = (bit == 0) if one == 0 else (bit != 0)
        if run and i < size - 1:
            i += 1
            j += 1
            if j == 8:
                j = 0
                k += 1
            bit2 = (src[k] >> j) & 1
            same = (bit2 == 0) if one == 0 else (bit2 != 0)
            if one == 0:
                if same:
                    cbit += 1
                    if cbit == 8:
                        cb += 1
                        cbit = 0
                else:
                    dst[cb] |= 1 << cbit
                    cbit += 1
                    if cbit == 8:
                        cb += 1
                        cbit = 0
                        dst[cb] = 0
                    dst[cb] |= 1 << cbit
                    cbit += 1
                    if cbit == 8:
                        cb += 1
                        cbit = 0
            else:
                if same:
                    dst[cb] |= 1 << cbit
                    cbit += 1
                    if cbit == 8:
                        cb += 1
                        cbit = 0
                else:
                    cbit += 1
                    if cbit == 8:
                        cb += 1
                        cbit = 0
                    dst[cb] |= 1 << cbit
                    cbit += 1
                    if cbit == 8:
                        cb += 1
                        cbit = 0
        elif run and i == size - 1:
            if one == 0:
                dst[cb] |= 1 << cbit
            cbit += 1
            if cbit == 8:
                cb += 1
                cbit = 0
        else:
            if one == 0:
                dst[cb] |= 1 << cbit
            cbit += 1
            if cbit == 8:
                cb += 1
                cbit = 0
            cbit += 1
            if cbit == 8:
                cb += 1
                cbit = 0
        j += 1
        if j == 8:
            j = 0
            k += 1
        i += 1
    return cb * 8 + cbit


@njit
def _decompress_bits(src, size, dst, one):
    """decompressZeroBits (one=0) / decompressOneBits (one=1)."""
    for i in range(dst.shape[0]):
        dst[i] = 0
    cb = 0
    cbit = 0
    last = dst.shape[0] - 1
    i = 0
    j = 0
    k = 0
    while i < size:
        bit = (src[k] >> j) & 1
        mark = (bit != 0) if one == 0 else (bit == 0)
        if mark and i < size - 1:
            i += 1
            j += 1
            if j == 8:
                j = 0
                k += 1
            bit2 = (src[k] >> j) & 1
            if one == 0:
                if bit2 != 0:                   # "11" -> "01"
                    cbit += 1
                    if cbit == 8:
                        cb += 1
                        cbit = 0
                    if cb >= last:
                        break
                    dst[cb] |= 1 << cbit
                    cbit += 1
                    if cbit == 8:
                        cb += 1
                        cbit = 0
                else:                           # "10" -> "1"
                    if cb >= last:
                        break
                    dst[cb] |= 1 << cbit
                    cbit += 1
                    if cbit == 8:
                        cb += 1
                        cbit = 0
            else:
                if bit2 == 0:                   # "00" -> "0"
                    cbit += 1
                    if cbit == 8:
                        cb += 1
                        cbit = 0
                else:                           # "01" -> "10"
                    if cb >= last:
                        break
                    dst[cb] |= 1 << cbit
                    cbit += 1
                    if cbit == 8:
                        cb += 1
                        cbit = 0
                    cbit += 1
                    if cbit == 8:
                        cb += 1
                        cbit = 0
        elif mark and i == size - 1:
            if one == 0:                        # "1" at end -> "0"
                cbit += 1
                if cbit == 8:
                    cb += 1
                    cbit = 0
            else:                               # "0" at end -> "1"
                if cb >= last:
                    break
                dst[cb] |= 1 << cbit
                cbit += 1
                if cbit == 8:
                    cb += 1
                    cbit = 0
        else:
            if one == 0:                        # "0" -> "00"
                cbit += 2
                if cbit >= 8:
                    cb += 1
                    cbit -= 8
            else:                               # "1" -> "11"
                if cb >= last:
                    break
                dst[cb] |= 1 << cbit
                cbit += 1
                if cbit == 8:
                    cb += 1
                    cbit = 0
                if cb >= last:
                    break
                dst[cb] |= 1 << cbit
                cbit += 1
                if cbit == 8:
                    cb += 1
                    cbit = 0
        j += 1
        if j == 8:
            j = 0
            k += 1
        i += 1
    return cb * 8 + cbit


@njit
def _compression_amount(string, bit_length, transform_type):
    positive = 0
    negative = 0
    byte_length = bit_length // 8
    total = byte_length * 8 + bit_length % 8
    previous = 1 if transform_type == 0 else 0
    for p in range(total):
        bit = (string[p >> 3] >> (p & 7)) & 1
        if transform_type == 0:
            if bit != 0 and previous != 0:
                positive += 1
            elif bit != 0:
                previous = 1
            elif previous != 0:
                previous = 0
            else:
                negative += 1
                previous = 1
        else:
            if bit == 0 and previous == 0:
                positive += 1
            elif bit == 0:
                previous = 0
            elif previous == 0:
                previous = 1
            else:
                negative += 1
                previous = 0
    return positive - negative


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
