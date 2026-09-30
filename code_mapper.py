"""
code_mapper.py -- the parts of CodeMapper.java the Delta programs use: the
Shannon-limit estimate, "regular" Huffman coding over the 256 byte values,
and the Deflate helpers.

Deflate: Java's Deflater/Inflater and Python's zlib produce and read the
same zlib-wrapped stream (RFC 1950), so files are interchangeable.
"""

import math
import zlib

import numpy as np

from numba_support import njit

REGULAR_MAX_CODE_LENGTH = 24


# ---- Estimates ----------------------------------------------------------------

@njit
def _shannon_limit(frequency):
    total = 0
    for v in frequency:
        total += v
    limit = 0.0
    for v in frequency:
        if v != 0:
            limit -= v * (math.log(v / total) / math.log(2.0))
    return limit


def get_shannon_limit(frequency):
    return float(_shannon_limit(np.asarray(frequency, dtype=np.int64)))


# ---- Deflate ------------------------------------------------------------------

def deflate(src, level=9):
    return zlib.compress(bytes(src), level)


def inflate(src, n):
    d = zlib.decompressobj()
    out = d.decompress(bytes(src), n)
    if len(out) < n:
        raise IOError("inflated %d of %d bytes" % (len(out), n))
    return out


# ---- Huffman ------------------------------------------------------------------

def get_huffman_length2(frequency):
    """Code lengths for counts sorted largest first (in-place Moffat-Katajainen,
    as CodeMapper.getHuffmanLength2)."""
    w = list(frequency)
    n = len(w)
    leaf = root = n - 1
    for nxt in range(n - 1, 0, -1):
        if leaf < 0 or (root > nxt and w[root] < w[leaf]):
            w[nxt] = w[root]; w[root] = nxt; root -= 1
        else:
            w[nxt] = w[leaf]; leaf -= 1
        if leaf < 0 or (root > nxt and w[root] < w[leaf]):
            w[nxt] += w[root]; w[root] = nxt; root -= 1
        else:
            w[nxt] += w[leaf]; leaf -= 1
    w[1] = 0
    for nxt in range(2, n):
        w[nxt] = w[w[nxt]] + 1
    avail, used, depth, root, nxt = 1, 0, 0, 1, 0
    while avail > 0:
        while root < n and w[root] == depth:
            used += 1; root += 1
        while avail > used:
            w[nxt] = depth; nxt += 1; avail -= 1
        avail, used, depth = 2 * used, 0, depth + 1
    return w


def _build_regular_length(f):
    length = np.zeros(256, dtype=np.int64)
    used = [s for s in range(256) if f[s] > 0]
    if not used:
        return length
    if len(used) == 1:
        length[used[0]] = 1
        return length
    order = sorted(used, key=lambda s: (-f[s], s))
    sorted_length = get_huffman_length2([f[s] for s in order])
    for k, s in enumerate(order):
        length[s] = sorted_length[k] & 0xFF
    return length


def get_regular_huffman_length(frequency):
    """Code length for each of the 256 byte values (0 = unused), at most
    REGULAR_MAX_CODE_LENGTH (counts are halved until they fit)."""
    f = [int(v) for v in frequency]
    while True:
        length = _build_regular_length(f)
        if int(length.max()) <= REGULAR_MAX_CODE_LENGTH:
            return length.astype(np.uint8)
        f = [max(1, v // 2) if v > 0 else 0 for v in f]


@njit
def _canonical_code(length, max_length):
    code = np.zeros(256, dtype=np.int64)
    nxt = 0
    for L in range(1, max_length + 1):
        for s in range(256):
            if length[s] == L:
                code[s] = nxt
                nxt += 1
        nxt <<= 1
    return code


def get_regular_canonical_code(length):
    return _canonical_code(np.asarray(length, dtype=np.int64), REGULAR_MAX_CODE_LENGTH)


@njit
def _pack_regular(data, length, code):
    bits = 0
    for b in data:
        bits += length[b]
    dst = np.zeros((bits + 7) // 8, dtype=np.uint8)
    position = 0
    for b in data:
        L = length[b]
        cw = code[b]
        for k in range(L - 1, -1, -1):
            if (cw >> k) & 1:
                dst[position >> 3] |= 0x80 >> (position & 7)
            position += 1
    return dst


def pack_regular_code(data, length):
    length = np.asarray(length, dtype=np.int64)
    if np.count_nonzero(length) <= 1:
        return np.zeros(0, dtype=np.uint8)
    return _pack_regular(np.asarray(data, dtype=np.uint8), length, get_regular_canonical_code(length))


@njit
def _unpack_regular(src, n, count, first_code, first_index, order):
    dst = np.zeros(n, dtype=np.uint8)
    position = 0
    for k in range(n):
        c = 0
        L = 1
        while True:
            byte = src[position >> 3] if (position >> 3) < src.shape[0] else 0
            c = (c << 1) | ((byte >> (7 - (position & 7))) & 1)
            position += 1
            offset = c - first_code[L]
            if offset < count[L]:
                dst[k] = order[first_index[L] + offset]
                break
            L += 1
            if L > 24:
                return dst              # corrupt data; Java would throw
    return dst


@njit
def _decode_tables(length, M):
    count = np.zeros(M + 1, dtype=np.int64)
    for L in length:
        if L > 0:
            count[L] += 1
    first_code = np.zeros(M + 1, dtype=np.int64)
    first_index = np.zeros(M + 1, dtype=np.int64)
    order = np.zeros(256, dtype=np.int64)
    code_word = 0
    index = 0
    for L in range(1, M + 1):
        first_code[L] = code_word
        first_index[L] = index
        for s in range(256):
            if length[s] == L:
                order[index] = s
                index += 1
        code_word = (code_word + count[L]) << 1
    return count, first_code, first_index, order


def unpack_regular_code(src, length, n):
    length = np.asarray(length, dtype=np.int64)
    used = np.nonzero(length)[0]
    if len(used) <= 1:
        return np.full(n, used[0] if len(used) else 0, dtype=np.uint8)
    count, first_code, first_index, order = _decode_tables(length, REGULAR_MAX_CODE_LENGTH)
    src = np.frombuffer(bytes(src), dtype=np.uint8) if not isinstance(src, np.ndarray) else src
    return _unpack_regular(src, n, count, first_code, first_index, order)


def pack_regular_tables(tables, level=9):
    """All tables (256 code lengths each), one after another, Deflated."""
    return zlib.compress(b"".join(bytes(np.asarray(t, dtype=np.uint8)) for t in tables), level)


def unpack_regular_tables(packed, n):
    raw = zlib.decompressobj().decompress(bytes(packed), n * 256)
    raw = raw + bytes(n * 256 - len(raw))
    return [np.frombuffer(raw[k * 256:(k + 1) * 256], dtype=np.uint8).copy() for k in range(n)]


def get_regular_code_bytes(frequency, length):
    """Coded size in bytes of pack_regular_code, without coding."""
    length = np.asarray(length, dtype=np.int64)
    if np.count_nonzero(length) <= 1:
        return 0
    bits = int(np.dot(np.asarray(frequency, dtype=np.int64), length))
    return (bits + 7) // 8


# ---- Varint lengths (the Packet programs' per-packet coded lengths) ----------

def pack_regular_lengths(coded):
    """Each item's length as a varint: 7 bits per byte, high bit = more."""
    out = bytearray()
    for c in coded:
        v = len(c)
        while v >= 128:
            out.append((v & 127) | 128)
            v >>= 7
        out.append(v)
    return bytes(out)


def unpack_regular_lengths(packed, n):
    length = [0] * n
    position = 0
    for k in range(n):
        v = shift = 0
        while True:
            b = packed[position]
            position += 1
            v |= (b & 127) << shift
            shift += 7
            if b < 128:
                break
        length[k] = v
    return length


def get_varint_bytes(v):
    n = 1
    while v >= 128:
        v >>= 7
        n += 1
    return n
