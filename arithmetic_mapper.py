"""
arithmetic_mapper.py -- the parts of ArithmeticMapper.java the Delta
programs use:

  - the byte-oriented range coder (LZMA-style carry propagation), shared by
  - the static coder (exact counts, counted down as coded: "Arithmetic"),
  - the adaptive coder (16 contexts from the previous byte: "Adaptive"),
  - the context coder (caller-chosen contexts: "Context" and the maps),
  - block splitting and the Deflated frequency tables.

Byte-for-byte compatible with the Java. All coder loops are Numba-compiled;
64-bit integers hold every intermediate exactly (the range stays under
2^40), so there is no overflow to emulate.

The Java context decoder takes a callback that computes each context from
the symbols decoded so far. Numba can't call back into Python cheaply, so
the two callers get their own decoders here: decode_map_context (maps) and
delta_mapper's context-delta decoder, both built on the helpers below.
"""

import struct
import zlib

import numpy as np

from numba_support import njit

RANGE_TOP = 1 << 40
RANGE_BOTTOM = 1 << 32

ADAPTIVE_CONTEXT_BITS = 4
ADAPTIVE_INCREMENT = 2
ADAPTIVE_LIMIT = 1 << 22

CONTEXT_INCREMENT = 32
CONTEXT_LIMIT = 1 << 16


# =============================================================================
# Range coder. Encoder state: st = [low, range, cache, cache_size, size].
# =============================================================================

@njit
def _enc_init(n):
    st = np.zeros(5, dtype=np.int64)
    st[1] = RANGE_TOP - 1
    st[3] = 1
    out = np.zeros(max(16, n + n // 4 + 16), dtype=np.uint8)
    return st, out


@njit
def _put(st, out, b):
    if st[4] == out.shape[0]:
        bigger = np.zeros(out.shape[0] * 2, dtype=np.uint8)
        bigger[:out.shape[0]] = out
        out = bigger
    out[st[4]] = b & 0xFF
    st[4] += 1
    return out


@njit
def _shift_low(st, out):
    low = st[0]
    if low < 0xFF00000000 or low >= RANGE_TOP:
        carry = low >> 40
        temp = st[2]
        while True:
            out = _put(st, out, temp + carry)
            temp = 0xFF
            st[3] -= 1
            if st[3] == 0:
                break
        st[2] = (low >> 32) & 0xFF
    st[3] += 1
    st[0] = (low & 0xFFFFFFFF) << 8
    return out


@njit
def _encode(st, out, start, count, total):
    r = st[1] // total
    st[0] += start * r
    st[1] = count * r
    while st[1] < RANGE_BOTTOM:
        st[1] <<= 8
        out = _shift_low(st, out)
    return out


@njit
def _finish(st, out):
    for _ in range(6):
        out = _shift_low(st, out)
    return out[:st[4]].copy()


# Decoder state: dt = [pos, code, range, r].
@njit
def _next(dt, data):
    p = dt[0]
    if p < data.shape[0]:
        dt[0] = p + 1
        return np.int64(data[p])
    return np.int64(0)


@njit
def _dec_init(data):
    dt = np.zeros(4, dtype=np.int64)
    dt[0] = 1                       # the first byte written is always the initial cache
    dt[2] = RANGE_TOP - 1
    for _ in range(5):
        dt[1] = (dt[1] << 8) | _next(dt, data)
    return dt


@njit
def _target(dt, total):
    dt[3] = dt[2] // total
    t = dt[1] // dt[3]
    return min(t, total - 1)


@njit
def _decode(dt, data, start, count):
    dt[1] -= start * dt[3]
    dt[2] = count * dt[3]
    while dt[2] < RANGE_BOTTOM:
        dt[2] <<= 8
        dt[1] = (dt[1] << 8) | _next(dt, data)


# =============================================================================
# Static coder: exact byte counts, counted down (getIntervalValueFastFenwick).
# The caller stores the count n; the frequencies are stored separately.
# =============================================================================

@njit
def _fenwick_build(f, n):
    bit = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        bit[i] += f[i - 1]
        j = i + (i & -i)
        if j <= n:
            bit[j] += bit[i]
    return bit


@njit
def _fenwick_add(bit, n, s, delta):
    i = s + 1
    while i <= n:
        bit[i] += delta
        i += i & -i


@njit
def _fenwick_start(bit, s):
    total = 0
    i = s
    while i > 0:
        total += bit[i]
        i -= i & -i
    return total


@njit
def _fenwick_find(bit, n, top, target):
    pos = 0
    b = top
    while b > 0:
        nxt = pos + b
        if nxt <= n and bit[nxt] <= target:
            target -= bit[nxt]
            pos = nxt
        b >>= 1
    return pos


@njit
def _static_encode(src, frequency):
    f = frequency.astype(np.int64).copy()
    bit = _fenwick_build(f, 256)
    m = f.sum()
    st, out = _enc_init(src.shape[0])
    for i in range(src.shape[0]):
        j = np.int64(src[i])
        sj = _fenwick_start(bit, j)
        out = _encode(st, out, sj, f[j], m)
        _fenwick_add(bit, 256, j, -1)
        f[j] -= 1
        m -= 1
    return _finish(st, out)


@njit
def _static_decode(data, frequency, n):
    f = frequency.astype(np.int64).copy()
    bit = _fenwick_build(f, 256)
    m = f.sum()
    dt = _dec_init(data)
    value = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        t = _target(dt, m)
        j = _fenwick_find(bit, 256, 256, t)
        sj = _fenwick_start(bit, j)
        _decode(dt, data, sj, f[j])
        value[i] = j
        _fenwick_add(bit, 256, j, -1)
        f[j] -= 1
        m -= 1
    return value


def get_interval_value_fast_fenwick(src, frequency):
    return _static_encode(_u8(src), np.asarray(frequency, dtype=np.int64))


def get_arithmetic_values_fast_fenwick(encoded, frequency, n):
    return _static_decode(_u8(encoded), np.asarray(frequency, dtype=np.int64), n)


# =============================================================================
# Adaptive models: per context, counts starting at 1, +increment per symbol,
# halved (staying >= 1) when the total passes limit. Shared by the Adaptive
# coder (256 symbols, 16 contexts) and the Context coder (any alphabet).
# Models: f[c, s], bit[c, 0..n], total[c].
# =============================================================================

def _highest_one_bit(n):
    return 1 << (n.bit_length() - 1) if n > 0 else 0


@njit
def _models(n_contexts, n):
    f = np.ones((n_contexts, n), dtype=np.int32)
    bit = np.zeros((n_contexts, n + 1), dtype=np.int32)
    for i in range(1, n + 1):
        low = i & -i
        bit[:, i] = low                 # Fenwick tree of all ones
    total = np.full(n_contexts, n, dtype=np.int64)
    return f, bit, total


@njit
def _model_start(bit, c, s):
    total = 0
    i = s
    while i > 0:
        total += bit[c, i]
        i -= i & -i
    return total


@njit
def _model_find(bit, c, n, top, target):
    pos = 0
    b = top
    while b > 0:
        nxt = pos + b
        if nxt <= n and bit[c, nxt] <= target:
            target -= bit[c, nxt]
            pos = nxt
        b >>= 1
    return pos


@njit
def _model_update(f, bit, total, c, n, s, increment, limit):
    f[c, s] += increment
    total[c] += increment
    i = s + 1
    while i <= n:
        bit[c, i] += increment
        i += i & -i
    if total[c] > limit:
        t = 0
        for k in range(n):
            f[c, k] = (f[c, k] + 1) >> 1
            t += f[c, k]
            bit[c, k + 1] = f[c, k]
        for i in range(1, n + 1):
            j = i + (i & -i)
            if j <= n:
                bit[c, j] += bit[c, i]
        total[c] = t


# ---- Adaptive coder ---------------------------------------------------------

@njit
def _adaptive_encode(src, context_bits, increment, limit):
    f, bit, total = _models(1 << context_bits, 256)
    st, out = _enc_init(src.shape[0])
    previous = 0
    for i in range(src.shape[0]):
        j = np.int64(src[i])
        c = previous >> (8 - context_bits)
        out = _encode(st, out, _model_start(bit, c, j), f[c, j], total[c])
        _model_update(f, bit, total, c, 256, j, increment, limit)
        previous = j
    return _finish(st, out)


@njit
def _adaptive_decode(data, n, context_bits, increment, limit):
    f, bit, total = _models(1 << context_bits, 256)
    dt = _dec_init(data)
    value = np.zeros(n, dtype=np.uint8)
    previous = 0
    for i in range(n):
        c = previous >> (8 - context_bits)
        j = _model_find(bit, c, 256, 256, _target(dt, total[c]))
        _decode(dt, data, _model_start(bit, c, j), f[c, j])
        value[i] = j
        _model_update(f, bit, total, c, 256, j, increment, limit)
        previous = j
    return value


def get_interval_value_adaptive(src):
    return _adaptive_encode(_u8(src), ADAPTIVE_CONTEXT_BITS, ADAPTIVE_INCREMENT, ADAPTIVE_LIMIT)


def get_arithmetic_values_adaptive(encoded, n):
    return _adaptive_decode(_u8(encoded), n, ADAPTIVE_CONTEXT_BITS, ADAPTIVE_INCREMENT, ADAPTIVE_LIMIT)


# ---- Context coder ------------------------------------------------------------

@njit
def _context_encode(symbol, n_symbols, context, n_contexts, top):
    f, bit, total = _models(n_contexts, n_symbols)
    st, out = _enc_init(symbol.shape[0] // 2)
    for k in range(symbol.shape[0]):
        c = context[k]
        s = symbol[k]
        out = _encode(st, out, _model_start(bit, c, s), f[c, s], total[c])
        _model_update(f, bit, total, c, n_symbols, s, CONTEXT_INCREMENT, CONTEXT_LIMIT)
    return _finish(st, out)


def get_interval_value_context(symbol, n_symbols, context, n_contexts):
    return _context_encode(np.asarray(symbol, dtype=np.int64), n_symbols,
                           np.asarray(context, dtype=np.int64), n_contexts, _highest_one_bit(n_symbols))


# Decoding one symbol at a time, for decoders whose contexts depend on what
# has been decoded (see delta_mapper).
@njit
def context_decoder_init(data, n_contexts, n_symbols):
    f, bit, total = _models(n_contexts, n_symbols)
    return f, bit, total, _dec_init(data)


@njit
def context_decode_one(data, dt, f, bit, total, c, n_symbols, top):
    s = _model_find(bit, c, n_symbols, top, _target(dt, total[c]))
    _decode(dt, data, _model_start(bit, c, s), f[c, s])
    _model_update(f, bit, total, c, n_symbols, s, CONTEXT_INCREMENT, CONTEXT_LIMIT)
    return s


# =============================================================================
# Blocks and frequency tables.
# =============================================================================

def _u8(a):
    if isinstance(a, np.ndarray) and a.dtype == np.uint8:
        return a
    if isinstance(a, (bytes, bytearray)):
        return np.frombuffer(bytes(a), dtype=np.uint8).copy()
    return np.asarray(a, dtype=np.uint8)


def get_block_lengths(length, n):
    lengths = [length // n] * n
    lengths[-1] += length % n
    return lengths


def get_blocks(src, n):
    blocks, pos = [], 0
    for L in get_block_lengths(len(src), n):
        blocks.append(src[pos:pos + L])
        pos += L
    return blocks


def join_blocks(blocks):
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.uint8)


def get_frequency(src):
    return np.bincount(_u8(src), minlength=256).astype(np.int64)


def get_frequency_type(frequency):
    mx = max((int(np.max(row)) for row in frequency), default=0)
    return 0 if mx <= 255 else 1 if mx <= 65535 else 2


def deflate_frequencies(frequency, type_, level=9):
    """n*256 counts, little-endian, 1, 2 or 4 bytes each (type_ 0, 1, 2),
    Deflated -- no header (pack_frequencies adds one)."""
    dtype = ("<u1", "<u2", "<u4")[type_]
    raw = np.asarray(frequency, dtype=np.int64).astype(dtype).tobytes()
    return zlib.compress(raw, level)


def inflate_frequencies(zipped, n, type_):
    width = (1, 2, 4)[type_]
    raw = zlib.decompressobj().decompress(bytes(zipped), n * 256 * width)
    if len(raw) < n * 256 * width:
        raise IOError("frequency tables: inflated %d of %d bytes" % (len(raw), n * 256 * width))
    dtype = ("<u1", "<u2", "<u4")[type_]
    return np.frombuffer(raw, dtype=dtype).astype(np.int64).reshape(n, 256)


def pack_frequencies(frequency, level=9):
    """int n, int type, int Deflated length, Deflated bytes: n*256 counts,
    little-endian, 1, 2 or 4 bytes each (the smallest that holds them)."""
    type_ = get_frequency_type(frequency)
    dtype = ("<u1", "<u2", "<u4")[type_]
    raw = np.asarray(frequency, dtype=np.int64).astype(dtype).tobytes()
    zipped = zlib.compress(raw, level)
    return struct.pack(">iii", len(frequency), type_, len(zipped)) + zipped


def read_frequencies(stream):
    n, type_, length = struct.unpack(">iii", stream.read_fully(12))
    raw = zlib.decompress(stream.read_fully(length))
    width = (1, 2, 4)[type_]
    if len(raw) < n * 256 * width:
        raise IOError("frequency tables: inflated %d of %d bytes" % (len(raw), n * 256 * width))
    dtype = ("<u1", "<u2", "<u4")[type_]
    return np.frombuffer(raw[:n * 256 * width], dtype=dtype).astype(np.int64).reshape(n, 256)
