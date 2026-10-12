"""
pixel_mapper.py -- PixelMapper.java (version 1.2): codes the pixels of the
red, green and blue channels with context-adaptive arithmetic coding, for
pixel_writer.py and pixel_reader.py.

The image is divided into square blocks (smaller at the right and bottom
edges), coded in rows of blocks, and the pixels of each block row by row.
Each channel is one arithmetic-coded stream whose models restart at the start
of every block, so a block is coded with what was learned inside it only.
Neighbours, though, are taken across block edges: they are already decoded
when the block is reached.

The channels are coded green, red, blue. Every coding starts from the same
prediction: the median edge predictor (MED) of the left, upper and upper-left
neighbours, corrected (when coding red, green and blue; see CHANNEL_SET_NAMES)
by the average of how far the earlier channels at the same pixel were from
their own MED predictions. "Activity" is how busy the neighbourhood is:
|left - upper-left| + |upper - upper-left| on the 0-255 scale.

Three codings (CODING_NAMES):
  0 Values, exact prediction: the pixel value, in a context of the predicted
    value x 3 activity bins.
  1 Values, brightness bands: the pixel value, in a context of the predicted
    value in bands of 4 (on the 0-255 scale) x 2 activity bins.
  2 Errors: value - prediction (wrapped into 0..max_value), in a context of
    6 activity bins x 5 bins of the size of the earlier channels' errors.

Counts start at 1 and go up by INCREMENT per symbol coded; a context whose
total passes LIMIT has its counts halved.

Byte-for-byte compatible with the Java. The coding loops are Numba-compiled
and use arithmetic_mapper's range coder.

Channels are flat int64 arrays; a channel set is a list of three, red, green,
blue (the writers' order: red holds bits 16-23 of a Java pixel).
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import arithmetic_mapper as am
import resize_mapper as rm
from numba_support import njit, jdiv

BLOCK_MIN, BLOCK_MAX = 8, 256
INCREMENT = 32
LIMIT = 1 << 13

VALUES, BANDS, ERRORS = 0, 1, 2
CODING_NAMES = ["Values, exact prediction", "Values, brightness bands", "Errors (value - prediction)"]

# Activity bins (lower limits, 0-255 scale) for each coding, and the bins for
# the size of the earlier channels' errors (Errors only).
ACTIVITY_LIMIT = [np.array([0, 5, 13], dtype=np.int64), np.array([0, 5], dtype=np.int64),
                  np.array([0, 2, 5, 10, 20, 40], dtype=np.int64)]
CROSS_LIMIT = np.array([0, 2, 5, 12, 30], dtype=np.int64)
BAND_BITS = 2   # bands of 4 on the 0-255 scale

# Coding order of the 3 channels, as indexes into [red, green, blue].
ORDER = [1, 0, 2]
CHANNEL_NAMES = ["red", "green", "blue"]

# Channels coded:
#   PRIMARY:    green, red, blue, with red's and blue's predictions corrected
#               by the earlier channels' errors.
#   DIFFERENCE: green, red - green, blue - green (each offset by max_value so
#               it is never negative), with no correction.
PRIMARY, DIFFERENCE = 0, 1
CHANNEL_SET_NAMES = ["Red, green, blue (corrected)", "Green, red - green, blue - green"]

# The block sizes find_best_block tries: powers of 2, then the steps halfway
# (by ratio) around the best of those.
SEARCH_SIZES = [8, 16, 32, 64, 128, 256]


# =============================================================================
# Contexts
# =============================================================================

@njit
def med(a, b, c):
    if c >= max(a, b):
        return min(a, b)
    if c <= min(a, b):
        return max(a, b)
    return a + b - c


@njit
def _bucket(v, limit):
    b = 0
    while b + 1 < limit.shape[0] and v >= limit[b + 1]:
        b += 1
    return b


@njit
def _neighbours(v, k, x, y, xdim, max_value):
    a = v[k - 1] if x > 0 else (v[k - xdim] if y > 0 else (max_value + 1) // 2)
    b = v[k - xdim] if y > 0 else a
    c = v[k - xdim - 1] if (x > 0 and y > 0) else b
    return a, b, c


@njit
def _context(v, i, error, x, y, xdim, max_value, shift, coding, correct, limit, cross_limit):
    """(context, prediction) for pixel (x, y) of channel i (coding order)."""
    k = y * xdim + x
    a, b, c = _neighbours(v, k, x, y, xdim, max_value)
    p = med(a, b, c)
    cross = 0
    if i > 0:
        total = 0
        size = 0
        for j in range(i):
            total += error[j, k]
            size += abs(error[j, k])
        if correct:
            p += jdiv(total, i)
            if p < 0:
                p = 0
            elif p > max_value:
                p = max_value
        cross = _bucket((size // i) << shift, cross_limit)
    activity = _bucket((abs(a - c) + abs(b - c)) << shift, limit)
    if coding == 0:
        ctx = p * limit.shape[0] + activity
    elif coding == 1:
        ctx = (p >> max(0, 2 - shift)) * limit.shape[0] + activity
    else:
        ctx = activity * cross_limit.shape[0] + cross
    return ctx, p


@njit
def _prediction_error(v, xdim, ydim, max_value):
    e = np.zeros(v.shape[0], dtype=np.int64)
    for y in range(ydim):
        for x in range(xdim):
            k = y * xdim + x
            a, b, c = _neighbours(v, k, x, y, xdim, max_value)
            e[k] = v[k] - med(a, b, c)
    return e


def prediction_error(v, xdim, ydim, max_value):
    """Each pixel's value minus its MED prediction."""
    return _prediction_error(np.asarray(v, dtype=np.int64), xdim, ydim, max_value)


def bucket(v, limit):
    return int(_bucket(v, np.asarray(limit, dtype=np.int64)))


# =============================================================================
# Models: one per context, counts with a Fenwick tree. A model is reset the
# first time it is used in a block (generation), as the Java's lazy models.
# =============================================================================

@njit
def _highest_one_bit(n):
    b = 1
    while b * 2 <= n:
        b *= 2
    return b


@njit
def _reset(f, bit, total, ctx, n):
    for s in range(n):
        f[ctx, s] = 1
    _rebuild(f, bit, total, ctx, n)


@njit
def _rebuild(f, bit, total, ctx, n):
    for q in range(n + 1):
        bit[ctx, q] = 0
    t = 0
    for s in range(n):
        t += f[ctx, s]
        q = s + 1
        while q <= n:
            bit[ctx, q] += f[ctx, s]
            q += q & -q
    total[ctx] = t


@njit
def _start(bit, ctx, s):
    total = 0
    q = s
    while q > 0:
        total += bit[ctx, q]
        q -= q & -q
    return total


@njit
def _find(bit, ctx, n, top, target):
    pos = 0
    b = top
    while b > 0:
        nxt = pos + b
        if nxt <= n and bit[ctx, nxt] <= target:
            target -= bit[ctx, nxt]
            pos = nxt
        b >>= 1
    return pos


@njit
def _update(f, bit, total, ctx, n, s, increment, limit):
    f[ctx, s] += increment
    total[ctx] += increment
    q = s + 1
    while q <= n:
        bit[ctx, q] += increment
        q += q & -q
    if total[ctx] > limit:
        for k in range(n):
            f[ctx, k] = (f[ctx, k] + 1) >> 1
        _rebuild(f, bit, total, ctx, n)


def _context_count(max_value, shift, coding):
    limit = ACTIVITY_LIMIT[coding]
    if coding == VALUES:
        return (max_value + 1) * len(limit)
    if coding == BANDS:
        return ((max_value >> max(0, BAND_BITS - shift)) + 1) * len(limit)
    return len(limit) * len(CROSS_LIMIT)


# =============================================================================
# Coding one channel
# =============================================================================

@njit
def _encode_channel(v, i, error, xdim, ydim, max_value, shift, block, coding, correct, limit, cross_limit, n_ctx):
    n = max_value + 1
    f = np.zeros((n_ctx, n), dtype=np.int64)
    bit = np.zeros((n_ctx, n + 1), dtype=np.int64)
    total = np.zeros(n_ctx, dtype=np.int64)
    gen = np.full(n_ctx, -1, dtype=np.int64)
    generation = 0
    st, out = am._enc_init(v.shape[0] // 2)
    for by in range(0, ydim, block):
        for bx in range(0, xdim, block):
            generation += 1
            y1 = min(ydim, by + block)
            x1 = min(xdim, bx + block)
            for y in range(by, y1):
                for x in range(bx, x1):
                    ctx, p = _context(v, i, error, x, y, xdim, max_value, shift, coding, correct, limit, cross_limit)
                    if gen[ctx] != generation:
                        _reset(f, bit, total, ctx, n)
                        gen[ctx] = generation
                    value = v[y * xdim + x]
                    s = (value - p) % n if coding == 2 else value
                    out = am._encode(st, out, _start(bit, ctx, s), f[ctx, s], total[ctx])
                    _update(f, bit, total, ctx, n, s, 32, 1 << 13)
    return am._finish(st, out)


@njit
def _decode_channel(data, i, error, xdim, ydim, max_value, shift, block, coding, correct, limit, cross_limit, n_ctx):
    n = max_value + 1
    v = np.zeros(xdim * ydim, dtype=np.int64)
    f = np.zeros((n_ctx, n), dtype=np.int64)
    bit = np.zeros((n_ctx, n + 1), dtype=np.int64)
    total = np.zeros(n_ctx, dtype=np.int64)
    gen = np.full(n_ctx, -1, dtype=np.int64)
    generation = 0
    top = _highest_one_bit(n)
    dt = am._dec_init(data)
    for by in range(0, ydim, block):
        for bx in range(0, xdim, block):
            generation += 1
            y1 = min(ydim, by + block)
            x1 = min(xdim, bx + block)
            for y in range(by, y1):
                for x in range(bx, x1):
                    ctx, p = _context(v, i, error, x, y, xdim, max_value, shift, coding, correct, limit, cross_limit)
                    if gen[ctx] != generation:
                        _reset(f, bit, total, ctx, n)
                        gen[ctx] = generation
                    s = _find(bit, ctx, n, top, am._target(dt, total[ctx]))
                    am._decode(dt, data, _start(bit, ctx, s), f[ctx, s])
                    v[y * xdim + x] = (p + s) % n if coding == 2 else s
                    _update(f, bit, total, ctx, n, s, 32, 1 << 13)
    return v


def channel_max(max_value, channels):
    """The largest value of each coded channel: a difference spans 0..2 * max_value."""
    if channels == DIFFERENCE:
        return [max_value, 2 * max_value, 2 * max_value]
    return [max_value, max_value, max_value]


def _map(n, body, parallel):
    if parallel:
        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(body, range(n)))
    return [body(i) for i in range(n)]


def encode(channel, xdim, ydim, max_value, shift, block, coding, channels, parallel=True):
    """Codes 3 channels (values 0..max_value, red, green, blue) of an xdim x
    ydim image. Returns the coded bytes (uint8 arrays) per channel, in ORDER.
    The channels are coded in parallel: the encoder has all of them."""
    ordered = [np.asarray(channel[ORDER[i]], dtype=np.int64) for i in range(3)]
    if channels == DIFFERENCE:
        for i in (1, 2):
            ordered[i] = ordered[i] - ordered[0] + max_value
    mx = channel_max(max_value, channels)
    error = np.zeros((3, xdim * ydim), dtype=np.int64)
    for i in range(2):
        error[i] = _prediction_error(ordered[i], xdim, ydim, mx[i])
    limit = ACTIVITY_LIMIT[coding]

    def one(i):
        return _encode_channel(ordered[i], i, error, xdim, ydim, mx[i], shift, block, coding, channels == PRIMARY,
                               limit, CROSS_LIMIT, _context_count(mx[i], shift, coding))
    return _map(3, one, parallel)


def decode(coded, xdim, ydim, max_value, shift, block, coding, channels):
    """Decodes what encode() made; returns red, green, blue. The channels are
    decoded in order, since each one's contexts use the ones before it."""
    mx = channel_max(max_value, channels)
    error = np.zeros((3, xdim * ydim), dtype=np.int64)
    ordered = [None] * 3
    limit = ACTIVITY_LIMIT[coding]
    for i in range(3):
        data = np.frombuffer(bytes(coded[i]), dtype=np.uint8)
        ordered[i] = _decode_channel(data, i, error, xdim, ydim, mx[i], shift, block, coding, channels == PRIMARY,
                                     limit, CROSS_LIMIT, _context_count(mx[i], shift, coding))
        if i < 2:
            error[i] = _prediction_error(ordered[i], xdim, ydim, mx[i])
    if channels == DIFFERENCE:
        for i in (1, 2):
            ordered[i] = ordered[i] + ordered[0] - max_value
    channel = [None] * 3
    for i in range(3):
        channel[ORDER[i]] = ordered[i]
    return channel


# =============================================================================
# Choosing the block size
# =============================================================================

def _java_round(x):
    return int(math.floor(x + 0.5))


def find_best_block(channel, xdim, ydim, max_value, shift, coding, channels, tried=None):
    """The block size (BLOCK_MIN..BLOCK_MAX) that codes the channels smallest:
    every size in SEARCH_SIZES, then the sizes halfway (by ratio) to each
    neighbour of the best (ties go to the smaller size). tried, if a list,
    gets (size, coded bytes of the 3 channels) for every size tried,
    smallest size first."""
    sizes = {}
    _measure(SEARCH_SIZES, channel, xdim, ydim, max_value, shift, coding, channels, sizes)
    best = _best_of(sizes)
    more = []
    lower, upper = _java_round(best / math.sqrt(2)), _java_round(best * math.sqrt(2))
    if lower >= BLOCK_MIN and lower not in sizes:
        more.append(lower)
    if upper <= BLOCK_MAX and upper not in sizes:
        more.append(upper)
    _measure(more, channel, xdim, ydim, max_value, shift, coding, channels, sizes)
    if tried is not None:
        tried.extend(sorted(sizes.items()))
    return _best_of(sizes)


def _measure(block_sizes, channel, xdim, ydim, max_value, shift, coding, channels, sizes):
    if not block_sizes:
        return

    def one(s):
        return sum(len(c) for c in encode(channel, xdim, ydim, max_value, shift, block_sizes[s], coding, channels, parallel=False))
    for b, n in zip(block_sizes, _map(len(block_sizes), one, True)):
        sizes[b] = n


def _best_of(sizes):
    best = -1
    for b in sorted(sizes):
        if best < 0 or sizes[b] < sizes[best]:
            best = b
    return best


# =============================================================================
# Quantizing and assembling (shared by the writer and reader)
# =============================================================================

def clamp(v):
    return np.clip(v, 0, 255)


def get_rgb(channel, qxdim, xdim, ydim, pixel_quant, pixel_shift):
    """The image the reader shows, as an (ydim, xdim, 3) uint8 array: each
    channel (red, green, blue, at the quantized size) resized back to xdim x
    ydim, then shifted back up and clamped to 0-255."""
    c = [rm.resize(ch, qxdim, xdim, ydim) if pixel_quant != 0 else np.asarray(ch, dtype=np.int64) for ch in channel]
    return np.stack([clamp(np.asarray(v, dtype=np.int64) << pixel_shift).reshape(ydim, xdim) for v in c], axis=2).astype(np.uint8)
