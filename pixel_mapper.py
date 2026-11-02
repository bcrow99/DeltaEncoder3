"""
pixel_mapper.py -- PixelMapper.java (version 1.4): codes the pixels of the
image's channels with context-adaptive arithmetic coding, for pixel_writer.py
and pixel_reader.py.

The image is divided into square blocks (smaller at the right and bottom
edges), coded in rows of blocks, and the pixels of each block row by row.
Each channel is one arithmetic-coded stream whose models restart at the start
of every block, so a block is coded with what was learned inside it only.
Neighbours, though, are taken across block edges: they are already decoded
when the block is reached.

The channels are coded green first, then red and blue or their differences
from green (CHANNEL_SET_NAMES). Every coding starts from a prediction
(PREDICTION_NAMES): the median edge predictor (MED) of the left, upper and
upper-left neighbours, or least squares (delta_mapper's, the wide clamp, as
Simple codes with). With MED and red, green and blue, the prediction is
corrected by the average of how far the earlier channels at the same pixel
were from their own MED predictions. "Activity" is how busy the
neighbourhood is: |left - upper-left| + |upper - upper-left| on the 0-255
scale.

With MED the blocks are coded one after another, each row by row. Least
squares fits from pixels above and to the right of the pixel, which in that
order may be in the next block, not yet decoded; so with least squares the
pixels are coded in raster order instead, each block keeping its own models
while the rows go through it -- the same models, in the same order within
each block, so the same probabilities.

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
import delta_mapper as dm
import image_mapper as im
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

CHANNEL_NAMES = ["red", "green", "blue"]

# Channels coded (PixelMapper.java has the measurements):
#   PRIMARY:     green, red, blue, with (MED) red's and blue's predictions
#                corrected by the earlier channels' errors.
#   DIFFERENCE:  green, red - green, blue - green, each difference plus
#                max_value so it is never negative (format versions 3 and 4;
#                GREEN_DIFFERENCES replaces it in the writer).
#   GREEN_DIFFERENCES: green, blue - green, red - green (Simple 2's set), each
#                difference less its minimum, so a channel's values run from 0
#                to the difference's range only. Not corrected.
PRIMARY, DIFFERENCE, GREEN_DIFFERENCES = 0, 1, 2
CHANNEL_SET_NAMES = ["Red, green, blue (corrected)", "Green, red - green, blue - green (plus the maximum)", "Green, blue - green, red - green"]
WRITER_CHANNEL_SETS = [GREEN_DIFFERENCES, PRIMARY]   # the sets the writer offers, in its menu's order

# Each set's coding order, as indexes into [red, green, blue]; after the
# first, each is coded less green in the difference sets.
ORDER = [[1, 0, 2], [1, 0, 2], [1, 2, 0]]

# The prediction: MED, or least squares with the window radius LS_RADIUS
# (stored in the file).
MED, LS = 0, 1
PREDICTION_NAMES = ["Median edge (MED)", "Least squares"]
LS_RADIUS = 8

# The block sizes find_best_block tries: powers of 2, then (if the largest is
# the best) doubling up to the whole frame, then the steps halfway (by ratio)
# around the best of those.
SEARCH_SIZES = [8, 16, 32, 64, 128, 256]


def block_limit(xdim, ydim):
    """The largest block size worth having for an xdim x ydim image (the
    Block Size slider's maximum): one block covering the whole frame, since
    any larger size codes the same, but never less than BLOCK_MAX."""
    return max(BLOCK_MAX, xdim, ydim)


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
def _context(v, i, error, x, y, xdim, max_value, shift, coding, correct, limit, cross_limit, given):
    """(context, prediction) for pixel (x, y) of channel i (coding order): the
    prediction given, or (given < 0) MED."""
    k = y * xdim + x
    a, b, c = _neighbours(v, k, x, y, xdim, max_value)
    p = given if given >= 0 else med(a, b, c)
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
def _prediction_error(v, pred, has_pred, xdim, y0, y1, max_value, e):
    """Rows y0..y1-1 of each pixel's value minus its prediction (pred's where
    it has one, least squares; MED elsewhere) into e."""
    for y in range(y0, y1):
        for x in range(xdim):
            k = y * xdim + x
            if has_pred and pred[k] >= 0:
                e[k] = v[k] - pred[k]
                continue
            a, b, c = _neighbours(v, k, x, y, xdim, max_value)
            e[k] = v[k] - med(a, b, c)


def prediction_error(v, xdim, ydim, max_value, pred=None):
    """Each pixel's value minus its prediction: pred's where it has one, MED
    elsewhere."""
    v = np.asarray(v, dtype=np.int64)
    e = np.zeros(v.shape[0], dtype=np.int64)
    has = pred is not None
    _prediction_error(v, np.asarray(pred, dtype=np.int64) if has else np.zeros(1, dtype=np.int64), has, xdim, 0, ydim, max_value, e)
    return e


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
                    ctx, p = _context(v, i, error, x, y, xdim, max_value, shift, coding, correct, limit, cross_limit, -1)
                    if gen[ctx] != generation:
                        _reset(f, bit, total, ctx, n)
                        gen[ctx] = generation
                    value = v[y * xdim + x]
                    s = (value - p) % n if coding == 2 else value
                    out = am._encode(st, out, _start(bit, ctx, s), f[ctx, s], total[ctx])
                    _update(f, bit, total, ctx, n, s, 32, 1 << 13)
    return am._finish(st, out)


@njit
def _decode_channel(data, i, error, v, xdim, ydim, max_value, shift, block, coding, correct, limit, cross_limit, n_ctx):
    """Decodes _encode_channel's coding into v, and (the first two channels)
    their MED errors into error[i], a row of blocks at a time."""
    n = max_value + 1
    f = np.zeros((n_ctx, n), dtype=np.int64)
    bit = np.zeros((n_ctx, n + 1), dtype=np.int64)
    total = np.zeros(n_ctx, dtype=np.int64)
    gen = np.full(n_ctx, -1, dtype=np.int64)
    generation = 0
    top = _highest_one_bit(n)
    dt = am._dec_init(data)
    none = np.zeros(1, dtype=np.int64)
    for by in range(0, ydim, block):
        y1 = min(ydim, by + block)
        for bx in range(0, xdim, block):
            generation += 1
            x1 = min(xdim, bx + block)
            for y in range(by, y1):
                for x in range(bx, x1):
                    ctx, p = _context(v, i, error, x, y, xdim, max_value, shift, coding, correct, limit, cross_limit, -1)
                    if gen[ctx] != generation:
                        _reset(f, bit, total, ctx, n)
                        gen[ctx] = generation
                    s = _find(bit, ctx, n, top, am._target(dt, total[ctx]))
                    am._decode(dt, data, _start(bit, ctx, s), f[ctx, s])
                    v[y * xdim + x] = (p + s) % n if coding == 2 else s
                    _update(f, bit, total, ctx, n, s, 32, 1 << 13)
        if i < 2:
            _prediction_error(v, none, False, xdim, by, y1, max_value, error[i])


@njit
def _pool_slot(slot, key, used, f, bit, total, n):
    """The pooled model of key (a block column's context in this row of
    blocks), reset when first used."""
    q = slot[key]
    if q < 0:
        q = used[0]
        used[0] += 1
        slot[key] = q
        _reset(f, bit, total, q, n)
    return q


@njit
def _encode_raster(v, i, error, pred, xdim, ydim, max_value, shift, block, coding, limit, cross_limit, n_ctx, cap):
    """Least squares: raster order, each block keeping its own models while
    the rows go through it (new ones at the top of each row of blocks; here
    from a pool of cap, one per block column and context used)."""
    n = max_value + 1
    bcols = (xdim + block - 1) // block
    f = np.zeros((cap, n), dtype=np.int32)
    bit = np.zeros((cap, n + 1), dtype=np.int32)
    total = np.zeros(cap, dtype=np.int32)
    slot = np.full(bcols * n_ctx, -1, dtype=np.int64)
    used = np.zeros(1, dtype=np.int64)
    st, out = am._enc_init(v.shape[0] // 2)
    for y in range(ydim):
        if y % block == 0:
            slot[:] = -1
            used[0] = 0
        for x in range(xdim):
            k = y * xdim + x
            ctx, p = _context(v, i, error, x, y, xdim, max_value, shift, coding, False, limit, cross_limit, pred[k])
            q = _pool_slot(slot, (x // block) * n_ctx + ctx, used, f, bit, total, n)
            s = (v[k] - p) % n if coding == 2 else v[k]
            out = am._encode(st, out, _start(bit, q, s), f[q, s], total[q])
            _update(f, bit, total, q, n, s, 32, 1 << 13)
    return am._finish(st, out)


@njit
def _decode_raster(data, i, error, v, ref, has_ref, xdim, ydim, max_value, shift, block, coding, limit, cross_limit, n_ctx, cap, radius,
                   row, col, win, ring, z, a, b, coef, dg):
    """Decodes _encode_raster's coding into v, fitting least squares as it
    goes (ref: the channel before, decoded), and (the first two channels)
    its errors into error[i]."""
    n = max_value + 1
    bcols = (xdim + block - 1) // block
    f = np.zeros((cap, n), dtype=np.int32)
    bit = np.zeros((cap, n + 1), dtype=np.int32)
    total = np.zeros(cap, dtype=np.int32)
    slot = np.full(bcols * n_ctx, -1, dtype=np.int64)
    used = np.zeros(1, dtype=np.int64)
    top = _highest_one_bit(n)
    dt = am._dec_init(data)
    for y in range(ydim):
        if y % block == 0:
            slot[:] = -1
            used[0] = 0
        for x in range(xdim):
            k = y * xdim + x
            given = -1
            if y > 0 and 0 < x < xdim - 1:
                w = dm._ls_next(v, ref, has_ref, xdim, k, row, col, win, ring, z, a, b, coef, dg, True, radius, 1)[1]
                given = 0 if w < 0 else (max_value if w > max_value else w)
            ctx, p = _context(v, i, error, x, y, xdim, max_value, shift, coding, False, limit, cross_limit, given)
            q = _pool_slot(slot, (x // block) * n_ctx + ctx, used, f, bit, total, n)
            s = _find(bit, q, n, top, am._target(dt, total[q]))
            am._decode(dt, data, _start(bit, q, s), f[q, s])
            v[k] = (p + s) % n if coding == 2 else s
            _update(f, bit, total, q, n, s, 32, 1 << 13)
            if i < 2:
                error[i, k] = v[k] - p


def _pool_size(xdim, ydim, block, n_ctx):
    """The most models a row of blocks can use: one per block column and
    context, and no more than its pixels."""
    return max(1, min(((xdim + block - 1) // block) * n_ctx, xdim * min(block, ydim)))


def channel_max(max_value, channels):
    """The largest value of each coded channel of format versions 1 to 4: a
    difference spans 0..2 * max_value."""
    if channels == DIFFERENCE:
        return [max_value, 2 * max_value, 2 * max_value]
    return [max_value, max_value, max_value]


def _map(n, body, parallel):
    if parallel:
        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(body, range(n)))
    return [body(i) for i in range(n)]


class Prepared:
    """The three channels as they are coded, and what doesn't depend on the
    block size or the coding (PixelMapper.Prepared): v, the coded channels in
    coding order, each 0..max[i]; offset, what was subtracted from each (a
    difference's minimum, or minus max_value for DIFFERENCE; 0 for the
    others); pred, with least squares, each channel's prediction at each pixel
    (-1 at the edges, where MED predicts), else None; error, the first two
    channels' prediction errors."""

    def __init__(self, xdim, ydim, channels, prediction, v, offset, mx, pred):
        self.xdim, self.ydim, self.channels, self.prediction, self.radius = xdim, ydim, channels, prediction, LS_RADIUS
        self.v, self.offset, self.max, self.pred = v, offset, mx, pred
        self.error = np.zeros((3, xdim * ydim), dtype=np.int64)


def _least_squares_prediction(v, ref, xdim, ydim, mx):
    """A channel's least-squares predictions (LS wide, clamped to 0..mx), -1
    at the edges (PixelMapper.leastSquaresPrediction)."""
    p = np.full(v.shape[0], -1, dtype=np.int64)
    if xdim < 3 or ydim < 2:
        return p
    has_ref = ref is not None
    wide = dm._ls_plane(v, ref if has_ref else np.zeros(1, dtype=np.int64), has_ref, xdim, ydim, LS_RADIUS)[1]
    inner = np.clip(np.asarray(wide, dtype=np.int64).reshape(ydim, xdim)[1:, 1:-1], 0, mx)
    p.reshape(ydim, xdim)[1:, 1:-1] = inner
    return p


def prepare(channel, xdim, ydim, max_value, channels, prediction):
    """The channels (values 0..max_value, red, green, blue) as the channel set
    codes them, with the least-squares predictions if asked for
    (PixelMapper.prepare)."""
    order = ORDER[channels]
    v, offset, mx = [None] * 3, [0] * 3, [0] * 3
    green = np.asarray(channel[order[0]], dtype=np.int64)
    for i in range(3):
        c = np.asarray(channel[order[i]], dtype=np.int64)
        if i == 0 or channels == PRIMARY:
            v[i], mx[i] = c, max_value
            continue
        d = c - green
        lo, hi = -max_value, max_value
        if channels == GREEN_DIFFERENCES:
            lo, hi = (int(d.min()), int(d.max())) if d.shape[0] else (0, 0)
        v[i], offset[i], mx[i] = d - lo, lo, hi - lo
    pred = None
    if prediction == LS:
        pred = _map(3, lambda i: _least_squares_prediction(v[i], v[i - 1] if i > 0 else None, xdim, ydim, mx[i]), True)
    pr = Prepared(xdim, ydim, channels, prediction, v, offset, mx, pred)
    for i in range(2):
        _prediction_error(v[i], pred[i] if pred is not None else np.zeros(1, dtype=np.int64), pred is not None, xdim, 0, ydim, mx[i], pr.error[i])
    return pr


def encode(pr, shift, block, coding, parallel=True):
    """Codes the prepared channels; returns the coded bytes (uint8 arrays)
    per channel, in coding order. The channels are coded in parallel: the
    encoder has all of them."""
    correct = pr.channels == PRIMARY and pr.prediction == MED
    limit = ACTIVITY_LIMIT[coding]

    def one(i):
        n_ctx = _context_count(pr.max[i], shift, coding)
        if pr.pred is None:
            return _encode_channel(pr.v[i], i, pr.error, pr.xdim, pr.ydim, pr.max[i], shift, block, coding, correct, limit, CROSS_LIMIT, n_ctx)
        return _encode_raster(pr.v[i], i, pr.error, pr.pred[i], pr.xdim, pr.ydim, pr.max[i], shift, block, coding, limit, CROSS_LIMIT, n_ctx,
                              _pool_size(pr.xdim, pr.ydim, block, n_ctx))
    return _map(3, one, parallel)


def decode_v4(coded, xdim, ydim, max_value, shift, block, coding, channels):
    """Decodes a file of format version 4 or before: MED, and the channel
    set's offsets and largest values (channel_max)."""
    offset = [0, -max_value, -max_value] if channels == DIFFERENCE else [0, 0, 0]
    return decode(coded, xdim, ydim, shift, block, coding, channels, MED, LS_RADIUS, offset, channel_max(max_value, channels))


def decode(coded, xdim, ydim, shift, block, coding, channels, prediction, radius, offset, mx):
    """Decodes what encode() made; returns red, green, blue. The channels are
    decoded in order, since each one's contexts (and least-squares fit) use
    the ones before it; when they don't (MED, the difference sets and a value
    coding), at the same time (the Java decodes them at the same time either
    way, each a row behind the one before)."""
    n = xdim * ydim
    ordered = np.zeros((3, n), dtype=np.int64)
    error = np.zeros((3, n), dtype=np.int64)
    limit = ACTIVITY_LIMIT[coding]
    correct = channels == PRIMARY and prediction == MED

    def one(i):
        data = np.frombuffer(bytes(coded[i]), dtype=np.uint8)
        n_ctx = _context_count(mx[i], shift, coding)
        if prediction == MED:
            _decode_channel(data, i, error, ordered[i], xdim, ydim, mx[i], shift, block, coding, correct, limit, CROSS_LIMIT, n_ctx)
        else:
            has_ref = i > 0
            ref = ordered[i - 1] if has_ref else np.zeros(1, dtype=np.int64)
            _decode_raster(data, i, error, ordered[i], ref, has_ref, xdim, ydim, mx[i], shift, block, coding, limit, CROSS_LIMIT, n_ctx,
                           _pool_size(xdim, ydim, block, n_ctx), radius, *dm._ls_state(xdim, has_ref, radius))
    if prediction == MED and channels != PRIMARY and coding != ERRORS:
        _map(3, one, True)
    else:
        for i in range(3):
            one(i)
    order = ORDER[channels]
    channel = [None] * 3
    for i in range(3):
        c = ordered[i] + offset[i]
        if i > 0 and channels != PRIMARY:
            c = c + ordered[0]
        channel[order[i]] = c
    return channel


# =============================================================================
# Choosing the block size
# =============================================================================

def _java_round(x):
    return int(math.floor(x + 0.5))


def find_best_block(pr, shift, coding, tried=None):
    """The block size (BLOCK_MIN..block_limit) that codes the prepared
    channels smallest: every size in SEARCH_SIZES; while the largest tried is the best
    and smaller than the frame, twice that (the last one the whole frame: the
    larger of xdim and ydim); then the sizes halfway (by ratio) to each
    neighbour of the best (ties go to the smaller size). tried, if a list,
    gets (size, coded bytes of the 3 channels) for every size tried,
    smallest size first."""
    xdim, ydim = pr.xdim, pr.ydim
    sizes = {}
    _measure(SEARCH_SIZES, pr, shift, coding, sizes)
    best = _best_of(sizes)
    frame, last = max(xdim, ydim), SEARCH_SIZES[-1]
    while best == last and last < frame:
        last = min(2 * last, frame)
        _measure([last], pr, shift, coding, sizes)
        best = _best_of(sizes)
    more = []
    lower = _java_round(best / math.sqrt(2))
    upper = min(_java_round(best * math.sqrt(2)), block_limit(xdim, ydim))
    if lower >= BLOCK_MIN and lower not in sizes:
        more.append(lower)
    if upper > best and upper not in sizes:
        more.append(upper)
    _measure(more, pr, shift, coding, sizes)
    if tried is not None:
        tried.extend(sorted(sizes.items()))
    return _best_of(sizes)


def _measure(block_sizes, pr, shift, coding, sizes):
    if not block_sizes:
        return

    def one(s):
        return sum(len(c) for c in encode(pr, shift, block_sizes[s], coding, parallel=False))
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


def get_rgb(channel, qxdim, xdim, ydim, pixel_quant, pixel_shift, centred):
    """The image the reader shows, as an (ydim, xdim, 3) uint8 array: each
    channel (red, green, blue, at the quantized size) resized back to xdim x
    ydim (centred, resize_mapper.resize2, from format version 5;
    resize_mapper.resize before), then shifted back up and clamped to 0-255."""
    def back(ch):
        if pixel_quant == 0:
            return np.asarray(ch, dtype=np.int64)
        return rm.resize2(ch, qxdim, xdim, ydim) if centred else rm.resize(ch, qxdim, xdim, ydim)
    c = [back(ch) for ch in channel]
    return np.stack([clamp(np.asarray(v, dtype=np.int64) << pixel_shift).reshape(ydim, xdim) for v in c], axis=2).astype(np.uint8)


# =============================================================================
# Averaging pyramid (the Quantization menu's Average; PixelMapper.java)
# =============================================================================
#
# Each level halves the width and height: every 2 x 2 square becomes its
# rounded average (image_mapper.shrink_avg_flat), and only the smallest level
# is coded. To expand, each level is predicted from the one below it by a
# gradient fit (expand_gradient_flat, or expand_gradient_saddle_flat with the
# cross term), then corrected with one sign bit per pixel: whether the pixel
# was >= its square's average (build_geq_bits_flat /
# refine_with_sign_bits_flat). Applied to the red, green and blue channels
# themselves (values 0..max_value), never to difference channels. The image
# is first padded (edge pixels repeated) to a multiple of 2^levels.

PYRAMID_MAX = 2


def pyramid_size(xdim, ydim, levels):
    """Size of the top level, the size the pixels are coded at."""
    if levels == 0:
        return xdim, ydim
    mult = 1 << levels
    return im.pad_to(xdim, mult) >> levels, im.pad_to(ydim, mult) >> levels


def shrink_pyramid(c, xdim, ydim, levels):
    """Shrinks c (xdim x ydim) levels times. Returns (the top level, the sign
    bits: one array per level, level lvl's bits against level lvl + 1)."""
    mult = 1 << levels
    level_xdim = im.pad_to(xdim, mult)
    level = im.pad_edge_replicate_flat(np.asarray(c, dtype=np.int64), xdim, ydim, level_xdim, im.pad_to(ydim, mult))
    bits = []
    for _ in range(levels):
        nxt = im.shrink_avg_flat(level, level_xdim)
        bits.append(np.asarray(im.build_geq_bits_flat(level, nxt, level_xdim), dtype=np.bool_))
        level = nxt
        level_xdim //= 2
    return np.asarray(level, dtype=np.int64), bits


def expand_pyramid(top, xdim, ydim, sign_bits, max_value, saddle):
    """Inverse of shrink_pyramid, as close as the sign bits allow: expands the
    top level back to xdim x ydim, correcting each level with its sign bits."""
    levels = len(sign_bits)
    mult = 1 << levels
    level = np.asarray(top, dtype=np.int64)
    level_xdim = im.pad_to(xdim, mult) >> levels
    for lvl in range(levels - 1, -1, -1):
        if saddle:
            predicted = im.expand_gradient_saddle_flat(level, level_xdim, max_value)
        else:
            predicted = im.expand_gradient_flat(level, level_xdim, max_value)
        level = im.refine_with_sign_bits_flat(level, predicted, sign_bits[lvl], level_xdim * 2, max_value)
        level_xdim *= 2
    return np.asarray(im.crop_flat(level, im.pad_to(xdim, mult), im.pad_to(ydim, mult), xdim, ydim), dtype=np.int64)


def write_sign_bits(out, sign_bits):
    """Sign bits, one bitmap per level: int length, then (length + 7) // 8
    bytes, bit q in byte q >> 3 at bit q & 7."""
    for bits in sign_bits:
        bits = np.asarray(bits, dtype=np.bool_)
        out.write_int(len(bits))
        out.write(np.packbits(bits, bitorder="little").tobytes())


def read_sign_bits(inp, levels):
    sign_bits = []
    for _ in range(levels):
        n = inp.read_int()
        packed = np.frombuffer(inp.read_fully((n + 7) // 8), dtype=np.uint8)
        sign_bits.append(np.unpackbits(packed, bitorder="little")[:n].astype(np.bool_))
    return sign_bits


def sign_bit_bytes(sign_bits):
    """Bytes the sign bits take in the file."""
    return sum(4 + (len(b) + 7) // 8 for b in sign_bits)
