"""
delta_mapper.py -- the parts of DeltaMapper.java the Delta programs use:
quantizing, the six candidate channels and ten channel sets, the fourteen
delta types (0-13) and their inverses, maps and tables on disk, context
coding of deltas, and the block-map settings search.

Channels are flat, row-major numpy int64 arrays (index = y * xdim + x).
Maps are numpy uint8 arrays. Byte-for-byte compatible with the Java; the
per-pixel loops are Numba-compiled. Java's truncating division is jdiv().
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from numba_support import njit, jdiv
import arithmetic_mapper as am
import string_mapper as sm
from code_mapper import _shannon_limit
from java_io import DataOutput, DataInput

SET_NAMES = [
    "blue, green, red", "blue, red, red-green", "blue, red, blue-green",
    "blue, blue-green, red-green", "blue, blue-green, red-blue",
    "green, red, blue-green", "red, blue-green, red-green",
    "green, blue-green, red-green", "green, red-green, red-blue",
    "red, red-green, red-blue"]

DELTA_TYPE_NAMES = [
    "horizontal", "vertical", "average", "med", "directional", "adaptive",
    "scanline (1)", "scanline (2)", "scanline (3)", "scanline (4)", "scanline (5)",
    "frame map (1)", "frame map (2)", "block map"]

DELTA_TYPES = len(DELTA_TYPE_NAMES)
MIN_DIM = 4

_CHANNELS = [(0, 1, 2), (0, 2, 4), (0, 2, 3), (0, 3, 4), (0, 3, 5),
             (1, 2, 3), (2, 3, 4), (1, 3, 4), (1, 4, 5), (2, 4, 5)]


def get_channels(set_id):
    return list(_CHANNELS[set_id])


def has_map(delta_type):
    return delta_type >= 6


# =============================================================================
# Quantizing and channels
# =============================================================================

def get_quantized_size(xdim, ydim, pixel_quant):
    """Size after Pixel Resolution (0-10) resizing."""
    if pixel_quant == 0 or xdim < MIN_DIM or ydim < MIN_DIM:
        return xdim, ydim
    factor = pixel_quant / 10.0
    return xdim - int(factor * (xdim // 2 - 2)), ydim - int(factor * (ydim // 2 - 2))


def quantize_channel(channel, pixel_shift):
    """Right shift by pixel_shift, rounding to nearest (clamped so the result
    never reconstructs past 255). Returns a new array."""
    channel = np.asarray(channel, dtype=np.int64)
    if pixel_shift == 0:
        return channel.copy()
    return np.minimum(channel + (1 << (pixel_shift - 1)), 255) >> pixel_shift


def shift(channel, amount):
    channel = np.asarray(channel, dtype=np.int64)
    return channel >> -amount if amount < 0 else channel << amount


def get_candidate_channels(blue, green, red):
    """The six candidate channels (blue, green, red, blue-green, red-green,
    red-blue), each difference shifted by its minimum to start at 0; returns
    (channels, minimums)."""
    c = [np.asarray(blue, dtype=np.int64), np.asarray(green, dtype=np.int64), np.asarray(red, dtype=np.int64)]
    c += [c[0] - c[1], c[2] - c[1], c[2] - c[0]]
    mins = [int(x.min()) for x in c]
    for i in range(3, 6):
        c[i] = c[i] - mins[i]
    return c, mins


def get_blue_green_red(set_id, c0, c1, c2):
    """Rebuilds the three colour channels from a set's channels (with the
    difference channels' minimums already added back)."""
    if set_id == 0:   b, g, r = c0, c1, c2
    elif set_id == 1: b, r = c0, c1; g = r - c2
    elif set_id == 2: b, r = c0, c1; g = b - c2
    elif set_id == 3: b = c0; g = b - c1; r = c2 + g
    elif set_id == 4: b = c0; g = b - c1; r = b + c2
    elif set_id == 5: g, r = c0, c1; b = c2 + g
    elif set_id == 6: r = c0; g = r - c2; b = c1 + g
    elif set_id == 7: g = c0; b = g + c1; r = g + c2
    elif set_id == 8: g = c0; r = g + c1; b = r - c2
    elif set_id == 9: r = c0; g = r - c1; b = r - c2
    else: raise ValueError("set_id %d" % set_id)
    return [b, g, r]


@njit
def _ideal_frequency2(src, xdim, ydim):
    lo = src.min()
    hi = src.max()
    span = hi - lo
    count = np.zeros(2 * span + 1, dtype=np.int64)
    for i in range(1, ydim):
        for j in range(1, xdim - 1):
            k = i * xdim + j
            a = src[k - 1]; b = src[k - xdim]; c = src[k - xdim - 1]; d = src[k - xdim + 1]; e = src[k]
            da = abs(a - e); db = abs(b - e); dc = abs(c - e); dd = abs(d - e)
            if da <= db and da <= dc and da <= dd:
                v = a - e
            elif db <= dc and db <= dd:
                v = b - e
            elif dc <= dd:
                v = c - e
            else:
                v = d - e
            count[v + span] += 1
    first = 0
    while count[first] == 0:
        first += 1
    last = count.shape[0] - 1
    while count[last] == 0:
        last -= 1
    return count[first:last + 1].copy()


def get_ideal_frequency2(src, xdim, ydim):
    if xdim < 3 or ydim < 2:
        return np.zeros(2, dtype=np.int64)
    return _ideal_frequency2(np.asarray(src, dtype=np.int64), xdim, ydim)


# =============================================================================
# Smoothing (Quantization menu)
# =============================================================================

@njit
def _bilateral(src, xdim, ydim, threshold):
    sigma_r = float(threshold * threshold)
    rw = np.zeros(256)
    r2 = 2.0 * sigma_r * sigma_r
    for d in range(256):
        rw[d] = math.exp(-(d * d) / r2)
    sw = np.zeros((5, 5))
    s2 = 2.0 * 1.5 * 1.5
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            sw[dy + 2, dx + 2] = math.exp(-(dx * dx + dy * dy) / s2)
    dst = np.zeros(src.shape[0], dtype=np.int64)
    for row in range(ydim):
        for col in range(xdim):
            center = src[row * xdim + col]
            sum_w = 0.0
            sum_v = 0.0
            for dy in range(-2, 3):
                ny = row + dy
                if ny < 0 or ny >= ydim:
                    continue
                for dx in range(-2, 3):
                    nx = col + dx
                    if nx < 0 or nx >= xdim:
                        continue
                    v = src[ny * xdim + nx]
                    w = sw[dy + 2, dx + 2] * rw[abs(v - center)]
                    sum_w += w
                    sum_v += w * v
            dst[row * xdim + col] = int(math.floor(sum_v / sum_w + 0.5))   # Math.round
    return dst


def bilateral_smooth(src, xdim, ydim, threshold):
    src = np.asarray(src, dtype=np.int64)
    return src.copy() if threshold == 0 else _bilateral(src, xdim, ydim, threshold)


@njit
def _anisotropic(src, xdim, ydim, threshold):
    K2 = (threshold * 3.0 + 5.0) * (threshold * 3.0 + 5.0)
    c = np.zeros(511)
    for d in range(-255, 256):
        c[d + 255] = math.exp(-(d * d) / K2)
    img = src.astype(np.float64)
    nxt = np.zeros(src.shape[0])
    for _ in range(threshold):
        for row in range(ydim):
            for col in range(xdim):
                k = row * xdim + col
                v = img[k]
                dN = img[k - xdim] - v if row > 0 else 0.0
                dS = img[k + xdim] - v if row < ydim - 1 else 0.0
                dE = img[k + 1] - v if col < xdim - 1 else 0.0
                dW = img[k - 1] - v if col > 0 else 0.0
                iN = max(0, min(510, int(dN + 255.5)))
                iS = max(0, min(510, int(dS + 255.5)))
                iE = max(0, min(510, int(dE + 255.5)))
                iW = max(0, min(510, int(dW + 255.5)))
                nxt[k] = v + 0.25 * (c[iN] * dN + c[iS] * dS + c[iE] * dE + c[iW] * dW)
        img, nxt = nxt, img
    dst = np.zeros(src.shape[0], dtype=np.int64)
    for i in range(src.shape[0]):
        dst[i] = max(0, min(255, int(math.floor(img[i] + 0.5))))
    return dst


def anisotropic_smooth(src, xdim, ydim, threshold):
    src = np.asarray(src, dtype=np.int64)
    return src.copy() if threshold == 0 else _anisotropic(src, xdim, ydim, threshold)


# =============================================================================
# Predictors
# =============================================================================

@njit
def _med(a, b, c):
    if c >= max(a, b):
        return min(a, b)
    if c <= min(a, b):
        return max(a, b)
    return a + b - c


@njit
def _directional(a, b, c, d):
    h = abs(b - c) + abs(b - d)
    v = abs(a - c) + abs(a - b)
    dl = abs(c - d)
    dr = abs(a - d)
    if h >= v and h >= dl and h >= dr:
        return b
    if v >= dl and v >= dr:
        return a
    if dl >= dr:
        return d
    return c


@njit
def _adaptive_pred(a, b, c, d):
    pa = abs(b - c)
    pb = abs(a - c)
    if pb > pa * 2:
        return a
    if pa > pb * 2:
        return b
    return _med(a, b, c)


@njit
def _pred16(a, b, c, d, p):
    if p == 0: return a
    if p == 1: return b
    if p == 2: return c
    if p == 3: return d
    if p == 4: return (a + b) >> 1
    if p == 5: return (b + c) >> 1
    if p == 6: return (a + c) >> 1
    if p == 7: return (b + d) >> 1
    if p == 8: return (c + d) >> 1
    if p == 9: return (a + b + c + d + 2) >> 2
    if p == 10: return a + b - c
    if p == 11: return _med(a, b, c)
    if p == 12: return (a * 3 + b + 2) >> 2
    if p == 13: return (a + b * 3 + 2) >> 2
    if p == 14: return (a * 3 + d + 2) >> 2
    if p == 15: return (b * 3 + a + 2) >> 2
    return a


FILTER_SETS_8 = np.array([[0, 1, 2, 3, 4, 10, 9, 5],
                          [0, 1, 4, 9, 10, 11, 12, 13],
                          [0, 1, 4, 9, 10, 11, 12, 7]], dtype=np.int64)


@njit
def _block_predictor(p, a, b, c, d):
    if p < 16: return _pred16(a, b, c, d, p)
    if p == 16: return (a + d) >> 1
    if p == 17: return a + ((b - c) >> 1)
    if p == 18: return b + ((a - c) >> 1)
    if p == 19: return a + d - b
    if p == 20: return jdiv(a + b + d + 1, 3)
    return b + ((d - c) >> 1)


@njit
def _set_predictor(entry, a, b, c, d):
    if entry < 32:
        return _block_predictor(entry, a, b, c, d)
    return (_block_predictor(entry // 32 - 1, a, b, c, d) + _block_predictor(entry % 32, a, b, c, d) + 1) >> 1


# =============================================================================
# Delta types 0-5 (no map)
# =============================================================================

@njit
def _horizontal(src, xdim, ydim):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    init = src[0]
    value = init
    k = 0
    for i in range(ydim):
        if i == 0:
            dst[k] = 0
            k += 1
        else:
            delta = src[k] - init
            dst[k] = delta
            k += 1
            init += delta
            value = init
        for j in range(1, xdim):
            delta = src[k] - value
            value += delta
            dst[k] = delta
            k += 1
    return dst


@njit
def _from_horizontal(src, xdim, ydim, init):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    k = 0
    value = init
    for i in range(ydim):
        if i != 0:
            value += src[k]
        cur = value
        dst[k] = cur
        k += 1
        for j in range(1, xdim):
            cur += src[k]
            dst[k] = cur
            k += 1
    return dst


@njit
def _vertical(src, xdim, ydim):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    value = src[0]
    for x in range(1, xdim):
        dst[x] = src[x] - value
        value = src[x]
    for k in range(xdim, xdim * ydim):
        dst[k] = src[k] - src[k - xdim]
    return dst


@njit
def _from_vertical(src, xdim, ydim, init):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    dst[0] = init
    value = init
    for i in range(1, xdim):
        value += src[i]
        dst[i] = value
    for k in range(xdim, xdim * ydim):
        dst[k] = dst[k - xdim] + src[k]
    return dst


@njit
def _average(src, xdim, ydim):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    for k in range(1, xdim):
        dst[k] = src[k] - src[k - 1]
    for i in range(1, ydim):
        k = i * xdim
        dst[k] = src[k] - src[k - xdim]
        for j in range(1, xdim):
            k = i * xdim + j
            dst[k] = src[k] - jdiv(src[k - 1] + src[k - xdim], 2)
    return dst


@njit
def _from_average(src, xdim, ydim, init):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    dst[0] = init
    for k in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
    for i in range(1, ydim):
        k = i * xdim
        dst[k] = dst[k - xdim] + src[k]
        for j in range(1, xdim):
            k = i * xdim + j
            dst[k] = jdiv(dst[k - 1] + dst[k - xdim], 2) + src[k]
    return dst


@njit
def _neighbour_deltas(src, xdim, ydim, kind):
    """MED (kind 3), directional (4) and adaptive (5): row 0 horizontal,
    column 0 vertical; the last column is MED (4), horizontal (5), or the
    predictor itself (3)."""
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    for k in range(1, xdim):
        dst[k] = src[k] - src[k - 1]
    for i in range(1, ydim):
        k = i * xdim
        dst[k] = src[k] - src[k - xdim]
        last = xdim if kind == 3 else xdim - 1
        for j in range(1, last):
            k = i * xdim + j
            a = src[k - 1]; b = src[k - xdim]; c = src[k - xdim - 1]
            if kind == 3:
                p = _med(a, b, c)
            elif kind == 4:
                p = _directional(a, b, c, src[k - xdim + 1])
            else:
                p = _adaptive_pred(a, b, c, src[k - xdim + 1])
            dst[k] = src[k] - p
        if kind != 3:
            k = i * xdim + xdim - 1
            if kind == 4:
                dst[k] = src[k] - _med(src[k - 1], src[k - xdim], src[k - xdim - 1])
            else:
                dst[k] = src[k] - src[k - 1]
    return dst


@njit
def _from_neighbour_deltas(src, xdim, ydim, init, kind):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    dst[0] = init
    for k in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
    for i in range(1, ydim):
        k = i * xdim
        dst[k] = dst[k - xdim] + src[k]
        last = xdim if kind == 3 else xdim - 1
        for j in range(1, last):
            k = i * xdim + j
            a = dst[k - 1]; b = dst[k - xdim]; c = dst[k - xdim - 1]
            if kind == 3:
                p = _med(a, b, c)
            elif kind == 4:
                p = _directional(a, b, c, dst[k - xdim + 1])
            else:
                p = _adaptive_pred(a, b, c, dst[k - xdim + 1])
            dst[k] = p + src[k]
        if kind != 3:
            k = i * xdim + xdim - 1
            if kind == 4:
                dst[k] = _med(dst[k - 1], dst[k - xdim], dst[k - xdim - 1]) + src[k]
            else:
                dst[k] = dst[k - 1] + src[k]
    return dst


# =============================================================================
# Scanline (1), (2), (3): one of four predictors per row
# =============================================================================

@njit
def _row_limit(row):
    lo = row.min()
    hi = row.max()
    freq = np.zeros(hi - lo + 1, dtype=np.int64)
    for v in row:
        freq[v - lo] += 1
    return int(math.floor(_shannon_limit(freq)))


@njit
def _scanline_pred(src, k, xdim, m, kind):
    """Row predictor m of scanline (1) (kind 1: left, above, average, MED)
    or scanline (3) (kind 3: left, average, MED, directional)."""
    a = src[k - 1]; b = src[k - xdim]
    if kind == 1:
        if m == 0: return a
        if m == 1: return b
        if m == 2: return jdiv(a + b, 2)
        return _med(a, b, src[k - xdim - 1])
    if m == 0: return a
    if m == 1: return jdiv(a + b, 2)
    if m == 2: return _med(a, b, src[k - xdim - 1])
    return _directional(a, b, src[k - xdim - 1], src[k - xdim + 1])


@njit
def _scanline13(src, xdim, ydim, kind):
    n = xdim - 2
    mp = np.zeros(ydim - 1, dtype=np.uint8)
    row = np.zeros(n, dtype=np.int64)
    for i in range(1, ydim):
        best = 0
        best_limit = 0
        for m in range(4):
            for j in range(1, xdim - 1):
                k = i * xdim + j
                row[j - 1] = src[k] - _scanline_pred(src, k, xdim, m, kind)
            limit = _row_limit(row)
            if m == 0 or limit < best_limit:
                best_limit = limit
                best = m
        mp[i - 1] = best
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    for k in range(1, xdim):
        dst[k] = src[k] - src[k - 1]
    for i in range(1, ydim):
        k = i * xdim
        dst[k] = src[k] - src[k - xdim]
        m = mp[i - 1]
        for j in range(1, xdim - 1):
            k = i * xdim + j
            dst[k] = src[k] - _scanline_pred(src, k, xdim, m, kind)
        k = i * xdim + xdim - 1
        if kind == 3 and m == 3:
            dst[k] = src[k] - _med(src[k - 1], src[k - xdim], src[k - xdim - 1])
        else:
            dst[k] = src[k] - src[k - 1]
    return dst, mp


@njit
def _from_scanline13(src, xdim, ydim, init, mp, kind):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    dst[0] = init
    for k in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
    for i in range(1, ydim):
        k = i * xdim
        dst[k] = dst[k - xdim] + src[k]
        m = mp[i - 1]
        for j in range(1, xdim - 1):
            k = i * xdim + j
            dst[k] = _scanline_pred(dst, k, xdim, m, kind) + src[k]
        k = i * xdim + xdim - 1
        if kind == 3 and m == 3:
            dst[k] = _med(dst[k - 1], dst[k - xdim], dst[k - xdim - 1]) + src[k]
        else:
            dst[k] = dst[k - 1] + src[k]
    return dst


@njit
def _scanline2_pred(src, k, xdim, m):
    if m == 0: return src[k - 1]
    if m == 1: return src[k - xdim]
    if m == 2: return jdiv(src[k - 1] + src[k - xdim], 2)
    return jdiv(src[k - 1] + src[k - xdim + 1], 2)


@njit
def _scanline2(src, xdim, ydim):
    mp = np.zeros(ydim - 1, dtype=np.uint8)
    for i in range(1, ydim):
        s = np.zeros(4, dtype=np.int64)
        for j in range(1, xdim - 1):
            k = i * xdim + j
            for m in range(4):
                s[m] += abs(src[k] - _scanline2_pred(src, k, xdim, m))
        best = 0
        for m in range(1, 4):
            if s[m] < s[best]:
                best = m
        mp[i - 1] = best
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    for k in range(1, xdim):
        dst[k] = src[k] - src[k - 1]
    for i in range(1, ydim):
        k = i * xdim
        dst[k] = src[k] - src[k - xdim]
        for j in range(1, xdim - 1):
            k = i * xdim + j
            dst[k] = src[k] - _scanline2_pred(src, k, xdim, mp[i - 1])
        k = i * xdim + xdim - 1
        dst[k] = src[k] - src[k - 1]
    return dst, mp


@njit
def _from_scanline2(src, xdim, ydim, init, mp):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    dst[0] = init
    for k in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
    for i in range(1, ydim):
        k = i * xdim
        dst[k] = dst[k - xdim] + src[k]
        for j in range(1, xdim - 1):
            k = i * xdim + j
            dst[k] = _scanline2_pred(dst, k, xdim, mp[i - 1]) + src[k]
        k = i * xdim + xdim - 1
        dst[k] = dst[k - 1] + src[k]
    return dst


# =============================================================================
# Scanline (4) and (5): one of 16 (or 8) predictors per row
# =============================================================================

@njit
def _row_pred(src, row, col, xdim, p, variant, eight):
    k = row * xdim + col
    a = src[k - 1] if col > 0 else 0
    b = src[k - xdim] if row > 0 else 0
    c = src[k - xdim - 1] if (row > 0 and col > 0) else 0
    d = src[k - xdim + 1] if (row > 0 and col < xdim - 1) else 0
    if eight:
        p = FILTER_SETS_8[variant, p]
    return _pred16(a, b, c, d, p)


@njit
def _scanline45(src, xdim, ydim, n_pred, variant):
    eight = n_pred == 8
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    mp = np.zeros(ydim, dtype=np.uint8)
    for row in range(ydim):
        best = 0
        best_sad = 1 << 62
        for p in range(n_pred):
            sad = 0
            for col in range(xdim):
                k = row * xdim + col
                if k == 0:
                    continue
                sad += abs(src[k] - _row_pred(src, row, col, xdim, p, variant, eight))
            if sad < best_sad:
                best_sad = sad
                best = p
        mp[row] = best
        for col in range(xdim):
            k = row * xdim + col
            if k == 0:
                continue
            dst[k] = src[k] - _row_pred(src, row, col, xdim, best, variant, eight)
    return dst, mp


@njit
def _from_scanline45(src, xdim, ydim, init, mp, n_pred, variant):
    eight = n_pred == 8
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    dst[0] = init
    for row in range(ydim):
        p = mp[row] & (0x7 if eight else 0xF)
        for col in range(xdim):
            k = row * xdim + col
            if k == 0:
                continue
            dst[k] = src[k] + _row_pred(dst, row, col, xdim, p, variant, eight)
    return dst


# =============================================================================
# Frame maps: one of 8 (1) or 16 (2) predictors per interior pixel
# =============================================================================

@njit
def _ideal8_pred(src, k, xdim, n):
    a = src[k - 1]; b = src[k - xdim]; c = src[k - xdim - 1]; d = src[k - xdim + 1]
    if n == 0: return a
    if n == 1: return jdiv(a + c, 2)
    if n == 2: return c
    if n == 3: return jdiv(c + b, 2)
    if n == 4: return b
    if n == 5: return jdiv(b + d, 2)
    if n == 6: return d
    return jdiv(d + a, 2)


@njit
def _ideal16_pred(a, b, c, d, n):
    if n == 0: return a
    if n == 1: return c
    if n == 2: return b
    if n == 3: return d
    if n == 4: return (a + c) >> 1
    if n == 5: return (c + b) >> 1
    if n == 6: return (b + d) >> 1
    if n == 7: return (d + a) >> 1
    if n == 8: return (a + b) >> 1
    if n == 9: return (c + d) >> 1
    if n == 10: return (a + b + c + d) >> 2
    if n == 11: return _med(a, b, c)
    if n == 12: return (a + b + c) >> 2
    if n == 13: return (a + b + d) >> 2
    if n == 14: return (a + c + d) >> 2
    return (b + c + d) >> 2


@njit
def _frame(src, xdim, ydim, n_pred):
    mp = np.zeros((xdim - 2) * (ydim - 1), dtype=np.uint8)
    # Frame map (1) leaves its last row at 0 (left), as the Java does.
    rows = ydim - 1 if n_pred == 8 else ydim
    m = 0
    for i in range(1, rows):
        for j in range(1, xdim - 1):
            k = i * xdim + j
            best = 0
            best_abs = 1 << 62
            for n in range(n_pred):
                if n_pred == 8:
                    p = _ideal8_pred(src, k, xdim, n)
                else:
                    p = _ideal16_pred(src[k - 1], src[k - xdim], src[k - xdim - 1], src[k - xdim + 1], n)
                e = abs(src[k] - p)
                if e < best_abs:
                    best_abs = e
                    best = n
            mp[m] = best
            m += 1
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    for k in range(1, xdim):
        dst[k] = src[k] - src[k - 1]
    m = 0
    for i in range(1, ydim):
        k = i * xdim
        dst[k] = src[k] - src[k - xdim]
        for j in range(1, xdim - 1):
            k = i * xdim + j
            n = mp[m]
            m += 1
            if n_pred == 8:
                p = _ideal8_pred(src, k, xdim, n)
            else:
                p = _ideal16_pred(src[k - 1], src[k - xdim], src[k - xdim - 1], src[k - xdim + 1], n)
            dst[k] = src[k] - p
        k = i * xdim + xdim - 1
        dst[k] = src[k] - src[k - 1]
    return dst, mp


@njit
def _from_frame(src, xdim, ydim, init, mp, n_pred):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    dst[0] = init
    for k in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
    m = 0
    for i in range(1, ydim):
        k = i * xdim
        dst[k] = dst[k - xdim] + src[k]
        for j in range(1, xdim - 1):
            k = i * xdim + j
            n = mp[m]
            m += 1
            if n_pred == 8:
                p = _ideal8_pred(dst, k, xdim, n)
            else:
                p = _ideal16_pred(dst[k - 1], dst[k - xdim], dst[k - xdim - 1], dst[k - xdim + 1], n)
            dst[k] = p + src[k]
        k = i * xdim + xdim - 1
        dst[k] = dst[k - 1] + src[k]
    return dst


# =============================================================================
# Block map (delta type 13)
# =============================================================================

BLOCK_MIN, BLOCK_MAX, BLOCK_DEFAULT = 4, 32, 8
BLOCK_SET_NAMES = ["Scanline 16", "No Neighbours 16", "Neighbours 8", "Basic 4", "Blends 32"]
BLOCK_SETS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    [0, 1, 2, 3, 11, 4, 10, 9],
    [11, 10, 4, 9],
    [11, 564, 37, 275, 1, 165, 46, 267, 102, 169, 368, 7, 178, 3, 0, 176,
     66, 75, 6, 145, 50, 69, 168, 403, 307, 181, 101, 38, 80, 561, 231, 4]]
_BLOCK_SETS = [np.array(s, dtype=np.int64) for s in BLOCK_SETS]
BLOCK_SEARCH_SIZES = [4, 6, 8, 12, 16, 24, 32]


def blocks_across(xdim, block):
    return (xdim - 2 + block - 1) // block


@njit
def _edges(src, dst, xdim, ydim):
    for x in range(1, xdim):
        dst[x] = src[x] - src[x - 1]
    for y in range(1, ydim):
        dst[y * xdim] = src[y * xdim] - src[(y - 1) * xdim]
        dst[y * xdim + xdim - 1] = src[y * xdim + xdim - 1] - src[y * xdim + xdim - 2]


@njit
def _block(src, xdim, ydim, block, ids, set_id):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    bw = (xdim - 2 + block - 1) // block
    bh = (ydim - 1 + block - 1) // block
    mp = np.zeros(2 + bw * bh, dtype=np.uint8)
    mp[0] = block
    mp[1] = set_id
    P = ids.shape[0]
    _edges(src, dst, xdim, ydim)
    sums = np.zeros(P, dtype=np.int64)
    prim = np.zeros(22, dtype=np.int64)
    for by in range(bh):
        for bx in range(bw):
            y0 = 1 + by * block; y1 = min(ydim, y0 + block)
            x0 = 1 + bx * block; x1 = min(xdim - 1, x0 + block)
            sums[:] = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    k = y * xdim + x
                    a = src[k - 1]; b = src[k - xdim]; c = src[k - xdim - 1]; d = src[k - xdim + 1]
                    for q in range(22):
                        prim[q] = _block_predictor(q, a, b, c, d)
                    for p in range(P):
                        e = ids[p]
                        v = prim[e] if e < 32 else (prim[e // 32 - 1] + prim[e % 32] + 1) >> 1
                        sums[p] += abs(src[k] - v)
            best = 0
            for p in range(1, P):
                if sums[p] < sums[best]:
                    best = p
            mp[2 + by * bw + bx] = best
            for y in range(y0, y1):
                for x in range(x0, x1):
                    k = y * xdim + x
                    dst[k] = src[k] - _set_predictor(ids[best], src[k - 1], src[k - xdim], src[k - xdim - 1], src[k - xdim + 1])
    return dst, mp


@njit
def _from_block(delta, xdim, ydim, init, mp, ids):
    block = np.int64(mp[0])
    bw = (xdim - 2 + block - 1) // block
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    dst[0] = init
    for x in range(1, xdim):
        dst[x] = dst[x - 1] + delta[x]
    for y in range(1, ydim):
        k = y * xdim
        dst[k] = dst[k - xdim] + delta[k]
        row = 2 + (y - 1) // block * bw
        for x in range(1, xdim - 1):
            k += 1
            e = ids[mp[row + (x - 1) // block]]
            dst[k] = delta[k] + _set_predictor(e, dst[k - 1], dst[k - xdim], dst[k - xdim - 1], dst[k - xdim + 1])
        k += 1
        dst[k] = dst[k - 1] + delta[k]
    return dst


# =============================================================================
# Dispatch
# =============================================================================

def get_deltas(src, xdim, ydim, delta_type, variant=0, block=BLOCK_DEFAULT, block_set=0):
    """(deltas, map or None, init value) for delta_type 0-13."""
    src = np.asarray(src, dtype=np.int64)
    init = int(src[0])
    t = delta_type
    if t == 0: return _horizontal(src, xdim, ydim), None, init
    if t == 1: return _vertical(src, xdim, ydim), None, init
    if t == 2: return _average(src, xdim, ydim), None, init
    if t in (3, 4, 5): return _neighbour_deltas(src, xdim, ydim, t), None, init
    if t == 6: d, m = _scanline13(src, xdim, ydim, 1)
    elif t == 7: d, m = _scanline2(src, xdim, ydim)
    elif t == 8: d, m = _scanline13(src, xdim, ydim, 3)
    elif t == 9: d, m = _scanline45(src, xdim, ydim, 16, 0)
    elif t == 10: d, m = _scanline45(src, xdim, ydim, 8, variant)
    elif t == 11: d, m = _frame(src, xdim, ydim, 8)
    elif t == 12: d, m = _frame(src, xdim, ydim, 16)
    elif t == 13: d, m = _block(src, xdim, ydim, block, _BLOCK_SETS[block_set], block_set)
    else: raise ValueError("delta_type %d" % t)
    return d, m, init


def get_values_from_deltas(delta, xdim, ydim, init, delta_type, mp=None, variant=0):
    delta = np.asarray(delta, dtype=np.int64)
    t = delta_type
    if t == 0: return _from_horizontal(delta, xdim, ydim, init)
    if t == 1: return _from_vertical(delta, xdim, ydim, init)
    if t == 2: return _from_average(delta, xdim, ydim, init)
    if t in (3, 4, 5): return _from_neighbour_deltas(delta, xdim, ydim, init, t)
    mp = np.asarray(mp, dtype=np.uint8)
    if t == 6: return _from_scanline13(delta, xdim, ydim, init, mp, 1)
    if t == 7: return _from_scanline2(delta, xdim, ydim, init, mp)
    if t == 8: return _from_scanline13(delta, xdim, ydim, init, mp, 3)
    if t == 9: return _from_scanline45(delta, xdim, ydim, init, mp, 16, 0)
    if t == 10: return _from_scanline45(delta, xdim, ydim, init, mp, 8, variant)
    if t == 11: return _from_frame(delta, xdim, ydim, init, mp, 8)
    if t == 12: return _from_frame(delta, xdim, ydim, init, mp, 16)
    if t == 13: return _from_block(delta, xdim, ydim, init, mp, _BLOCK_SETS[int(mp[1])])
    raise ValueError("delta_type %d" % t)


# =============================================================================
# Tables and maps on disk
# =============================================================================

def write_table(out, table):
    """short length, then one unsigned byte per entry (length <= 255) or one
    short per entry."""
    table = np.asarray(table)
    out.write_short(len(table))
    if len(table) <= 255:
        out.write(table.astype(np.uint8).tobytes())
    else:
        out.write(table.astype(">u2").tobytes())


def read_table(inp):
    n = inp.read_short()
    if n <= 255:
        return np.frombuffer(inp.read_fully(n), dtype=np.uint8).astype(np.int64)
    return np.frombuffer(inp.read_fully(2 * n), dtype=">i2").astype(np.int64)


def _map_width(delta_type, xdim):
    return 1 if delta_type <= 10 else xdim - 2


@njit
def _map_contexts(m, previous, has_previous, width, K):
    n = m.shape[0]
    ctx = np.zeros(n, dtype=np.int64)
    for k in range(n):
        x = k % width
        left = np.int64(m[k - 1]) if x > 0 else 0
        up = np.int64(m[k - width]) if k >= width else 0
        ul = np.int64(m[k - width - 1]) if (x > 0 and k >= width) else 0
        prev = np.int64(previous[k]) if has_previous else 0
        shape = 0 if ul == left else (1 if ul == up else 2)
        ctx[k] = ((left * K + up) * K + prev) * 3 + shape
    return ctx


@njit
def _decode_map_context(data, n, K, previous, has_previous, width, top):
    n_contexts = K * K * K * 3
    f, bit, total, dt = am.context_decoder_init(data, n_contexts, K)
    m = np.zeros(n, dtype=np.uint8)
    for k in range(n):
        x = k % width
        left = np.int64(m[k - 1]) if x > 0 else 0
        up = np.int64(m[k - width]) if k >= width else 0
        ul = np.int64(m[k - width - 1]) if (x > 0 and k >= width) else 0
        prev = np.int64(previous[k]) if has_previous else 0
        shape = 0 if ul == left else (1 if ul == up else 2)
        c = ((left * K + up) * K + prev) * 3 + shape
        m[k] = am.context_decode_one(data, dt, f, bit, total, c, K, top)
    return m


def _prev_array(previous, n):
    if previous is None:
        return np.zeros(n, dtype=np.uint8), False
    return np.asarray(previous, dtype=np.uint8), True


def _map_to_string(m):
    lo, bits, table, string = sm.get_string_list(m.astype(np.int64), False)
    bits = sm.get_bitlength(string)
    out = DataOutput()
    out.write_int(len(m)); write_table(out, table); out.write_int(lo); out.write_int(bits)
    out.write_byte(m[0]); out.write(string[:sm.get_bytelength(bits)].tobytes())
    return out.to_bytes()


def _map_to_arithmetic(m):
    freq = np.bincount(m, minlength=256)
    used = np.nonzero(freq)[0]
    out = DataOutput()
    out.write_int(len(m)); out.write_short(len(used))
    for v in used:
        out.write_byte(v); out.write_int(freq[v])
    if len(used) <= 1:
        out.write_int(0)
    else:
        coded = am.get_interval_value_fast_fenwick(m, freq)
        out.write_int(len(coded)); out.write(coded.tobytes())
    return out.to_bytes()


def _map_to_context(m, previous, width):
    K = max(1, int(m.max()) + 1 if len(m) else 1)
    if previous is not None and len(previous):
        K = max(K, int(np.max(previous)) + 1)
    prev, has = _prev_array(previous, len(m))
    ctx = _map_contexts(m, prev, has, width, K)
    coded = am.get_interval_value_context(m.astype(np.int64), K, ctx, K * K * K * 3)
    out = DataOutput()
    out.write_int(len(m)); out.write_byte(K); out.write_int(len(coded)); out.write(coded.tobytes())
    return out.to_bytes()


def write_map(out, delta_type, mp, previous, xdim):
    """A delta-type map. Types 6-8 (values 0-3): int length, int packed
    length, 4 values per byte, low bits first. Types 9-13: a flag byte and
    the smallest of three forms -- 0 unary string, 1 arithmetic, 2 context
    coded (contexts use the previous channel's map). A block map (13) starts
    with its block size and predictor set bytes."""
    mp = np.asarray(mp, dtype=np.uint8)
    if delta_type <= 8:
        packed = np.zeros((len(mp) + 3) // 4, dtype=np.uint8)
        for r in range(4):
            part = mp[r::4] & 3
            packed[:len(part)] |= (part << (2 * r)).astype(np.uint8)
        out.write_int(len(mp)); out.write_int(len(packed)); out.write(packed.tobytes())
        return
    width = _map_width(delta_type, xdim)
    if delta_type == 13:
        out.write_byte(mp[0]); out.write_byte(mp[1])
        width = blocks_across(xdim, int(mp[0]))
        mp = mp[2:]
        previous = None if previous is None else np.asarray(previous, dtype=np.uint8)[2:]
    forms = [_map_to_string(mp), _map_to_arithmetic(mp), _map_to_context(mp, previous, width)]
    best = min(range(3), key=lambda f: (len(forms[f]), f))
    out.write_byte(best)
    out.write(forms[best])


def map_bytes(delta_type, mp, previous, xdim):
    out = DataOutput()
    write_map(out, delta_type, mp, previous, xdim)
    return out.size()


def _read_map_forms(inp, previous, width):
    coding = inp.read_byte()
    n = inp.read_int()
    if coding == 2:
        K = inp.read_unsigned_byte()
        coded = np.frombuffer(inp.read_fully(inp.read_int()), dtype=np.uint8)
        prev, has = _prev_array(previous, n)
        return _decode_map_context(coded, n, K, prev, has, width, am._highest_one_bit(K))
    if coding == 1:
        K = inp.read_short()
        freq = np.zeros(256, dtype=np.int64)
        only = 0
        for _ in range(K):
            only = inp.read_unsigned_byte()
            freq[only] = inp.read_int()
        coded = np.frombuffer(inp.read_fully(inp.read_int()), dtype=np.uint8)
        if K > 1:
            return am.get_arithmetic_values_fast_fenwick(coded, freq, n)
        return np.full(n, only, dtype=np.uint8)
    table = read_table(inp)
    lo = inp.read_int()
    bits = inp.read_int()
    first = inp.read_unsigned_byte()
    string = np.frombuffer(inp.read_fully(sm.get_bytelength(bits)), dtype=np.uint8)
    unpacked = sm.decompress_strings(string)
    value = sm.unpack_strings(unpacked, table, n, sm.get_bitlength(unpacked))
    mp = ((value + lo) & 0xFF).astype(np.uint8)
    mp[0] = first
    return mp


def read_map(inp, delta_type, previous, xdim):
    if delta_type <= 8:
        n = inp.read_int()
        packed = np.frombuffer(inp.read_fully(inp.read_int()), dtype=np.uint8)
        q = np.arange(n)
        return ((packed[q >> 2] >> ((q & 3) << 1)) & 3).astype(np.uint8)
    if delta_type == 13:
        block = inp.read_byte()
        set_id = inp.read_byte()
        if not (0 <= set_id < len(BLOCK_SET_NAMES)) or not (BLOCK_MIN <= block <= BLOCK_MAX):
            raise IOError("Block map with predictor set %d and block size %d: not one this version reads." % (set_id, block))
        prev = None if previous is None else np.asarray(previous, dtype=np.uint8)[2:]
        body = _read_map_forms(inp, prev, blocks_across(xdim, block))
        return np.concatenate([np.array([block, set_id], dtype=np.uint8), body])
    return _read_map_forms(inp, previous, _map_width(delta_type, xdim))


# =============================================================================
# Context coding of deltas (the Context entropy type)
#
# Deltas are coded as ranks (most frequent value = 0). Context: activity =
# |left| + |above| + |above-left| + |above-right| of the deltas already
# coded, in 12 buckets, times 5 buckets of the deltas at the same pixel in
# the channels coded before (so channels decode in order).
# =============================================================================

ACTIVITY_LIMIT = np.array([0, 1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 60], dtype=np.int64)
CROSS_LIMIT = np.array([0, 1, 3, 6, 12], dtype=np.int64)
CONTEXTS = len(ACTIVITY_LIMIT) * len(CROSS_LIMIT)


@njit
def _bucket(v, limit):
    b = 0
    while b + 1 < limit.shape[0] and v >= limit[b + 1]:
        b += 1
    return b


@njit
def _delta_context(d, previous, k, xdim, act_limit, cross_limit):
    x = k % xdim
    activity = 0
    cross = 0
    if x > 0:
        activity += abs(d[k - 1])
    if k >= xdim:
        activity += abs(d[k - xdim])
        if x > 0:
            activity += abs(d[k - xdim - 1])
        if x < xdim - 1:
            activity += abs(d[k - xdim + 1])
    for p in range(previous.shape[0]):
        cross += abs(previous[p, k])
    return _bucket(activity, act_limit) * cross_limit.shape[0] + _bucket(cross, cross_limit)


@njit
def _delta_contexts(d, previous, xdim, act_limit, cross_limit):
    ctx = np.zeros(d.shape[0], dtype=np.int64)
    for k in range(d.shape[0]):
        ctx[k] = _delta_context(d, previous, k, xdim, act_limit, cross_limit)
    return ctx


@njit
def _decode_context_deltas(data, n, value, previous, xdim, act_limit, cross_limit, top):
    n_symbols = value.shape[0]
    f, bit, total, dt = am.context_decoder_init(data, act_limit.shape[0] * cross_limit.shape[0], n_symbols)
    d = np.zeros(n, dtype=np.int64)
    for k in range(n):
        c = _delta_context(d, previous, k, xdim, act_limit, cross_limit)
        d[k] = value[am.context_decode_one(data, dt, f, bit, total, c, n_symbols, top)]
    return d


def _previous_2d(previous, n):
    if not previous:
        return np.zeros((0, n), dtype=np.int64)
    return np.stack([np.asarray(p, dtype=np.int64) for p in previous])


def pack_context_deltas(delta, previous, xdim):
    """int min, the rank table (value -> rank, see write_table), int coded
    length, coded bytes. previous: the deltas of the channels before."""
    delta = np.asarray(delta, dtype=np.int64)
    lo = int(delta.min())
    count = np.bincount(delta - lo)
    order = sorted(range(len(count)), key=lambda v: (-count[v], v))
    rank = np.zeros(len(count), dtype=np.int64)
    rank[order] = np.arange(len(count))
    symbol = rank[delta - lo]
    ctx = _delta_contexts(delta, _previous_2d(previous, len(delta)), xdim, ACTIVITY_LIMIT, CROSS_LIMIT)
    coded = am.get_interval_value_context(symbol, len(rank), ctx, CONTEXTS)
    out = DataOutput()
    out.write_int(lo); write_table(out, rank); out.write_int(len(coded)); out.write(coded.tobytes())
    return out.to_bytes()


def read_context_deltas(inp, n, previous, xdim):
    lo = inp.read_int()
    rank = read_table(inp)
    coded = np.frombuffer(inp.read_fully(inp.read_int()), dtype=np.uint8)
    value = np.zeros(len(rank), dtype=np.int64)
    value[rank] = np.arange(len(rank)) + lo
    return _decode_context_deltas(coded, n, value, _previous_2d(previous, n), xdim,
                                  ACTIVITY_LIMIT, CROSS_LIMIT, am._highest_one_bit(len(rank)))


# =============================================================================
# Block-map settings search
# =============================================================================

def get_block_map_bytes(channel, xdim, ydim, block, block_set):
    """Bytes the three channels take as block-map deltas (Context coded)
    plus their maps, as the writers save them."""
    d, m = [], []
    for c in channel:
        delta, mp, _ = get_deltas(c, xdim, ydim, 13, 0, block, block_set)
        d.append(delta); m.append(mp)
    total = 0
    for i in range(3):
        total += len(pack_context_deltas(d[i], d[:i], xdim))
        total += map_bytes(13, m[i], m[i - 1] if i > 0 else None, xdim)
    return total


def find_best_block(channel, xdim, ydim, workers=None):
    """The block size and predictor set that code the three channels
    smallest: every set at sizes 4, 8 and 16, then the best set at the
    untried sizes next to its best. Returns ((block, set), bytes) where
    bytes[set][size index] is -1 for sizes not tried."""
    n_sets = len(BLOCK_SET_NAMES)
    first = [0, 2, 4]
    table = [[-1] * len(BLOCK_SEARCH_SIZES) for _ in range(n_sets)]
    jobs = [(s, i) for s in range(n_sets) for i in first]
    with ThreadPoolExecutor(workers) as pool:
        for (s, i), v in zip(jobs, pool.map(lambda j: get_block_map_bytes(channel, xdim, ydim, BLOCK_SEARCH_SIZES[j[1]], j[0]), jobs)):
            table[s][i] = v
    best_set, best_i = 0, first[0]
    for s in range(n_sets):
        for i in first:
            if table[s][i] < table[best_set][best_i]:
                best_set, best_i = s, i
    more = [i for i in (best_i - 1, best_i + 1) if 0 <= i < len(BLOCK_SEARCH_SIZES) and table[best_set][i] < 0]
    with ThreadPoolExecutor(workers) as pool:
        for i, v in zip(more, pool.map(lambda i: get_block_map_bytes(channel, xdim, ydim, BLOCK_SEARCH_SIZES[i], best_set), more)):
            table[best_set][i] = v
    for i in range(len(BLOCK_SEARCH_SIZES)):
        if 0 <= table[best_set][i] < table[best_set][best_i]:
            best_i = i
    return (BLOCK_SEARCH_SIZES[best_i], best_set), table


def get_block_table(table, best):
    lines = ["Block map, bytes coded (Context-coded deltas + maps, 3 channels):",
             "  %-18s" % "set \\ block" + "".join(" %9d " % b for b in BLOCK_SEARCH_SIZES)]
    for s, name in enumerate(BLOCK_SET_NAMES):
        row = "  %-18s" % name
        for i, b in enumerate(BLOCK_SEARCH_SIZES):
            v = table[s][i]
            chosen = s == best[1] and b == best[0]
            row += " %9s " % "-" if v < 0 else " %9d" % v + ("*" if chosen else " ")
        lines.append(row)
    return "\n".join(lines) + "\n"
