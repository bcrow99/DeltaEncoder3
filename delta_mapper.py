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
import os
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
    if p == 21: return b + ((d - c) >> 1)
    # 22-27: selection predictors
    if p == 22:
        q = a + b - c
        pa = abs(q - a); pb = abs(q - b); pc = abs(q - c)
        return a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)       # Paeth (PNG)
    if p == 23: return _median3(a, b, c)
    if p == 24: return _median3(a, b, d)
    if p == 25: return _median3(a, d, a + d - b)
    if p == 26: return a if abs(b - c) <= abs(a - c) else b
    if p == 27:
        h = abs(b - c); v = abs(a - c); g = abs(a - b)
        return a if (h <= v and h <= g) else (b if v <= g else d)
    return b + ((d - c) >> 1)


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
#
# The interior is split into block x block squares, and each uses one
# predictor from its channel's set. The map is [block size, n, n predictor
# ids (PREDICTOR_TABLE), then one entry per block, row by row: an index
# 0..n-1 into the ids]. Some predictors use the previous channel (in coding
# order) at the same pixel, so a channel is coded and decoded with the one
# before it (ref). With a named set a block takes the predictor with the
# smallest sum of |delta|; survey_block_sets chooses by estimated coded size.
# See DeltaMapper.java and predictor.txt.
# =============================================================================

BLOCK_MIN, BLOCK_MAX, BLOCK_DEFAULT = 4, 32, 8
# Block shapes, a block map's first byte (DeltaMapper.BLOCK_SHAPES):
# BLOCK_MIN..BLOCK_MAX is a square of that size; SHAPE_FIRST + i is
# BLOCK_SHAPES[i], width x height (width 0: the whole row). Rows of pixels
# -- "lines" -- are the 1-high ones. Blocks are numbered row by row.
SHAPE_FIRST = 64
BLOCK_SHAPES = [(16, 4), (32, 2), (64, 1), (256, 1), (0, 1), (4, 16), (1, 64)]
_SHAPE_W = np.array([w for w, _ in BLOCK_SHAPES], dtype=np.int64)
_SHAPE_H = np.array([h for _, h in BLOCK_SHAPES], dtype=np.int64)
BLOCK_SEARCH_SIZES = [4, 6, 8, 12, 16, 24, 32]

# Primitives: 0-21 on a, b, c, d (_block_predictor); 22-27 selection; 28-35
# cross-channel; 36-39 longer reach; 40-41 clamped; 42-45 bias-corrected.
PRIMITIVES = 22
SELECTION_FIRST, CROSS_FIRST, REACH_FIRST, CLAMP_FIRST, BIAS_FIRST, ALL_PRIMITIVES = 22, 28, 36, 40, 42, 46
GROUP_NAMES = ["selection", "cross-channel", "longer reach", "clamped", "bias-corrected"]
GROUP_FIRST = [SELECTION_FIRST, CROSS_FIRST, REACH_FIRST, CLAMP_FIRST, BIAS_FIRST, ALL_PRIMITIVES]
PRIMITIVE_NAMES = [
    "a", "b", "c", "d", "(a+b)/2", "(b+c)/2", "(a+c)/2", "(b+d)/2",
    "(c+d)/2", "(a+b+c+d)/4", "a+b-c", "MED", "(3a+b)/4", "(a+3b)/4", "(3a+d)/4", "(3b+a)/4",
    "(a+d)/2", "a+(b-c)/2", "b+(a-c)/2", "a+d-b", "(a+b+d)/3", "b+(d-c)/2"]
EXTRA_PRIMITIVE_NAMES = [
    "Paeth", "median(a,b,c)", "median(a,b,d)", "median(a,d,a+d-b)", "switch(a,b)", "switch(a,b,d)",
    "a+(R-Ra)", "b+(R-Rb)", "d+(R-Rd)", "MED+(R-MED_R)", "(a+b)/2+(R-..)", "a+b-c+(R-..)", "MED+(R-MED_R)/2", "(a+b)/2+(R-..)/2",
    "2a-aa", "2b-bb", "GAP", "GAP/8",
    "clamp(a+b-c)", "clamp(a+d-b)",
    "MED+bias", "(a+b)/2+bias", "a+bias", "b+bias"]
CROSS_BASE = np.array([0, 1, 3, 11, 4, 10, 11, 4], dtype=np.int64)   # the last two with half the error
BIAS_BASE = np.array([11, 4, 0, 1], dtype=np.int64)

# PREDICTOR_TABLE as pairs of primitives (equal for a primitive alone).
PREDICTOR_PAIRS = [
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 11, 11, 13, 13,
    16, 16, 18, 18, 19, 19, 23, 23, 24, 24, 25, 25, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 34, 34,
    35, 35, 36, 36, 37, 37, 39, 39, 40, 40, 42, 42, 45, 45, 2, 0, 2, 1, 3, 0, 3, 1, 4, 0,
    4, 3, 5, 0, 5, 1, 5, 2, 5, 3, 5, 4, 6, 0, 6, 1, 6, 2, 6, 3, 6, 4, 7, 0,
    7, 1, 7, 4, 7, 5, 7, 6, 8, 0, 8, 1, 8, 4, 9, 4, 9, 6, 11, 0, 11, 1, 11, 7,
    11, 8, 13, 0, 16, 0, 16, 1, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7, 16, 8, 16, 9, 16, 11,
    18, 0, 18, 1, 18, 3, 18, 4, 18, 5, 18, 6, 18, 7, 18, 8, 18, 9, 18, 13, 18, 16, 19, 3,
    19, 5, 19, 6, 19, 7, 19, 8, 19, 18, 23, 0, 23, 1, 23, 2, 23, 3, 23, 4, 23, 5, 23, 6,
    24, 0, 24, 1, 24, 4, 24, 5, 24, 6, 24, 7, 24, 9, 24, 18, 24, 23, 25, 1, 25, 3, 25, 5,
    25, 6, 28, 0, 28, 5, 28, 6, 28, 7, 28, 8, 28, 16, 29, 1, 29, 4, 29, 5, 29, 6, 29, 7,
    29, 8, 29, 16, 29, 28, 30, 6, 30, 7, 30, 16, 30, 25, 30, 28, 30, 29, 31, 0, 31, 5, 31, 6,
    31, 7, 31, 11, 31, 16, 32, 4, 32, 5, 32, 6, 32, 7, 32, 8, 32, 9, 32, 16, 32, 28, 32, 29,
    32, 30, 34, 0, 34, 2, 34, 3, 34, 6, 34, 7, 34, 8, 34, 16, 34, 23, 34, 25, 34, 28, 34, 29,
    34, 30, 34, 31, 35, 0, 35, 1, 35, 2, 35, 3, 35, 4, 35, 5, 35, 6, 35, 7, 35, 8, 35, 9,
    35, 16, 35, 23, 35, 24, 35, 28, 35, 29, 35, 30, 35, 32, 36, 2, 36, 4, 36, 6, 36, 11, 36, 16,
    36, 25, 37, 5, 37, 7, 37, 8, 37, 9, 37, 13, 37, 16, 37, 18, 37, 19, 37, 25, 39, 0, 39, 1,
    39, 2, 39, 3, 39, 4, 39, 5, 39, 6, 39, 7, 39, 8, 39, 9, 39, 11, 39, 13, 39, 16, 39, 18,
    39, 19, 39, 23, 39, 24, 39, 25, 39, 28, 39, 29, 39, 30, 39, 31, 39, 32, 39, 34, 39, 35, 39, 36,
    39, 37, 40, 0, 40, 1, 40, 3, 40, 7, 40, 11, 40, 16, 40, 19, 40, 25, 40, 34, 40, 36, 40, 37,
    40, 39, 42, 0, 42, 1, 42, 3, 42, 4, 42, 6, 42, 7, 42, 16, 42, 18, 42, 23, 42, 29, 42, 34,
    42, 35, 42, 36, 42, 37, 42, 39, 42, 40, 45, 0, 45, 3, 45, 4, 45, 5, 45, 6, 45, 7, 45, 16,
    45, 23, 45, 34, 45, 39]
TABLE_FIRST = np.array(PREDICTOR_PAIRS[0::2], dtype=np.int64)
TABLE_SECOND = np.array(PREDICTOR_PAIRS[1::2], dtype=np.int64)
PREDICTORS = len(TABLE_FIRST)   # 255
TABLE_PRIMITIVES = int(max(TABLE_FIRST.max(), TABLE_SECOND.max())) + 1

# Named sets: the first 4, 8, 16 and 32 of one greedy order (DeltaMapper.BLOCK_ORDER).
BLOCK_SET_NAMES = ["Top 4", "Top 8", "Top 16", "Top 32"]
BLOCK_ORDER = [243, 21, 167, 127, 209, 37, 0, 231, 171, 1, 39, 149, 141, 172, 205, 28, 44, 173, 122, 206, 208, 184, 177, 86, 213, 67, 182, 150, 27, 137, 30, 121]
BLOCK_SET_SIZES = [4, 8, 16, 32]


def primitive_name(p):
    return PRIMITIVE_NAMES[p] if p < PRIMITIVES else EXTRA_PRIMITIVE_NAMES[p - PRIMITIVES]


def predictor_name(pid):
    i, j = int(TABLE_FIRST[pid]), int(TABLE_SECOND[pid])
    return primitive_name(i) if i == j else "avg(%s, %s)" % (primitive_name(i), primitive_name(j))


def table_uses(pid, p):
    return int(TABLE_FIRST[pid]) == p or int(TABLE_SECOND[pid]) == p


def get_set_ids(block_set):
    """The predictor ids of a named set (BLOCK_SET_NAMES)."""
    return BLOCK_ORDER[:BLOCK_SET_SIZES[block_set]]


def map_header_length(mp):
    return 2 + int(mp[1])


def get_map_ids(mp):
    return [int(v) for v in mp[2:map_header_length(mp)]]


def describe_map_set(mp):
    ids = get_map_ids(mp)
    for s, name in enumerate(BLOCK_SET_NAMES):
        if ids == get_set_ids(s):
            return name
    return "%d predictors" % len(ids)


@njit
def _block_wh(block, xdim):
    """(width, height) of a block shape."""
    if block < SHAPE_FIRST:
        return block, block
    w = _SHAPE_W[block - SHAPE_FIRST]
    if w == 0:
        w = max(1, xdim - 2)
    return w, _SHAPE_H[block - SHAPE_FIRST]


def block_width(block, xdim):
    return int(_block_wh(int(block), int(xdim))[0])


def block_height(block):
    block = int(block)
    return block if block < SHAPE_FIRST else BLOCK_SHAPES[block - SHAPE_FIRST][1]


def valid_shape(block):
    return BLOCK_MIN <= block <= BLOCK_MAX or SHAPE_FIRST <= block < SHAPE_FIRST + len(BLOCK_SHAPES)


def shape_name(block):
    """"8" for a square, "16x4", "row" for a whole-row line."""
    block = int(block)
    if block < SHAPE_FIRST:
        return str(block)
    w, h = BLOCK_SHAPES[block - SHAPE_FIRST]
    return "row" if w == 0 else "%dx%d" % (w, h)


def blocks_across(xdim, block):
    """Blocks across the interior (columns 1..xdim-2)."""
    w = block_width(block, xdim)
    return (xdim - 2 + w - 1) // w


def blocks_down(ydim, block):
    """Blocks down the interior (rows 1..ydim-1)."""
    h = block_height(block)
    return (ydim - 1 + h - 1) // h


# Row bands: the Numba-compiled loops release the GIL, so splitting their
# rows over threads runs them on all cores (as Java's parallel streams do).
# The band tasks never wait on other tasks, so the pool can't deadlock when
# callers on other threads use it at the same time.
_CPUS = os.cpu_count() or 1
_ROW_POOL = ThreadPoolExecutor(_CPUS)


def _available_memory():
    """Bytes of memory free for new work, or None if it can't be told."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:                                    # Linux
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    try:                                    # Windows
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total_phys", ctypes.c_ulonglong),
                        ("avail_phys", ctypes.c_ulonglong), ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong), ("avail_extended", ctypes.c_ulonglong)]
        st = _Status()
        st.length = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.avail_phys)
    except Exception:
        pass
    return None


def _bounded_map(fn, items, task_bytes):
    """[fn(x) for x in items], as many at once as there are cores and as free
    memory allows (each task needing about task_bytes): big images run fewer
    trials at once instead of pushing the machine into swap. (Python has no
    heap limit to raise; this is DeltaMapper.boundedParallel.)"""
    items = list(items)
    free = _available_memory()
    if free is None:
        free = 4 << 30                      # can't tell: assume 4 GB
    workers = max(1, min(len(items), _CPUS, int(free * 0.6) // max(1, task_bytes)))
    if workers == 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fn, items))


def _in_bands(n, body):
    """body(lo, hi) over 0..n-1 in up to 2 x cores bands, in parallel."""
    k = max(1, min(n, 2 * _CPUS))
    cuts = [n * t // k for t in range(k + 1)]
    if k == 1:
        body(0, n)
        return
    for f in [_ROW_POOL.submit(body, cuts[t], cuts[t + 1]) for t in range(k)]:
        f.result()


@njit
def _edges(src, dst, xdim, ydim):
    for x in range(1, xdim):
        dst[x] = src[x] - src[x - 1]
    for y in range(1, ydim):
        dst[y * xdim] = src[y * xdim] - src[(y - 1) * xdim]
        dst[y * xdim + xdim - 1] = src[y * xdim + xdim - 1] - src[y * xdim + xdim - 2]


@njit
def _median3(x, y, z):
    return max(min(x, y), min(max(x, y), z))


@njit
def _gap(w, ww, n, nn, nw, ne, nne, t1, t2, t3):
    dh = abs(w - ww) + abs(n - nw) + abs(n - ne)
    dv = abs(w - nw) + abs(n - nn) + abs(ne - nne)
    if dv - dh > t1:
        return w
    if dh - dv > t1:
        return n
    p = (2 * (w + n) + ne - nw) >> 2
    if dv - dh > t2:
        p = (p + w) >> 1
    elif dv - dh > t3:
        p = (3 * p + w) >> 2
    elif dh - dv > t2:
        p = (p + n) >> 1
    elif dh - dv > t3:
        p = (3 * p + n) >> 2
    return p


@njit
def _gradient_level(g):
    m = abs(g)
    l = 0 if m == 0 else (1 if m < 3 else (2 if m < 7 else (3 if m < 21 else 4)))
    return -l if g < 0 else l


@njit
def _bias_context(a, b, c, d):
    return (_gradient_level(d - b) + 4) * 81 + (_gradient_level(b - c) + 4) * 9 + _gradient_level(c - a) + 4


@njit
def _bias_prepare(v, k, xdim, ctx, corr, bsum, bcount):
    a = v[k - 1]; b = v[k - xdim]; c = v[k - xdim - 1]; d = v[k - xdim + 1]
    t = _bias_context(a, b, c, d)
    ctx[0] = t
    for s in range(4):
        n = bcount[s, t]
        e = bsum[s, t]
        if n == 0:
            corr[s, k] = 0
        elif e >= 0:
            corr[s, k] = (e + n // 2) // n
        else:
            corr[s, k] = -((-e + n // 2) // n)


@njit
def _bias_commit(v, k, xdim, ctx, bsum, bcount, bias_base):
    a = v[k - 1]; b = v[k - xdim]; c = v[k - xdim - 1]; d = v[k - xdim + 1]
    t = ctx[0]
    for s in range(4):
        bsum[s, t] += v[k] - _block_predictor(bias_base[s], a, b, c, d)
        bcount[s, t] += 1
        if bcount[s, t] == 64:
            bsum[s, t] = jdiv(bsum[s, t], 2)
            bcount[s, t] = 32


@njit
def _bias_corrections(v, xdim, ydim, bias_base):
    """The encoder's bias corrections for the whole channel, [4, n]."""
    n = v.shape[0]
    corr = np.zeros((4, n), dtype=np.int16)   # |corr| <= the channel's range
    ctx = np.zeros(1, dtype=np.int64)          # the current pixel's bias context
    bsum = np.zeros((4, 729), dtype=np.int64)
    bcount = np.zeros((4, 729), dtype=np.int64)
    for y in range(1, ydim):
        for x in range(1, xdim - 1):
            k = y * xdim + x
            _bias_prepare(v, k, xdim, ctx, corr, bsum, bcount)
            _bias_commit(v, k, xdim, ctx, bsum, bcount, bias_base)
    return corr


@njit
def _primitive(q, v, ref, has_ref, corr, k, xdim, cross_base, bias_base):
    """Primitive q at interior pixel k of channel v (ref: the previous channel)."""
    a = v[k - 1]; b = v[k - xdim]; c = v[k - xdim - 1]; d = v[k - xdim + 1]
    if q < CROSS_FIRST:
        return _block_predictor(q, a, b, c, d)
    if q < REACH_FIRST:
        s = q - CROSS_FIRST
        base = cross_base[s]
        p = _block_predictor(base, a, b, c, d)
        if not has_ref:
            return p
        e = ref[k] - _block_predictor(base, ref[k - 1], ref[k - xdim], ref[k - xdim - 1], ref[k - xdim + 1])
        return p + (e >> 1) if s >= 6 else p + e
    if q < CLAMP_FIRST:
        x = k % xdim
        y = k // xdim
        aa = v[k - 2] if x >= 2 else a
        bb = v[k - 2 * xdim] if y >= 2 else b
        if q == REACH_FIRST:
            return 2 * a - aa
        if q == REACH_FIRST + 1:
            return 2 * b - bb
        nne = v[k - 2 * xdim + 1] if y >= 2 else d
        if q == REACH_FIRST + 2:
            return _gap(a, aa, b, bb, c, d, nne, 80, 32, 8)
        return _gap(a, aa, b, bb, c, d, nne, 10, 4, 1)
    if q < BIAS_FIRST:
        lo = min(min(a, b), min(c, d))
        hi = max(max(a, b), max(c, d))
        p = a + b - c if q == CLAMP_FIRST else a + d - b
        return max(lo, min(hi, p))
    s = q - BIAS_FIRST
    return _block_predictor(bias_base[s], a, b, c, d) + np.int64(corr[s, k])


@njit
def _all_primitives(v, ref, has_ref, corr, k, xdim, prim):
    """Every primitive (0..ALL_PRIMITIVES-1) at interior pixel k, into prim:
    the same values as _primitive, in one straight pass (the neighbours read
    once, no dispatch)."""
    a = v[k - 1]; b = v[k - xdim]; c = v[k - xdim - 1]; d = v[k - xdim + 1]
    prim[0] = a; prim[1] = b; prim[2] = c; prim[3] = d
    prim[4] = (a + b) >> 1; prim[5] = (b + c) >> 1; prim[6] = (a + c) >> 1; prim[7] = (b + d) >> 1
    prim[8] = (c + d) >> 1; prim[9] = (a + b + c + d + 2) >> 2; prim[10] = a + b - c
    med = _med(a, b, c)
    prim[11] = med
    prim[12] = (a * 3 + b + 2) >> 2; prim[13] = (a + b * 3 + 2) >> 2; prim[14] = (a * 3 + d + 2) >> 2; prim[15] = (b * 3 + a + 2) >> 2
    prim[16] = (a + d) >> 1; prim[17] = a + ((b - c) >> 1); prim[18] = b + ((a - c) >> 1); prim[19] = a + d - b
    prim[20] = jdiv(a + b + d + 1, 3); prim[21] = b + ((d - c) >> 1)
    q = a + b - c
    pa = abs(q - a); pb = abs(q - b); pc = abs(q - c)
    prim[22] = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
    prim[23] = _median3(a, b, c); prim[24] = _median3(a, b, d); prim[25] = _median3(a, d, a + d - b)
    prim[26] = a if abs(b - c) <= abs(a - c) else b
    h = abs(b - c); vv = abs(a - c); g = abs(a - b)
    prim[27] = a if (h <= vv and h <= g) else (b if vv <= g else d)
    # cross-channel: base P = a, b, d, MED, (a+b)/2, a+b-c, then MED and (a+b)/2 with half the error
    if has_ref:
        r = ref[k]; ra = ref[k - 1]; rb = ref[k - xdim]; rc = ref[k - xdim - 1]; rd = ref[k - xdim + 1]
        e_med = r - _med(ra, rb, rc)
        e_avg = r - ((ra + rb) >> 1)
        prim[28] = a + (r - ra); prim[29] = b + (r - rb); prim[30] = d + (r - rd); prim[31] = med + e_med
        prim[32] = prim[4] + e_avg; prim[33] = prim[10] + (r - (ra + rb - rc))
        prim[34] = med + (e_med >> 1); prim[35] = prim[4] + (e_avg >> 1)
    else:
        prim[28] = a; prim[29] = b; prim[30] = d; prim[31] = med; prim[32] = prim[4]; prim[33] = prim[10]
        prim[34] = med; prim[35] = prim[4]
    x = k % xdim
    y = k // xdim
    aa = v[k - 2] if x >= 2 else a
    bb = v[k - 2 * xdim] if y >= 2 else b
    nne = v[k - 2 * xdim + 1] if y >= 2 else d
    prim[36] = 2 * a - aa; prim[37] = 2 * b - bb
    prim[38] = _gap(a, aa, b, bb, c, d, nne, 80, 32, 8)
    prim[39] = _gap(a, aa, b, bb, c, d, nne, 10, 4, 1)
    lo = min(min(a, b), min(c, d))
    hi = max(max(a, b), max(c, d))
    prim[40] = max(lo, min(hi, a + b - c)); prim[41] = max(lo, min(hi, a + d - b))
    prim[42] = med + np.int64(corr[0, k]); prim[43] = prim[4] + np.int64(corr[1, k])
    prim[44] = a + np.int64(corr[2, k]); prim[45] = b + np.int64(corr[3, k])


@njit
def _predict(pid, first, second, v, ref, has_ref, corr, k, xdim, cross_base, bias_base):
    i = first[pid]
    j = second[pid]
    p = _primitive(i, v, ref, has_ref, corr, k, xdim, cross_base, bias_base)
    if i == j:
        return p
    return (p + _primitive(j, v, ref, has_ref, corr, k, xdim, cross_base, bias_base) + 1) >> 1


class PixelSource:
    """What the table's primitives need for a channel (DeltaMapper.PixelSource,
    encoder side): the channel, the previous channel, the bias corrections."""

    def __init__(self, v, ref, xdim, ydim):
        self.v = np.asarray(v, dtype=np.int64)
        self.has_ref = ref is not None
        self.ref = np.asarray(ref, dtype=np.int64) if ref is not None else np.zeros(1, dtype=np.int64)
        self.xdim, self.ydim = xdim, ydim
        self.corr = _bias_corrections(self.v, xdim, ydim, BIAS_BASE)


@njit
def _block_from_map(src, ref, has_ref, corr, xdim, y0, y1, mp, ids, first, second, cross_base, bias_base, dst):
    """Rows y0..y1-1 (>= 1) of the block deltas, into dst."""
    bwd, bht = _block_wh(np.int64(mp[0]), xdim)
    bw = (xdim - 2 + bwd - 1) // bwd
    head = 2 + np.int64(mp[1])
    prim = np.zeros(ALL_PRIMITIVES, dtype=np.int64)   # all of them: faster than computing two by dispatch
    for y in range(y0, y1):
        row = head + (y - 1) // bht * bw
        for x in range(1, xdim - 1):
            k = y * xdim + x
            _all_primitives(src, ref, has_ref, corr, k, xdim, prim)
            pid = ids[mp[row + (x - 1) // bwd]]
            i = first[pid]
            j = second[pid]
            dst[k] = src[k] - (prim[i] if i == j else (prim[i] + prim[j] + 1) >> 1)


def get_block_deltas_from_map(ps, mp, ids=None):
    """The block deltas for a given block map (DeltaMapper.getBlockDeltasFromMap);
    rows in parallel."""
    mp = np.asarray(mp, dtype=np.uint8)
    ids = np.array(get_map_ids(mp) if ids is None else list(ids), dtype=np.int64)
    dst = np.zeros(ps.xdim * ps.ydim, dtype=np.int64)
    _edges(ps.v, dst, ps.xdim, ps.ydim)
    _in_bands(ps.ydim - 1, lambda lo, hi: _block_from_map(ps.v, ps.ref, ps.has_ref, ps.corr, ps.xdim, 1 + lo, 1 + hi, mp, ids,
                                                         TABLE_FIRST, TABLE_SECOND, CROSS_BASE, BIAS_BASE, dst))
    return dst


@njit
def _block_choices(src, ref, has_ref, corr, xdim, ydim, block, ids, used, first, second, n_prim, cross_base, bias_base, by0, by1, choice):
    """Block rows by0..by1-1: each block's index of the id with the smallest
    sum of |delta|, into choice. Each primitive in used is computed once per pixel."""
    bwd, bht = _block_wh(block, xdim)
    bw = (xdim - 2 + bwd - 1) // bwd
    n = ids.shape[0]
    sums = np.zeros(bw * n, dtype=np.int64)
    prim = np.zeros(ALL_PRIMITIVES, dtype=np.int64)
    for by in range(by0, by1):
        sums[:] = 0
        y0 = 1 + by * bht
        y1 = min(ydim, y0 + bht)
        for y in range(y0, y1):
            for x in range(1, xdim - 1):
                k = y * xdim + x
                _all_primitives(src, ref, has_ref, corr, k, xdim, prim)
                base = (x - 1) // bwd * n
                for p in range(n):
                    i = first[ids[p]]
                    j = second[ids[p]]
                    v = prim[i] if i == j else (prim[i] + prim[j] + 1) >> 1
                    sums[base + p] += abs(src[k] - v)
        for bx in range(bw):
            best = 0
            for p in range(1, n):
                if sums[bx * n + p] < sums[bx * n + best]:
                    best = p
            choice[by * bw + bx] = best


def get_block_deltas_from_values(src, ref, xdim, ydim, block, ids):
    """(deltas, map): each block takes the id with the smallest sum of |delta|
    (DeltaMapper.getBlockDeltasFromValues); ref is the previous channel or None."""
    return _block_deltas_from_values(PixelSource(src, ref, xdim, ydim), block, ids)


def _block_deltas_from_values(ps, block, ids):
    xdim, ydim = ps.xdim, ps.ydim
    ids = [int(i) for i in ids]
    arr = np.array(ids, dtype=np.int64)
    used = np.array(sorted(set(int(TABLE_FIRST[i]) for i in ids) | set(int(TABLE_SECOND[i]) for i in ids)), dtype=np.int64)
    bh = blocks_down(ydim, block)
    choice = np.zeros(blocks_across(xdim, block) * bh, dtype=np.uint8)
    _in_bands(bh, lambda lo, hi: _block_choices(ps.v, ps.ref, ps.has_ref, ps.corr, xdim, ydim, block, arr, used,
                                                TABLE_FIRST, TABLE_SECOND, TABLE_PRIMITIVES, CROSS_BASE, BIAS_BASE, lo, hi, choice))
    mp = np.concatenate([np.array([block, len(ids)] + ids, dtype=np.uint8), choice])
    return get_block_deltas_from_map(ps, mp, ids), mp


@njit
def _from_block(delta, ref, has_ref, xdim, ydim, init, mp, ids, first, second, cross_base, bias_base):
    bwd, bht = _block_wh(np.int64(mp[0]), xdim)
    bw = (xdim - 2 + bwd - 1) // bwd
    head = 2 + np.int64(mp[1])
    n = xdim * ydim
    dst = np.zeros(n, dtype=np.int64)
    corr = np.zeros((4, n), dtype=np.int16)
    ctx = np.zeros(1, dtype=np.int64)
    bsum = np.zeros((4, 729), dtype=np.int64)
    bcount = np.zeros((4, 729), dtype=np.int64)
    prim = np.zeros(ALL_PRIMITIVES, dtype=np.int64)
    dst[0] = init
    for x in range(1, xdim):
        dst[x] = dst[x - 1] + delta[x]
    for y in range(1, ydim):
        k = y * xdim
        dst[k] = dst[k - xdim] + delta[k]
        row = head + (y - 1) // bht * bw
        for x in range(1, xdim - 1):
            k += 1
            _bias_prepare(dst, k, xdim, ctx, corr, bsum, bcount)
            _all_primitives(dst, ref, has_ref, corr, k, xdim, prim)
            pid = ids[mp[row + (x - 1) // bwd]]
            i = first[pid]
            j = second[pid]
            dst[k] = delta[k] + (prim[i] if i == j else (prim[i] + prim[j] + 1) >> 1)
            _bias_commit(dst, k, xdim, ctx, bsum, bcount, bias_base)
        k += 1
        dst[k] = dst[k - 1] + delta[k]
    return dst


def get_values_from_block_deltas(delta, xdim, ydim, init, mp, ref):
    mp = np.asarray(mp, dtype=np.uint8)
    has_ref = ref is not None
    r = np.asarray(ref, dtype=np.int64) if has_ref else np.zeros(1, dtype=np.int64)
    return _from_block(np.asarray(delta, dtype=np.int64), r, has_ref, xdim, ydim, init, mp,
                       np.array(get_map_ids(mp), dtype=np.int64), TABLE_FIRST, TABLE_SECOND, CROSS_BASE, BIAS_BASE)


# =============================================================================
# Dispatch
# =============================================================================

def get_deltas(src, xdim, ydim, delta_type, variant=0, block=BLOCK_DEFAULT, block_set=0, set_ids=None, ref=None):
    """(deltas, map or None, init value) for delta_type 0-13. For the block
    map (13) the predictors are the named set block_set, or the predictor
    ids set_ids if given, and ref is the previous channel (None for the
    first)."""
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
    elif t == 13:
        ids = get_set_ids(block_set) if set_ids is None else [int(i) for i in set_ids]
        d, m = get_block_deltas_from_values(src, ref, xdim, ydim, block, ids)
    else: raise ValueError("delta_type %d" % t)
    return d, m, init


def get_values_from_deltas(delta, xdim, ydim, init, delta_type, mp=None, variant=0, ref=None):
    """Inverse of get_deltas; ref (the previous channel as decoded, before
    any minimum is added back) is used by the block map only."""
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
    if t == 13: return get_values_from_block_deltas(delta, xdim, ydim, init, mp, ref)
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


CONTEXT_MAP_MAX = 32   # the context form sets up K^3 x 3 contexts of K counts, so only small sets use it


def _map_to_context(m, previous, width):
    K = max(1, int(m.max()) + 1 if len(m) else 1)
    if previous is not None and len(previous):
        K = max(K, int(np.max(previous)) + 1)
    if K > CONTEXT_MAP_MAX:
        return None
    prev, has = _prev_array(previous, len(m))
    ctx = _map_contexts(m, prev, has, width, K)
    coded = am.get_interval_value_context(m.astype(np.int64), K, ctx, K * K * K * 3)
    out = DataOutput()
    out.write_int(len(m)); out.write_byte(K); out.write_int(len(coded)); out.write(coded.tobytes())
    return out.to_bytes()


NEIGHBOUR_CONTEXTS = 32


@njit
def _neighbour(v, previous, has_previous, k, width, which):
    x = k % width
    if which == 0:
        return np.int64(v[k - 1]) if x > 0 else -1
    if which == 1:
        return np.int64(v[k - width]) if k >= width else -1
    if which == 2:
        return np.int64(previous[k]) if has_previous else -1
    return np.int64(v[k - width - 1]) if (x > 0 and k >= width) else -1


@njit
def _neighbour_context(v, previous, has_previous, k, width):
    L = _neighbour(v, previous, has_previous, k, width, 0)
    U = _neighbour(v, previous, has_previous, k, width, 1)
    V = _neighbour(v, previous, has_previous, k, width, 2)
    D = _neighbour(v, previous, has_previous, k, width, 3)
    c = 0
    if L >= 0 and L == U: c |= 1
    if V >= 0 and V == L: c |= 2
    if V >= 0 and V == U: c |= 4
    if D >= 0 and D == L: c |= 8
    if D >= 0 and D == U: c |= 16
    return c


@njit
def _neighbour_tokens(v, previous, has_previous, width):
    n = v.shape[0]
    token = np.zeros(n, dtype=np.int64)
    ctx = np.zeros(n, dtype=np.int64)
    for k in range(n):
        s = v[k]
        L = _neighbour(v, previous, has_previous, k, width, 0)
        U = _neighbour(v, previous, has_previous, k, width, 1)
        V = _neighbour(v, previous, has_previous, k, width, 2)
        if L >= 0 and s == L: token[k] = 0
        elif U >= 0 and s == U: token[k] = 1
        elif V >= 0 and s == V: token[k] = 2
        else: token[k] = 3 + s
        ctx[k] = _neighbour_context(v, previous, has_previous, k, width)
    return token, ctx


@njit
def _decode_map_neighbours(data, n, K, previous, has_previous, width, top):
    f, bit, total, dt = am.context_decoder_init(data, NEIGHBOUR_CONTEXTS, K + 3)
    v = np.zeros(n, dtype=np.int64)
    for k in range(n):
        c = _neighbour_context(v, previous, has_previous, k, width)
        t = am.context_decode_one(data, dt, f, bit, total, c, K + 3, top)
        v[k] = _neighbour(v, previous, has_previous, k, width, t) if t < 3 else t - 3
    return v.astype(np.uint8)


def _map_to_neighbours(m, previous, width):
    """Form 3: each entry as "same as left" (0), "same as above" (1), "same
    as the previous channel's map" (2), or 3 + its value; context: which of
    those neighbours and the upper-left one agree (32 contexts). Any number
    of values works, so it suits large predictor sets."""
    v = m.astype(np.int64)
    K = max(1, int(v.max()) + 1 if len(v) else 1)
    prev, has = _prev_array(previous, len(m))
    token, ctx = _neighbour_tokens(v, prev, has, width)
    coded = am.get_interval_value_context(token, K + 3, ctx, NEIGHBOUR_CONTEXTS)
    out = DataOutput()
    out.write_int(len(m)); out.write_short(K); out.write_int(len(coded)); out.write(coded.tobytes())
    return out.to_bytes()


def write_map(out, delta_type, mp, previous, xdim):
    """A delta-type map. Types 6-8 (values 0-3): int length, int packed
    length, 4 values per byte, low bits first. Types 9-13: a flag byte and
    the smallest of three forms -- 0 unary string, 1 arithmetic, 2 context
    coded (contexts use the previous channel's map), 3 neighbour coded (format
    version 2; see _map_to_neighbours). A block map (13) starts
    with its block size, the number of predictors n and their n ids."""
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
        head = map_header_length(mp)
        out.write(mp[:head].tobytes())
        width = blocks_across(xdim, int(mp[0]))
        mp = mp[head:]
        if previous is not None:
            previous = np.asarray(previous, dtype=np.uint8)
            previous = previous[map_header_length(previous):]
    forms = [_map_to_string(mp), _map_to_arithmetic(mp), _map_to_context(mp, previous, width), _map_to_neighbours(mp, previous, width)]
    best = min((f for f in range(4) if forms[f] is not None), key=lambda f: (len(forms[f]), f))
    out.write_byte(best)
    out.write(forms[best])


def map_bytes(delta_type, mp, previous, xdim):
    out = DataOutput()
    write_map(out, delta_type, mp, previous, xdim)
    return out.size()


def _read_map_forms(inp, previous, width):
    coding = inp.read_byte()
    n = inp.read_int()
    if coding == 3:
        K = inp.read_short()
        coded = np.frombuffer(inp.read_fully(inp.read_int()), dtype=np.uint8)
        prev, has = _prev_array(previous, n)
        return _decode_map_neighbours(coded, n, K, prev, has, width, am._highest_one_bit(K + 3))
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
        block = inp.read_unsigned_byte()
        n = inp.read_unsigned_byte()
        if not (1 <= n <= PREDICTORS):
            raise IOError("Block map with %d predictors: not one this version reads." % n)
        ids = [inp.read_unsigned_byte() for _ in range(n)]
        if max(ids) >= PREDICTORS:
            raise IOError("Block map with predictor id %d: not one this version reads." % max(ids))
        if not valid_shape(block):
            raise IOError("Block map with block shape %d: not one this version reads." % block)
        prev = None
        if previous is not None:
            prev = np.asarray(previous, dtype=np.uint8)
            prev = prev[map_header_length(prev):]
        body = _read_map_forms(inp, prev, blocks_across(xdim, block))
        if len(body) and int(body.max()) >= len(ids):
            raise IOError("Block map entry %d with %d predictors." % (int(body.max()), len(ids)))
        return np.concatenate([np.array([block, len(ids)] + ids, dtype=np.uint8), body])
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
def _delta_contexts(d, previous, xdim, act_limit, cross_limit, k0, k1, ctx):
    """Contexts of deltas k0..k1-1, into ctx."""
    for k in range(k0, k1):
        ctx[k] = _delta_context(d, previous, k, xdim, act_limit, cross_limit)


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
    prev2d = _previous_2d(previous, len(delta))
    ctx = np.zeros(len(delta), dtype=np.int64)
    _in_bands(len(delta), lambda k0, k1: _delta_contexts(delta, prev2d, xdim, ACTIVITY_LIMIT, CROSS_LIMIT, k0, k1, ctx))
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

def get_block_map_bytes(channel, xdim, ydim, block, block_set, ps=None):
    """Bytes the three channels take as block-map deltas (Context coded)
    plus their maps, as the writers save them (ps: their PixelSources)."""
    ps = pixel_sources(channel, xdim, ydim) if ps is None else ps
    d, m = [], []
    for i in range(len(channel)):
        delta, mp = _block_deltas_from_values(ps[i], block, get_set_ids(block_set))
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
    ps = pixel_sources(channel, xdim, ydim)
    n_sets = len(BLOCK_SET_NAMES)
    first = [0, 2, 4]
    table = [[-1] * len(BLOCK_SEARCH_SIZES) for _ in range(n_sets)]
    task = 64 * xdim * ydim                 # a trial's deltas, maps and Context-coder arrays, 3 channels
    jobs = [(s, i) for s in range(n_sets) for i in first]
    for (s, i), v in zip(jobs, _bounded_map(lambda j: get_block_map_bytes(channel, xdim, ydim, BLOCK_SEARCH_SIZES[j[1]], j[0], ps), jobs, task)):
        table[s][i] = v
    best_set, best_i = 0, first[0]
    for s in range(n_sets):
        for i in first:
            if table[s][i] < table[best_set][best_i]:
                best_set, best_i = s, i
    more = [i for i in (best_i - 1, best_i + 1) if 0 <= i < len(BLOCK_SEARCH_SIZES) and table[best_set][i] < 0]
    for i, v in zip(more, _bounded_map(lambda i: get_block_map_bytes(channel, xdim, ydim, BLOCK_SEARCH_SIZES[i], best_set, ps), more, task)):
        table[best_set][i] = v
    for i in range(len(BLOCK_SEARCH_SIZES)):
        if 0 <= table[best_set][i] < table[best_set][best_i]:
            best_i = i
    return (BLOCK_SEARCH_SIZES[best_i], best_set), table


# ---- Surveying predictor sets (DeltaMapper.surveyBlockSets) -------------------
#
# Predictors are ranked by an estimate of their coded size: a cost per delta
# value per coding context, from the deltas of a reference set (Top 32);
# each predictor's estimated bits per block; a greedy order that charges for
# the map (entropy of the block counts plus a byte per id); real coded sizes
# at SURVEY_COUNTS; blocks assigned by the estimate. See DeltaMapper.java.
# Costs are integers (1/256 bits) so the result matches Java exactly.

SURVEY_COUNTS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 255]
SURVEY_REFERENCE_SET = 3            # Top 32
SURVEY_ID_COST = 8 * 256            # a predictor id in the map header, 1/256 bits
SURVEY_SAMPLE_BLOCKS = 1 << 17      # blocks the greedy order is found on, at most
_LOG2 = math.log(2)


def _cost256(r):
    return int(math.floor(256 * math.log(r) / _LOG2 + 0.5))


@njit
def _cost_counts(ref, previous, xdim, ydim, lo, rng, act_limit, cross_limit, n_ctx):
    ctx = np.zeros(xdim * ydim, dtype=np.int64)
    count = np.zeros((n_ctx, rng), dtype=np.int64)
    total = np.zeros(n_ctx, dtype=np.int64)
    for y in range(1, ydim):
        for x in range(1, xdim - 1):
            k = y * xdim + x
            c = _delta_context(ref, previous, k, xdim, act_limit, cross_limit)
            ctx[k] = c
            v = min(rng - 1, max(0, ref[k] - lo))
            count[c, v] += 1
            total[c] += 1
    return ctx, count, total


class DeltaCost:
    """Estimated cost (1/256 bits) of each delta value in each coding
    context, from the reference deltas ref (DeltaMapper.DeltaCost)."""

    def __init__(self, src, ref, previous, xdim, ydim):
        src = np.asarray(src, dtype=np.int64)
        span = int(src.max()) - int(src.min())
        self.lo = -2 * span - 2
        self.range = 4 * span + 5
        self.context, count, total = _cost_counts(np.asarray(ref, dtype=np.int64), _previous_2d(previous, len(src)),
                                                  xdim, ydim, self.lo, self.range, ACTIVITY_LIMIT, CROSS_LIMIT, CONTEXTS)
        cost = np.zeros((CONTEXTS, self.range), dtype=np.int64)
        cache = {}
        for c in range(CONTEXTS):
            t = 32.0 * int(total[c]) + self.range
            row = count[c].tolist()
            for v in range(self.range):
                key = (t, row[v])
                if key not in cache:
                    cache[key] = _cost256(t / (32.0 * row[v] + 1))
                cost[c, v] = cache[key]
        self.cost = cost


@njit
def _block_cost_table(src, ref, has_ref, corr, xdim, ydim, block, cost, ctx, lo, rng, pids, first, second, n_prim, cross_base, bias_base, rows, r0, r1, est):
    """Block rows rows[r0..r1-1] of the cost table, into est[(r * bw + bx) * n + q]."""
    n = pids.shape[0]
    bwd, bht = _block_wh(block, xdim)
    bw = (xdim - 2 + bwd - 1) // bwd
    prim = np.zeros(ALL_PRIMITIVES, dtype=np.int64)
    for r in range(r0, r1):
        by = rows[r]
        y0 = 1 + by * bht
        y1 = min(ydim, y0 + bht)
        for y in range(y0, y1):
            for x in range(1, xdim - 1):
                k = y * xdim + x
                s = src[k]
                c = ctx[k]
                _all_primitives(src, ref, has_ref, corr, k, xdim, prim)
                base = (r * bw + (x - 1) // bwd) * n
                for q in range(n):
                    i = first[pids[q]]
                    j = second[pids[q]]
                    v = prim[i] if i == j else (prim[i] + prim[j] + 1) >> 1
                    est[base + q] += cost[c, min(rng - 1, max(0, s - v - lo))]


@njit
def _prefix_maps(src, ref, has_ref, corr, xdim, ydim, block, cost, ctx, lo, rng, pids, first, second, n_prim, cross_base, bias_base,
                 counts, heads, by0, by1, maps):
    """Block rows by0..by1-1 of the prefix maps (DeltaMapper.prefixMaps): row c
    of maps is the map for the first counts[c] ids, entries from heads[c]."""
    n = pids.shape[0]
    bwd, bht = _block_wh(block, xdim)
    bw = (xdim - 2 + bwd - 1) // bwd
    est = np.zeros(bw * n, dtype=np.int64)
    rows = np.zeros(1, dtype=np.int64)
    for by in range(by0, by1):
        est[:] = 0
        rows[0] = by
        _block_cost_table(src, ref, has_ref, corr, xdim, ydim, block, cost, ctx, lo, rng, pids, first, second, n_prim,
                          cross_base, bias_base, rows, 0, 1, est)
        for bx in range(bw):
            best = 0
            c = 0
            for q in range(n):
                if c >= counts.shape[0]:
                    break
                if est[bx * n + q] < est[bx * n + best]:
                    best = q
                while c < counts.shape[0] and counts[c] == q + 1:
                    maps[c, heads[c] + by * bw + bx] = best
                    c += 1


def block_cost_table(ps, block, dc, ids=None, rows=None):
    """Estimated bits (1/256) of each predictor in each block, as
    [block * n + q] for the ids given, or for every id (ids None); for the
    given block rows only (rows), or all of them."""
    ids = np.array(list(range(PREDICTORS)) if ids is None else ids, dtype=np.int64)
    bh = blocks_down(ps.ydim, block)
    rows = np.arange(bh, dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
    est = np.zeros(blocks_across(ps.xdim, block) * len(rows) * len(ids), dtype=np.int32)   # as Java's int
    _in_bands(len(rows), lambda lo, hi: _block_cost_table(ps.v, ps.ref, ps.has_ref, ps.corr, ps.xdim, ps.ydim, block, dc.cost, dc.context,
                                                          dc.lo, dc.range, ids, TABLE_FIRST, TABLE_SECOND,
                                                          TABLE_PRIMITIVES, CROSS_BASE, BIAS_BASE, rows, lo, hi, est))
    return est


def prefix_maps(ps, block, dc, ids, counts):
    """Block maps for the first counts[c] of ids, each block taking the first
    id of the prefix with the smallest estimate, one block row at a time
    (DeltaMapper.prefixMaps)."""
    bw = blocks_across(ps.xdim, block)
    bh = blocks_down(ps.ydim, block)
    ids = [int(i) for i in ids]
    width = 2 + max(counts) + bw * bh
    maps = np.zeros((len(counts), width), dtype=np.uint8)
    heads = np.array([2 + c for c in counts], dtype=np.int64)
    arr = np.array(ids, dtype=np.int64)
    _in_bands(bh, lambda lo, hi: _prefix_maps(ps.v, ps.ref, ps.has_ref, ps.corr, ps.xdim, ps.ydim, block, dc.cost, dc.context,
                                              dc.lo, dc.range, arr, TABLE_FIRST, TABLE_SECOND, TABLE_PRIMITIVES, CROSS_BASE, BIAS_BASE,
                                              np.array(counts, dtype=np.int64), heads, lo, hi, maps))
    out = []
    for c, m in enumerate(counts):
        mp = maps[c, :2 + m + bw * bh].copy()
        mp[0] = block
        mp[1] = m
        mp[2:2 + m] = ids[:m]
        out.append(mp)
    return out


def _nlog2n_table(n):
    t = np.zeros(n + 1, dtype=np.int64)
    for c in range(2, n + 1):
        t[c] = int(math.floor(256.0 * c * math.log(c) / _LOG2 + 0.5))
    return t


@njit
def _greedy_cost_gains(est, blocks, cur, assign, count, chosen, n, NL, map_old, id_cost, p0, p1, gain):
    """Gains of predictors p0..p1-1, into gain."""
    P = chosen.shape[0]
    moved_from = np.zeros(n, dtype=np.int64)
    for p in range(p0, p1):
        if chosen[p]:
            continue
        moved_from[:] = 0
        save = 0
        moved = 0
        for b in range(blocks):
            e = est[b * P + p]
            if e < cur[b]:
                save += cur[b] - e
                moved_from[assign[b]] += 1
                moved += 1
        map_new = NL[blocks] - NL[moved]
        for q in range(n):
            map_new -= NL[count[q] - moved_from[q]]
        gain[p] = save - (map_new - map_old) - id_cost


def greedy_cost_order(est, blocks, id_cost=None):
    """Predictor ids in greedy order over a block_cost_table of every id
    (DeltaMapper.greedyCostOrder); id_cost, the cost of an id in the map
    header, is scaled down when est covers only a sample of the blocks."""
    id_cost = SURVEY_ID_COST if id_cost is None else id_cost
    P = PREDICTORS
    NL = _nlog2n_table(blocks)
    table = est.reshape(blocks, P)
    total = table.sum(axis=0, dtype=np.int64)
    first = int(np.argmin(total))
    order = [first]
    chosen = np.zeros(P, dtype=np.bool_)
    chosen[first] = True
    cur = table[:, first].astype(np.int64)
    assign = np.zeros(blocks, dtype=np.int64)
    count = np.zeros(P, dtype=np.int64)
    count[0] = blocks
    while len(order) < min(P, 255):     # a set has at most 255 ids
        n = len(order)
        map_old = int(NL[blocks]) - int(NL[count[:n]].sum())   # (n numbers: cheap)
        gain = np.zeros(P, dtype=np.int64)
        _in_bands(P, lambda lo, hi: _greedy_cost_gains(est, blocks, cur, assign, count, chosen, n, NL, map_old,
                                                       id_cost, lo, hi, gain))
        best = _greedy_step(est, P, blocks, gain, chosen, n, cur, assign, count)
        if best < 0:
            break
        order.append(best)
    return order


@njit
def _greedy_step(est, P, blocks, gain, chosen, n, cur, assign, count):
    """The predictor with the largest gain (first on ties), or -1 if no gain
    is positive; else adds it: blocks it estimates cheaper move to it (index n)."""
    best = -1
    for p in range(P):
        if not chosen[p] and (best < 0 or gain[p] > gain[best]):
            best = p
    if gain[best] <= 0:
        return -1
    chosen[best] = True
    for b in range(blocks):
        e = est[b * P + best]
        if e < cur[b]:
            cur[b] = e
            count[assign[b]] -= 1
            assign[b] = n
            count[n] += 1
    return best


def _assign_by_cost(est, stride, cols, ids, xdim, ydim, block):
    """A block map for ids: each block uses the first of them with the
    smallest estimate (cols: each id's column in est)."""
    bw = blocks_across(xdim, block)
    bh = blocks_down(ydim, block)
    n = len(ids)
    sub = est.reshape(bw * bh, stride)[:, cols]
    choice = np.argmin(sub, axis=1)          # first minimum
    mp = np.zeros(2 + n + bw * bh, dtype=np.uint8)
    mp[0] = block
    mp[1] = n
    mp[2:2 + n] = ids
    mp[2 + n:] = choice
    return mp


def pixel_sources(channel, xdim, ydim):
    """Each channel's PixelSource, with the channel before it as reference."""
    ps = [None] * len(channel)

    def one(i):
        ps[i] = PixelSource(channel[i], channel[i - 1] if i > 0 else None, xdim, ydim)
    for f in [_ROW_POOL.submit(one, i) for i in range(len(channel))]:
        f.result()
    return ps


def _channel_cost(ps, block, chosen_before):
    ref, _ = _block_deltas_from_values(ps, block, get_set_ids(SURVEY_REFERENCE_SET))
    return DeltaCost(ps.v, ref, chosen_before, ps.xdim, ps.ydim)


def get_block_deltas_by_cost(channel, xdim, ydim, block, ids):
    """Each channel's (deltas, map) for its predictor ids, blocks assigned
    by estimated cost as the survey does (DeltaMapper.getBlockDeltasByCost)."""
    result, delta = [], []
    ps = pixel_sources(channel, xdim, ydim)
    for i in range(len(channel)):
        dc = _channel_cost(ps[i], block, delta[:i])
        mp = prefix_maps(ps[i], block, dc, ids[i], [len(ids[i])])[0]
        d = get_block_deltas_from_map(ps[i], mp, ids[i])
        result.append((d, mp))
        delta.append(d)
    return result


class BlockSurvey:
    def __init__(self, block):
        self.block = block
        self.ids = [None] * 3
        self.bytes = 0
        self.report = ""
        self.tried = None     # the full survey: every shape tried, by shape


def _survey_block_size(ps, block, log):
    xdim, ydim = ps[0].xdim, ps[0].ydim
    result = BlockSurvey(block)
    delta, mp = [None] * 3, [None] * 3
    bw = blocks_across(xdim, block)
    bh = blocks_down(ydim, block)
    blocks = bw * bh
    # Big images: the greedy order is found on every step-th block row (at
    # most SURVEY_SAMPLE_BLOCKS blocks), the id's cost scaled to match; the
    # maps are then made over every block.
    step = (blocks + SURVEY_SAMPLE_BLOCKS - 1) // SURVEY_SAMPLE_BLOCKS if blocks > SURVEY_SAMPLE_BLOCKS else 1
    rows = np.arange(0, bh, step, dtype=np.int64)
    sampled = len(rows) * bw
    log.append("  block %s%s:" % (shape_name(block), " (order from %s of %s blocks)" % (format(sampled, ","), format(blocks, ",")) if step > 1 else ""))
    for i in range(3):
        dc = _channel_cost(ps[i], block, delta[:i])
        est = block_cost_table(ps[i], block, dc, rows=rows)
        order = greedy_cost_order(est, sampled, SURVEY_ID_COST * sampled // blocks)
        counts = [c for c in SURVEY_COUNTS if c < len(order)] + [len(order)]
        prefix = None
        if step > 1:
            est = None
            prefix = prefix_maps(ps[i], block, dc, order, counts)
        best_bytes, worse = None, 0
        line = "    channel %d (order %d):" % (i, len(order))
        for c, n in enumerate(counts):
            ids = order[:n]
            m = prefix[c] if prefix is not None else _assign_by_cost(est, PREDICTORS, np.array(ids, dtype=np.int64), ids, xdim, ydim, block)
            d = get_block_deltas_from_map(ps[i], m, ids)
            size = map_bytes(13, m, mp[i - 1] if i > 0 else None, xdim) + len(pack_context_deltas(d, delta[:i], xdim))
            line += " %d:%s" % (n, format(size, ","))
            if best_bytes is None or size < best_bytes:
                best_bytes, result.ids[i], delta[i], mp[i], worse = size, list(ids), d, m, 0
                line += "*"
            else:
                worse += 1
                if worse == 2:
                    break
        result.bytes += best_bytes
        log.append(line)
    return result


def survey_at_block_size(channel, xdim, ydim, block):
    """The survey at one block shape only: the same sets survey_block_sets
    would choose for that shape (DeltaMapper.surveyAtBlockSize)."""
    return _survey_block_size(pixel_sources(channel, xdim, ydim), block, [])


def survey_block_sets(channel, xdim, ydim, survey_shapes=False):
    """The block size and each channel's predictor ids that code smallest
    (DeltaMapper.surveyBlockSets): per channel, predictors in greedy order
    by estimated coded bits (with the map's cost), with the real coded size
    measured for the first n at SURVEY_COUNTS until two in a row are larger
    than the best; block sizes 4, 8, 16, then the untried sizes next to the
    best; with survey_shapes, then the rectangles and lines (BLOCK_SHAPES)."""
    tried, log = {}, []
    ps = pixel_sources(channel, xdim, ydim)

    blocks4 = blocks_across(xdim, 4) * ((ydim - 1 + 3) // 4)
    task = 160 * xdim * ydim + 4 * PREDICTORS * min(SURVEY_SAMPLE_BLOCKS, blocks4)

    def survey_sizes(sizes):
        # The block sizes are independent: surveyed at the same time, as many
        # as memory allows.
        logs = {b: [] for b in sizes}
        got = dict(zip(sizes, _bounded_map(lambda b: _survey_block_size(ps, b, logs[b]), sizes, task)))
        for b in sizes:
            tried[b] = got[b]
            log.extend(logs[b])
    survey_sizes([4, 8, 16])

    def best_of():
        best = None
        for b in sorted(tried):
            if best is None or tried[b].bytes < tried[best].bytes:
                best = b
        return best
    s = BLOCK_SEARCH_SIZES.index(best_of())
    more = [BLOCK_SEARCH_SIZES[t] for t in (s - 1, s + 1) if 0 <= t < len(BLOCK_SEARCH_SIZES) and BLOCK_SEARCH_SIZES[t] not in tried]
    if more:
        survey_sizes(more)
    if survey_shapes:
        survey_sizes([SHAPE_FIRST + t for t in range(len(BLOCK_SHAPES))])
    result = tried[best_of()]
    r = ["Predictor survey: bytes coded (Context-coded deltas + maps) for the first n predictors",
         "in each channel's greedy order (by estimated bits); * marks the n kept for the channel."] + log
    r.append("Block size:" + "".join("  %s -> %s" % (shape_name(b), format(tried[b].bytes, ",")) for b in sorted(tried))
             + "   chose %s (%s bytes)" % (shape_name(result.block), format(result.bytes, ",")))
    for i in range(3):
        names = [predictor_name(p) for p in result.ids[i]]
        r.append("Channel %d, %d predictors:" % (i, len(names)))
        for k in range(0, len(names), 6):
            r.append("    " + ", ".join(names[k:k + 6]))
    result.report = "\n".join(r) + "\n"
    result.tried = tried
    return result


def describe_group_use(survey, coded):
    """Which primitives past the basic 22 a survey's sets use, by group
    (DeltaMapper.describeGroupUse); coded: get_block_deltas_by_cost's result."""
    total = 0
    blocks = np.zeros((3, PREDICTORS), dtype=np.int64)
    for i in range(3):
        mp = coded[i][1]
        idx = np.asarray(survey.ids[i], dtype=np.int64)[np.asarray(mp[map_header_length(mp):], dtype=np.int64)]
        blocks[i] += np.bincount(idx, minlength=PREDICTORS)
        total += len(idx)
    r = ["Predictors past the basic 22 in the surveyed sets (share of all blocks):"]
    for g, gname in enumerate(GROUP_NAMES):
        lo, hi = GROUP_FIRST[g], GROUP_FIRST[g + 1]
        anyb = 0
        for i in range(3):
            for pid in survey.ids[i]:
                if any(table_uses(pid, p) for p in range(lo, hi)):
                    anyb += int(blocks[i][pid])
        used, unused = [], []
        for p in range(lo, hi):
            if not any(table_uses(pid, p) for pid in range(PREDICTORS)):
                continue
            alone = avg = chosen = 0
            for i in range(3):
                for pid in survey.ids[i]:
                    if table_uses(pid, p):
                        chosen += 1
                        if TABLE_FIRST[pid] == TABLE_SECOND[pid]:
                            alone += int(blocks[i][pid])
                        else:
                            avg += int(blocks[i][pid])
            if chosen == 0:
                unused.append(primitive_name(p))
            else:
                used.append("      %-20s %2d ids chosen, %5.2f%% alone, %5.2f%% in averages"
                            % (primitive_name(p), chosen, 100.0 * alone / total, 100.0 * avg / total))
        if not used:
            r.append("  %s: unused." % gname)
        else:
            r.append("  %s: used, in %.2f%% of blocks." % (gname, 100.0 * anyb / total))
            r += used
            if unused:
                r.append("      unused: " + ", ".join(unused))
    return "\n".join(r) + "\n"


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
