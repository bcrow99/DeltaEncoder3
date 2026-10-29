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

import threading

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


@njit
def _pack_image(c0, c1, c2, amount, out):
    """out[k] = each channel's value at k shifted by amount (shift) and
    clamped to 0..255."""
    for k in range(c0.shape[0]):
        for c in range(3):
            v = c0[k] if c == 0 else (c1[k] if c == 1 else c2[k])
            v = v >> -amount if amount < 0 else v << amount
            out[k, c] = 0 if v < 0 else (255 if v > 255 else v)


def pack_image(channels, xdim, ydim, amount):
    """The three channels shifted by amount (shift) and clamped to 0..255, as
    an (ydim, xdim, 3) uint8 image, in one pass (np.stack of
    np.clip(shift(...)) gives the same, several times as slowly)."""
    c = [np.ascontiguousarray(v, dtype=np.int64) for v in channels]
    out = np.zeros((xdim * ydim, 3), dtype=np.uint8)
    _pack_image(c[0], c[1], c[2], amount, out)
    return out.reshape(ydim, xdim, 3)


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
# before it (ref). The sets and the blocks' choices come from the survey
# (survey_channels), from the whole table, by estimated coded size.
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
# cross-channel; 36-39 longer reach; 40-41 clamped; 42-45 bias-corrected;
# 46 least squares (LS: see _ls_next); 47 the same fit with a wider clamp (LS
# wide: 2 beyond the range of all 8 neighbours rather than of a, b, c, d).
PRIMITIVES = 22
SELECTION_FIRST, CROSS_FIRST, REACH_FIRST, CLAMP_FIRST, BIAS_FIRST, LS_FIRST, ALL_PRIMITIVES = 22, 28, 36, 40, 42, 46, 48
LS_WIDE = 47
LS_RADIUS = 10   # the least-squares window: rows above, columns either side, pixels to the left
GROUP_NAMES = ["selection", "cross-channel", "longer reach", "clamped", "bias-corrected", "least squares"]
GROUP_FIRST = [SELECTION_FIRST, CROSS_FIRST, REACH_FIRST, CLAMP_FIRST, BIAS_FIRST, LS_FIRST, ALL_PRIMITIVES]
PRIMITIVE_NAMES = [
    "a", "b", "c", "d", "(a+b)/2", "(b+c)/2", "(a+c)/2", "(b+d)/2",
    "(c+d)/2", "(a+b+c+d)/4", "a+b-c", "MED", "(3a+b)/4", "(a+3b)/4", "(3a+d)/4", "(3b+a)/4",
    "(a+d)/2", "a+(b-c)/2", "b+(a-c)/2", "a+d-b", "(a+b+d)/3", "b+(d-c)/2"]
EXTRA_PRIMITIVE_NAMES = [
    "Paeth", "median(a,b,c)", "median(a,b,d)", "median(a,d,a+d-b)", "switch(a,b)", "switch(a,b,d)",
    "a+(R-Ra)", "b+(R-Rb)", "d+(R-Rd)", "MED+(R-MED_R)", "(a+b)/2+(R-..)", "a+b-c+(R-..)", "MED+(R-MED_R)/2", "(a+b)/2+(R-..)/2",
    "2a-aa", "2b-bb", "GAP", "GAP/8",
    "clamp(a+b-c)", "clamp(a+d-b)",
    "MED+bias", "(a+b)/2+bias", "a+bias", "b+bias",
    "LS", "LS wide"]
CROSS_BASE = np.array([0, 1, 3, 11, 4, 10, 11, 4], dtype=np.int64)   # the last two with half the error
BIAS_BASE = np.array([11, 4, 0, 1], dtype=np.int64)

# PREDICTOR_TABLE_256 as pairs of primitives (equal for a primitive alone):
# the table of Block formats 3 to 8 and Delta, Packet and Simple formats 3
# and 4; id 255 is LS.
PREDICTOR_PAIRS_256 = [
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
    45, 23, 45, 34, 45, 39, 46, 46]
# PREDICTOR_TABLE (DeltaMapper.PREDICTOR_PAIRS): 225 entries, the 224 of the
# 256 that BlockWriter's survey chose at least once over 19 test images at four
# settings, in their old order (LS last, 223), then LS wide (224).
PREDICTOR_PAIRS = [
    0, 0, 1, 1, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 11, 11, 13, 13, 16, 16,
    19, 19, 23, 23, 24, 24, 25, 25, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 34, 34, 35, 35, 39, 39,
    40, 40, 42, 42, 45, 45, 2, 0, 2, 1, 3, 0, 4, 0, 4, 3, 5, 0, 5, 1, 5, 4, 6, 0,
    6, 1, 6, 2, 6, 4, 7, 0, 7, 4, 7, 5, 7, 6, 8, 0, 8, 1, 8, 4, 9, 4, 11, 0,
    11, 1, 11, 7, 11, 8, 13, 0, 16, 0, 16, 1, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7, 16, 9,
    16, 11, 18, 0, 18, 1, 18, 4, 18, 5, 18, 6, 18, 7, 18, 8, 18, 9, 18, 13, 18, 16, 19, 3,
    19, 18, 23, 0, 23, 2, 23, 4, 23, 5, 23, 6, 24, 0, 24, 1, 24, 4, 24, 5, 24, 6, 24, 7,
    24, 9, 24, 18, 24, 23, 25, 1, 25, 3, 25, 5, 28, 0, 28, 5, 28, 6, 28, 7, 28, 8, 28, 16,
    29, 1, 29, 4, 29, 5, 29, 6, 29, 7, 29, 8, 29, 16, 29, 28, 30, 6, 30, 7, 30, 16, 30, 25,
    30, 28, 30, 29, 31, 5, 31, 6, 31, 11, 31, 16, 32, 4, 32, 5, 32, 6, 32, 7, 32, 8, 32, 9,
    32, 16, 32, 28, 32, 30, 34, 0, 34, 2, 34, 3, 34, 6, 34, 7, 34, 8, 34, 16, 34, 23, 34, 25,
    34, 28, 34, 29, 34, 30, 34, 31, 35, 0, 35, 1, 35, 2, 35, 3, 35, 4, 35, 5, 35, 6, 35, 7,
    35, 8, 35, 9, 35, 16, 35, 23, 35, 24, 35, 28, 35, 29, 35, 30, 36, 4, 36, 11, 36, 16, 36, 25,
    37, 9, 37, 13, 37, 16, 37, 18, 37, 19, 37, 25, 39, 0, 39, 1, 39, 2, 39, 3, 39, 4, 39, 5,
    39, 6, 39, 7, 39, 8, 39, 9, 39, 11, 39, 13, 39, 16, 39, 23, 39, 24, 39, 25, 39, 28, 39, 29,
    39, 30, 39, 31, 39, 32, 39, 34, 39, 35, 39, 36, 39, 37, 40, 0, 40, 1, 40, 3, 40, 7, 40, 11,
    40, 16, 40, 19, 40, 25, 40, 34, 40, 36, 40, 37, 40, 39, 42, 0, 42, 1, 42, 3, 42, 4, 42, 6,
    42, 7, 42, 16, 42, 18, 42, 23, 42, 29, 42, 34, 42, 35, 42, 36, 42, 37, 42, 39, 42, 40, 45, 0,
    45, 4, 45, 5, 45, 6, 45, 7, 45, 16, 45, 34, 45, 39, 46, 46, 47, 47]
TABLE_FIRST = np.array(PREDICTOR_PAIRS[0::2], dtype=np.int64)
TABLE_SECOND = np.array(PREDICTOR_PAIRS[1::2], dtype=np.int64)
PREDICTORS = len(TABLE_FIRST)   # 225
TABLE = (TABLE_FIRST, TABLE_SECOND)
TABLE_256 = (np.array(PREDICTOR_PAIRS_256[0::2], dtype=np.int64), np.array(PREDICTOR_PAIRS_256[1::2], dtype=np.int64))
MAX_PREDICTORS = 256   # the most ids a block map may carry, one more than the largest id
TABLE_PRIMITIVES = int(max(TABLE_FIRST.max(), TABLE_SECOND.max())) + 1


def primitive_name(p):
    return PRIMITIVE_NAMES[p] if p < PRIMITIVES else EXTRA_PRIMITIVE_NAMES[p - PRIMITIVES]


def predictor_name(pid, table=None):
    first, second = TABLE if table is None else table
    i, j = int(first[pid]), int(second[pid])
    return primitive_name(i) if i == j else "avg(%s, %s)" % (primitive_name(i), primitive_name(j))


def table_uses(pid, p):
    return int(TABLE_FIRST[pid]) == p or int(TABLE_SECOND[pid]) == p


def map_header_length(mp):
    return 2 + int(mp[1])


def get_map_ids(mp):
    return [int(v) for v in mp[2:map_header_length(mp)]]


def channel_name(set_id, i):
    """A channel's name in a channel set: blue, green, red, blue-green,
    red-green or red-blue (DeltaMapper.channelName)."""
    return SET_NAMES[set_id].split(", ")[i]


def describe_block_size(set_id, i, mp, xdim):
    """A channel's block map in a line: "Channel 0 - blue: 8 x 8, 2,400 blocks"
    (DeltaMapper.describeBlockSize)."""
    block = int(mp[0])
    return "Channel %d - %s: %d x %d, %s blocks" % (i, channel_name(set_id, i), block_width(block, xdim), block_height(block),
                                                     format(len(mp) - map_header_length(mp), ","))


def describe_block_predictors(mp, indent, table=None):
    """The predictors a block map's blocks use, most used first (ties in the
    map's order), one per line: the number of blocks, then the name
    (DeltaMapper.describeBlockPredictors)."""
    mp = np.asarray(mp, dtype=np.uint8)
    ids = get_map_ids(mp)
    count = np.bincount(mp[map_header_length(mp):], minlength=len(ids))
    order = sorted(range(len(ids)), key=lambda p: (-int(count[p]), p))
    return "".join("%s%9s  %s\n" % (indent, format(int(count[p]), ","), predictor_name(ids[p], table)) for p in order if count[p] > 0)


def least_squares_share(maps, xdim, ydim):
    """The share of the channels' predicted pixels (all but the first row and
    the first and last columns) whose block's predictor uses least squares (LS
    or LS wide, alone or averaged) (DeltaMapper.leastSquaresShare)."""
    ls = total = 0
    for mp in maps:
        mp = np.asarray(mp, dtype=np.int64)
        block = int(mp[0])
        bwd, bht = _block_wh(block, xdim)
        bw, bh = blocks_across(xdim, block), blocks_down(ydim, block)
        head = map_header_length(mp)
        pid = mp[2 + mp[head:head + bw * bh]].reshape(bh, bw)
        width = np.minimum(bwd, xdim - 2 - np.arange(bw) * bwd)
        height = np.minimum(bht, ydim - 1 - np.arange(bh) * bht)
        area = height[:, None] * width[None, :]
        uses = (TABLE_FIRST[pid] >= LS_FIRST) | (TABLE_SECOND[pid] >= LS_FIRST)
        ls += int(area[uses].sum())
        total += int(area.sum())
    return ls / total if total else 0.0


def describe_block_maps(set_id, maps, xdim, table=None):
    """The channel set, then each channel's block size, number of blocks and
    predictors (DeltaMapper.describeBlockMaps); table: the one the maps' ids
    index (TABLE_256 for older files)."""
    t = "Channel set: " + SET_NAMES[set_id] + "\n"
    for i, mp in enumerate(maps):
        t += describe_block_size(set_id, i, mp, xdim) + "\n" + describe_block_predictors(mp, "  ", table)
    return t


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


@njit(inline="always")   # a call per pixel passing arrays is slow in Numba
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


@njit(inline="always")   # a call per pixel passing arrays is slow in Numba
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


# ---- The least-squares primitive (DeltaMapper.LeastSquares) -------------------
#
# At each pixel, the linear predictor (with a constant term) that best fits
# the pixels in a causal window -- the r rows above, r columns either side,
# and the r pixels to the left (r = LS_RADIUS for the table's LS) -- from 8
# neighbours and, after the first channel, the previous channel at the
# pixel, to its left and above. The window's sums are exact integers, kept
# as it slides: per column, the sums of the window's rows; the current
# row's pixels' products are kept (one row of them) and added to the columns
# when the row joins them, and computed again when the row leaves them, so
# the memory moved through per row stays small; the fit (ridge 1, solved by L D L^T) is done in the
# Java's order, so both get the same values.
# _ls_next must be called for every interior pixel in raster order, from
# row 1 or from a row given to _ls_start; because the sums are exact, the
# encoder computes bands of rows independently (_ls_plane).

@njit(inline="always")   # a call per pixel passing arrays is slow in Numba
def _ls_at(v, y, x, xdim):
    if y < 0:
        y = 0
    if x < 0:
        x = 0
    elif x > xdim - 1:
        x = xdim - 1
    return v[y * xdim + x]


@njit(inline="always")   # a call per pixel passing arrays is slow in Numba
def _ls_features(v, ref, has_ref, xdim, nf, y, x, z):
    z[0] = _ls_at(v, y, x - 1, xdim); z[1] = _ls_at(v, y - 1, x, xdim)
    z[2] = _ls_at(v, y - 1, x - 1, xdim); z[3] = _ls_at(v, y - 1, x + 1, xdim)
    z[4] = _ls_at(v, y, x - 2, xdim); z[5] = _ls_at(v, y - 2, x, xdim)
    z[6] = _ls_at(v, y - 1, x + 2, xdim); z[7] = _ls_at(v, y - 2, x + 1, xdim)
    if has_ref:
        k = y * xdim + x
        z[8] = ref[k]; z[9] = ref[k - 1]; z[10] = ref[k - xdim]
    z[nf] = 1
    z[nf + 1] = v[y * xdim + x]


@njit(inline="always")   # a call per pixel passing arrays is slow in Numba
def _ls_slot(x, nt):
    """Where the products of the current row's pixel x are kept in ring (by column)."""
    return x * nt


@njit(inline="always")   # a call per pixel passing arrays is slow in Numba
def _ls_products(v, ref, has_ref, xdim, nf, y, x, z, out, at, row):
    """Pixel (y, x)'s products into out from at. The fit at a pixel leaves
    its features in z (row[2] is the pixel): only its value (unknown to a
    decoder then) is needed for its products."""
    k = y * xdim + x
    if row[2] == k:
        z[nf + 1] = v[k]
    else:
        _ls_features(v, ref, has_ref, xdim, nf, y, x, z)
        row[2] = k
    m = nf + 2
    t = at
    for i in range(m):
        zi = z[i]
        for j in range(i, m):
            out[t] = zi * z[j]
            t += 1


@njit
def _ls_state(xdim, has_ref, r):
    """(row, col, win, ring, z, a, b, coef, dg): the window's state
    (DeltaMapper.LeastSquares). row holds the row of the last pixel, the row
    whose products ring holds, the pixel whose features z holds (-1 for
    none) and whether coef holds a solution from this row (for _ls_next's
    every > 1); col per column the window rows' sums of products (an int32 holds
    r rows of them for any radius up to 255); win the window's sums (the
    columns' part and the current row's); ring the products of the current
    row's pixels so far, by column (kept so that when the row joins the
    columns they needn't be computed again), and after them room for one
    pixel's products computed again (when its row joins or leaves the
    columns)."""
    nf = 11 if has_ref else 8
    m = nf + 2
    nt = m * (m + 1) // 2
    return (np.full(4, -1, dtype=np.int64), np.zeros(xdim * nt, dtype=np.int32), np.zeros(nt, dtype=np.int64),
            np.zeros((xdim + 1) * nt, dtype=np.int32), np.zeros(m, dtype=np.int64),
            np.zeros((nf + 1, nf + 1)), np.zeros(nf + 1), np.zeros(nf + 1), np.zeros(nf + 1))


@njit
def _ls_start(v, ref, has_ref, xdim, y0, row, col, ring, z, r):
    """Sets the window up as if the rows before y0 had been gone through:
    the columns hold rows y0 - 1 - r .. y0 - 2 (from 1); row y0 - 1 joins
    them at the first _ls_next."""
    nf = 11 if has_ref else 8
    m = nf + 2
    nt = m * (m + 1) // 2
    gone = xdim * nt
    for ty in range(max(1, y0 - 1 - r), y0 - 1):
        for tx in range(1, xdim - 1):
            _ls_products(v, ref, has_ref, xdim, nf, ty, tx, z, ring, gone, row)
            c = tx * nt
            for t in range(nt):
                col[c + t] += ring[gone + t]
    row[0] = y0 - 1


@njit(inline="always")   # a call per pixel passing arrays is slow in Numba
def _ls_next(v, ref, has_ref, xdim, k, row, col, win, ring, z, a, b, coef, dg, fit, r, every):
    """Moves the window (radius r) to interior pixel k and, if fit, returns
    the least-squares primitives there, (LS, LS wide) (the decoder fits only
    where they are used), solving the fit at every every-th pixel along a
    row (1: at every pixel)."""
    nf = 11 if has_ref else 8
    m = nf + 2
    nt = m * (m + 1) // 2
    y = k // xdim
    x = k % xdim
    # The sums are indexed with unsigned offsets: computed signed ones make
    # Numba check each index for wraparound, about 3 times as slow.
    if y != row[0]:
        # The window's rows become y - r .. y - 1 (from 1): add row y - 1
        # (its products kept in ring as it went by, but for its last pixel's,
        # which no _ls_next adds to the current row, and unless the window
        # started at row y), drop row y - 1 - r (its products computed again,
        # into the room after the row's pixels in ring).
        gone = xdim * nt
        ug = np.uint64(gone)
        if y - 1 >= 1:
            kept = row[1] == y - 1
            for tx in range(1, xdim - 1):
                c = np.uint64(tx * nt)
                if kept and tx != xdim - 2:
                    for t in range(nt):
                        ut = np.uint64(t)
                        col[c + ut] += ring[c + ut]
                else:
                    _ls_products(v, ref, has_ref, xdim, nf, y - 1, tx, z, ring, gone, row)
                    for t in range(nt):
                        ut = np.uint64(t)
                        col[c + ut] += ring[ug + ut]
        if y - 1 - r >= 1:
            for tx in range(1, xdim - 1):
                _ls_products(v, ref, has_ref, xdim, nf, y - 1 - r, tx, z, ring, gone, row)
                c = np.uint64(tx * nt)
                for t in range(nt):
                    ut = np.uint64(t)
                    col[c + ut] -= ring[ug + ut]
        row[0] = y
        row[1] = y
        row[3] = 0
        win[:] = 0
        for tx in range(1, min(xdim - 2, 1 + r) + 1):
            c = np.uint64(tx * nt)
            for t in range(nt):
                ut = np.uint64(t)
                win[ut] += col[c + ut]
    else:
        if x + r <= xdim - 2:
            c = np.uint64((x + r) * nt)
            for t in range(nt):
                ut = np.uint64(t)
                win[ut] += col[c + ut]
        if x - r - 1 >= 1:
            c = np.uint64((x - r - 1) * nt)
            for t in range(nt):
                ut = np.uint64(t)
                win[ut] -= col[c + ut]
        _ls_products(v, ref, has_ref, xdim, nf, y, x - 1, z, ring, _ls_slot(x - 1, nt), row)
        sl = np.uint64(_ls_slot(x - 1, nt))
        for t in range(nt):
            ut = np.uint64(t)
            win[ut] += ring[sl + ut]
        if x - 1 - r >= 1:
            sl = np.uint64(_ls_slot(x - 1 - r, nt))
            for t in range(nt):
                ut = np.uint64(t)
                win[ut] -= ring[sl + ut]
    if not fit:
        return 0, 0

    n = nf + 1
    pa = _ls_at(v, y, x - 1, xdim); pb = _ls_at(v, y - 1, x, xdim)
    pc = _ls_at(v, y - 1, x - 1, xdim); pd = _ls_at(v, y - 1, x + 1, xdim)
    # The fit is solved at every every-th pixel along a row (row[3]: coef
    # holds a solution from this row), the pixels between using the last.
    if row[3] != 1 or (x - 1) % every == 0:
        row[3] = 0
        cnt = win[nf * m - nf * (nf - 1) // 2]
        if cnt < 2 * n:
            return pa, pa
        # The normal equations' lower triangle (ridge 1), then L D L^T as in the Java.
        for i in range(n):
            base = i * m - i * (i - 1) // 2
            for j in range(i, n):
                a[j, i] = float(win[base + j - i])
            b[i] = float(win[base + m - 1 - i])
            a[i, i] += 1.0
        for j in range(n):
            sv = a[j, j]
            for q in range(j):
                sv -= a[j, q] * a[j, q] * dg[q]
            if sv < 1e-9:
                return pa, pa
            dg[j] = sv
            for i in range(j + 1, n):
                t_ = a[i, j]
                for q in range(j):
                    t_ -= a[i, q] * a[j, q] * dg[q]
                a[i, j] = t_ / sv
        for i in range(n):
            sv = b[i]
            for q in range(i):
                sv -= a[i, q] * coef[q]
            coef[i] = sv
        for i in range(n):
            coef[i] /= dg[i]
        for i in range(n - 1, -1, -1):
            sv = coef[i]
            for q in range(i + 1, n):
                sv -= a[q, i] * coef[q]
            coef[i] = sv
        if every > 1:
            row[3] = 1
    _ls_features(v, ref, has_ref, xdim, nf, y, x, z)
    row[2] = k
    p = 0.0
    for i in range(n):
        p += coef[i] * z[i]
    lo = min(min(pa, pb), min(pc, pd))
    hi = max(max(pa, pb), max(pc, pd))
    wlo = lo
    whi = hi
    for i in range(4, 8):
        wlo = min(wlo, z[i])
        whi = max(whi, z[i])
    w = max(wlo - 2, min(whi + 2, p))
    if p < lo - 2:
        p = lo - 2
    if p > hi + 2:
        p = hi + 2
    return int(np.floor(p + 0.5)), int(np.floor(w + 0.5))


@njit
def _ls_band(v, ref, has_ref, xdim, y0, y1, out, wide, r, every):
    row, col, win, ring, z, a, b, coef, dg = _ls_state(xdim, has_ref, r)
    _ls_start(v, ref, has_ref, xdim, y0, row, col, ring, z, r)
    for y in range(y0, y1):
        for x in range(1, xdim - 1):
            k = y * xdim + x
            out[k], wide[k] = _ls_next(v, ref, has_ref, xdim, k, row, col, win, ring, z, a, b, coef, dg, True, r, every)


_LS_CACHE = []        # recently computed planes: (xdim, radius, channel copy, previous channel copy or None, (LS, LS wide))
_LS_CACHE_SIZE = 3    # a channel set's three
_LS_CACHE_LOCK = threading.Lock()
_LS_POOL = ThreadPoolExecutor(_CPUS)   # its own: PixelSources are made in _ROW_POOL's threads


def _ls_planes(v, ref, has_ref, xdim, ydim, radius, every=1):
    """(LS, LS wide) for the whole channel with the window of the given
    radius, in bands of rows computed in parallel (DeltaMapper.LeastSquares.plane)."""
    out = np.zeros(v.shape[0], dtype=np.int16)
    wide = np.zeros(v.shape[0], dtype=np.int16)
    band = max(4 * (radius + 1), (ydim - 1 + _CPUS - 1) // _CPUS)
    for f in [_LS_POOL.submit(_ls_band, v, ref, has_ref, xdim, y0, min(ydim, y0 + band), out, wide, radius, every) for y0 in range(1, ydim, band)]:
        f.result()
    return out, wide


def _ls_plane(v, ref, has_ref, xdim, ydim, radius=LS_RADIUS):
    """The encoder's least-squares primitives for the whole channel, (LS, LS
    wide), with the window of the given radius (DeltaMapper.leastSquaresPlanes):
    bands of rows in parallel, reused if the same channel (and previous
    channel) was just done."""
    r = ref if has_ref else None
    with _LS_CACHE_LOCK:
        for xd, rd, cv, cr, plane in _LS_CACHE:
            if xd == xdim and rd == radius and np.array_equal(cv, v) and ((cr is None and r is None) or (cr is not None and r is not None and np.array_equal(cr, r))):
                return plane
    out = _ls_planes(v, ref, has_ref, xdim, ydim, radius)
    with _LS_CACHE_LOCK:
        _LS_CACHE.append((xdim, radius, v.copy(), None if r is None else r.copy(), out))
        del _LS_CACHE[:-_LS_CACHE_SIZE]
    return out


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
    if q == LS_WIDE:
        return np.int64(corr[5, k])
    if q >= LS_FIRST:
        return np.int64(corr[4, k])
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
    prim[46] = np.int64(corr[4, k]) if corr.shape[0] > 4 else a   # LS (row 4: see PixelSource)
    prim[47] = np.int64(corr[5, k]) if corr.shape[0] > 5 else a   # LS wide (row 5)


_CROSS_BASE_T = tuple(int(b) for b in CROSS_BASE)   # as constants in _primitive_value
_BIAS_BASE_T = tuple(int(b) for b in BIAS_BASE)


@njit
def _primitive_value(q, a, b, c, d, aa, bb, nne, has_ref, r, ra, rb, rc, rd, c0, c1, c2, c3, ls, lsw):
    """Primitive q from the neighbours already read (scalars only: a call
    passing arrays per pixel is slow in Numba): a, b, c, d; aa, bb, nne as
    _all_primitives has them; the previous channel's r, ra, rb, rc, rd (if
    has_ref); bias corrections c0-c3; LS and LS wide. The same values as
    _primitive."""
    if q < CROSS_FIRST:
        return _block_predictor(q, a, b, c, d)
    if q < REACH_FIRST:
        s = q - CROSS_FIRST
        base = _CROSS_BASE_T[s]
        p = _block_predictor(base, a, b, c, d)
        if not has_ref:
            return p
        e = r - _block_predictor(base, ra, rb, rc, rd)
        return p + (e >> 1) if s >= 6 else p + e
    if q < CLAMP_FIRST:
        if q == REACH_FIRST:
            return 2 * a - aa
        if q == REACH_FIRST + 1:
            return 2 * b - bb
        if q == REACH_FIRST + 2:
            return _gap(a, aa, b, bb, c, d, nne, 80, 32, 8)
        return _gap(a, aa, b, bb, c, d, nne, 10, 4, 1)
    if q < BIAS_FIRST:
        lo = min(min(a, b), min(c, d))
        hi = max(max(a, b), max(c, d))
        p = a + b - c if q == CLAMP_FIRST else a + d - b
        return max(lo, min(hi, p))
    if q == LS_WIDE:
        return lsw
    if q >= LS_FIRST:
        return ls
    s = q - BIAS_FIRST
    cs = c0 if s == 0 else (c1 if s == 1 else (c2 if s == 2 else c3))
    return _block_predictor(_BIAS_BASE_T[s], a, b, c, d) + cs


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
    encoder side): the channel, the previous channel, the bias corrections
    (corr rows 0-3) and the least-squares primitives (rows 4, LS, and 5, LS
    wide)."""

    def __init__(self, v, ref, xdim, ydim):
        self.v = np.asarray(v, dtype=np.int64)
        self.has_ref = ref is not None
        self.ref = np.asarray(ref, dtype=np.int64) if ref is not None else np.zeros(1, dtype=np.int64)
        self.xdim, self.ydim = xdim, ydim
        corr = _bias_corrections(self.v, xdim, ydim, BIAS_BASE)
        ls, wide = _ls_plane(self.v, self.ref, self.has_ref, xdim, ydim)
        self.corr = np.concatenate([corr, ls[None, :], wide[None, :]])


@njit
def _block_from_map(src, ref, has_ref, corr, xdim, y0, y1, mp, ids, first, second, cross_base, bias_base, dst):
    """Rows y0..y1-1 (>= 1) of the block deltas, into dst."""
    bwd, bht = _block_wh(np.int64(mp[0]), xdim)
    bw = (xdim - 2 + bwd - 1) // bwd
    head = 2 + np.int64(mp[1])
    nc = corr.shape[0]
    # The entry's one or two primitives only, from neighbours read here
    # (_primitive_value takes scalars).
    for y in range(y0, y1):
        row = head + (y - 1) // bht * bw
        for x in range(1, xdim - 1):
            k = y * xdim + x
            pid = ids[mp[row + (x - 1) // bwd]]
            i = first[pid]
            j = second[pid]
            a = src[k - 1]; b = src[k - xdim]; c = src[k - xdim - 1]; d = src[k - xdim + 1]
            aa = src[k - 2] if x >= 2 else a
            bb = src[k - 2 * xdim] if y >= 2 else b
            nne = src[k - 2 * xdim + 1] if y >= 2 else d
            r = ra = rb = rc = rd = 0
            if has_ref:
                r = ref[k]; ra = ref[k - 1]; rb = ref[k - xdim]; rc = ref[k - xdim - 1]; rd = ref[k - xdim + 1]
            c0 = np.int64(corr[0, k]); c1 = np.int64(corr[1, k]); c2 = np.int64(corr[2, k]); c3 = np.int64(corr[3, k])
            ls = np.int64(corr[4, k]) if nc > 4 else a
            lsw = np.int64(corr[5, k]) if nc > 5 else a
            p = _primitive_value(i, a, b, c, d, aa, bb, nne, has_ref, r, ra, rb, rc, rd, c0, c1, c2, c3, ls, lsw)
            if i != j:
                p = (p + _primitive_value(j, a, b, c, d, aa, bb, nne, has_ref, r, ra, rb, rc, rd, c0, c1, c2, c3, ls, lsw) + 1) >> 1
            dst[k] = src[k] - p


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


def _first_channel_primitive(q):
    """What primitive q is on the first channel: a cross-channel primitive's
    base predictor (CROSS_BASE), any other primitive itself."""
    return int(CROSS_BASE[q - CROSS_FIRST]) if CROSS_FIRST <= q < REACH_FIRST else q


def same_prediction(has_ref, pids, first=None, second=None):
    """(same, at) for the ids pids (DeltaMapper.samePrediction and
    distinctPositions): on the first channel (no ref) entries that differ only
    in cross-channel primitives predict alike; same[q] is the first position
    predicting as pids[q] does, at the positions that are their own first. The
    survey computes only those: the others' sums and estimates are copies, and
    a strict-less argmin over them picks what it would over all."""
    first = TABLE_FIRST if first is None else first
    second = TABLE_SECOND if second is None else second
    n = len(pids)
    same = np.arange(n, dtype=np.int64)
    if not has_ref:
        seen = {}
        for q in range(n):
            i = _first_channel_primitive(int(first[pids[q]]))
            j = _first_channel_primitive(int(second[pids[q]]))
            same[q] = seen.setdefault((min(i, j), max(i, j)), q)
    return same, np.flatnonzero(same == np.arange(n)).astype(np.int64)


@njit
def _block_choices(src, ref, has_ref, corr, xdim, ydim, block, ids, used, at, first, second, n_prim, cross_base, bias_base, by0, by1, choice):
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
                for p in at:
                    i = first[ids[p]]
                    j = second[ids[p]]
                    v = prim[i] if i == j else (prim[i] + prim[j] + 1) >> 1
                    sums[base + p] += abs(src[k] - v)
        for bx in range(bw):
            best = 0
            for p in at:
                if sums[bx * n + p] < sums[bx * n + best]:
                    best = p
            choice[by * bw + bx] = best


def get_block_deltas_from_values(src, ref, xdim, ydim, block, ids):
    """(deltas, map): each block takes the id with the smallest sum of |delta|
    (DeltaMapper.getBlockDeltasFromValues); ref is the previous channel or None."""
    return _block_deltas_from_values(PixelSource(src, ref, xdim, ydim), block, ids)


def _block_choices_for(ps, block, ids):
    """Each block's choice (an index into ids) by smallest sum of |delta|, the first on a tie."""
    xdim, ydim = ps.xdim, ps.ydim
    arr = np.array(ids, dtype=np.int64)
    used = np.array(sorted(set(int(TABLE_FIRST[i]) for i in ids) | set(int(TABLE_SECOND[i]) for i in ids)), dtype=np.int64)
    bh = blocks_down(ydim, block)
    choice = np.zeros(blocks_across(xdim, block) * bh, dtype=np.uint8)
    at = same_prediction(ps.has_ref, ids)[1]
    _in_bands(bh, lambda lo, hi: _block_choices(ps.v, ps.ref, ps.has_ref, ps.corr, xdim, ydim, block, arr, used, at,
                                                TABLE_FIRST, TABLE_SECOND, TABLE_PRIMITIVES, CROSS_BASE, BIAS_BASE, lo, hi, choice))
    return choice


def _block_deltas_from_values(ps, block, ids):
    ids = [int(i) for i in ids]
    choice = _block_choices_for(ps, block, ids)
    mp = np.concatenate([np.array([block, len(ids)] + ids, dtype=np.uint8), choice])
    return get_block_deltas_from_map(ps, mp, ids), mp


@njit
def _nested_choices(src, ref, has_ref, corr, xdim, ydim, sizes, offsets, n, at, first, second, cross_base, bias_base, t0, t1, choice):
    """Bands t0..t1-1 (each a row of the largest blocks): each base (smallest)
    block's sums of |delta| for ids 0..n-1, then each shape's blocks' sums
    added up from them; each block's id with the smallest sum (the first on a
    tie) into choice[offsets[shape] + block]."""
    base = sizes[0]
    top = sizes[sizes.shape[0] - 1]
    bw_base = (xdim - 2 + base - 1) // base
    rows_base = top // base
    sums = np.zeros(rows_base * bw_base * n, dtype=np.int64)
    total = np.zeros(n, dtype=np.int64)
    prim = np.zeros(ALL_PRIMITIVES, dtype=np.int64)
    for band in range(t0, t1):
        sums[:] = 0
        y0 = 1 + band * top
        y1 = min(ydim, y0 + top)
        for y in range(y0, y1):
            r = (y - y0) // base
            for x in range(1, xdim - 1):
                k = y * xdim + x
                _all_primitives(src, ref, has_ref, corr, k, xdim, prim)
                s0 = (r * bw_base + (x - 1) // base) * n
                for p in at:
                    i = first[p]
                    j = second[p]
                    v = prim[i] if i == j else (prim[i] + prim[j] + 1) >> 1
                    sums[s0 + p] += abs(src[k] - v)
        for t in range(sizes.shape[0]):
            size = sizes[t]
            f = size // base
            rows = top // size
            bw = (xdim - 2 + size - 1) // size
            bh = (ydim - 1 + size - 1) // size
            for rs in range(rows):
                by = band * rows + rs
                if by >= bh:
                    break
                for bx in range(bw):
                    total[:] = 0
                    for rr in range(rs * f, rs * f + f):
                        for cc in range(bx * f, min(bw_base, bx * f + f)):
                            s0 = (rr * bw_base + cc) * n
                            for p in at:
                                total[p] += sums[s0 + p]
                    best = 0
                    for p in at:
                        if total[p] < total[best]:
                            best = p
                    choice[offsets[t] + by * bw + bx] = best


def _nested_families(sizes):
    """The shapes split into families of nested squares, smallest first: each
    square joins the first family whose smallest divides it and whose largest
    divides it or is divided by it (4, 6, 8, 12, 16: {4, 8, 16} and {6, 12});
    other shapes are families of one (DeltaMapper.nestedFamilies). Returns
    lists of indices into sizes."""
    family = []
    for t in sorted(range(len(sizes)), key=lambda t: sizes[t]):
        b = sizes[t]
        home = None
        if b < SHAPE_FIRST:
            for f in family:
                base, top = sizes[f[0]], sizes[f[-1]]
                if base < SHAPE_FIRST and b % base == 0 and (top % b == 0 or b % top == 0):
                    home = f
                    break
        if home is None:
            home = []
            family.append(home)
        home.append(t)
    return family


def _nested_squares(sizes):
    """Whether the shapes are squares, each dividing the largest and divided
    by the smallest (DeltaMapper.nestedSquares)."""
    if any(b >= SHAPE_FIRST for b in sizes):
        return False
    base, top = min(sizes), max(sizes)
    return all(b % base == 0 and top % b == 0 for b in sizes)


def _table_reference_deltas_at(ps, sizes):
    """The whole table's reference deltas at several block shapes
    (DeltaMapper.tableReferenceDeltas): nested squares in one pass over the
    pixels, each family of them (4, 8, 16; 6, 12) in its own pass, the larger
    blocks' sums added up from the smallest's; other shapes one at a time.
    Either way the result is the same."""
    sizes = [int(b) for b in sizes]
    if not _nested_squares(sizes):
        # Each family of nested squares (4, 8, 16; 6, 12) in its own pass;
        # other shapes one at a time.
        ref = [None] * len(sizes)
        for family in _nested_families(sizes):
            shapes = [sizes[f] for f in family]
            got = [_table_reference_deltas(ps, shapes[0])] if len(family) == 1 else _table_reference_deltas_at(ps, shapes)
            for f, d in zip(family, got):
                ref[f] = d
        return ref
    xdim, ydim = ps.xdim, ps.ydim
    order = sorted(range(len(sizes)), key=lambda t: sizes[t])
    srt = np.array([sizes[t] for t in order], dtype=np.int64)
    counts = [blocks_across(xdim, b) * blocks_down(ydim, b) for b in srt]
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    choice = np.zeros(int(sum(counts)), dtype=np.uint8)
    bands = blocks_down(ydim, int(srt[-1]))
    at = same_prediction(ps.has_ref, list(range(PREDICTORS)))[1]
    _in_bands(bands, lambda lo, hi: _nested_choices(ps.v, ps.ref, ps.has_ref, ps.corr, xdim, ydim, srt, offsets, PREDICTORS, at,
                                                    TABLE_FIRST, TABLE_SECOND, CROSS_BASE, BIAS_BASE, lo, hi, choice))
    ids = list(range(PREDICTORS))
    by_size = {}
    for t, b in enumerate(srt):
        mp = np.concatenate([np.array([b, 0], dtype=np.uint8), choice[offsets[t]:offsets[t] + counts[t]]])
        by_size[int(b)] = get_block_deltas_from_map(ps, mp, ids)
    return [by_size[b] for b in sizes]


def _table_reference_deltas(ps, block):
    """The survey's reference deltas from the whole table: each block takes
    whichever of the table's predictors gives the smallest sum of |delta|
    (DeltaMapper.tableReferenceDeltas)."""
    ids = list(range(PREDICTORS))
    choice = _block_choices_for(ps, block, ids)
    # No id list in this map's header (256 ids don't fit its count byte):
    # the choices start at 2 and the ids are passed directly.
    mp = np.concatenate([np.array([block, 0], dtype=np.uint8), choice])
    return get_block_deltas_from_map(ps, mp, ids)


@njit
def _from_block(delta, ref, has_ref, xdim, ydim, init, mp, ids, first, second, cross_base, bias_base, use_ls,
                mixed, data, value, cb, fwd, c9b, m, act_table, n_contexts, opinions):
    # mixed: each delta is decoded (into delta) just before it is used, from
    # the deltas and values so far (get_values_from_mixed_deltas); with
    # opinions, given the block's prediction at interior pixels.
    state = _mix_state(data, m, value.shape[0], n_contexts)
    bwd, bht = _block_wh(np.int64(mp[0]), xdim)
    bw = (xdim - 2 + bwd - 1) // bwd
    head = 2 + np.int64(mp[1])
    n = xdim * ydim
    dst = np.zeros(n, dtype=np.int64)
    corr = np.zeros((6, n), dtype=np.int16)   # bias corrections, then LS and LS wide (computed only if the map uses them)
    ctx = np.zeros(1, dtype=np.int64)
    bsum = np.zeros((4, 729), dtype=np.int64)
    bcount = np.zeros((4, 729), dtype=np.int64)
    prim = np.zeros(ALL_PRIMITIVES, dtype=np.int64)
    row, col, win, ring, z, a, b, coef, dg = _ls_state(xdim, has_ref, LS_RADIUS)
    # One pixel at a time, with one call site of each helper (the per-pixel
    # ones are inlined): the prediction (row 0 horizontal, column 0
    # vertical, the last column horizontal, the interior the block's
    # predictor), then the delta.
    for y in range(ydim):
        mrow = head + (y - 1) // bht * bw
        for x in range(xdim):
            k = y * xdim + x
            pred = 0                    # the second opinions' prediction (interior only)
            interior = y > 0 and 0 < x < xdim - 1
            if y == 0:
                p = init if x == 0 else dst[k - 1]
            elif x == 0:
                p = dst[k - xdim]
            elif x == xdim - 1:
                p = dst[k - 1]
            else:
                pid = ids[mp[mrow + (x - 1) // bwd]]
                if use_ls:
                    corr[4, k], corr[5, k] = _ls_next(dst, ref, has_ref, xdim, k, row, col, win, ring, z, a, b, coef, dg,
                                                      first[pid] == LS_FIRST or second[pid] == LS_FIRST
                                                      or first[pid] == LS_WIDE or second[pid] == LS_WIDE, LS_RADIUS, 1)
                _bias_prepare(dst, k, xdim, ctx, corr, bsum, bcount)
                _all_primitives(dst, ref, has_ref, corr, k, xdim, prim)
                i = first[pid]
                j = second[pid]
                p = prim[i] if i == j else (prim[i] + prim[j] + 1) >> 1
                pred = p
            if mixed:
                _mix_decode_one(state, data, k, delta, dst, value, xdim, cb, fwd, c9b, m, act_table, opinions, pred)
            dst[k] = init if k == 0 else delta[k] + p
            if interior:
                _bias_commit(dst, k, xdim, ctx, bsum, bcount, bias_base)
    return dst


def _from_block_args(xdim, mp, ref, table=None):
    """(map, ref, has_ref, ids, use_ls, first, second) for decoding a block
    map whose ids index table (TABLE if None; TABLE_256 for older files)."""
    first, second = TABLE if table is None else table
    mp = np.asarray(mp, dtype=np.uint8)
    has_ref = ref is not None
    r = np.asarray(ref, dtype=np.int64) if has_ref else np.zeros(1, dtype=np.int64)
    ids = np.array(get_map_ids(mp), dtype=np.int64)
    if np.any(ids >= len(first)):
        raise ValueError("Block map with predictor id %d: not in a %d-entry table." % (int(ids.max()), len(first)))
    ls = (first[ids] == LS_FIRST) | (second[ids] == LS_FIRST) | (first[ids] == LS_WIDE) | (second[ids] == LS_WIDE)
    return mp, r, has_ref, ids, bool(np.any(ls)), first, second


def get_values_from_block_deltas(delta, xdim, ydim, init, mp, ref, table=None):
    """Inverse of the block deltas (DeltaMapper.getValuesFromBlockDeltas);
    table: the one the map's ids index (TABLE if None; TABLE_256 for older
    files)."""
    mp, r, has_ref, ids, use_ls, first, second = _from_block_args(xdim, mp, ref, table)
    none = np.zeros(1, dtype=np.int64)
    return _from_block(np.asarray(delta, dtype=np.int64), r, has_ref, xdim, ydim, init, mp,
                       ids, first, second, CROSS_BASE, BIAS_BASE, use_ls,
                       False, np.zeros(1, dtype=np.uint8), none, none, none, none, 2, MIX_ACTIVITY_TABLE, 1, False)


def get_values_from_mixed_deltas(value, coded, previous, xdim, ydim, init, mp, ref, opinions=False, table=None):
    """Decodes a channel pack_mixed_deltas coded (value, coded: from
    read_mixed_deltas) while rebuilding it from its block map, as
    get_values_from_block_deltas does. previous: the deltas of the channels
    decoded before it; ref: the previous channel's values; opinions: whether
    it was coded with the second opinions; table: the one the map's ids index
    (TABLE if None; TABLE_256 for older files). Returns (deltas, values)."""
    mp, r, has_ref, ids, use_ls, first, second = _from_block_args(xdim, mp, ref, table)
    n = xdim * ydim
    cb, fwd, c9b, m = _mix_setup(previous, xdim, n)
    delta = np.zeros(n, dtype=np.int64)
    values = _from_block(delta, r, has_ref, xdim, ydim, init, mp, ids, first, second, CROSS_BASE, BIAS_BASE,
                         use_ls, True, coded, value, cb, fwd, c9b, m, MIX_ACTIVITY_TABLE, _mix_contexts_count(opinions), opinions)
    return delta, values


# =============================================================================
# Least squares alone (the Simple format; DeltaMapper.surveyWindows)
# =============================================================================
#
# Every interior pixel predicted by the least-squares fit with the wide clamp
# (LS wide), with no blocks and no map: per channel only the window radius.
# The edges are as the block map's, and the deltas are mixed-context coded
# with the second opinions (the prediction is the fit's). survey_windows
# picks each channel's radius by the real coded size.

WINDOW_RADII = [4, 8, 16]   # the radii survey_windows tries, smallest first


def get_least_squares_deltas(v, ref, xdim, ydim, radius, every=1):
    """A channel's deltas from the least-squares fit alone (the wide clamp),
    with the window of the given radius (DeltaMapper.getLeastSquaresDeltas)."""
    v = np.asarray(v, dtype=np.int64)
    has_ref = ref is not None
    r = np.asarray(ref, dtype=np.int64) if has_ref else np.zeros(1, dtype=np.int64)
    p = _ls_planes(v, r, has_ref, xdim, ydim, radius, every)[1]
    d = np.zeros(v.shape[0], dtype=np.int64)
    _edges(v, d, xdim, ydim)
    if ydim > 1 and xdim > 2:
        d.reshape(ydim, xdim)[1:, 1:-1] = v.reshape(ydim, xdim)[1:, 1:-1] - p.reshape(ydim, xdim)[1:, 1:-1]
    return d


def get_least_squares_band_deltas(whole, v, ref, xdim, y0, y1, radius, every=1):
    """A band's least-squares deltas -- what get_least_squares_deltas gives for
    rows y0 to y1 - 1 alone, coded as an image of their own -- from the whole
    channel's (DeltaMapper.getLeastSquaresBandDeltas). They differ only in
    the band's first radius + 2 rows (the band's top row is an edge, and the
    window's first row uses the row above it), so only those are fitted
    again; the sums are exact, so the rest are the same."""
    h = y1 - y0
    k = min(h, radius + 2)
    v = np.asarray(v, dtype=np.int64)
    top = get_least_squares_deltas(v[y0 * xdim:(y0 + k) * xdim],
                                   None if ref is None else np.asarray(ref, dtype=np.int64)[y0 * xdim:(y0 + k) * xdim], xdim, k, radius, every)
    return np.concatenate([top, np.asarray(whole, dtype=np.int64)[(y0 + k) * xdim:y1 * xdim]])


@njit
def _least_squares_rows(ref, has_ref, xdim, ydim, init, radius, every, data, value, previous, cross, cb, fwd, c9b, earlier, m,
                        act_table, state, ls, delta, dst, y0, y1):
    """Decodes rows y0..y1-1 of a channel's least-squares deltas (into delta,
    each just before it is used) and rebuilds them (into dst), continuing
    from the rows before: state is the delta decoder's (_mix_state), ls the
    fit's (_ls_state); cross, cb, fwd, c9b and earlier hold the earlier
    channels' terms, filled here as far as row y1 (_mix_earlier_rows; the
    earlier channels must be decoded to row y1)."""
    row, col, win, ring, z, a, b, coef, dg = ls
    if m == 3:
        _mix_earlier_rows(previous, xdim, ydim, MIX_CROSS_TABLE, cross, cb, fwd, c9b, earlier, y1)
    # One pixel at a time, with one call site of each helper (they are
    # inlined): the prediction (row 0 horizontal, column 0 vertical, the last
    # column horizontal, the interior the fit's), then the delta.
    for y in range(y0, y1):
        for x in range(xdim):
            k = y * xdim + x
            pred = 0                    # the second opinions' prediction (interior only)
            if y == 0:
                p = init if x == 0 else dst[k - 1]
            elif x == 0:
                p = dst[k - xdim]
            elif x == xdim - 1:
                p = dst[k - 1]
            else:
                p = _ls_next(dst, ref, has_ref, xdim, k, row, col, win, ring, z, a, b, coef, dg, True, radius, every)[1]
                pred = p
            _mix_decode_one(state, data, k, delta, dst, value, xdim, cb, fwd, c9b, m, act_table, True, pred)
            dst[k] = init if k == 0 else delta[k] + p


LS_DECODE_ROWS = 1   # the rows a channel decodes between looking at the channel before (get_values_from_least_squares_channels): as the Java, a row at a time


def get_values_from_least_squares_channels(mixed, xdim, ydim, init, radius, every=1):
    """Decodes channels pack_mixed_deltas coded (with the second opinions;
    mixed: each channel's (value, coded) from read_mixed_deltas, in coding
    order) from their least-squares deltas, rebuilding them as it goes
    (DeltaMapper.getValuesFromLeastSquaresChannels): each channel's fit uses
    the one before it, and its contexts the deltas of all the ones before
    it. Each channel decodes in its own thread, LS_DECODE_ROWS rows at a time,
    once the channel before it has decoded a row further (its contexts look
    one row down into the earlier channels), so the channels decode at the
    same time. Returns (deltas, values), each per channel."""
    nch, n = len(mixed), xdim * ydim
    delta = np.zeros((nch, n), dtype=np.int64)     # rows of one array: a channel's previous is delta[:i], no copy
    value = np.zeros((nch, n), dtype=np.int64)
    done = [0] * nch
    cond = threading.Condition()
    failed = []

    def run(i):
        try:
            table, coded = mixed[i]
            m = 3 if i > 0 else 2
            state = _mix_state(coded, m, len(table), _mix_contexts_count(True))
            ls = _ls_state(xdim, i > 0, radius[i])
            ref = value[i - 1] if i > 0 else np.zeros(1, dtype=np.int64)
            terms = [np.zeros(n, dtype=np.int64) for _ in range(4)]   # cross, cb, fwd, c9b (zero for the first channel)
            earlier = np.zeros(2, dtype=np.int64)
            for y0 in range(0, ydim, LS_DECODE_ROWS):
                y1 = min(ydim, y0 + LS_DECODE_ROWS)
                if i > 0:
                    need = min(ydim, y1 + 1)
                    with cond:
                        while done[i - 1] < need and not failed:
                            cond.wait()
                        if failed:
                            return
                _least_squares_rows(ref, i > 0, xdim, ydim, int(init[i]), radius[i], every, coded, table, delta[:i], terms[0], terms[1], terms[2], terms[3],
                                    earlier, m, MIX_ACTIVITY_TABLE, state, ls, delta[i], value[i], y0, y1)
                with cond:
                    done[i] = y1
                    cond.notify_all()
        except BaseException as e:
            with cond:
                failed.append(e)
                cond.notify_all()

    threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(nch)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        raise failed[0]
    return list(delta), list(value)


# ---- Least squares from the rows above (the Simple 2 format) -----------------
#
# DeltaMapper.RowFit: the least-squares primitive with a window of the r rows
# above only (r columns either side; no pixels of the pixel's own row), so a
# row's fits depend only on the rows before it and are solved, in parallel,
# before the row is decoded. Solved at every every-th pixel along a row, from
# column 1 (the pixels up to the next use its solution); where a fit can't be
# solved, those pixels are predicted by a (the pixel to the left). The wide
# clamp. The column sums are exact integers, and the solution is in the
# Java's order, so both get the same values.

@njit(inline="always")   # a call per pixel passing arrays is slow in Numba
def _rf_products(v, ref, has_ref, xdim, nf, y, x, z, out):
    """Pixel (y, x)'s products into out."""
    _ls_features(v, ref, has_ref, xdim, nf, y, x, z)
    m = nf + 2
    t = 0
    for i in range(m):
        zi = z[i]
        for j in range(i, m):
            out[t] = zi * z[j]
            t += 1


@njit
def _rf_start(v, ref, has_ref, xdim, y0, r, col, z, p):
    """Sets the columns to row y0's window (rows y0 - r .. y0 - 1, from 1)
    (RowFit.start)."""
    nf = 11 if has_ref else 8
    m = nf + 2
    nt = m * (m + 1) // 2
    col[:] = 0
    for ty in range(max(1, y0 - r), y0):
        for tx in range(1, xdim - 1):
            _rf_products(v, ref, has_ref, xdim, nf, ty, tx, z, p)
            c = np.uint64(tx * nt)
            for t in range(nt):
                ut = np.uint64(t)
                col[c + ut] += p[ut]


@njit
def _rf_advance(v, ref, has_ref, xdim, y, r, col, c0, c1, z, p):
    """Columns c0..c1-1 from row y - 1's window to row y's: adds row y - 1,
    drops row y - 1 - r (RowFit.moveTo; y - 1 >= 1)."""
    nf = 11 if has_ref else 8
    m = nf + 2
    nt = m * (m + 1) // 2
    drop = y - 1 - r
    for tx in range(c0, c1):
        c = np.uint64(tx * nt)
        _rf_products(v, ref, has_ref, xdim, nf, y - 1, tx, z, p)
        for t in range(nt):
            ut = np.uint64(t)
            col[c + ut] += p[ut]
        if drop >= 1:
            _rf_products(v, ref, has_ref, xdim, nf, drop, tx, z, p)
            for t in range(nt):
                ut = np.uint64(t)
                col[c + ut] -= p[ut]


@njit
def _rf_solve(col, xdim, r, nf, every, p0, p1, coef, valid, win, a, b, dg):
    """Solves solved pixels p0..p1-1 of a row (column 1 + every * p) from the
    columns (RowFit.solveRow and solve): each one's window sums, slid from
    the one before, then L D L^T as in the Java."""
    m = nf + 2
    nt = m * (m + 1) // 2
    n = nf + 1
    lo = 0
    hi = -1
    for p in range(p0, p1):
        x = 1 + p * every
        nlo = max(1, x - r)
        nhi = min(xdim - 2, x + r)
        if p == p0:
            win[:] = 0
            for c in range(nlo, nhi + 1):
                o = np.uint64(c * nt)
                for s in range(nt):
                    us = np.uint64(s)
                    win[us] += col[o + us]
        else:
            for c in range(lo, nlo):
                o = np.uint64(c * nt)
                for s in range(nt):
                    us = np.uint64(s)
                    win[us] -= col[o + us]
            for c in range(hi + 1, nhi + 1):
                o = np.uint64(c * nt)
                for s in range(nt):
                    us = np.uint64(s)
                    win[us] += col[o + us]
        lo = nlo
        hi = nhi
        valid[p] = False
        if win[nf * m - nf * (nf - 1) // 2] < 2 * n:
            continue
        for i in range(n):
            base = i * m - i * (i - 1) // 2
            for j in range(i, n):
                a[j, i] = float(win[base + j - i])
            b[i] = float(win[base + m - 1 - i])
            a[i, i] += 1.0
        ok = True
        for j in range(n):
            sv = a[j, j]
            for q in range(j):
                sv -= a[j, q] * a[j, q] * dg[q]
            if sv < 1e-9:
                ok = False
                break
            dg[j] = sv
            for i in range(j + 1, n):
                t_ = a[i, j]
                for q in range(j):
                    t_ -= a[i, q] * a[j, q] * dg[q]
                a[i, j] = t_ / sv
        if not ok:
            continue
        o = p * n
        for i in range(n):
            sv = b[i]
            for q in range(i):
                sv -= a[i, q] * coef[o + q]
            coef[o + i] = sv
        for i in range(n):
            coef[o + i] /= dg[i]
        for i in range(n - 1, -1, -1):
            sv = coef[o + i]
            for q in range(i + 1, n):
                sv -= a[q, i] * coef[o + q]
            coef[o + i] = sv
        valid[p] = True


@njit(inline="always")   # a call per pixel passing arrays is slow in Numba
def _rf_predict(v, ref, has_ref, xdim, nf, y, x, every, coef, valid, z):
    """The prediction at interior pixel (y, x), with the wide clamp, from the
    row's solved fits (RowFit.predict)."""
    n = nf + 1
    p = (x - 1) // every
    pa = _ls_at(v, y, x - 1, xdim); pb = _ls_at(v, y - 1, x, xdim)
    pc = _ls_at(v, y - 1, x - 1, xdim); pd = _ls_at(v, y - 1, x + 1, xdim)
    if not valid[p]:
        return pa
    _ls_features(v, ref, has_ref, xdim, nf, y, x, z)
    s = 0.0
    o = p * n
    for i in range(n):
        s += coef[o + i] * z[i]
    lo = min(min(pa, pb), min(pc, pd))
    hi = max(max(pa, pb), max(pc, pd))
    for i in range(4, 8):
        lo = min(lo, z[i])
        hi = max(hi, z[i])
    w = max(lo - 2, min(hi + 2, s))
    return int(np.floor(w + 0.5))


def _rf_state(xdim, has_ref, every):
    """(col, coef, valid, z, p, win, a, b, dg, points): a RowFit's columns
    and a row's solutions, and one task's working arrays (_rf_work)."""
    nf = 11 if has_ref else 8
    m = nf + 2
    nt = m * (m + 1) // 2
    n = nf + 1
    points = max(0, (xdim - 2 + every - 1) // every)
    return (np.zeros(xdim * nt, dtype=np.int32), np.zeros(max(1, points) * n), np.zeros(max(1, points), dtype=np.bool_)) + _rf_work(nf) + (points,)


def _rf_work(nf):
    """One task's working arrays: (z, products, window sums, a, b, dg)."""
    m = nf + 2
    nt = m * (m + 1) // 2
    n = nf + 1
    return (np.zeros(m, dtype=np.int64), np.zeros(nt, dtype=np.int32), np.zeros(nt, dtype=np.int64),
            np.zeros((n, n)), np.zeros(n), np.zeros(n))


@njit
def _rf_band(v, ref, has_ref, xdim, y0, y1, out, r, every, col, coef, valid, z, p, win, a, b, dg, points):
    """The encoder's predictions for rows y0..y1-1, the window started at
    y0 on its own (RowFit.plane's bands)."""
    nf = 11 if has_ref else 8
    _rf_start(v, ref, has_ref, xdim, y0, r, col, z, p)
    for y in range(y0, y1):
        if y > y0 and y - 1 >= 1:
            _rf_advance(v, ref, has_ref, xdim, y, r, col, 1, xdim - 1, z, p)
        _rf_solve(col, xdim, r, nf, every, 0, points, coef, valid, win, a, b, dg)
        for x in range(1, xdim - 1):
            out[y * xdim + x] = _rf_predict(v, ref, has_ref, xdim, nf, y, x, every, coef, valid, z)


def _rf_planes(v, ref, has_ref, xdim, ydim, radius, every):
    """The encoder's predictions for the whole channel, in bands of rows
    computed in parallel (RowFit.plane)."""
    out = np.zeros(v.shape[0], dtype=np.int16)
    band = max(4 * (radius + 1), (ydim - 1 + _CPUS - 1) // _CPUS)

    def one(y0):
        _rf_band(v, ref, has_ref, xdim, y0, min(ydim, y0 + band), out, radius, every, *_rf_state(xdim, has_ref, every))
    for f in [_LS_POOL.submit(one, y0) for y0 in range(1, ydim, band)]:
        f.result()
    return out


def get_row_fit_deltas(v, ref, xdim, ydim, radius, every):
    """A channel's deltas from the least-squares fit from the rows above
    (the wide clamp) (DeltaMapper.getRowFitDeltas)."""
    v = np.asarray(v, dtype=np.int64)
    has_ref = ref is not None
    r = np.asarray(ref, dtype=np.int64) if has_ref else np.zeros(1, dtype=np.int64)
    p = _rf_planes(v, r, has_ref, xdim, ydim, radius, every)
    d = np.zeros(v.shape[0], dtype=np.int64)
    _edges(v, d, xdim, ydim)
    if ydim > 1 and xdim > 2:
        d.reshape(ydim, xdim)[1:, 1:-1] = v.reshape(ydim, xdim)[1:, 1:-1] - p.reshape(ydim, xdim)[1:, 1:-1]
    return d


def get_row_fit_band_deltas(whole, v, ref, xdim, y0, y1, radius, every):
    """A band's RowFit deltas -- what get_row_fit_deltas gives for rows y0 to
    y1 - 1 alone -- from the whole channel's: they differ only in the band's
    first radius + 2 rows, so only those are fitted again
    (DeltaMapper.getRowFitBandDeltas)."""
    h = y1 - y0
    k = min(h, radius + 2)
    v = np.asarray(v, dtype=np.int64)
    top = get_row_fit_deltas(v[y0 * xdim:(y0 + k) * xdim],
                             None if ref is None else np.asarray(ref, dtype=np.int64)[y0 * xdim:(y0 + k) * xdim], xdim, k, radius, every)
    return np.concatenate([top, np.asarray(whole, dtype=np.int64)[(y0 + k) * xdim:y1 * xdim]])


@njit
def _row_fit_row(ref, has_ref, xdim, ydim, init, every, data, value, previous, cross, cb, fwd, c9b, earlier, m,
                 act_table, state, coef, valid, z, delta, dst, y):
    """Decodes row y of a channel's RowFit deltas (into delta, each just
    before it is used) and rebuilds it (into dst), the row's fits solved
    (coef, valid); state is the delta decoder's (_mix_state); cross, cb, fwd,
    c9b and earlier hold the earlier channels' terms, filled here as far as
    row y + 1 (the earlier channels must be decoded to row y + 1)."""
    nf = 11 if has_ref else 8
    if m == 3:
        _mix_earlier_rows(previous, xdim, ydim, MIX_CROSS_TABLE, cross, cb, fwd, c9b, earlier, min(ydim, y + 1))
    for x in range(xdim):
        k = y * xdim + x
        pred = 0                    # the second opinions' prediction (interior only)
        if y == 0:
            p = init if x == 0 else dst[k - 1]
        elif x == 0:
            p = dst[k - xdim]
        elif x == xdim - 1:
            p = dst[k - 1]
        else:
            p = _rf_predict(dst, ref, has_ref, xdim, nf, y, x, every, coef, valid, z)
            pred = p
        _mix_decode_one(state, data, k, delta, dst, value, xdim, cb, fwd, c9b, m, act_table, True, pred)
        dst[k] = init if k == 0 else delta[k] + p


_RF_POOL = ThreadPoolExecutor(_CPUS)   # RowFit's decoder tasks (they never wait, so the pool can't block)
RF_TASKS = 4                           # the most tasks a row's RowFit work is split into in the decoder (each is a Python call)


def _rf_split(items, least):
    """How many tasks to split items into: at most RF_TASKS, with at least
    least items each."""
    return max(1, min(RF_TASKS, items // least))


def get_values_from_row_fit_channels(mixed, xdim, ydim, init, radius, every):
    """Decodes channels pack_mixed_deltas coded (with the second opinions)
    from their RowFit deltas, rebuilding them as it goes
    (DeltaMapper.getValuesFromRowFitChannels): as
    get_values_from_least_squares_channels, a thread per channel, a row at a
    time once the channel before has decoded a row further; before each row
    its fits are solved in tasks run at the same time (_rf_advance,
    _rf_solve), so decoding a row is left with the predictions and the delta
    decoder. mixed[i] may be a Future, for channels still being read: the
    channel's thread starts once it is done (init[i] set before it). Returns
    (deltas, values), each per channel."""
    nch, n = len(mixed), xdim * ydim
    delta = np.zeros((nch, n), dtype=np.int64)     # rows of one array: a channel's previous is delta[:i], no copy
    value = np.zeros((nch, n), dtype=np.int64)
    done = [0] * nch
    cond = threading.Condition()
    failed = []

    def run(i):
        try:
            table, coded = mixed[i].result() if hasattr(mixed[i], "result") else mixed[i]
            m = 3 if i > 0 else 2
            has_ref = i > 0
            nf = 11 if has_ref else 8
            r = radius[i]
            state = _mix_state(coded, m, len(table), _mix_contexts_count(True))
            col, coef, valid, z, _, _, _, _, _, points = _rf_state(xdim, has_ref, every)
            ref = value[i - 1] if i > 0 else np.zeros(1, dtype=np.int64)
            dst = value[i]
            terms = [np.zeros(n, dtype=np.int64) for _ in range(4)]   # cross, cb, fwd, c9b (zero for the first channel)
            earlier = np.zeros(2, dtype=np.int64)
            ca = _rf_split(max(0, xdim - 2), 128)
            cs = _rf_split(points, 32)
            work = [_rf_work(nf) for _ in range(max(ca, cs))]

            def advance(t, y):
                c0, c1 = 1 + t * (xdim - 2) // ca, 1 + (t + 1) * (xdim - 2) // ca
                _rf_advance(dst, ref, has_ref, xdim, y, r, col, c0, c1, work[t][0], work[t][1])

            def solve(t):
                p0, p1 = t * points // cs, (t + 1) * points // cs
                w = work[t]
                _rf_solve(col, xdim, r, nf, every, p0, p1, coef, valid, w[2], w[3], w[4], w[5])

            for y in range(ydim):
                if i > 0:
                    need = min(ydim, y + 2)
                    with cond:
                        while done[i - 1] < need and not failed:
                            cond.wait()
                        if failed:
                            return
                if y >= 1:
                    if y >= 2:
                        jobs = [_RF_POOL.submit(advance, t, y) for t in range(1, ca)]
                        advance(0, y)
                        for j in jobs:
                            j.result()
                    jobs = [_RF_POOL.submit(solve, t) for t in range(1, cs)]
                    solve(0)
                    for j in jobs:
                        j.result()
                _row_fit_row(ref, has_ref, xdim, ydim, int(init[i]), every, coded, table, delta[:i], terms[0], terms[1], terms[2], terms[3],
                             earlier, m, MIX_ACTIVITY_TABLE, state, coef, valid, z, delta[i], dst, y)
                with cond:
                    done[i] = y + 1
                    cond.notify_all()
        except BaseException as e:
            with cond:
                failed.append(e)
                cond.notify_all()

    threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(nch)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        raise failed[0]
    return list(delta), list(value)


class WindowSurvey:
    """What survey_windows chose for each channel, and what it measured
    (DeltaMapper.WindowSurvey): radius, deltas, coded (pack_mixed_deltas with
    the second opinions) and bytes per channel (coding order), their total, the radii tried, and tried[channel][radius index],
    each channel's coded size at each."""

    def __init__(self, n, radii):
        self.radius = [0] * n
        self.deltas = [None] * n
        self.coded = [None] * n
        self.bytes = [0] * n
        self.total = 0
        self.radii = list(radii)
        self.tried = [[0] * len(radii) for _ in range(n)]


def survey_windows(channel, xdim, ydim, radii=None):
    """Each channel's window radius, in turn: the one from radii whose deltas
    code smallest with the mixed coder and second opinions, given the deltas
    chosen for the channels before (the first smallest on a tie)
    (DeltaMapper.surveyWindows). The radii are fitted at the same time, as
    memory allows (each fit also splits itself across the cores: _ls_planes),
    and a channel's radii are coded, in a thread of their own, while the
    next channel's are fitted: the coding needs only the deltas chosen
    before, and the fit only the channels' values."""
    radii = WINDOW_RADII if radii is None else radii
    channel = [np.asarray(c, dtype=np.int64) for c in channel]
    n, nr = len(channel), len(radii)
    task = 80 * xdim * ydim
    s = WindowSurvey(n, radii)

    def fit(i):
        v, ref = channel[i], channel[i - 1] if i > 0 else None
        return _bounded_map(lambda r: get_least_squares_deltas(v, ref, xdim, ydim, radii[r]), range(nr), task)

    def code(i, got):
        previous = s.deltas[:i]
        coded = _bounded_map(lambda r: pack_mixed_deltas(got[r], channel[i], previous, xdim, True), range(nr), task)
        size = [len(c) for c in coded]
        s.tried[i] = size
        best = 0
        for r in range(1, nr):
            if size[r] < size[best]:
                best = r
        s.radius[i] = radii[best]
        s.deltas[i] = got[best]
        s.coded[i] = coded[best]
        s.bytes[i] = size[best]
        s.total += size[best]

    got = fit(0)
    with ThreadPoolExecutor(1) as coder:
        for i in range(n):
            job = coder.submit(code, i, got)
            try:
                following = fit(i + 1) if i + 1 < n else None
            finally:
                job.result()
            got = following
    return s


# =============================================================================
# Dispatch
# =============================================================================

def get_deltas(src, xdim, ydim, delta_type, variant=0):
    """(deltas, map or None, init value) for delta_type 0-12. The block map's
    (13) deltas come from the survey (surveyed_block_maps)."""
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
    elif t == 13: raise ValueError("delta type 13: the block map's deltas come from the survey (surveyed_block_maps)")
    else: raise ValueError("delta_type %d" % t)
    return d, m, init


def get_values_from_deltas(delta, xdim, ydim, init, delta_type, mp=None, variant=0, ref=None, table=None):
    """Inverse of get_deltas; ref (the previous channel as decoded, before
    any minimum is added back) and table (the one a block map's ids index:
    TABLE_256 for older files) are used by the block map only."""
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
    if t == 13: return get_values_from_block_deltas(delta, xdim, ydim, init, mp, ref, table)
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
        block = int(mp[0])
        width = blocks_across(xdim, block)
        mp = mp[head:]
        if previous is not None:
            previous = np.asarray(previous, dtype=np.uint8)
            previous = align_map(previous[map_header_length(previous):], int(previous[0]), block, xdim, len(mp))
    forms = [_map_to_string(mp), _map_to_arithmetic(mp), _map_to_context(mp, previous, width), _map_to_neighbours(mp, previous, width)]
    best = min((f for f in range(4) if forms[f] is not None), key=lambda f: (len(forms[f]), f))
    out.write_byte(best)
    out.write(forms[best])


def map_bytes(delta_type, mp, previous, xdim):
    out = DataOutput()
    write_map(out, delta_type, mp, previous, xdim)
    return out.size()


def align_map(previous, prev_block, block, xdim, n):
    """The previous channel's block map entries lined up with this channel's
    blocks: each block takes the entry of the previous channel's block under
    its centre (the nearest one at the edges); with the same block shape,
    entry for entry, as before format 5 (DeltaMapper.alignMap)."""
    if prev_block == block:
        return previous
    w, pw = blocks_across(xdim, block), blocks_across(xdim, prev_block)
    ph = len(previous) // pw
    bw, bh = block_width(block, xdim), block_height(block)
    pbw, pbh = block_width(prev_block, xdim), block_height(prev_block)
    k = np.arange(n)
    px = np.minimum(pw - 1, ((k % w) * bw + bw // 2) // pbw)
    py = np.minimum(ph - 1, ((k // w) * bh + bh // 2) // pbh)
    return np.asarray(previous, dtype=np.uint8)[py * pw + px]


def _read_map_forms(inp, previous, width):
    """previous: the previous channel's entries, or a function giving them
    lined up with this map's n entries, or None."""
    coding = inp.read_byte()
    n = inp.read_int()
    if callable(previous):
        previous = previous(n)
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
        if not (1 <= n <= MAX_PREDICTORS):
            raise IOError("Block map with %d predictors: not one this version reads." % n)
        ids = [inp.read_unsigned_byte() for _ in range(n)]
        if max(ids) >= MAX_PREDICTORS:
            raise IOError("Block map with predictor id %d: not one this version reads." % max(ids))
        if not valid_shape(block):
            raise IOError("Block map with block shape %d: not one this version reads." % block)
        prev = None
        if previous is not None:
            full = np.asarray(previous, dtype=np.uint8)
            entries, prev_block = full[map_header_length(full):], int(full[0])
            prev = lambda n_entries: align_map(entries, prev_block, block, xdim, n_entries)   # noqa: E731
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
def _delta_contexts(d, previous, xdim, act_table, cross_table, n_cross, k0, k1, ctx):
    """Contexts of deltas k0..k1-1, into ctx: _delta_context's, written out
    here with its buckets looked up (act_table, cross_table: _bucket_table of
    ACTIVITY_LIMIT and CROSS_LIMIT), as a call per delta passing arrays is
    slow in Numba."""
    n_prev = previous.shape[0]
    na = act_table.shape[0] - 1
    nc = cross_table.shape[0] - 1
    for k in range(k0, k1):
        y = k // xdim
        x = k - y * xdim
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
        for p in range(n_prev):
            cross += abs(previous[p, k])
        ctx[k] = act_table[min(activity, na)] * n_cross + cross_table[min(cross, nc)]


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
    symbol, lo, rank = _rank_symbols(delta)
    prev2d = _previous_2d(previous, len(delta))
    ctx = np.zeros(len(delta), dtype=np.int64)
    _in_bands(len(delta), lambda k0, k1: _delta_contexts(delta, prev2d, xdim, MIX_ACTIVITY_TABLE, MIX_CROSS_TABLE, len(CROSS_LIMIT), k0, k1, ctx))
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
# Mixed context coding (Block format 4; DeltaMapper.packMixedDeltas)
# =============================================================================
#
# The deltas as ranks (as pack_context_deltas), coded with a mixture of
# adaptive context models. Each model is the context coder's (counts from 1,
# +CONTEXT_INCREMENT, halved above CONTEXT_LIMIT), with its own context:
#   0  activity (as _delta_context) plus CALIC's gradients of this channel's
#      values, dh + dv, halved; cross as _delta_context
#   1  activity plus half the earlier channels' |deltas| to the E, SW, S and
#      SE (those channels are fully known); cross as _delta_context
#   2  activity; cross from the earlier channels' 3x3 neighbourhood, / 3
# The first channel mixes model 0 and _delta_context's context (models 1
# and 2 would both equal that one).
# Mixture: C_i(s) = cum_i(s) * 2^MIX_R_BITS // total_i,
# M(s) = (sum_i w_i C_i(s)) >> MIX_W_BITS + s, out of 2^MIX_R_BITS + n_symbols
# (so every symbol has at least 1). The weights are ints summing to
# 2^MIX_W_BITS, one set per channel, moved after each symbol along the
# code-length gradient: w_i += w_i (q_i - q) / (q MIX_RATE) (Java division),
# at least MIX_WMIN, then rescaled (the remainder to the largest, first on a
# tie). Cumulative counts by a linear scan from rank 0 (ranks are most
# frequent first, so the scan is short). Model 0 needs the values decoded so
# far, so the decoder decodes each delta as it rebuilds the channel
# (get_values_from_mixed_deltas).
#
# With second opinions (Block formats 7 and 8): each model's context is also
# split MIX_OPINION_LEVELS ways by how far a simple predictor's value lies
# from the block's prediction, signed (_gradient_level of opinion -
# prediction): GAP/8 for model 0, MED for model 1 and a (left) for model 2;
# interior pixels only (the rest use the middle level). See DeltaMapper.java.

MIX_OPINION_LEVELS = 9
MIX_R_BITS = 16
MIX_W_BITS = 16
MIX_WMIN = 256
MIX_RATE = 32


def _bucket_table(limit, size):
    """The bucket of each value 0..size-1 (values past the end use the last entry)."""
    t = np.zeros(size, dtype=np.int64)
    for v in range(size):
        b = 0
        while b + 1 < len(limit) and v >= limit[b + 1]:
            b += 1
        t[v] = b
    return t


MIX_ACTIVITY_TABLE = _bucket_table(ACTIVITY_LIMIT, 256)
MIX_CROSS_TABLE = _bucket_table(CROSS_LIMIT, 64)


def _rank_symbols(delta):
    """(each delta's rank, most frequent value first, ties: smaller first; the minimum; the rank table)."""
    delta = np.asarray(delta, dtype=np.int64)
    lo = int(delta.min())
    count = np.bincount(delta - lo)
    order = sorted(range(len(count)), key=lambda v: (-count[v], v))
    rank = np.zeros(len(count), dtype=np.int64)
    rank[order] = np.arange(len(count))
    return rank[delta - lo], lo, rank


@njit
def _mix_earlier(previous, xdim, ydim, cross_table):
    """From the earlier channels' deltas: the cross bucket, the forward sum
    (E, SW, S, SE) and the 3x3 cross bucket, per pixel."""
    n = xdim * ydim
    cross = np.zeros(n, dtype=np.int64)
    cb = np.zeros(n, dtype=np.int64)
    fwd = np.zeros(n, dtype=np.int64)
    c9b = np.zeros(n, dtype=np.int64)
    done = np.zeros(2, dtype=np.int64)
    _mix_earlier_rows(previous, xdim, ydim, cross_table, cross, cb, fwd, c9b, done, ydim)
    return cb, fwd, c9b


@njit
def _mix_earlier_rows(previous, xdim, ydim, cross_table, cross, cb, fwd, c9b, done, y1):
    """_mix_earlier's cb, fwd and c9b up to row y1 (not included), which
    needs the earlier channels' deltas up to row y1 (included); done holds
    the rows of cross and of the rest filled so far (DeltaMapper.MixedModels.earlierRows):
    so a decoder can decode a channel while the earlier ones are still being
    decoded, a row or two ahead."""
    y1 = min(y1, ydim)
    c1 = min(ydim, y1 + 1)
    for k in range(done[0] * xdim, c1 * xdim):
        t = 0
        for p in range(previous.shape[0]):
            t += abs(previous[p, k])
        cross[k] = t
    done[0] = max(done[0], c1)
    top = cross_table.shape[0] - 1
    for y in range(done[1], y1):
        for x in range(xdim):
            k = y * xdim + x
            s9 = 0
            for yy in range(max(0, y - 1), min(ydim, y + 2)):
                for xx in range(max(0, x - 1), min(xdim, x + 2)):
                    s9 += cross[yy * xdim + xx]
            f = 0
            if x + 1 < xdim:
                f += cross[k + 1]
            if y + 1 < ydim:
                f += cross[k + xdim]
                if x > 0:
                    f += cross[k + xdim - 1]
                if x + 1 < xdim:
                    f += cross[k + xdim + 1]
            cb[k] = cross_table[min(cross[k], top)]
            c9b[k] = cross_table[min(s9 // 3, top)]
            fwd[k] = f
    done[1] = max(done[1], y1)


@njit(inline="always")   # a call per pixel passing arrays is slow in Numba
def _mix_contexts(k, d, v, xdim, cb, fwd, c9b, m, act_table, ctx, opinions, pred):
    """The models' contexts at pixel k (d and v are read below k only); with
    opinions, split by the second opinions about pred, the block's prediction."""
    x = k % xdim
    act = 0
    if x > 0:
        act += abs(d[k - 1])
    if k >= xdim:
        act += abs(d[k - xdim])
        if x > 0:
            act += abs(d[k - xdim - 1])
        if x < xdim - 1:
            act += abs(d[k - xdim + 1])
    g = 0
    if x >= 2 and k >= 2 * xdim and x < xdim - 1:
        W = v[k - 1]
        N = v[k - xdim]
        NW = v[k - xdim - 1]
        NE = v[k - xdim + 1]
        g = (abs(W - v[k - 2]) + abs(N - NW) + abs(N - NE)
             + abs(W - NW) + abs(N - v[k - 2 * xdim]) + abs(NE - v[k - 2 * xdim + 1]))
    top = act_table.shape[0] - 1
    c = cb[k]
    ctx[0] = act_table[min((act + g) >> 1, top)] * 5 + c
    if m == 3:
        ctx[1] = act_table[min(act + (fwd[k] >> 1), top)] * 5 + c
        ctx[2] = act_table[min(act, top)] * 5 + c9b[k]
    else:
        ctx[1] = act_table[min(act, top)] * 5 + c
    if opinions:
        mid = MIX_OPINION_LEVELS // 2
        o0 = mid
        o1 = mid
        o2 = mid
        if k >= xdim and x >= 1 and x < xdim - 1:
            a = v[k - 1]
            b = v[k - xdim]
            cc = v[k - xdim - 1]
            dd = v[k - xdim + 1]
            up2 = k >= 2 * xdim
            aa = v[k - 2] if x >= 2 else a
            bb = v[k - 2 * xdim] if up2 else b
            nne = v[k - 2 * xdim + 1] if up2 else dd
            o0 = _gradient_level(_gap(a, aa, b, bb, cc, dd, nne, 10, 4, 1) - pred) + mid
            o1 = _gradient_level(_block_predictor(11, a, b, cc, dd) - pred) + mid
            o2 = _gradient_level(a - pred) + mid
        ctx[0] = ctx[0] * MIX_OPINION_LEVELS + o0
        ctx[1] = ctx[1] * MIX_OPINION_LEVELS + o1
        if m == 3:
            ctx[2] = ctx[2] * MIX_OPINION_LEVELS + o2


def _mix_contexts_count(opinions):
    return CONTEXTS * MIX_OPINION_LEVELS if opinions else CONTEXTS


@njit
def _mix_init(m, n_symbols, n_contexts):
    f = np.ones((m, n_contexts, n_symbols), dtype=np.int32)
    total = np.full((m, n_contexts), n_symbols, dtype=np.int64)
    w = np.full(m, (1 << MIX_W_BITS) // m, dtype=np.int64)
    w[0] += (1 << MIX_W_BITS) - w.sum()
    return f, total, w


@njit(inline="always")   # a call per pixel passing arrays is slow in Numba
def _mix_update(f, total, w, ctx, m, s, n_symbols, q):
    """Moves the weights (q: each model's scaled count of s), then counts s."""
    qm = 0
    for j in range(m):
        qm += w[j] * q[j]
    qm >>= MIX_W_BITS
    if qm < 1:
        qm = 1
    t = 0
    for j in range(m):
        w[j] += jdiv(w[j] * (q[j] - qm), qm * MIX_RATE)
        if w[j] < MIX_WMIN:
            w[j] = MIX_WMIN
        t += w[j]
    t2 = 0
    big = 0
    for j in range(m):
        w[j] = (w[j] << MIX_W_BITS) // t
        t2 += w[j]
        if w[j] > w[big]:
            big = j
    w[big] += (1 << MIX_W_BITS) - t2
    for j in range(m):
        c = ctx[j]
        f[j, c, s] += am.CONTEXT_INCREMENT
        total[j, c] += am.CONTEXT_INCREMENT
        if total[j, c] > am.CONTEXT_LIMIT:
            u = 0
            for i in range(n_symbols):
                f[j, c, i] = (f[j, c, i] + 1) >> 1
                u += f[j, c, i]
            total[j, c] = u


@njit
def _mix_encode(symbol, d, v, xdim, n_symbols, cb, fwd, c9b, m, act_table, n_contexts, opinions):
    n = symbol.shape[0]
    f, total, w = _mix_init(m, n_symbols, n_contexts)
    st, out = am._enc_init(n)
    ctx = np.zeros(3, dtype=np.int64)
    q = np.zeros(m, dtype=np.int64)
    T = (1 << MIX_R_BITS) + n_symbols
    for k in range(n):
        _mix_contexts(k, d, v, xdim, cb, fwd, c9b, m, act_table, ctx, opinions, v[k] - d[k])
        s = symbol[k]
        a = 0
        b = 0
        for j in range(m):
            c = ctx[j]
            cs = 0
            for i in range(s):
                cs += f[j, c, i]
            C0 = (cs << MIX_R_BITS) // total[j, c]
            C1 = ((cs + f[j, c, s]) << MIX_R_BITS) // total[j, c]
            a += w[j] * C0
            b += w[j] * C1
            q[j] = C1 - C0
        lo = (a >> MIX_W_BITS) + s
        hi = (b >> MIX_W_BITS) + s + 1
        out = am._encode(st, out, lo, hi - lo, T)
        _mix_update(f, total, w, ctx, m, s, n_symbols, q)
    return am._finish(st, out)


@njit
def _mix_state(data, m, n_symbols, n_contexts):
    f, total, w = _mix_init(m, n_symbols, n_contexts)
    return (am._dec_init(data), f, total, w, np.zeros(3, dtype=np.int64), np.zeros(m, dtype=np.int64),
            np.zeros(m, dtype=np.int64), np.zeros(m, dtype=np.int64), np.zeros(m, dtype=np.int64))


@njit(inline="always")   # a call per pixel passing arrays is slow in Numba
def _mix_decode_one(state, data, k, d, v, value, xdim, cb, fwd, c9b, m, act_table, opinions, pred):
    """Decodes delta k into d[k] (d and v must be filled below k; pred: the
    block's prediction at k, used with opinions at interior pixels)."""
    dt, f, total, w, ctx, cum, C0, C1, q = state
    n_symbols = value.shape[0]
    _mix_contexts(k, d, v, xdim, cb, fwd, c9b, m, act_table, ctx, opinions, pred)
    t = am._target(dt, (1 << MIX_R_BITS) + n_symbols)
    for j in range(m):
        cum[j] = 0
        C1[j] = 0
    s = 0
    lo = 0
    while True:
        a = 0
        for j in range(m):
            c = ctx[j]
            C0[j] = C1[j]
            cum[j] += f[j, c, s]
            C1[j] = (cum[j] << MIX_R_BITS) // total[j, c]
            a += w[j] * C1[j]
        hi = (a >> MIX_W_BITS) + s + 1
        if t < hi or s == n_symbols - 1:
            break
        lo = hi
        s += 1
    for j in range(m):
        q[j] = C1[j] - C0[j]
    am._decode(dt, data, lo, hi - lo)
    _mix_update(f, total, w, ctx, m, s, n_symbols, q)
    d[k] = value[s]


@njit
def _mix_decode_known(data, value, v, xdim, cb, fwd, c9b, m, act_table, n_contexts, opinions, pred):
    """Decodes every delta with the channel's values (and, with opinions, the
    block's predictions) already known (a check of the coding)."""
    n = v.shape[0]
    d = np.zeros(n, dtype=np.int64)
    state = _mix_state(data, m, value.shape[0], n_contexts)
    for k in range(n):
        _mix_decode_one(state, data, k, d, v, value, xdim, cb, fwd, c9b, m, act_table, opinions, pred[k] if opinions else 0)
    return d


def _mix_setup(previous, xdim, n):
    """(cb, fwd, c9b, number of models) for a channel with the earlier channels' deltas previous."""
    if not previous:
        z = np.zeros(n, dtype=np.int64)
        return z, z, z, 2
    prev2d = _previous_2d(previous, n)
    cb, fwd, c9b = _mix_earlier(prev2d, xdim, n // xdim, MIX_CROSS_TABLE)
    return cb, fwd, c9b, 3


def pack_mixed_deltas(delta, values, previous, xdim, opinions=False):
    """A channel's deltas, mixed-context coded: int min, the rank table (as
    pack_context_deltas), int coded length, coded bytes. values: the channel
    as the decoder rebuilds it (before any minimum is added back); previous:
    the deltas of the channels coded before it; opinions: with the second
    opinions (Block formats 7 and 8; the block's prediction is values - delta)."""
    delta = np.asarray(delta, dtype=np.int64)
    symbol, lo, rank = _rank_symbols(delta)
    cb, fwd, c9b, m = _mix_setup(previous, xdim, len(delta))
    coded = _mix_encode(symbol, delta, np.asarray(values, dtype=np.int64), xdim, len(rank),
                        cb, fwd, c9b, m, MIX_ACTIVITY_TABLE, _mix_contexts_count(opinions), opinions)
    out = DataOutput()
    out.write_int(lo); write_table(out, rank); out.write_int(len(coded)); out.write(coded.tobytes())
    return out.to_bytes()


def read_mixed_deltas(inp):
    """Reads what pack_mixed_deltas wrote, without decoding it:
    (the value of each rank, the coded bytes)."""
    lo = inp.read_int()
    rank = read_table(inp)
    coded = np.frombuffer(inp.read_fully(inp.read_int()), dtype=np.uint8)
    value = np.zeros(len(rank), dtype=np.int64)
    value[rank] = np.arange(len(rank)) + lo
    return value, coded


def unpack_mixed_deltas(packed, values, previous, xdim, prediction=None):
    """Decodes pack_mixed_deltas' bytes given the channel's values (for
    checking the coding; a reader rebuilds the values as it decodes) and,
    for bytes coded with the second opinions, the block's prediction at
    each pixel."""
    value, coded = read_mixed_deltas(DataInput(packed))
    values = np.asarray(values, dtype=np.int64)
    cb, fwd, c9b, m = _mix_setup(previous, xdim, len(values))
    opinions = prediction is not None
    pred = np.asarray(prediction, dtype=np.int64) if opinions else np.zeros(1, dtype=np.int64)
    return _mix_decode_known(coded, value, values, xdim, cb, fwd, c9b, m, MIX_ACTIVITY_TABLE,
                             _mix_contexts_count(opinions), opinions, pred)


# ---- Surveying predictor sets (DeltaMapper.surveyBlockSets) -------------------
#
# Predictors are ranked by an estimate of their coded size: a cost per delta
# value per coding context, from reference deltas (each block's best of the
# whole table by sum of |delta|); each predictor's estimated bits per block; a greedy order that charges for
# the map (entropy of the block counts plus a byte per id); real coded sizes
# at SURVEY_COUNTS; blocks assigned by the estimate. survey_channels (what
# the writers use) gives each channel its own block size from
# SURVEY_CHANNEL_SIZES; surveyed_block_maps keeps the last survey. See
# DeltaMapper.java.
# Costs are integers (1/256 bits) so the result matches Java exactly.

SURVEY_COUNTS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 255]
SURVEY_ID_COST = 8 * 256            # a predictor id in the map header, 1/256 bits
SURVEY_SAMPLE_BLOCKS = 1 << 17      # blocks the greedy order is found on, at most
_LOG2 = math.log(2)


def _cost256(r):
    return int(math.floor(256 * math.log(r) / _LOG2 + 0.5))


@njit
def _cost_counts(ref, previous, xdim, ydim, lo, rng, act_table, cross_table, n_cross, n_ctx):
    every = np.zeros(xdim * ydim, dtype=np.int64)
    _delta_contexts(ref, previous, xdim, act_table, cross_table, n_cross, xdim, xdim * ydim, every)
    ctx = np.zeros(xdim * ydim, dtype=np.int64)
    count = np.zeros((n_ctx, rng), dtype=np.int64)
    total = np.zeros(n_ctx, dtype=np.int64)
    for y in range(1, ydim):
        for x in range(1, xdim - 1):
            k = y * xdim + x
            c = every[k]
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
                                                  xdim, ydim, self.lo, self.range, MIX_ACTIVITY_TABLE, MIX_CROSS_TABLE,
                                                  len(CROSS_LIMIT), CONTEXTS)
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
def _block_cost_table(src, ref, has_ref, corr, xdim, ydim, block, cost, ctx, lo, rng, pids, same, at, first, second, n_prim, cross_base, bias_base, rows, r0, r1, est):
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
                for q in at:
                    i = first[pids[q]]
                    j = second[pids[q]]
                    v = prim[i] if i == j else (prim[i] + prim[j] + 1) >> 1
                    est[base + q] += cost[c, min(rng - 1, max(0, s - v - lo))]
        # The rest of each group as its first (each row's estimates start at 0).
        if at.shape[0] < n and y0 < y1:
            for bx in range(bw):
                for q in range(n):
                    if same[q] != q:
                        est[(r * bw + bx) * n + q] = est[(r * bw + bx) * n + same[q]]


@njit
def _prefix_maps(src, ref, has_ref, corr, xdim, ydim, block, cost, ctx, lo, rng, pids, same, at, first, second, n_prim, cross_base, bias_base,
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
        _block_cost_table(src, ref, has_ref, corr, xdim, ydim, block, cost, ctx, lo, rng, pids, same, at, first, second, n_prim,
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
    same, at = same_prediction(ps.has_ref, ids)
    _in_bands(len(rows), lambda lo, hi: _block_cost_table(ps.v, ps.ref, ps.has_ref, ps.corr, ps.xdim, ps.ydim, block, dc.cost, dc.context,
                                                          dc.lo, dc.range, ids, same, at, TABLE_FIRST, TABLE_SECOND,
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
    same, at = same_prediction(ps.has_ref, ids)
    _in_bands(bh, lambda lo, hi: _prefix_maps(ps.v, ps.ref, ps.has_ref, ps.corr, ps.xdim, ps.ydim, block, dc.cost, dc.context,
                                              dc.lo, dc.range, arr, same, at, TABLE_FIRST, TABLE_SECOND, TABLE_PRIMITIVES, CROSS_BASE, BIAS_BASE,
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


def _channel_cost(ps, block, chosen_before, ref=None):
    """The cost model; its reference deltas given (ref), or from the whole
    table (_table_reference_deltas)."""
    if ref is None:
        ref = _table_reference_deltas(ps, block)
    return DeltaCost(ps.v, ref, chosen_before, ps.xdim, ps.ydim)


def get_block_deltas_by_cost(channel, xdim, ydim, block, ids):
    """Each channel's (deltas, map) for its predictor ids, blocks assigned
    by estimated cost as the survey does (DeltaMapper.getBlockDeltasByCost);
    block: one shape for every channel, or a list of each channel's."""
    result, delta = [], []
    ps = pixel_sources(channel, xdim, ydim)
    blocks = list(block) if isinstance(block, (list, tuple)) else [block] * len(channel)
    for i in range(len(channel)):
        block = blocks[i]
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
        self.blocks = None    # survey_channels: each channel's block shape,
        self.deltas = None    # and its deltas and block map as chosen (what
        self.maps = None      # get_block_deltas_by_cost would make from blocks and ids)


def _survey_step(blocks):
    """Big images find the greedy order on every step-th block row (at most
    SURVEY_SAMPLE_BLOCKS blocks), the id's cost scaled to match; the maps are
    then made over every block."""
    return (blocks + SURVEY_SAMPLE_BLOCKS - 1) // SURVEY_SAMPLE_BLOCKS if blocks > SURVEY_SAMPLE_BLOCKS else 1


class ChannelSurvey:
    """One channel's survey at one block shape (DeltaMapper.ChannelSurvey)."""

    def __init__(self, block):
        self.block = block
        self.ids = None
        self.bytes = None
        self.delta = None
        self.map = None


def _survey_channel(ps, block, chosen_before, previous_map, ref=None):
    """One channel of the survey at one block shape: the greedy order by
    estimated bits, then the real coded size of its prefixes at SURVEY_COUNTS
    until two in a row are larger than the best (DeltaMapper.surveyChannel).
    Returns (the ChannelSurvey, its report line)."""
    xdim, ydim = ps.xdim, ps.ydim
    bw = blocks_across(xdim, block)
    bh = blocks_down(ydim, block)
    blocks = bw * bh
    step = _survey_step(blocks)
    rows = np.arange(0, bh, step, dtype=np.int64)
    sampled = len(rows) * bw
    dc = _channel_cost(ps, block, chosen_before, ref)
    est = block_cost_table(ps, block, dc, rows=rows)
    order = greedy_cost_order(est, sampled, SURVEY_ID_COST * sampled // blocks)
    counts = [c for c in SURVEY_COUNTS if c < len(order)] + [len(order)]
    prefix = None
    if step > 1:
        est = None
        prefix = prefix_maps(ps, block, dc, order, counts)
    result = ChannelSurvey(block)
    worse = 0
    line = " (order %d):" % len(order)

    def coded(c):
        ids = order[:counts[c]]
        m = prefix[c] if prefix is not None else _assign_by_cost(est, PREDICTORS, np.array(ids, dtype=np.int64), ids, xdim, ydim, block)
        d = get_block_deltas_from_map(ps, m, ids)
        return ids, d, m, map_bytes(13, m, previous_map, xdim) + len(pack_context_deltas(d, chosen_before, xdim))

    # The counts are coded _survey_width at a time, on cores the other
    # block shapes leave idle; those past where the stopping rule ends are
    # thrown away, so the result (and the report) is the same.
    width = _survey_width(xdim * ydim)
    done = False
    for c0 in range(0, len(counts), width):
        for ids, d, m, size in _bounded_map(coded, range(c0, min(len(counts), c0 + width)), 64 * xdim * ydim):
            line += " %d:%s" % (len(ids), format(size, ","))
            if result.bytes is None or size < result.bytes:
                result.bytes, result.ids, result.delta, result.map, worse = size, list(ids), d, m, 0
                line += "*"
            else:
                worse += 1
                if worse == 2:
                    done = True
                    break
        if done:
            break
    return result, line


SURVEY_WIDTH_PIXELS = 1 << 20     # pixels of work each extra count coded at once needs (see _survey_width)


def _survey_width(pixels):
    """How many of a channel's predictor counts the survey codes at once at
    each block shape: the cores left over when the five block shapes run at
    once, shared out (1 on up to 9 cores, 2 on 10-14, 3 on 15-19, at most 4),
    and no more than one per SURVEY_WIDTH_PIXELS of the channel (so 1 below
    about 2 million pixels): on less, the counts the stopping rule throws
    away cost more than the idle cores save (DeltaMapper.surveyWidth)."""
    return max(1, min(4, _CPUS // 5, pixels // SURVEY_WIDTH_PIXELS))


def _survey_block_size(ps, block, log):
    xdim, ydim = ps[0].xdim, ps[0].ydim
    result = BlockSurvey(block)
    delta, mp = [None] * 3, [None] * 3
    bw = blocks_across(xdim, block)
    bh = blocks_down(ydim, block)
    blocks = bw * bh
    step = _survey_step(blocks)
    sampled = (bh + step - 1) // step * bw
    log.append("  block %s%s:" % (shape_name(block), " (order from %s of %s blocks)" % (format(sampled, ","), format(blocks, ",")) if step > 1 else ""))
    for i in range(3):
        c, line = _survey_channel(ps[i], block, delta[:i], mp[i - 1] if i > 0 else None)
        result.ids[i], delta[i], mp[i] = c.ids, c.delta, c.map
        result.bytes += c.bytes
        log.append("    channel %d" % i + line)
    return result


SURVEY_CHANNEL_SIZES = [4, 6, 8, 12, 16]


def survey_channels(channel, xdim, ydim, sizes=SURVEY_CHANNEL_SIZES):
    """The survey with a block shape per channel (DeltaMapper.surveyChannels):
    channels in order, each surveyed at every shape in sizes (at once, as
    memory allows) given the deltas and map chosen for the channels before
    it, keeping the shape that codes it smallest (the first on a tie).
    result.block is channel 0's; result.blocks has each channel's."""
    ps = pixel_sources(channel, xdim, ydim)
    blocks4 = blocks_across(xdim, 4) * ((ydim - 1 + 3) // 4)
    task = 160 * xdim * ydim + 4 * PREDICTORS * min(SURVEY_SAMPLE_BLOCKS, blocks4)
    result = BlockSurvey(None)
    result.blocks = [None] * len(channel)
    delta, mp, log = [None] * len(channel), [None] * len(channel), []
    for i in range(len(channel)):
        before, prev = delta[:i], mp[i - 1] if i > 0 else None
        # The whole table's reference deltas for every size in one pass.
        ref = dict(zip(sizes, _table_reference_deltas_at(ps[i], sizes)))
        got = _bounded_map(lambda b: _survey_channel(ps[i], b, before, prev, ref[b]), list(sizes), task)
        best = 0
        for t, b in enumerate(sizes):
            log.append("  channel %d, block %s" % (i, shape_name(b)) + got[t][1])
            if got[t][0].bytes < got[best][0].bytes:
                best = t
        c = got[best][0]
        result.blocks[i], result.ids[i] = c.block, c.ids
        result.bytes += c.bytes
        delta[i], mp[i] = c.delta, c.map
    result.block = result.blocks[0]
    result.report = "\n".join(log) + "\n"
    result.deltas, result.maps = delta, mp
    return result


def best_channel_set(qc, xdim, ydim):
    """The channel set whose three channels have the smallest summed entropy
    estimate (get_ideal_frequency2; the first on a tie), as the writers pick
    it (DeltaMapper.bestChannelSet). qc: the six candidate channels."""
    total = [0] * 6

    def one(i):
        total[i] = int(math.floor(float(_shannon_limit(np.asarray(get_ideal_frequency2(qc[i], xdim, ydim), dtype=np.int64)))))
    for f in [_ROW_POOL.submit(one, i) for i in range(6)]:
        f.result()
    sums = [sum(total[c] for c in get_channels(s)) for s in range(10)]
    return min(range(10), key=lambda s: (sums[s], s))


def _copy(a):
    return a.copy() if hasattr(a, "copy") else a


def surveyed_deltas(survey, i, first):
    """A channel's block map from a survey, as get_deltas gives other types:
    (deltas, map, the first value), copies, as the survey is kept for the
    next call (surveyed_block_maps; DeltaMapper.surveyedDeltas)."""
    return _copy(survey.deltas[i]), _copy(survey.maps[i]), int(first)


_SURVEYED = None                # the last surveyed_block_maps: (xdim, channel copies, survey)
_SURVEYED_LOCK = threading.Lock()


def surveyed_block_maps(channel, xdim, ydim):
    """The block maps the survey chooses for three channels (survey_channels
    at SURVEY_CHANNEL_SIZES): each channel's block size, predictor set,
    deltas and map. The last result is kept, and returned again for the same
    channels, so a writer can ask whenever it needs the block map's deltas
    (DeltaMapper.surveyedBlockMaps)."""
    global _SURVEYED
    with _SURVEYED_LOCK:
        channel = [np.asarray(c, dtype=np.int64) for c in channel]
        if _SURVEYED is not None and _SURVEYED[0] == xdim and len(_SURVEYED[1]) == len(channel) \
                and all(np.array_equal(a, b) for a, b in zip(_SURVEYED[1], channel)):
            return _SURVEYED[2]
        survey = survey_channels(channel, xdim, ydim, SURVEY_CHANNEL_SIZES)
        _SURVEYED = (xdim, [c.copy() for c in channel], survey)
        return survey


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
