"""
resize_mapper.py -- ResizeMapper.java (resize, resizeX, resizeY): the
Pixel Resolution resize, by dropping (shrink) or inserting averaged (grow)
columns and rows at evenly spaced points; and resize2 (resizeX2, resizeY2),
the same with those columns and rows centred.

Written with while loops so the loop variables end where Java's for loops
leave them (several places read src[k] or src[k - 1] after a loop).
Numba-compiled; flat row-major int64 arrays in and out.
"""

import numpy as np

from numba_support import njit, jdiv


@njit
def _long_segments(number_of_segments, remainder):
    is_long = np.zeros(number_of_segments, dtype=np.bool_)
    increment = number_of_segments // (remainder + 1)
    index = increment
    for _ in range(remainder):
        is_long[index] = True
        index += increment
    return is_long


@njit
def resize_x(src, xdim, new_xdim):
    ydim = src.shape[0] // xdim
    dst = np.zeros(new_xdim * ydim, dtype=np.int64)
    if new_xdim == xdim:
        dst[:] = src[:xdim * ydim]
    elif new_xdim < xdim:
        number_of_segments = xdim - new_xdim + 1
        remainder = new_xdim % number_of_segments
        if remainder == 0:
            segment_length = new_xdim // number_of_segments
            m = 0
            for i in range(ydim):
                start = i * xdim
                stop = start + segment_length
                for j in range(number_of_segments):
                    k = start
                    while k < stop:
                        dst[m] = src[k]; m += 1; k += 1
                    start += segment_length + 1
                    stop = start + segment_length
        else:
            number_of_segments = xdim - new_xdim
            remainder = new_xdim % number_of_segments
            segment_length = new_xdim // number_of_segments
            if remainder == 0:
                m = 0
                for i in range(ydim):
                    start = i * xdim
                    stop = start + segment_length
                    for j in range(number_of_segments):
                        k = start
                        while k < stop:
                            dst[m] = src[k]; m += 1; k += 1
                        start += segment_length + 1
                        stop = start + segment_length
            else:
                is_long = _long_segments(number_of_segments, remainder)
                m = 0
                for i in range(ydim):
                    start = i * xdim
                    stop = start + segment_length
                    for j in range(number_of_segments):
                        if is_long[j]:
                            stop += 1
                        k = start
                        while k < stop:
                            dst[m] = src[k]; m += 1; k += 1
                        start = stop + 1
                        stop = start + segment_length
    else:
        number_of_segments = new_xdim - xdim + 1
        remainder = xdim % number_of_segments
        if remainder == 0:
            segment_length = xdim // number_of_segments
            m = 0
            k = 0
            for i in range(ydim):
                start = i * xdim
                stop = start + segment_length
                for j in range(number_of_segments - 1):
                    k = start
                    while k < stop:
                        dst[m] = src[k]; m += 1; k += 1
                    dst[m] = jdiv(src[k] + src[k - 1], 2); m += 1
                    start += segment_length
                    stop = start + segment_length
                jj = start
                while jj < stop:
                    dst[m] = src[jj]; m += 1; jj += 1
        else:
            number_of_segments = new_xdim - xdim
            remainder = xdim % number_of_segments
            segment_length = xdim // number_of_segments
            if remainder == 0:
                m = 0
                k = 0
                for i in range(ydim):
                    start = i * xdim
                    stop = start + segment_length
                    for j in range(number_of_segments - 1):
                        k = start
                        while k < stop:
                            dst[m] = src[k]; m += 1; k += 1
                        dst[m] = jdiv(src[k] + src[k - 1], 2); m += 1
                        start += segment_length
                        stop = start + segment_length
                    k = start
                    while k < stop:
                        dst[m] = src[k]; m += 1; k += 1
                    dst[m] = src[k - 1]; m += 1
            else:
                is_long = _long_segments(number_of_segments, remainder)
                m = 0
                k = 0
                for i in range(ydim):
                    start = i * xdim
                    stop = start + segment_length
                    for j in range(number_of_segments - 1):
                        if is_long[j]:
                            stop += 1
                        k = start
                        while k < stop:
                            dst[m] = src[k]; m += 1; k += 1
                        dst[m] = jdiv(src[k] + src[k - 1], 2); m += 1
                        start = stop
                        stop = start + segment_length
                    if is_long[number_of_segments - 1]:   # the last segment can be a long one too
                        stop += 1
                    k = start
                    while k < stop:
                        dst[m] = src[k]; m += 1; k += 1
                    dst[m] = src[k - 1]; m += 1
    return dst


@njit
def resize_y(src, xdim, new_ydim):
    ydim = src.shape[0] // xdim
    dst = np.zeros(xdim * new_ydim, dtype=np.int64)
    if new_ydim == ydim:
        dst[:] = src[:xdim * ydim]
    elif new_ydim < ydim:
        number_of_segments = ydim - new_ydim + 1
        remainder = new_ydim % number_of_segments
        if remainder == 0:
            segment_length = new_ydim // number_of_segments
            for i in range(xdim):
                m = i
                start = i
                stop = start + segment_length * xdim
                for j in range(number_of_segments):
                    k = start
                    while k < stop:
                        dst[m] = src[k]; m += xdim; k += xdim
                    start = stop + xdim
                    stop = start + segment_length * xdim
        else:
            number_of_segments = ydim - new_ydim
            remainder = new_ydim % number_of_segments
            segment_length = new_ydim // number_of_segments
            if remainder == 0:
                for i in range(xdim):
                    m = i
                    start = i
                    stop = start + segment_length * xdim
                    for j in range(number_of_segments):
                        k = start
                        while k < stop:
                            dst[m] = src[k]; m += xdim; k += xdim
                        start = stop + xdim
                        stop = start + segment_length * xdim
            else:
                is_long = _long_segments(number_of_segments, remainder)
                for i in range(xdim):
                    m = i
                    start = i
                    stop = start + segment_length * xdim
                    for j in range(number_of_segments):
                        if is_long[j]:
                            stop += xdim
                        k = start
                        while k < stop:
                            dst[m] = src[k]; m += xdim; k += xdim
                        start = stop + xdim
                        stop = start + segment_length * xdim
    else:
        number_of_segments = new_ydim - ydim + 1
        remainder = ydim % number_of_segments
        if remainder == 0:
            segment_length = ydim // number_of_segments
            for i in range(xdim):
                m = i
                start = i
                stop = start + segment_length * xdim
                for j in range(number_of_segments - 1):
                    k = start
                    while k < stop:
                        dst[m] = src[k]; m += xdim; k += xdim
                    dst[m] = jdiv(src[stop] + src[stop - xdim], 2)
                    m += xdim
                    start = stop
                    stop = start + segment_length * xdim
                stop = start + segment_length * xdim
                k = start
                while k < stop:
                    dst[m] = src[k]; m += xdim; k += xdim
        else:
            number_of_segments = new_ydim - ydim
            remainder = ydim % number_of_segments
            segment_length = ydim // number_of_segments
            if remainder == 0:
                for i in range(xdim):
                    m = i
                    start = i
                    stop = start + segment_length * xdim
                    k = 0
                    for j in range(number_of_segments - 1):
                        k = start
                        while k < stop:
                            dst[m] = src[k]; m += xdim; k += xdim
                        dst[m] = jdiv(src[k] + src[k - xdim], 2)
                        m += xdim
                        start = stop
                        stop = start + segment_length * xdim
                    k = start
                    while k < stop:
                        dst[m] = src[k]; m += xdim; k += xdim
                    dst[m] = src[k - xdim]
            else:
                is_long = _long_segments(number_of_segments, remainder)
                for i in range(xdim):
                    m = i
                    start = i
                    stop = start + segment_length * xdim
                    k = 0
                    j = 0
                    while j < number_of_segments - 1:
                        if is_long[j]:
                            stop += xdim
                        k = start
                        while k < stop:
                            dst[m] = src[k]; m += xdim; k += xdim
                        dst[m] = jdiv(src[k] + src[k - xdim], 2)
                        m += xdim
                        start = stop
                        stop = start + segment_length * xdim
                        j += 1
                    if is_long[j]:
                        stop += xdim
                    k = start
                    while k < stop:
                        dst[m] = src[k]; m += xdim; k += xdim
                    dst[m] = src[k - xdim]
    return dst


def resize(src, xdim, new_xdim, new_ydim):
    """Resizes a flat xdim-wide channel to new_xdim x new_ydim."""
    src = np.asarray(src, dtype=np.int64)
    if new_xdim < xdim:
        return resize_y(resize_x(src, xdim, new_xdim), new_xdim, new_ydim)
    return resize_x(resize_y(src, xdim, new_ydim), xdim, new_xdim)


# ---- Centred (ResizeMapper.resize2) ---------------------------------------------

def centred_positions(n, d):
    """The centred positions of d columns (or rows) in a line of n: spread
    evenly with half a gap at each end, (2j + 1) n / (2d) for the left half
    and mirrored for the right; with an odd d the middle one at (n - 1) // 2
    (ResizeMapper.centredPositions)."""
    p = [0] * d
    for j in range(d // 2):
        p[j] = (2 * j + 1) * n // (2 * d)
        p[d - 1 - j] = n - 1 - p[j]
    if d % 2 == 1:
        p[d // 2] = (n - 1) // 2
    return p


def _centred_map(n, new_n):
    """(a, b): each resized pixel is the average of source pixels a[i] and
    b[i] (the same pixel where it is kept) (ResizeMapper.centredMap)."""
    a = np.zeros(new_n, dtype=np.int64)
    b = np.zeros(new_n, dtype=np.int64)
    if new_n <= n:
        drop = np.zeros(n, dtype=np.bool_)
        drop[centred_positions(n, n - new_n)] = True
        kept = np.flatnonzero(~drop)
        a[:] = kept
        b[:] = kept
    else:
        insert = np.zeros(new_n, dtype=np.bool_)
        insert[centred_positions(new_n, new_n - n)] = True
        kept = np.full(new_n, -1, dtype=np.int64)
        kept[~insert] = np.arange(n)
        for o in range(new_n):
            if kept[o] >= 0:
                a[o] = b[o] = kept[o]
                continue
            l, r = o - 1, o + 1
            while l >= 0 and kept[l] < 0:
                l -= 1
            while r < new_n and kept[r] < 0:
                r += 1
            a[o] = kept[l] if l >= 0 else kept[r]
            b[o] = kept[r] if r < new_n else kept[l]
    return a, b


def _half(s):
    """s / 2 with Java's truncation toward zero."""
    return np.where(s >= 0, s // 2, -((-s) // 2))


def _quarter(s):
    """s / 4 with Java's truncation toward zero."""
    return np.where(s >= 0, s // 4, -((-s) // 4))


def resize_x2(src, xdim, new_xdim):
    """Changes the width with centred columns (ResizeMapper.resizeX2)."""
    src = np.asarray(src, dtype=np.int64)
    ydim = src.shape[0] // xdim
    a, b = _centred_map(xdim, new_xdim)
    m = src.reshape(ydim, xdim)
    return _half(m[:, a] + m[:, b]).reshape(-1)


def resize_y2(src, xdim, new_ydim):
    """Changes the height with centred rows (ResizeMapper.resizeY2)."""
    src = np.asarray(src, dtype=np.int64)
    ydim = src.shape[0] // xdim
    a, b = _centred_map(ydim, new_ydim)
    m = src.reshape(ydim, xdim)
    return _half(m[a, :] + m[b, :]).reshape(-1)


def resize2(src, xdim, new_xdim, new_ydim):
    """As resize, with the dropped (shrinking) or inserted (growing) columns
    and rows centred, so mirroring or rotating before shrinking keeps the same
    original pixels and the edges are kept (ResizeMapper.resize2). Both
    directions in one pass, each pixel the average of up to four source
    pixels with one rounding, so the order of the directions doesn't matter."""
    src = np.asarray(src, dtype=np.int64)
    ydim = src.shape[0] // xdim
    ca, cb = _centred_map(xdim, new_xdim)
    ra, rb = _centred_map(ydim, new_ydim)
    m = src.reshape(ydim, xdim)
    return _quarter(m[np.ix_(ra, ca)] + m[np.ix_(ra, cb)] + m[np.ix_(rb, ca)] + m[np.ix_(rb, cb)]).reshape(-1)
