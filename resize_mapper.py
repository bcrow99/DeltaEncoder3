"""
resize_mapper.py -- ResizeMapper.java (resize, resizeX2, resizeY2): the
Pixel Resolution resize, by dropping (shrink) or inserting averaged (grow)
columns and rows at evenly spaced points.

Written with while loops so the loop variables end where Java's for loops
leave them (several places read src[k] or src[k - 1] after a loop).
Numba-compiled; flat row-major int64 arrays in and out.
"""

import numpy as np

from numba_support import njit, jdiv


@njit
def _long_segments(number_of_segments, remainder):
    is_long = np.zeros(number_of_segments, dtype=np.bool_)
    interval = 1.0 / (remainder + 1)
    increment = int(interval * number_of_segments)
    index = increment
    for _ in range(remainder):
        is_long[index] = True
        index += increment
    return is_long


@njit
def resize_x2(src, xdim, new_xdim):
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
                    k = start
                    while k < stop:
                        dst[m] = src[k]; m += 1; k += 1
                    dst[m] = src[k - 1]; m += 1
    return dst


@njit
def resize_y2(src, xdim, new_ydim):
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
        return resize_y2(resize_x2(src, xdim, new_xdim), new_xdim, new_ydim)
    return resize_x2(resize_y2(src, xdim, new_ydim), xdim, new_xdim)
