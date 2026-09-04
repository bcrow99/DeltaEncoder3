"""
Python translation of DeltaMapper.java

Notes on the translation:
  - Java's `int` arithmetic wraps at 32 bits; this translation uses Python's
    arbitrary-precision ints and does not emulate overflow/wraparound. For the
    pixel-value ranges this code is meant to operate on (8-bit image data and
    small deltas), this makes no observable difference.
  - Java's integer division ("/") truncates toward zero, which differs from
    Python's "//" (floors toward negative infinity) for negative operands.
    A small helper, `jdiv`, reproduces Java's truncating semantics wherever
    the original code used "/" between two ints.
  - Java's ">>" is an arithmetic (sign-propagating) right shift, which is
    equivalent to Python's ">>" for the same operands, so ">>" is translated
    directly.
  - `ArrayList` return values that mix an int, an int[]/list, and possibly a
    byte[]/list are translated to plain Python lists, matching the original
    element order (so callers can still do result[0], result[1], ... exactly
    as in the Java code).
  - `byte[]` maps (values 0-15) are translated to ordinary Python lists of
    ints; Python has no fixed-width byte semantics to preserve here since all
    stored values are small non-negative indices.
  - `Hashtable<Double, ...>` / tie-breaking logic that relied on Java's
    Double object identity semantics is translated using plain Python dicts,
    which behave equivalently for this purpose.
  - `CodeMapper.getShannonLimit(...)` is provided by code_mapper.py (the
    Python translation of CodeMapper.java) as a plain module-level function,
    `code_mapper.get_shannon_limit(frequency)`. It's imported below and used
    at every site that called `CodeMapper.getShannonLimit(...)` in the Java
    source (get_med_scanline_frequency, get_scanline2_frequency,
    get_mixed_deltas4_frequency, get_mixed_deltas_from_values,
    get_mixed_deltas_from_values4). code_mapper.py must be importable
    (i.e. on sys.path alongside this file, together with its own
    string_mapper.py / segment_mapper.py dependencies) for those five
    functions to run; every other function in this module has no such
    dependency.
"""

import math
from typing import List

import numpy as np

import code_mapper

# ---------------------------------------------------------------------------
# Optional Numba acceleration.
#
# Several functions below (the ones profiling showed dominate real-world
# runtime -- see the conversation this was produced in) are pure per-pixel
# numeric loops with no Python-object logic, which CPython interprets one
# bytecode at a time but which a JIT compiler turns into native machine
# code. Numba is an optional accelerator, not a hard dependency: if it
# isn't installed, `njit` below becomes a no-op decorator and every
# function runs exactly as pure Python, at the same correctness (just
# without the speedup). No function's behavior depends on whether numba
# is present -- only its speed does.
#
# Two settings on by default here: `cache=True` persists compiled machine
# code to disk, so the (multi-second, one-time) JIT compilation cost is
# paid once ever on a given machine, not once per process launch.
# `nogil=True` releases Python's GIL for the duration of each compiled
# call, which matters specifically because delta_writer.py/delta_reader.py
# run per-channel and per-segment work on separate threading.Thread
# objects expecting real parallelism (mirroring Java's real OS threads) --
# without nogil=True, @njit alone makes each thread's own work fast but
# the GIL still serializes the threads relative to each other, so multiple
# cores never actually get used at once. Every accelerated function here
# is either pure-numeric on already-typed numpy arrays, or (for the few
# that also accept a plain Python list/bytes and convert it internally)
# confirmed safe with nogil=True: numba transparently reacquires the GIL
# just for that conversion step and releases it again for the rest of the
# call, without changing any result.
try:
    from numba import njit as _njit
    NUMBA_AVAILABLE = True

    def njit(*args, **kwargs):
        kwargs.setdefault("cache", True)
        kwargs.setdefault("nogil", True)
        return _njit(*args, **kwargs)
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        # No-op fallback so every @njit-decorated function below still
        # runs (as plain Python) when numba isn't installed.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn
        return _wrap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@njit
def jdiv(a: int, b: int) -> int:
    """Java-style integer division: truncates toward zero."""
    q, r = divmod(a, b)
    if r != 0 and ((a < 0) != (b < 0)):
        q += 1
    return q


def _to_int64_array(x):
    """Converts src/map_/etc. to an int64 numpy array regardless of
    whether the caller passed a numpy array, a plain Python list, or a
    bytes/bytearray object (delta_reader.py passes bytes for map_ in
    particular -- see the module docstring's note on why this can't just
    happen inside the @njit functions themselves: nopython mode doesn't
    support np.asarray() on a raw bytes object)."""
    if isinstance(x, np.ndarray):
        return x if x.dtype == np.int64 else x.astype(np.int64)
    if isinstance(x, (bytes, bytearray)):
        return np.frombuffer(bytes(x), dtype=np.uint8).astype(np.int64)
    return np.asarray(list(x), dtype=np.int64)


# ---------------------------------------------------------------------------
# Basic array ops
# ---------------------------------------------------------------------------

def get_difference(src1: List[int], src2: List[int]) -> List[int]:
    length = len(src1)
    if len(src2) != length:
        raise ValueError(
            f"get_difference: len(src1) ({length}) != len(src2) ({len(src2)})"
        )
    return [src1[i] - src2[i] for i in range(length)]


def get_sum(src1: List[int], src2: List[int]) -> List[int]:
    length = len(src1)
    if len(src2) != length:
        raise ValueError(
            f"get_sum: len(src1) ({length}) != len(src2) ({len(src2)})"
        )
    return [src1[i] + src2[i] for i in range(length)]


def shift(src: List[int], amount: int) -> List[int]:
    length = len(src)
    shifted_value = [0] * length
    if amount < 0:
        for i in range(length):
            shifted_value[i] = src[i] >> -amount
    else:
        for i in range(length):
            shifted_value[i] = src[i] << amount
    return shifted_value


def get_pixel(blue: List[int], green: List[int], red: List[int], xdim: int, pixel_shift: int) -> List[int]:
    ydim = len(blue) // xdim
    pixel = [0] * len(blue)

    blue_shift = pixel_shift + 16
    green_shift = pixel_shift + 8
    red_shift = pixel_shift

    k = 0
    for i in range(ydim):
        for j in range(xdim):
            pixel[k] = (blue[k] << blue_shift) + (green[k] << green_shift) + (red[k] << red_shift)
            k += 1
    return pixel


# ---------------------------------------------------------------------------
# Unified frequency estimator for delta types 0-4.
# ---------------------------------------------------------------------------

def get_frequency(src: List[int], xdim: int, ydim: int, delta_type: int) -> List[int]:
    delta_list: List[int] = []

    for i in range(1, ydim):
        k = i * xdim + 1
        for j in range(1, xdim - 1):
            if delta_type == 0:
                delta = src[k] - src[k - 1]
            elif delta_type == 1:
                delta = src[k] - src[k - xdim]
            elif delta_type == 2:
                delta = src[k] - jdiv(src[k - 1] + src[k - xdim], 2)
            elif delta_type == 3:
                a = src[k - 1]
                b = src[k - xdim]
                c = src[k - xdim - 1]

                if c >= max(a, b):
                    pred = min(a, b)
                elif c <= min(a, b):
                    pred = max(a, b)
                else:
                    pred = a + b - c

                delta = src[k] - pred
            else:  # delta_type == 4
                a = src[k - 1]
                b = src[k - xdim]
                c = src[k - xdim - 1]
                d = src[k - xdim + 1]

                h_edge = abs(b - c) + abs(b - d)
                v_edge = abs(a - c) + abs(a - b)
                dl_edge = abs(c - d)
                dr_edge = abs(a - d)

                if h_edge >= v_edge and h_edge >= dl_edge and h_edge >= dr_edge:
                    pred = b
                elif v_edge >= dl_edge and v_edge >= dr_edge:
                    pred = a
                elif dl_edge >= dr_edge:
                    pred = d
                else:
                    pred = c

                delta = src[k] - pred

            delta_list.append(delta)
            k += 1

    delta_min = min(delta_list)
    delta_max = max(delta_list)

    range_ = delta_max - delta_min
    frequency = [0] * (range_ + 1)
    for current_value in delta_list:
        frequency[current_value - delta_min] += 1

    return frequency


# ---------------------------------------------------------------------------
# Compact Huffman map encoder / decoder.
# ---------------------------------------------------------------------------

def encode_map_huffman(map_: bytes, n_sym: int, bit_count_out: List[int]) -> bytes:
    """
    bit_count_out must be a length-1 list; bit_count_out[0] is set to the
    total payload bit count (mirrors the Java int[] bit_count_out param).
    """
    freq = [0] * n_sym
    for b in map_:
        freq[b & 0xFF] += 1

    length = huffman_lengths(freq, n_sym)

    max_len = max(length) if length else 0

    codes = huffman_codes(length, n_sym, max_len)

    total_bits = 0
    for s in range(n_sym):
        total_bits += freq[s] * length[s]
    bit_count_out[0] = total_bits

    header = n_sym // 2
    out = bytearray(header + (total_bits + 7) // 8)

    for s in range(n_sym):
        out[s >> 1] |= (length[s] & 0xF) << ((s & 1) << 2)

    bit_pos = 0
    for b in map_:
        s = b & 0xFF
        code = codes[s]
        clen = length[s]
        for bit in range(clen - 1, -1, -1):
            if ((code >> bit) & 1) == 1:
                out[header + (bit_pos >> 3)] |= 1 << (7 - (bit_pos & 7))
            bit_pos += 1

    return bytes(out)


def decode_map_huffman(encoded: bytes, n_sym: int, map_length: int, bit_count: int) -> bytes:
    header = n_sym // 2

    length = [0] * n_sym
    for s in range(n_sym):
        length[s] = (encoded[s >> 1] >> ((s & 1) << 2)) & 0xF

    max_len = max(length) if length else 0
    codes = huffman_codes(length, n_sym, max_len)

    map_ = bytearray(map_length)
    bit_pos = 0

    for q in range(map_length):
        acc = 0
        acc_len = 0
        while True:
            byte_idx = header + (bit_pos >> 3)
            acc = (acc << 1) | ((encoded[byte_idx] >> (7 - (bit_pos & 7))) & 1)
            acc_len += 1
            bit_pos += 1
            found = False
            for s in range(n_sym):
                if length[s] == acc_len and codes[s] == acc:
                    map_[q] = s
                    found = True
                    break
            if found:
                break

    return bytes(map_)


def huffman_lengths(freq: List[int], n_sym: int) -> List[int]:
    length = [0] * n_sym
    used = sum(1 for f in freq if f > 0)

    if used <= 1:
        for s in range(n_sym):
            if freq[s] > 0:
                length[s] = 1
                break
        return length

    node_freq = [0] * (2 * n_sym)
    parent = [-1] * (2 * n_sym)
    active = [False] * (2 * n_sym)

    for s in range(n_sym):
        node_freq[s] = freq[s]
        active[s] = freq[s] > 0
        parent[s] = -1

    next_ = n_sym
    while True:
        m1 = -1
        m2 = -1
        for n in range(next_):
            if not active[n]:
                continue
            if m1 == -1 or node_freq[n] < node_freq[m1]:
                m2 = m1
                m1 = n
            elif m2 == -1 or node_freq[n] < node_freq[m2]:
                m2 = n
        if m2 == -1:
            break

        node_freq[next_] = node_freq[m1] + node_freq[m2]
        parent[m1] = next_
        active[m1] = False
        parent[m2] = next_
        active[m2] = False
        active[next_] = True
        parent[next_] = -1
        next_ += 1

    root = next_ - 1
    for s in range(n_sym):
        if freq[s] == 0:
            continue
        depth = 0
        node = s
        while node != root:
            depth += 1
            node = parent[node]
        length[s] = depth
    return length


def huffman_codes(length: List[int], n_sym: int, max_len: int) -> List[int]:
    if max_len == 0:
        return [0] * n_sym

    bl_count = [0] * (max_len + 1)
    for s in range(n_sym):
        bl_count[length[s]] += 1
    bl_count[0] = 0

    next_code = [0] * (max_len + 2)
    code = 0
    for bits in range(1, max_len + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code

    codes = [0] * n_sym
    for s in range(n_sym):
        if length[s] > 0:
            codes[s] = next_code[length[s]]
            next_code[length[s]] += 1

    return codes


# ---------------------------------------------------------------------------

@njit
def _ideal_frequency_core(src, xdim, ydim):
    n = (ydim - 1) * (xdim - 2)
    delta_list = np.empty(n, dtype=np.int64)
    idx = 0

    for i in range(1, ydim):
        k = i * xdim + 1
        for j in range(1, xdim - 1):
            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]
            d = src[k - xdim + 1]
            e = src[k]

            delta_a = abs(a - e)
            delta_b = abs(b - e)
            delta_c = abs(c - e)
            delta_d = abs(d - e)

            if delta_a <= delta_b and delta_a <= delta_c and delta_a <= delta_d:
                delta = a - e
            elif delta_b <= delta_c and delta_b <= delta_d:
                delta = b - e
            elif delta_c <= delta_d:
                delta = c - e
            else:
                delta = d - e
            delta_list[idx] = delta
            idx += 1

            k += 1

    delta_min = delta_list.min()
    delta_max = delta_list.max()

    range_ = delta_max - delta_min
    frequency = np.zeros(range_ + 1, dtype=np.int64)
    for i in range(n):
        frequency[delta_list[i] - delta_min] += 1

    return frequency


def get_ideal_frequency(src: List[int], xdim: int, ydim: int) -> List[int]:
    src_arr = np.asarray(src, dtype=np.int64)
    return _ideal_frequency_core(src_arr, xdim, ydim).tolist()


@njit
def _ideal_frequency8_core(src, xdim, ydim):
    n = (ydim - 1) * (xdim - 2)
    delta_list = np.empty(n, dtype=np.int64)
    map_freq = np.zeros(8, dtype=np.int64)
    idx = 0
    pred = np.empty(8, dtype=np.int64)

    for i in range(1, ydim):
        for j in range(1, xdim - 1):
            k = i * xdim + j
            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]
            d = src[k - xdim + 1]
            e = src[k]

            pred[0] = a
            pred[1] = jdiv(a + c, 2)
            pred[2] = c
            pred[3] = jdiv(c + b, 2)
            pred[4] = b
            pred[5] = jdiv(b + d, 2)
            pred[6] = d
            pred[7] = jdiv(d + a, 2)

            best_abs = -1
            best_n = 0
            best_delta = 0
            for m in range(8):
                delta = e - pred[m]
                abs_delta = abs(delta)
                if best_abs < 0 or abs_delta < best_abs:
                    best_abs = abs_delta
                    best_n = m
                    best_delta = delta
            delta_list[idx] = best_delta
            idx += 1
            map_freq[best_n] += 1

    delta_min = delta_list.min()
    delta_max = delta_list.max()
    delta_freq = np.zeros(delta_max - delta_min + 1, dtype=np.int64)
    for i in range(n):
        delta_freq[delta_list[i] - delta_min] += 1

    return delta_freq, map_freq


def get_ideal_frequency8(src: List[int], xdim: int, ydim: int):
    src_arr = np.asarray(src, dtype=np.int64)
    delta_freq, map_freq = _ideal_frequency8_core(src_arr, xdim, ydim)
    return [delta_freq.tolist(), map_freq.tolist()]


@njit
def _ideal_frequency16_core(src, xdim, ydim):
    n = (ydim - 1) * (xdim - 2)
    delta_list = np.empty(n, dtype=np.int64)
    map_freq = np.zeros(16, dtype=np.int64)
    idx = 0
    pred = np.empty(16, dtype=np.int64)

    for i in range(1, ydim):
        for j in range(1, xdim - 1):
            k = i * xdim + j
            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]
            d = src[k - xdim + 1]
            e = src[k]

            if c >= max(a, b):
                med = min(a, b)
            elif c <= min(a, b):
                med = max(a, b)
            else:
                med = a + b - c

            pred[0] = a
            pred[1] = c
            pred[2] = b
            pred[3] = d
            pred[4] = (a + c) >> 1
            pred[5] = (c + b) >> 1
            pred[6] = (b + d) >> 1
            pred[7] = (d + a) >> 1
            pred[8] = (a + b) >> 1
            pred[9] = (c + d) >> 1
            pred[10] = (a + b + c + d) >> 2
            pred[11] = med
            pred[12] = (a + b + c) >> 2
            pred[13] = (a + b + d) >> 2
            pred[14] = (a + c + d) >> 2
            pred[15] = (b + c + d) >> 2

            best_abs = -1
            best_delta = 0
            best_n = 0
            for m in range(16):
                delta = e - pred[m]
                abs_delta = abs(delta)
                if best_abs < 0 or abs_delta < best_abs:
                    best_abs = abs_delta
                    best_delta = delta
                    best_n = m
            delta_list[idx] = best_delta
            idx += 1
            map_freq[best_n] += 1

    delta_min = delta_list.min()
    delta_max = delta_list.max()
    delta_freq = np.zeros(delta_max - delta_min + 1, dtype=np.int64)
    for i in range(n):
        delta_freq[delta_list[i] - delta_min] += 1

    return delta_freq, map_freq


def get_ideal_frequency16(src: List[int], xdim: int, ydim: int):
    src_arr = np.asarray(src, dtype=np.int64)
    delta_freq, map_freq = _ideal_frequency16_core(src_arr, xdim, ydim)
    return [delta_freq.tolist(), map_freq.tolist()]


@njit
def _med_scanline_frequency_core(src, xdim, ydim):
    n_out = (ydim - 1) * (xdim - 2)
    delta_list = np.empty(n_out, dtype=np.int64)
    out_idx = 0
    map_ = np.zeros(ydim - 1, dtype=np.int64)

    delta = np.empty((4, xdim - 2), dtype=np.int64)
    limit = np.empty(4, dtype=np.float64)

    # Pass 1: choose best filter per row using Shannon entropy
    # Filters: 0=horizontal, 1=vertical, 2=average, 3=MED
    for i in range(1, ydim):
        k = i * xdim + 1
        for j in range(1, xdim - 1):
            delta[0, j - 1] = src[k] - src[k - 1]
            delta[1, j - 1] = src[k] - src[k - xdim]
            delta[2, j - 1] = src[k] - jdiv(src[k - 1] + src[k - xdim], 2)

            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]

            if c >= max(a, b):
                pred = min(a, b)
            elif c <= min(a, b):
                pred = max(a, b)
            else:
                pred = a + b - c

            delta[3, j - 1] = src[k] - pred
            k += 1

        for f in range(4):
            row = delta[f]
            delta_min = row[0]
            delta_max = row[0]
            for kk in range(1, row.shape[0]):
                if row[kk] < delta_min:
                    delta_min = row[kk]
                elif row[kk] > delta_max:
                    delta_max = row[kk]

            frequency = np.zeros(delta_max - delta_min + 1, dtype=np.float64)
            for kk in range(row.shape[0]):
                frequency[row[kk] - delta_min] += 1
            shannon_limit = code_mapper._shannon_limit_core(frequency)
            limit[f] = math.floor(shannon_limit)

        value = limit[0]
        index = 0
        for kk in range(1, 4):
            if limit[kk] < value:
                value = limit[kk]
                index = kk
        map_[i - 1] = index

    # Pass 2: collect deltas using chosen filter per row
    for i in range(1, ydim):
        k = i * xdim + 1
        m = map_[i - 1]

        if m == 0:
            for j in range(1, xdim - 1):
                delta_list[out_idx] = src[k] - src[k - 1]
                out_idx += 1
                k += 1
        elif m == 1:
            for j in range(1, xdim - 1):
                delta_list[out_idx] = src[k] - src[k - xdim]
                out_idx += 1
                k += 1
        elif m == 2:
            for j in range(1, xdim - 1):
                delta_list[out_idx] = src[k] - jdiv(src[k - 1] + src[k - xdim], 2)
                out_idx += 1
                k += 1
        else:  # m == 3
            for j in range(1, xdim - 1):
                a = src[k - 1]
                b = src[k - xdim]
                c = src[k - xdim - 1]

                if c >= max(a, b):
                    pred = min(a, b)
                elif c <= min(a, b):
                    pred = max(a, b)
                else:
                    pred = a + b - c

                delta_list[out_idx] = src[k] - pred
                out_idx += 1
                k += 1

    delta_min = delta_list.min()
    delta_max = delta_list.max()

    delta_frequency = np.zeros(delta_max - delta_min + 1, dtype=np.int64)
    for i in range(n_out):
        delta_frequency[delta_list[i] - delta_min] += 1

    map_frequency = np.zeros(4, dtype=np.int64)
    for i in range(ydim - 1):
        map_frequency[map_[i]] += 1

    return delta_frequency, map_frequency


def get_med_scanline_frequency(src: List[int], xdim: int, ydim: int):
    src_arr = np.asarray(src, dtype=np.int64)
    delta_frequency, map_frequency = _med_scanline_frequency_core(src_arr, xdim, ydim)
    return [delta_frequency.tolist(), map_frequency.tolist()]


@njit
def _scanline2_frequency_core(src, xdim, ydim):
    n_out = (ydim - 1) * (xdim - 2)
    delta_list = np.empty(n_out, dtype=np.int64)
    out_idx = 0
    map_ = np.zeros(ydim - 1, dtype=np.int64)

    delta = np.empty((4, xdim - 2), dtype=np.int64)
    limit = np.empty(4, dtype=np.float64)

    for i in range(1, ydim):
        k = i * xdim + 1
        for j in range(1, xdim - 1):
            delta[0, j - 1] = src[k] - src[k - 1]
            delta[1, j - 1] = src[k] - src[k - xdim]
            delta[2, j - 1] = src[k] - jdiv(src[k - 1] + src[k - xdim], 2)
            delta[3, j - 1] = src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2)
            k += 1

        for f in range(4):
            row = delta[f]
            delta_min = row[0]
            delta_max = row[0]
            for kk in range(1, row.shape[0]):
                if row[kk] < delta_min:
                    delta_min = row[kk]
                elif row[kk] > delta_max:
                    delta_max = row[kk]

            frequency = np.zeros(delta_max - delta_min + 1, dtype=np.float64)
            for kk in range(row.shape[0]):
                frequency[row[kk] - delta_min] += 1
            shannon_limit = code_mapper._shannon_limit_core(frequency)
            limit[f] = math.floor(shannon_limit)

        value = limit[0]
        index = 0
        for kk in range(1, 4):
            if limit[kk] < value:
                value = limit[kk]
                index = kk
        map_[i - 1] = index

    for i in range(1, ydim):
        k = i * xdim + 1
        m = map_[i - 1]

        if m == 0:
            for j in range(1, xdim - 1):
                delta_list[out_idx] = src[k] - src[k - 1]
                out_idx += 1
                k += 1
        elif m == 1:
            for j in range(1, xdim - 1):
                delta_list[out_idx] = src[k] - src[k - xdim]
                out_idx += 1
                k += 1
        elif m == 2:
            for j in range(1, xdim - 1):
                delta_list[out_idx] = src[k] - jdiv(src[k - 1] + src[k - xdim], 2)
                out_idx += 1
                k += 1
        else:  # m == 3
            for j in range(1, xdim - 1):
                delta_list[out_idx] = src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2)
                out_idx += 1
                k += 1

    delta_min = delta_list.min()
    delta_max = delta_list.max()

    frequency = np.zeros(delta_max - delta_min + 1, dtype=np.int64)
    for i in range(n_out):
        frequency[delta_list[i] - delta_min] += 1

    map_frequency = np.zeros(4, dtype=np.int64)
    for i in range(ydim - 1):
        map_frequency[map_[i]] += 1

    return frequency, map_frequency


def get_scanline2_frequency(src: List[int], xdim: int, ydim: int):
    src_arr = np.asarray(src, dtype=np.int64)
    frequency, map_frequency = _scanline2_frequency_core(src_arr, xdim, ydim)
    return [frequency.tolist(), map_frequency.tolist()]


@njit
def _mixed_deltas4_frequency_core(src, xdim, ydim):
    n_out = (ydim - 1) * (xdim - 2)
    delta_list = np.empty(n_out, dtype=np.int64)
    out_idx = 0
    map_ = np.zeros(ydim - 1, dtype=np.int64)

    delta = np.empty((4, xdim - 2), dtype=np.int64)
    limit = np.empty(4, dtype=np.float64)

    # Pass 1: choose best filter per row using Shannon entropy
    # Filters: 0=horizontal, 1=average, 2=MED, 3=directional
    for i in range(1, ydim):
        k = i * xdim + 1
        for j in range(1, xdim - 1):
            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]
            d = src[k - xdim + 1]

            delta[0, j - 1] = src[k] - a
            delta[1, j - 1] = src[k] - jdiv(a + b, 2)

            if c >= max(a, b):
                med_pred = min(a, b)
            elif c <= min(a, b):
                med_pred = max(a, b)
            else:
                med_pred = a + b - c
            delta[2, j - 1] = src[k] - med_pred

            h_edge = abs(b - c) + abs(b - d)
            v_edge = abs(a - c) + abs(a - b)
            dl_edge = abs(c - d)
            dr_edge = abs(a - d)

            if h_edge >= v_edge and h_edge >= dl_edge and h_edge >= dr_edge:
                dir_pred = b
            elif v_edge >= dl_edge and v_edge >= dr_edge:
                dir_pred = a
            elif dl_edge >= dr_edge:
                dir_pred = d
            else:
                dir_pred = c
            delta[3, j - 1] = src[k] - dir_pred

            k += 1

        for f in range(4):
            row = delta[f]
            delta_min = row[0]
            delta_max = row[0]
            for kk in range(1, row.shape[0]):
                if row[kk] < delta_min:
                    delta_min = row[kk]
                elif row[kk] > delta_max:
                    delta_max = row[kk]

            frequency = np.zeros(delta_max - delta_min + 1, dtype=np.float64)
            for kk in range(row.shape[0]):
                frequency[row[kk] - delta_min] += 1
            shannon_limit = code_mapper._shannon_limit_core(frequency)
            limit[f] = math.floor(shannon_limit)

        value = limit[0]
        index = 0
        for kk in range(1, 4):
            if limit[kk] < value:
                value = limit[kk]
                index = kk
        map_[i - 1] = index

    # Pass 2: collect deltas using chosen filter per row
    for i in range(1, ydim):
        k = i * xdim + 1
        m = map_[i - 1]

        if m == 0:
            for j in range(1, xdim - 1):
                delta_list[out_idx] = src[k] - src[k - 1]
                out_idx += 1
                k += 1
        elif m == 1:
            for j in range(1, xdim - 1):
                delta_list[out_idx] = src[k] - jdiv(src[k - 1] + src[k - xdim], 2)
                out_idx += 1
                k += 1
        elif m == 2:
            for j in range(1, xdim - 1):
                a = src[k - 1]
                b = src[k - xdim]
                c = src[k - xdim - 1]

                if c >= max(a, b):
                    pred = min(a, b)
                elif c <= min(a, b):
                    pred = max(a, b)
                else:
                    pred = a + b - c

                delta_list[out_idx] = src[k] - pred
                out_idx += 1
                k += 1
        else:  # m == 3
            for j in range(1, xdim - 1):
                a = src[k - 1]
                b = src[k - xdim]
                c = src[k - xdim - 1]
                d = src[k - xdim + 1]

                h_edge = abs(b - c) + abs(b - d)
                v_edge = abs(a - c) + abs(a - b)
                dl_edge = abs(c - d)
                dr_edge = abs(a - d)

                if h_edge >= v_edge and h_edge >= dl_edge and h_edge >= dr_edge:
                    pred = b
                elif v_edge >= dl_edge and v_edge >= dr_edge:
                    pred = a
                elif dl_edge >= dr_edge:
                    pred = d
                else:
                    pred = c

                delta_list[out_idx] = src[k] - pred
                out_idx += 1
                k += 1

    delta_min = delta_list.min()
    delta_max = delta_list.max()

    delta_frequency = np.zeros(delta_max - delta_min + 1, dtype=np.int64)
    for i in range(n_out):
        delta_frequency[delta_list[i] - delta_min] += 1

    map_frequency = np.zeros(4, dtype=np.int64)
    for i in range(ydim - 1):
        map_frequency[map_[i]] += 1

    return delta_frequency, map_frequency


def get_mixed_deltas4_frequency(src: List[int], xdim: int, ydim: int):
    src_arr = np.asarray(src, dtype=np.int64)
    delta_frequency, map_frequency = _mixed_deltas4_frequency_core(src_arr, xdim, ydim)
    return [delta_frequency.tolist(), map_frequency.tolist()]


# ---------------------------------------------------------------------------
# Delta encoders / decoders
# ---------------------------------------------------------------------------

def get_horizontal_deltas_from_values(src: List[int], xdim: int, ydim: int) -> list:
    dst = [0] * (xdim * ydim)
    total_sum = 0
    init_value = src[0]
    original_init_value = src[0]
    value = init_value

    k = 0
    for i in range(ydim):
        if i == 0:
            dst[k] = 0
            k += 1
        else:
            delta = src[k] - init_value
            dst[k] = delta
            k += 1
            init_value += delta
            total_sum += abs(delta)
            value = init_value

        for j in range(1, xdim):
            delta = src[k] - value
            value += delta
            total_sum += abs(delta)
            dst[k] = delta
            k += 1

    return [total_sum, dst, original_init_value]


def get_values_from_horizontal_deltas(src: List[int], xdim: int, ydim: int, init_value: int) -> List[int]:
    dst = [0] * (xdim * ydim)

    k = 0
    value = init_value
    for i in range(ydim):
        if i != 0:
            value += src[k]
        current_value = value
        dst[k] = current_value
        k += 1
        for j in range(1, xdim):
            current_value += src[k]
            dst[k] = current_value
            k += 1
    return dst


def get_vertical_deltas_from_values(src: List[int], xdim: int, ydim: int) -> list:
    dst = [0] * (xdim * ydim)
    init_value = src[0]
    value = init_value
    delta = 0
    total_sum = 0

    k = 0
    for i in range(ydim):
        for j in range(xdim):
            if i == 0:
                if j == 0:
                    dst[k] = 0
                    k += 1
                else:
                    delta = src[k] - value
                    value += delta
                    dst[k] = delta
                    k += 1
                    total_sum += abs(delta)
            else:
                delta = src[k] - src[k - xdim]
                dst[k] = delta
                k += 1
                total_sum += abs(delta)

    return [total_sum, dst, init_value]


def get_values_from_vertical_deltas(src: List[int], xdim: int, ydim: int, init_value: int) -> List[int]:
    dst = [0] * (xdim * ydim)
    dst[0] = init_value
    value = init_value

    for i in range(1, xdim):
        value += src[i]
        dst[i] = value

    for i in range(1, ydim):
        for j in range(xdim):
            index = i * xdim + j
            dst[index] = dst[index - xdim] + src[index]

    return dst


def get_average_deltas_from_values(src: List[int], xdim: int, ydim: int) -> list:
    dst = [0] * (xdim * ydim)
    total_sum = 0
    init_value = src[0]

    k = 0
    dst[k] = 0
    k += 1
    for i in range(1, xdim):
        delta = src[k] - src[k - 1]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

    for i in range(1, ydim):
        delta = src[k] - src[k - xdim]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

        for j in range(1, xdim):
            delta = src[k] - jdiv(src[k - 1] + src[k - xdim], 2)
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

    return [total_sum, dst, init_value]


def get_values_from_average_deltas(src: List[int], xdim: int, ydim: int, init_value: int) -> List[int]:
    dst = [0] * (xdim * ydim)
    k = 0
    dst[k] = init_value
    k += 1

    for i in range(1, xdim):
        value = dst[k - 1] + src[k]
        dst[k] = value
        k += 1

    for i in range(1, ydim):
        value = dst[k - xdim] + src[k]
        dst[k] = value
        k += 1
        for j in range(1, xdim):
            value = jdiv(dst[k - 1] + dst[k - xdim], 2) + src[k]
            dst[k] = value
            k += 1

    return dst


def get_paeth_deltas_from_values(src: List[int], xdim: int, ydim: int) -> list:
    dst = [0] * (xdim * ydim)
    init_value = src[0]
    original_init_value = src[0]
    value = init_value
    delta = 0
    total_sum = 0
    k = 0

    for i in range(ydim):
        if i == 0:
            for j in range(xdim):
                if j == 0:
                    dst[k] = 0
                    k += 1
                else:
                    delta = src[k] - value
                    value += delta
                    dst[k] = delta
                    k += 1
                    total_sum += abs(delta)
        else:
            for j in range(xdim):
                if j == 0:
                    delta = src[k] - init_value
                    init_value = src[k]
                    dst[k] = delta
                    k += 1
                    total_sum += abs(delta)
                else:
                    a = src[k - 1]
                    b = src[k - xdim]
                    c = src[k - xdim - 1]
                    d = a + b - c

                    delta_a = abs(a - d)
                    delta_b = abs(b - d)
                    delta_c = abs(c - d)

                    if delta_a <= delta_b and delta_a <= delta_c:
                        delta = src[k] - src[k - 1]
                    elif delta_b <= delta_c:
                        delta = src[k] - src[k - xdim]
                    else:
                        delta = src[k] - src[k - xdim - 1]

                    dst[k] = delta
                    k += 1
                    total_sum += abs(delta)

    return [total_sum, dst, original_init_value]


def get_values_from_paeth_deltas(src: List[int], xdim: int, ydim: int, init_value: int) -> List[int]:
    dst = [0] * (xdim * ydim)
    dst[0] = init_value
    value = init_value

    for i in range(1, xdim):
        value += src[i]
        dst[i] = value

    for i in range(1, ydim):
        for j in range(xdim):
            if j == 0:
                init_value += src[i * xdim]
                dst[i * xdim] = init_value
                value = init_value
            else:
                a = dst[i * xdim + j - 1]
                b = dst[(i - 1) * xdim + j]
                c = dst[(i - 1) * xdim + j - 1]
                d = a + b - c

                delta_a = abs(a - d)
                delta_b = abs(b - d)
                delta_c = abs(c - d)

                if delta_a <= delta_b and delta_a <= delta_c:
                    dst[i * xdim + j] = a + src[i * xdim + j]
                elif delta_b <= delta_c:
                    dst[i * xdim + j] = b + src[i * xdim + j]
                else:
                    dst[i * xdim + j] = c + src[i * xdim + j]
    return dst


@njit
def _med_deltas_from_values_core(src, xdim, ydim):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    total_sum = 0
    k = 0

    dst[k] = 0
    k += 1
    for j in range(1, xdim):
        delta = src[k] - src[k - 1]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

    for i in range(1, ydim):
        delta = src[k] - src[k - xdim]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

        for j in range(1, xdim):
            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]

            if c >= max(a, b):
                pred = min(a, b)
            elif c <= min(a, b):
                pred = max(a, b)
            else:
                pred = a + b - c

            delta = src[k] - pred
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

    return total_sum, dst


def get_med_deltas_from_values(src: List[int], xdim: int, ydim: int) -> list:
    src_arr = np.asarray(src, dtype=np.int64)
    total_sum, dst = _med_deltas_from_values_core(src_arr, xdim, ydim)
    return [int(total_sum), dst.tolist(), int(src_arr[0])]


@njit
def get_values_from_med_deltas(src, xdim: int, ydim: int, init_value: int) -> List[int]:
    src = np.asarray(src, dtype=np.int64)
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    k = 0

    dst[k] = init_value
    k += 1
    for j in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
        k += 1

    for i in range(1, ydim):
        dst[k] = dst[k - xdim] + src[k]
        k += 1

        for j in range(1, xdim):
            a = dst[k - 1]
            b = dst[k - xdim]
            c = dst[k - xdim - 1]

            if c >= max(a, b):
                pred = min(a, b)
            elif c <= min(a, b):
                pred = max(a, b)
            else:
                pred = a + b - c

            dst[k] = pred + src[k]
            k += 1

    return dst


@njit
def _directional_deltas_from_values_core(src, xdim, ydim):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    total_sum = 0
    k = 0

    dst[k] = 0
    k += 1
    for j in range(1, xdim):
        delta = src[k] - src[k - 1]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

    for i in range(1, ydim):
        delta = src[k] - src[k - xdim]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

        for j in range(1, xdim - 1):
            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]
            d = src[k - xdim + 1]

            h_edge = abs(b - c) + abs(b - d)
            v_edge = abs(a - c) + abs(a - b)
            dl_edge = abs(c - d)
            dr_edge = abs(a - d)

            if h_edge >= v_edge and h_edge >= dl_edge and h_edge >= dr_edge:
                pred = b
            elif v_edge >= dl_edge and v_edge >= dr_edge:
                pred = a
            elif dl_edge >= dr_edge:
                pred = d
            else:
                pred = c

            delta = src[k] - pred
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

        # Last column: no above-right, fall back to MED
        a = src[k - 1]
        b = src[k - xdim]
        c = src[k - xdim - 1]

        if c >= max(a, b):
            pred = min(a, b)
        elif c <= min(a, b):
            pred = max(a, b)
        else:
            pred = a + b - c

        delta = src[k] - pred
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

    return total_sum, dst


def get_directional_deltas_from_values(src: List[int], xdim: int, ydim: int) -> list:
    src_arr = np.asarray(src, dtype=np.int64)
    total_sum, dst = _directional_deltas_from_values_core(src_arr, xdim, ydim)
    return [int(total_sum), dst.tolist(), int(src_arr[0])]


@njit
def get_values_from_directional_deltas(src, xdim: int, ydim: int, init_value: int) -> List[int]:
    src = np.asarray(src, dtype=np.int64)
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    k = 0

    dst[k] = init_value
    k += 1
    for j in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
        k += 1

    for i in range(1, ydim):
        dst[k] = dst[k - xdim] + src[k]
        k += 1

        for j in range(1, xdim - 1):
            a = dst[k - 1]
            b = dst[k - xdim]
            c = dst[k - xdim - 1]
            d = dst[k - xdim + 1]

            h_edge = abs(b - c) + abs(b - d)
            v_edge = abs(a - c) + abs(a - b)
            dl_edge = abs(c - d)
            dr_edge = abs(a - d)

            if h_edge >= v_edge and h_edge >= dl_edge and h_edge >= dr_edge:
                pred = b
            elif v_edge >= dl_edge and v_edge >= dr_edge:
                pred = a
            elif dl_edge >= dr_edge:
                pred = d
            else:
                pred = c

            dst[k] = pred + src[k]
            k += 1

        # Last column: MED
        a = dst[k - 1]
        b = dst[k - xdim]
        c = dst[k - xdim - 1]

        if c >= max(a, b):
            pred = min(a, b)
        elif c <= min(a, b):
            pred = max(a, b)
        else:
            pred = a + b - c

        dst[k] = pred + src[k]
        k += 1

    return dst


def get_gradient_deltas_from_values(src: List[int], xdim: int, ydim: int) -> list:
    dst = [0] * (xdim * ydim)
    gradient = [0] * 4
    init_value = src[0]
    original_init_value = src[0]
    total_sum = 0
    delta = 0
    k = 0

    dst[k] = 0
    k += 1
    for i in range(1, xdim):
        delta = src[k] - src[k - 1]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

    delta = src[k] - init_value
    dst[k] = delta
    k += 1
    init_value += delta
    total_sum += abs(delta)

    for i in range(1, xdim):
        delta = src[k] - src[k - 1]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

    for i in range(2, ydim):
        delta = src[k] - init_value
        dst[k] = delta
        k += 1
        init_value += delta
        total_sum += abs(delta)

        for j in range(1, xdim - 1):
            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]
            d = src[k - xdim + 1]
            e = src[k - 2 * xdim - 1]

            gradient[0] = abs(a - e)
            gradient[1] = abs(c - d)
            gradient[2] = abs(a - d)
            gradient[3] = abs(b - e)

            max_value = gradient[0]
            max_index = 0
            for m in range(1, 4):
                if gradient[m] > max_value:
                    max_value = gradient[m]
                    max_index = m

            if max_index == 0:
                delta = src[k] - src[k - 1]
            elif max_index == 1:
                delta = src[k] - src[k - xdim]
            elif max_index == 2:
                delta = src[k] - src[k - xdim - 1]
            else:
                delta = src[k] - src[k - xdim + 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

        delta = src[k] - src[k - 1]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

    return [total_sum, dst, original_init_value]


def get_values_from_gradient_deltas(src: List[int], xdim: int, ydim: int, init_value: int) -> List[int]:
    dst = [0] * (xdim * ydim)
    gradient = [0] * 4
    k = 0

    dst[k] = init_value
    k += 1
    for i in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
        k += 1

    init_value += src[k]
    dst[k] = init_value
    k += 1
    for i in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
        k += 1

    for i in range(2, ydim):
        init_value += src[k]
        dst[k] = init_value
        k += 1

        for j in range(1, xdim - 1):
            a = dst[k - 1]
            b = dst[k - xdim]
            c = dst[k - xdim - 1]
            d = dst[k - xdim + 1]
            e = dst[k - 2 * xdim - 1]

            gradient[0] = abs(a - e)
            gradient[1] = abs(c - d)
            gradient[2] = abs(a - d)
            gradient[3] = abs(b - e)

            max_value = gradient[0]
            max_index = 0
            for m in range(1, 4):
                if gradient[m] > max_value:
                    max_value = gradient[m]
                    max_index = m

            if max_index == 0:
                dst[k] = dst[k - 1] + src[k]
            elif max_index == 1:
                dst[k] = dst[k - xdim] + src[k]
            elif max_index == 2:
                dst[k] = dst[k - xdim - 1] + src[k]
            else:
                dst[k] = dst[k - xdim + 1] + src[k]
            k += 1

        dst[k] = dst[k - 1] + src[k]
        k += 1

    return dst


def get_gradient_deltas_from_values2(src: List[int], xdim: int, ydim: int) -> list:
    dst = [0] * (xdim * ydim)
    gradient = [0] * 4
    init_value = src[0]
    original_init_value = src[0]
    total_sum = 0
    k = 0

    dst[k] = 0
    k += 1
    for i in range(1, xdim):
        delta = src[k] - src[k - 1]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

    for i in range(1, ydim):
        delta = src[k] - init_value
        dst[k] = delta
        k += 1
        init_value += delta
        total_sum += abs(delta)

        delta = src[k] - src[k - 1]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

        for j in range(2, xdim - 1):
            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]
            d = src[k - xdim + 1]
            e = src[k - xdim - 2]

            gradient[0] = abs(c - b)
            gradient[1] = abs(c - a)
            gradient[2] = abs(a - b)
            gradient[3] = abs(a - e)

            max_value = gradient[0]
            max_index = 0
            for m in range(1, 4):
                if gradient[m] > max_value:
                    max_value = gradient[m]
                    max_index = m

            if max_index == 0:
                delta = src[k] - src[k - 1]
            elif max_index == 1:
                delta = src[k] - src[k - xdim]
            elif max_index == 2:
                delta = src[k] - src[k - xdim - 1]
            else:
                delta = src[k] - src[k - xdim + 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

        delta = src[k] - src[k - 1]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

    return [total_sum, dst, original_init_value]


def get_values_from_gradient_deltas2(src: List[int], xdim: int, ydim: int, init_value: int) -> List[int]:
    dst = [0] * (xdim * ydim)
    gradient = [0] * 4
    k = 0

    dst[k] = init_value
    k += 1
    for i in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
        k += 1

    for i in range(1, ydim):
        init_value += src[k]
        dst[k] = init_value
        k += 1
        dst[k] = dst[k - 1] + src[k]
        k += 1

        for j in range(2, xdim - 1):
            a = dst[k - 1]
            b = dst[k - xdim]
            c = dst[k - xdim - 1]
            d = dst[k - xdim + 1]
            e = dst[k - xdim - 2]

            gradient[0] = abs(c - b)
            gradient[1] = abs(c - a)
            gradient[2] = abs(a - b)
            gradient[3] = abs(a - e)

            max_value = gradient[0]
            max_index = 0
            for m in range(1, 4):
                if gradient[m] > max_value:
                    max_value = gradient[m]
                    max_index = m

            if max_index == 0:
                dst[k] = dst[k - 1] + src[k]
            elif max_index == 1:
                dst[k] = dst[k - xdim] + src[k]
            elif max_index == 2:
                dst[k] = dst[k - xdim - 1] + src[k]
            else:
                dst[k] = dst[k - xdim + 1] + src[k]
            k += 1

        dst[k] = dst[k - 1] + src[k]
        k += 1

    return dst


def get_mixed_deltas_from_values(src: List[int], xdim: int, ydim: int) -> list:
    map_ = [0] * (ydim - 1)

    for i in range(1, ydim):
        delta = [[0] * (xdim - 2) for _ in range(4)]

        k = i * xdim + 1
        for j in range(1, xdim - 1):
            delta[0][j - 1] = src[k] - src[k - 1]
            delta[1][j - 1] = src[k] - src[k - xdim]
            delta[2][j - 1] = src[k] - jdiv(src[k - 1] + src[k - xdim], 2)

            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]

            if c >= max(a, b):
                pred = min(a, b)
            elif c <= min(a, b):
                pred = max(a, b)
            else:
                pred = a + b - c

            delta[3][j - 1] = src[k] - pred
            k += 1

        limit = [0] * 4
        for j in range(4):
            current_delta = delta[j]

            delta_min = current_delta[0]
            delta_max = current_delta[0]
            for kk in range(1, len(current_delta)):
                if current_delta[kk] < delta_min:
                    delta_min = current_delta[kk]
                elif current_delta[kk] > delta_max:
                    delta_max = current_delta[kk]

            for kk in range(len(current_delta)):
                current_delta[kk] -= delta_min
            range_ = delta_max - delta_min
            frequency = [0] * (range_ + 1)
            for kk in range(len(current_delta)):
                frequency[current_delta[kk]] += 1
            shannon_limit = code_mapper.get_shannon_limit(frequency)
            limit[j] = math.floor(shannon_limit)

        value = limit[0]
        index = 0
        for kk in range(1, 4):
            if limit[kk] < value:
                value = limit[kk]
                index = kk
        map_[i - 1] = index

    dst = [0] * (xdim * ydim)
    init_value = src[0]
    original_init_value = src[0]
    total_sum = 0
    k = 0

    dst[k] = 0
    k += 1
    delta = src[k] - init_value
    value = src[k]
    dst[k] = delta
    k += 1
    total_sum += abs(delta)

    for i in range(2, xdim):
        delta = src[k] - value
        value = src[k]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

    for i in range(1, ydim):
        delta = src[k] - init_value
        init_value = src[k]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

        m = map_[i - 1]

        if m == 0:
            for j in range(1, xdim - 1):
                delta = src[k] - src[k - 1]
                dst[k] = delta
                k += 1
            delta = src[k] - src[k - 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)
        elif m == 1:
            for j in range(1, xdim - 1):
                delta = src[k] - src[k - xdim]
                dst[k] = delta
                k += 1
            delta = src[k] - src[k - 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)
        elif m == 2:
            for j in range(1, xdim - 1):
                delta = src[k] - jdiv(src[k - 1] + src[k - xdim], 2)
                dst[k] = delta
                k += 1
            delta = src[k] - src[k - 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)
        elif m == 3:
            for j in range(1, xdim - 1):
                a = src[k - 1]
                b = src[k - xdim]
                c = src[k - xdim - 1]

                if c >= max(a, b):
                    pred = min(a, b)
                elif c <= min(a, b):
                    pred = max(a, b)
                else:
                    pred = a + b - c

                delta = src[k] - pred
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
            delta = src[k] - src[k - 1]
            dst[k] = delta
            k += 1

    return [total_sum, dst, map_, original_init_value]


def get_values_from_mixed_deltas(src: List[int], xdim: int, ydim: int, init_value: int, map_: List[int]) -> List[int]:
    dst = [0] * (xdim * ydim)
    k = 0
    value = init_value

    dst[k] = init_value
    k += 1
    for i in range(1, xdim):
        value += src[k]
        dst[k] = value
        k += 1

    for i in range(1, ydim):
        init_value += src[k]
        dst[k] = init_value
        k += 1

        m = map_[i - 1]
        if m == 0:
            for j in range(1, xdim - 1):
                value = dst[k - 1] + src[k]
                dst[k] = value
                k += 1
        elif m == 1:
            for j in range(1, xdim - 1):
                value = dst[k - xdim] + src[k]
                dst[k] = value
                k += 1
        elif m == 2:
            for j in range(1, xdim - 1):
                value = jdiv(dst[k - xdim] + dst[k - 1], 2) + src[k]
                dst[k] = value
                k += 1
        elif m == 3:
            for j in range(1, xdim - 1):
                a = dst[k - 1]
                b = dst[k - xdim]
                c = dst[k - xdim - 1]

                if c >= max(a, b):
                    pred = min(a, b)
                elif c <= min(a, b):
                    pred = max(a, b)
                else:
                    pred = a + b - c

                value = pred + src[k]
                dst[k] = value
                k += 1
        value = dst[k - 1] + src[k]
        dst[k] = value
        k += 1

    return dst


def get_mixed_deltas_from_values2(src: List[int], xdim: int, ydim: int) -> list:
    map_ = [0] * (ydim - 1)

    for i in range(1, ydim):
        sum_ = [0] * 4

        for j in range(1, xdim - 1):
            k = i * xdim + j
            sum_[0] += abs(src[k] - src[k - 1])
            sum_[1] += abs(src[k] - src[k - xdim])
            sum_[2] += abs(src[k] - jdiv(src[k - 1] + src[k - xdim], 2))
            sum_[3] += abs(src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2))

        value = sum_[0]
        index = 0
        for k in range(1, 4):
            if sum_[k] < value:
                value = sum_[k]
                index = k
        map_[i - 1] = index

    dst = [0] * (xdim * ydim)
    init_value = src[0]
    original_init_value = src[0]
    value = init_value
    total_sum = 0

    for i in range(ydim):
        if i == 0:
            for j in range(xdim):
                if j == 0:
                    dst[j] = 0
                else:
                    delta = src[j] - value
                    value += delta
                    dst[j] = delta
        else:
            k = i * xdim
            delta = src[k] - init_value
            init_value = src[k]
            dst[k] = delta
            k += 1

            m = map_[i - 1]

            for j in range(1, xdim - 1):
                if m == 0:
                    delta = src[k] - src[k - 1]
                elif m == 1:
                    delta = src[k] - src[k - xdim]
                elif m == 2:
                    delta = src[k] - jdiv(src[k - 1] + src[k - xdim], 2)
                else:
                    delta = src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2)  # m == 3

                dst[k] = delta
                k += 1
                total_sum += abs(delta)

            delta = src[k] - src[k - 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

    return [total_sum, dst, map_, original_init_value]


def get_values_from_mixed_deltas2(src: List[int], xdim: int, ydim: int, init_value: int, map_: List[int]) -> List[int]:
    dst = [0] * (xdim * ydim)
    dst[0] = init_value
    value = init_value

    for i in range(1, xdim):
        value += src[i]
        dst[i] = value

    for i in range(1, ydim):
        m = map_[i - 1]
        for j in range(xdim):
            k = i * xdim + j
            if j == 0:
                init_value += src[k]
                dst[k] = init_value
            elif j < xdim - 1:
                if m == 0:
                    value = dst[k - 1]
                elif m == 1:
                    value = dst[k - xdim]
                elif m == 2:
                    value = jdiv(dst[k - 1] + dst[k - xdim], 2)
                elif m == 3:
                    value = jdiv(dst[k - 1] + dst[k - xdim + 1], 2)

                value += src[k]
                dst[k] = value
            else:
                value = dst[k - 1] + src[k]
                dst[k] = value
    return dst


def get_mixed_deltas_from_values3(src: List[int], xdim: int, ydim: int) -> list:
    line_map = [0] * (ydim - 1)
    m = 0
    for i in range(1, ydim):
        sum_ = [0] * 5

        for j in range(1, xdim - 1):
            k = i * xdim + j
            sum_[0] += abs(src[k] - src[k - 1])
            sum_[1] += abs(src[k] - src[k - xdim])
            sum_[2] += abs(src[k] - src[k - xdim - 1])
            sum_[3] += abs(src[k] - jdiv(src[k - 1] + src[k - xdim], 2))
            sum_[4] += abs(src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2))

        key_list = []
        delta_table = {}

        current_key = float(sum_[0])
        addend = 0.00000001

        key_list.append(current_key)
        delta_table[current_key] = 0
        for k in range(1, 5):
            current_key = float(sum_[k])
            if current_key in key_list:
                current_key += addend
                addend *= 2.0
            key_list.append(current_key)
            delta_table[current_key] = k

        key_list.sort()

        first_key = key_list[0]
        first_type = delta_table[first_key]
        second_key = key_list[1]
        second_type = delta_table[second_key]

        type_list = [first_type, second_type]

        if 0 in type_list and 1 in type_list:
            line_map[m] = 0
        elif 0 in type_list and 2 in type_list:
            line_map[m] = 1
        elif 0 in type_list and 3 in type_list:
            line_map[m] = 2
        elif 0 in type_list and 4 in type_list:
            line_map[m] = 3
        elif 1 in type_list and 2 in type_list:
            line_map[m] = 4
        elif 1 in type_list and 3 in type_list:
            line_map[m] = 5
        elif 1 in type_list and 4 in type_list:
            line_map[m] = 6
        elif 2 in type_list and 3 in type_list:
            line_map[m] = 7
        elif 2 in type_list and 4 in type_list:
            line_map[m] = 8
        else:
            line_map[m] = 9
        m += 1

    pixel_map = [0] * ((xdim - 2) * (ydim - 1))
    n = 0
    for i in range(1, ydim):
        m = line_map[i - 1]
        value = [0, 0]
        for j in range(1, xdim - 1):
            k = i * xdim + j
            if m == 0 or m == 1:
                value[0] += abs(src[k] - src[k - 1])
                value[1] += abs(src[k] - src[k - xdim])
            elif m == 2:
                value[0] += abs(src[k] - src[k - 1])
                value[1] += abs(src[k] - jdiv(src[k - 1] + src[k - xdim], 2))
            elif m == 3:
                value[0] += abs(src[k] - src[k - 1])
                value[1] += abs(src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2))
            elif m == 4:
                value[0] += abs(src[k] - src[k - xdim])
                value[1] += abs(src[k] - src[k - xdim - 1])
            elif m == 5:
                value[0] += abs(src[k] - src[k - xdim])
                value[1] += abs(src[k] - jdiv(src[k - 1] + src[k - xdim], 2))
            elif m == 6:
                value[0] += abs(src[k] - src[k - xdim])
                value[1] += abs(src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2))
            elif m == 7:
                value[0] += abs(src[k] - src[k - xdim - 1])
                value[1] += abs(src[k] - jdiv(src[k - 1] + src[k - xdim], 2))
            elif m == 8:
                value[0] += abs(src[k] - src[k - xdim - 1])
                value[1] += abs(src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2))
            else:
                value[0] += abs(src[k] - jdiv(src[k - 1] + src[k - xdim], 2))
                value[1] += abs(src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2))
            pixel_map[n] = 0 if value[0] <= value[1] else 1
            n += 1

    dst = [0] * (xdim * ydim)
    init_value = src[0]
    original_init_value = src[0]
    value = init_value
    total_sum = 0
    p = 0

    for i in range(ydim):
        if i == 0:
            for j in range(xdim):
                if j == 0:
                    dst[j] = 0
                else:
                    delta = src[j] - value
                    value += delta
                    dst[j] = delta
        else:
            k = i * xdim
            delta = src[k] - init_value
            init_value = src[k]
            dst[k] = delta
            k += 1

            m = line_map[i - 1]

            for j in range(1, xdim - 1):
                n = pixel_map[p]
                p += 1
                if m == 0:
                    delta = (src[k] - src[k - 1]) if n == 0 else (src[k] - src[k - xdim])
                elif m == 1:
                    delta = (src[k] - src[k - 1]) if n == 0 else (src[k] - src[k - xdim - 1])
                elif m == 2:
                    delta = (src[k] - src[k - 1]) if n == 0 else (src[k] - jdiv(src[k - 1] + src[k - xdim], 2))
                elif m == 3:
                    delta = (src[k] - src[k - 1]) if n == 0 else (src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2))
                elif m == 4:
                    delta = (src[k] - src[k - xdim]) if n == 0 else (src[k] - src[k - xdim - 1])
                elif m == 5:
                    delta = (src[k] - src[k - xdim]) if n == 0 else (src[k] - jdiv(src[k - 1] + src[k - xdim], 2))
                elif m == 6:
                    delta = (src[k] - src[k - xdim]) if n == 0 else (src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2))
                elif m == 7:
                    delta = (src[k] - src[k - xdim - 1]) if n == 0 else (src[k] - jdiv(src[k - 1] + src[k - xdim], 2))
                elif m == 8:
                    delta = (src[k] - src[k - xdim - 1]) if n == 0 else (src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2))
                else:
                    delta = (src[k] - jdiv(src[k - 1] + src[k - xdim], 2)) if n == 0 else (src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2))
                dst[k] = delta
                k += 1
                total_sum += abs(delta)

            delta = src[k] - src[k - 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

    return [total_sum, dst, line_map, pixel_map, original_init_value]


def get_values_from_mixed_deltas3(src: List[int], xdim: int, ydim: int, init_value: int,
                                   line_map: List[int], pixel_map: List[int]) -> List[int]:
    dst = [0] * (xdim * ydim)
    dst[0] = init_value
    value = init_value

    for i in range(1, xdim):
        value += src[i]
        dst[i] = value

    p = 0
    for i in range(1, ydim):
        m = line_map[i - 1]
        for j in range(xdim):
            k = i * xdim + j
            if j == 0:
                init_value += src[k]
                dst[k] = init_value
            elif j < xdim - 1:
                n = pixel_map[p]
                p += 1
                if m == 0:
                    value = dst[k - 1] if n == 0 else dst[k - xdim]
                elif m == 1:
                    value = dst[k - 1] if n == 0 else dst[k - xdim - 1]
                elif m == 2:
                    value = dst[k - 1] if n == 0 else jdiv(dst[k - 1] + dst[k - xdim], 2)
                elif m == 3:
                    value = dst[k - 1] if n == 0 else jdiv(dst[k - 1] + dst[k - xdim + 1], 2)
                elif m == 4:
                    value = dst[k - xdim] if n == 0 else dst[k - xdim - 1]
                elif m == 5:
                    value = dst[k - xdim] if n == 0 else jdiv(dst[k - 1] + dst[k - xdim], 2)
                elif m == 6:
                    value = dst[k - xdim] if n == 0 else jdiv(dst[k - 1] + dst[k - xdim + 1], 2)
                elif m == 7:
                    value = dst[k - xdim - 1] if n == 0 else jdiv(dst[k - 1] + dst[k - xdim], 2)
                elif m == 8:
                    value = dst[k - xdim - 1] if n == 0 else jdiv(dst[k - 1] + dst[k - xdim + 1], 2)
                else:
                    value = jdiv(dst[k - 1] + dst[k - xdim], 2) if n == 0 else jdiv(dst[k - 1] + dst[k - xdim + 1], 2)
                value += src[k]
                dst[k] = value
            else:
                value = dst[k - 1] + src[k]
                dst[k] = value
    return dst


def get_mixed_deltas_from_values4(src: List[int], xdim: int, ydim: int) -> list:
    map_ = [0] * (ydim - 1)

    for i in range(1, ydim):
        delta = [[0] * (xdim - 2) for _ in range(4)]

        k = i * xdim + 1
        for j in range(1, xdim - 1):
            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]
            d = src[k - xdim + 1]

            delta[0][j - 1] = src[k] - a
            delta[1][j - 1] = src[k] - jdiv(a + b, 2)

            if c >= max(a, b):
                med_pred = min(a, b)
            elif c <= min(a, b):
                med_pred = max(a, b)
            else:
                med_pred = a + b - c
            delta[2][j - 1] = src[k] - med_pred

            h_edge = abs(b - c) + abs(b - d)
            v_edge = abs(a - c) + abs(a - b)
            dl_edge = abs(c - d)
            dr_edge = abs(a - d)

            if h_edge >= v_edge and h_edge >= dl_edge and h_edge >= dr_edge:
                dir_pred = b
            elif v_edge >= dl_edge and v_edge >= dr_edge:
                dir_pred = a
            elif dl_edge >= dr_edge:
                dir_pred = d
            else:
                dir_pred = c
            delta[3][j - 1] = src[k] - dir_pred

            k += 1

        limit = [0] * 4
        for j in range(4):
            current_delta = delta[j]

            delta_min = current_delta[0]
            delta_max = current_delta[0]
            for kk in range(1, len(current_delta)):
                if current_delta[kk] < delta_min:
                    delta_min = current_delta[kk]
                elif current_delta[kk] > delta_max:
                    delta_max = current_delta[kk]

            for kk in range(len(current_delta)):
                current_delta[kk] -= delta_min
            range_ = delta_max - delta_min
            frequency = [0] * (range_ + 1)
            for kk in range(len(current_delta)):
                frequency[current_delta[kk]] += 1
            shannon_limit = code_mapper.get_shannon_limit(frequency)
            limit[j] = math.floor(shannon_limit)

        value = limit[0]
        index = 0
        for kk in range(1, 4):
            if limit[kk] < value:
                value = limit[kk]
                index = kk
        map_[i - 1] = index

    dst = [0] * (xdim * ydim)
    init_value = src[0]
    original_init_value = src[0]
    total_sum = 0
    k = 0

    dst[k] = 0
    k += 1
    delta = src[k] - init_value
    value = src[k]
    dst[k] = delta
    k += 1
    total_sum += abs(delta)

    for i in range(2, xdim):
        delta = src[k] - value
        value = src[k]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

    for i in range(1, ydim):
        delta = src[k] - init_value
        init_value = src[k]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

        m = map_[i - 1]

        if m == 0:
            for j in range(1, xdim - 1):
                delta = src[k] - src[k - 1]
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
            delta = src[k] - src[k - 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)
        elif m == 1:
            for j in range(1, xdim - 1):
                delta = src[k] - jdiv(src[k - 1] + src[k - xdim], 2)
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
            delta = src[k] - src[k - 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)
        elif m == 2:
            for j in range(1, xdim - 1):
                a = src[k - 1]
                b = src[k - xdim]
                c = src[k - xdim - 1]

                if c >= max(a, b):
                    pred = min(a, b)
                elif c <= min(a, b):
                    pred = max(a, b)
                else:
                    pred = a + b - c

                delta = src[k] - pred
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
            delta = src[k] - src[k - 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)
        elif m == 3:
            for j in range(1, xdim - 1):
                a = src[k - 1]
                b = src[k - xdim]
                c = src[k - xdim - 1]
                d = src[k - xdim + 1]

                h_edge = abs(b - c) + abs(b - d)
                v_edge = abs(a - c) + abs(a - b)
                dl_edge = abs(c - d)
                dr_edge = abs(a - d)

                if h_edge >= v_edge and h_edge >= dl_edge and h_edge >= dr_edge:
                    pred = b
                elif v_edge >= dl_edge and v_edge >= dr_edge:
                    pred = a
                elif dl_edge >= dr_edge:
                    pred = d
                else:
                    pred = c

                delta = src[k] - pred
                dst[k] = delta
                k += 1
                total_sum += abs(delta)

            # Last column: MED fallback
            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]

            if c >= max(a, b):
                pred = min(a, b)
            elif c <= min(a, b):
                pred = max(a, b)
            else:
                pred = a + b - c

            delta = src[k] - pred
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

    return [total_sum, dst, map_, original_init_value]


def get_values_from_mixed_deltas4(src: List[int], xdim: int, ydim: int, init_value: int, map_: List[int]) -> List[int]:
    dst = [0] * (xdim * ydim)
    k = 0
    value = init_value

    dst[k] = init_value
    k += 1
    for i in range(1, xdim):
        value += src[k]
        dst[k] = value
        k += 1

    for i in range(1, ydim):
        init_value += src[k]
        dst[k] = init_value
        k += 1

        m = map_[i - 1]

        if m == 0:
            for j in range(1, xdim - 1):
                value = dst[k - 1] + src[k]
                dst[k] = value
                k += 1
        elif m == 1:
            for j in range(1, xdim - 1):
                value = jdiv(dst[k - 1] + dst[k - xdim], 2) + src[k]
                dst[k] = value
                k += 1
        elif m == 2:
            for j in range(1, xdim - 1):
                a = dst[k - 1]
                b = dst[k - xdim]
                c = dst[k - xdim - 1]

                if c >= max(a, b):
                    pred = min(a, b)
                elif c <= min(a, b):
                    pred = max(a, b)
                else:
                    pred = a + b - c

                value = pred + src[k]
                dst[k] = value
                k += 1
        elif m == 3:
            for j in range(1, xdim - 1):
                a = dst[k - 1]
                b = dst[k - xdim]
                c = dst[k - xdim - 1]
                d = dst[k - xdim + 1]

                h_edge = abs(b - c) + abs(b - d)
                v_edge = abs(a - c) + abs(a - b)
                dl_edge = abs(c - d)
                dr_edge = abs(a - d)

                if h_edge >= v_edge and h_edge >= dl_edge and h_edge >= dr_edge:
                    pred = b
                elif v_edge >= dl_edge and v_edge >= dr_edge:
                    pred = a
                elif dl_edge >= dr_edge:
                    pred = d
                else:
                    pred = c

                value = pred + src[k]
                dst[k] = value
                k += 1

            # Last column: MED fallback
            a = dst[k - 1]
            b = dst[k - xdim]
            c = dst[k - xdim - 1]

            if c >= max(a, b):
                pred = min(a, b)
            elif c <= min(a, b):
                pred = max(a, b)
            else:
                pred = a + b - c

            value = pred + src[k]
            dst[k] = value
            k += 1
            continue

        value = dst[k - 1] + src[k]
        dst[k] = value
        k += 1

    return dst


# ---------------------------------------------------------------------------
# Scanline (4) -- per-row selection from 16 predictors
# ---------------------------------------------------------------------------

@njit
def pred16(a: int, b: int, c: int, d: int, p: int) -> int:
    if p == 0:
        return a
    if p == 1:
        return b
    if p == 2:
        return c
    if p == 3:
        return d
    if p == 4:
        return (a + b) >> 1
    if p == 5:
        return (b + c) >> 1
    if p == 6:
        return (a + c) >> 1
    if p == 7:
        return (b + d) >> 1
    if p == 8:
        return (c + d) >> 1
    if p == 9:
        return (a + b + c + d + 2) >> 2
    if p == 10:
        return a + b - c
    if p == 11:
        if c >= max(a, b):
            return min(a, b)
        if c <= min(a, b):
            return max(a, b)
        return a + b - c
    if p == 12:
        return (a * 3 + b + 2) >> 2
    if p == 13:
        return (a + b * 3 + 2) >> 2
    if p == 14:
        return (a * 3 + d + 2) >> 2
    if p == 15:
        return (b * 3 + a + 2) >> 2
    return a


@njit
def _mixed_deltas16_frequency_core(src, xdim, ydim):
    delta_freq = np.zeros(511, dtype=np.int64)
    map_freq = np.zeros(16, dtype=np.int64)

    for row in range(ydim):
        best_pred = 0
        best_sad = -1
        for p in range(16):
            sad = 0
            for col in range(xdim):
                k = row * xdim + col
                if k == 0:
                    continue
                a = src[k - 1] if col > 0 else 0
                b = src[k - xdim] if row > 0 else 0
                c = src[k - xdim - 1] if (row > 0 and col > 0) else 0
                d = src[k - xdim + 1] if (row > 0 and col < xdim - 1) else 0
                sad += abs(src[k] - pred16(a, b, c, d, p))
            if best_sad < 0 or sad < best_sad:
                best_sad = sad
                best_pred = p
        map_freq[best_pred] += 1
        for col in range(xdim):
            k = row * xdim + col
            if k == 0:
                continue
            a = src[k - 1] if col > 0 else 0
            b = src[k - xdim] if row > 0 else 0
            c = src[k - xdim - 1] if (row > 0 and col > 0) else 0
            d = src[k - xdim + 1] if (row > 0 and col < xdim - 1) else 0
            delta = src[k] - pred16(a, b, c, d, best_pred)
            idx = delta + 255
            if 0 <= idx < 511:
                delta_freq[idx] += 1

    return delta_freq, map_freq


def get_mixed_deltas16_frequency(src: List[int], xdim: int, ydim: int):
    src_arr = np.asarray(src, dtype=np.int64)
    delta_freq, map_freq = _mixed_deltas16_frequency_core(src_arr, xdim, ydim)
    return [delta_freq.tolist(), map_freq.tolist()]


@njit
def _mixed_deltas_from_values16_rows_core(src, xdim, ydim):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    map_ = np.zeros(ydim, dtype=np.int64)  # one entry per row, value 0-15

    dst[0] = 0
    for row in range(ydim):
        best_pred = 0
        best_sad = -1
        for p in range(16):
            sad = 0
            for col in range(xdim):
                k = row * xdim + col
                if k == 0:
                    continue
                a = src[k - 1] if col > 0 else 0
                b = src[k - xdim] if row > 0 else 0
                c = src[k - xdim - 1] if (row > 0 and col > 0) else 0
                d = src[k - xdim + 1] if (row > 0 and col < xdim - 1) else 0
                sad += abs(src[k] - pred16(a, b, c, d, p))
            if best_sad < 0 or sad < best_sad:
                best_sad = sad
                best_pred = p
        map_[row] = best_pred
        for col in range(xdim):
            k = row * xdim + col
            if k == 0:
                continue
            a = src[k - 1] if col > 0 else 0
            b = src[k - xdim] if row > 0 else 0
            c = src[k - xdim - 1] if (row > 0 and col > 0) else 0
            d = src[k - xdim + 1] if (row > 0 and col < xdim - 1) else 0
            dst[k] = src[k] - pred16(a, b, c, d, best_pred)

    return dst, map_


def get_mixed_deltas_from_values16_rows(src: List[int], xdim: int, ydim: int) -> list:
    src_arr = np.asarray(src, dtype=np.int64)
    dst, map_ = _mixed_deltas_from_values16_rows_core(src_arr, xdim, ydim)
    total = int(np.abs(dst).sum())
    return [total, dst.tolist(), map_.tolist(), int(src_arr[0])]


@njit
def _values_from_mixed_deltas16_rows_core(src, xdim, ydim, init_value, map_):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    dst[0] = init_value

    for row in range(ydim):
        p = map_[row] & 0xF
        for col in range(xdim):
            k = row * xdim + col
            if k == 0:
                continue
            a = dst[k - 1] if col > 0 else 0
            b = dst[k - xdim] if row > 0 else 0
            c = dst[k - xdim - 1] if (row > 0 and col > 0) else 0
            d = dst[k - xdim + 1] if (row > 0 and col < xdim - 1) else 0
            dst[k] = src[k] + pred16(a, b, c, d, p)
    return dst


def get_values_from_mixed_deltas16_rows(src, xdim: int, ydim: int, init_value: int, map_) -> List[int]:
    return _values_from_mixed_deltas16_rows_core(_to_int64_array(src), xdim, ydim, init_value, _to_int64_array(map_))


# ---------------------------------------------------------------------------
# Bilateral smoothing -- preserves edges, suppresses noise.
# threshold 0 = no-op; 1-10 maps range sigma 10-100.
# ---------------------------------------------------------------------------

def bilateral_smooth(src: List[int], xdim: int, ydim: int, threshold: int) -> List[int]:
    if threshold == 0:
        return list(src)

    sigma_r = threshold * threshold
    sigma_s = 1.5
    radius = 2

    rw = [0.0] * 256
    r2 = 2.0 * sigma_r * sigma_r
    for d in range(256):
        rw[d] = math.exp(-(d * d) / r2)

    ksize = 2 * radius + 1
    sw = [[0.0] * ksize for _ in range(ksize)]
    s2 = 2.0 * sigma_s * sigma_s
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sw[dy + radius][dx + radius] = math.exp(-(dx * dx + dy * dy) / s2)

    dst = [0] * len(src)
    for row in range(ydim):
        for col in range(xdim):
            center = src[row * xdim + col]
            sum_w = 0.0
            sum_v = 0.0
            for dy in range(-radius, radius + 1):
                ny = row + dy
                if ny < 0 or ny >= ydim:
                    continue
                for dx in range(-radius, radius + 1):
                    nx = col + dx
                    if nx < 0 or nx >= xdim:
                        continue
                    v = src[ny * xdim + nx]
                    w = sw[dy + radius][dx + radius] * rw[abs(v - center)]
                    sum_w += w
                    sum_v += w * v
            dst[row * xdim + col] = round(sum_v / sum_w)
    return dst


# ---------------------------------------------------------------------------
# Anisotropic diffusion (Perona-Malik) -- iterative edge-preserving smooth.
# threshold 0 = no-op; iterations = threshold, K = threshold*3+5 (8-35).
# lambda = 0.25 (stability limit for 4-directional scheme).
# ---------------------------------------------------------------------------

def anisotropic_smooth(src: List[int], xdim: int, ydim: int, threshold: int) -> List[int]:
    if threshold == 0:
        return list(src)

    iterations = threshold
    K2 = (threshold * 3.0 + 5.0) * (threshold * 3.0 + 5.0)
    lambda_ = 0.25

    c = [0.0] * 511
    for d in range(-255, 256):
        c[d + 255] = math.exp(-(d * d) / K2)

    img = [float(v) for v in src]
    next_ = [0.0] * len(src)

    for _ in range(iterations):
        for row in range(ydim):
            for col in range(xdim):
                k = row * xdim + col
                v = img[k]
                dN = img[k - xdim] - v if row > 0 else 0
                dS = img[k + xdim] - v if row < ydim - 1 else 0
                dE = img[k + 1] - v if col < xdim - 1 else 0
                dW = img[k - 1] - v if col > 0 else 0
                iN = max(0, min(510, int(dN + 255.5)))
                iS = max(0, min(510, int(dS + 255.5)))
                iE = max(0, min(510, int(dE + 255.5)))
                iW = max(0, min(510, int(dW + 255.5)))
                next_[k] = v + lambda_ * (c[iN] * dN + c[iS] * dS + c[iE] * dE + c[iW] * dW)
        img, next_ = next_, img

    dst = [0] * len(src)
    for i in range(len(src)):
        dst[i] = max(0, min(255, round(img[i])))
    return dst


# ---------------------------------------------------------------------------

# Each row maps local predictor index 0-7 to a pred16 index. A numpy array
# (not a Python list of lists) so pred8 below can be numba-compiled --
# nopython mode needs a typed, homogeneous array for a module-level
# lookup table, not a reflected list-of-lists.
# Add new rows here to define new variants; variant 0 is the default.
FILTER_SETS_8 = np.array([
    # variant 0: spread coverage -- directional anchors + best composites
    [0, 1, 2, 3, 4, 10, 9, 5],
    # variant 1: averaging focus -- left, above, avg(l,a), avg-all-4, gradient, MED, weighted blends
    [0, 1, 4, 9, 10, 11, 12, 13],
    # variant 2: variant 1 with weighted 1:3 swapped for avg(above, above-right)
    [0, 1, 4, 9, 10, 11, 12, 7],
], dtype=np.int64)


@njit
def pred8(a: int, b: int, c: int, d: int, p: int, variant: int) -> int:
    return pred16(a, b, c, d, FILTER_SETS_8[variant, p])


@njit
def _mixed_deltas8_frequency_core(src, xdim, ydim, variant):
    delta_freq = np.zeros(511, dtype=np.int64)
    map_freq = np.zeros(8, dtype=np.int64)

    for row in range(ydim):
        best_pred = 0
        best_sad = -1
        for p in range(8):
            sad = 0
            for col in range(xdim):
                k = row * xdim + col
                if k == 0:
                    continue
                a = src[k - 1] if col > 0 else 0
                b = src[k - xdim] if row > 0 else 0
                c = src[k - xdim - 1] if (row > 0 and col > 0) else 0
                d = src[k - xdim + 1] if (row > 0 and col < xdim - 1) else 0
                sad += abs(src[k] - pred8(a, b, c, d, p, variant))
            if best_sad < 0 or sad < best_sad:
                best_sad = sad
                best_pred = p
        map_freq[best_pred] += 1
        for col in range(xdim):
            k = row * xdim + col
            if k == 0:
                continue
            a = src[k - 1] if col > 0 else 0
            b = src[k - xdim] if row > 0 else 0
            c = src[k - xdim - 1] if (row > 0 and col > 0) else 0
            d = src[k - xdim + 1] if (row > 0 and col < xdim - 1) else 0
            delta = src[k] - pred8(a, b, c, d, best_pred, variant)
            idx = delta + 255
            if 0 <= idx < 511:
                delta_freq[idx] += 1

    return delta_freq, map_freq


def get_mixed_deltas8_frequency(src: List[int], xdim: int, ydim: int, variant: int):
    src_arr = np.asarray(src, dtype=np.int64)
    delta_freq, map_freq = _mixed_deltas8_frequency_core(src_arr, xdim, ydim, variant)
    return [delta_freq.tolist(), map_freq.tolist()]


@njit
def _mixed_deltas_from_values8_rows_core(src, xdim, ydim, variant):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    map_ = np.zeros(ydim, dtype=np.int64)

    dst[0] = 0
    for row in range(ydim):
        best_pred = 0
        best_sad = -1
        for p in range(8):
            sad = 0
            for col in range(xdim):
                k = row * xdim + col
                if k == 0:
                    continue
                a = src[k - 1] if col > 0 else 0
                b = src[k - xdim] if row > 0 else 0
                c = src[k - xdim - 1] if (row > 0 and col > 0) else 0
                d = src[k - xdim + 1] if (row > 0 and col < xdim - 1) else 0
                sad += abs(src[k] - pred8(a, b, c, d, p, variant))
            if best_sad < 0 or sad < best_sad:
                best_sad = sad
                best_pred = p
        map_[row] = best_pred
        for col in range(xdim):
            k = row * xdim + col
            if k == 0:
                continue
            a = src[k - 1] if col > 0 else 0
            b = src[k - xdim] if row > 0 else 0
            c = src[k - xdim - 1] if (row > 0 and col > 0) else 0
            d = src[k - xdim + 1] if (row > 0 and col < xdim - 1) else 0
            dst[k] = src[k] - pred8(a, b, c, d, best_pred, variant)

    return dst, map_


def get_mixed_deltas_from_values8_rows(src: List[int], xdim: int, ydim: int, variant: int) -> list:
    src_arr = np.asarray(src, dtype=np.int64)
    dst, map_ = _mixed_deltas_from_values8_rows_core(src_arr, xdim, ydim, variant)
    total = int(np.abs(dst).sum())
    return [total, dst.tolist(), map_.tolist(), int(src_arr[0])]


@njit
def _values_from_mixed_deltas8_rows_core(src, xdim, ydim, init_value, map_, variant):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    dst[0] = init_value

    for row in range(ydim):
        p = map_[row] & 0x7
        for col in range(xdim):
            k = row * xdim + col
            if k == 0:
                continue
            a = dst[k - 1] if col > 0 else 0
            b = dst[k - xdim] if row > 0 else 0
            c = dst[k - xdim - 1] if (row > 0 and col > 0) else 0
            d = dst[k - xdim + 1] if (row > 0 and col < xdim - 1) else 0
            dst[k] = src[k] + pred8(a, b, c, d, p, variant)
    return dst


def get_values_from_mixed_deltas8_rows(src, xdim: int, ydim: int,
                                        init_value: int, map_, variant: int) -> List[int]:
    return _values_from_mixed_deltas8_rows_core(_to_int64_array(src), xdim, ydim, init_value, _to_int64_array(map_), variant)


# ---------------------------------------------------------------------------
# Ideal delta helpers (pixel-map variants)
# ---------------------------------------------------------------------------

def get_ideal_deltas_from_values(src: List[int], xdim: int, ydim: int) -> list:
    dst = [0] * (xdim * ydim)
    map_ = [0] * (xdim * (ydim - 1))

    init_value = src[0]
    m = 0

    for i in range(1, ydim - 1):
        for j in range(xdim):
            k = i * xdim + j
            if j == 0:
                map_[m] = 1 if (abs(src[k] - src[k - xdim]) <= abs(src[k] - src[k - xdim + 1])) else 3
                m += 1
            elif j < xdim - 1:
                da = abs(src[k] - src[k - 1])
                db = abs(src[k] - src[k - xdim])
                dc = abs(src[k] - src[k - xdim - 1])
                dd = abs(src[k] - src[k - xdim + 1])
                if da <= db and da <= dc and da <= dd:
                    map_[m] = 0
                elif db <= dc and db <= dd:
                    map_[m] = 1
                elif dc <= dd:
                    map_[m] = 2
                else:
                    map_[m] = 3
                m += 1
            else:
                da = abs(src[k] - src[k - 1])
                db = abs(src[k] - src[k - xdim])
                dc = abs(src[k] - src[k - xdim - 1])
                if da <= db and da <= dc:
                    map_[m] = 0
                elif db <= dc:
                    map_[m] = 1
                else:
                    map_[m] = 2
                m += 1

    k = xdim * (ydim - 1)
    for j in range(xdim):
        if j == 0:
            map_[m] = 1 if (abs(src[k] - src[k - xdim]) <= abs(src[k] - src[k - xdim + 1])) else 3
            m += 1
        elif j < xdim - 1:
            da = abs(src[k] - src[k - 1])
            db = abs(src[k] - src[k - xdim])
            dc = abs(src[k] - src[k - xdim - 1])
            dd = abs(src[k] - src[k - xdim + 1])
            if da <= db and da <= dc and da <= dd:
                map_[m] = 0
            elif db <= dc and db <= dd:
                map_[m] = 1
            elif dc <= dd:
                map_[m] = 2
            else:
                map_[m] = 3
            m += 1
        else:
            da = abs(src[k] - src[k - 1])
            db = abs(src[k] - src[k - xdim])
            dc = abs(src[k] - src[k - xdim - 1])
            if da <= db and da <= dc:
                map_[m] = 0
            elif db <= dc:
                map_[m] = 1
            else:
                map_[m] = 2
            m += 1
        k += 1

    k = 0
    m = 0
    delta = 0
    for i in range(ydim):
        if i == 0:
            for j in range(xdim):
                if j == 0:
                    dst[k] = delta
                    k += 1
                else:
                    delta = src[k] - src[k - 1]
                    dst[k] = delta
                    k += 1
        else:
            for j in range(xdim):
                n = map_[m]
                m += 1
                if n == 0:
                    delta = src[k] - src[k - 1]
                elif n == 1:
                    delta = src[k] - src[k - xdim]
                elif n == 2:
                    delta = src[k] - src[k - xdim - 1]
                else:
                    delta = src[k] - src[k - xdim + 1]
                dst[k] = delta
                k += 1

    return [0, dst, map_, init_value]


def get_values_from_ideal_deltas(src: List[int], xdim: int, ydim: int, init_value: int, map_: List[int]) -> List[int]:
    dst = [0] * (xdim * ydim)
    k = 0
    dst[k] = init_value
    k += 1

    for i in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
        k += 1

    m = 0
    for i in range(1, ydim):
        for j in range(xdim):
            n = map_[m]
            m += 1
            if n == 0:
                dst[k] = dst[k - 1] + src[k]
            elif n == 1:
                dst[k] = dst[k - xdim] + src[k]
            elif n == 2:
                dst[k] = dst[k - xdim - 1] + src[k]
            else:
                dst[k] = dst[k - xdim + 1] + src[k]
            k += 1
    return dst


def get_ideal_deltas_from_values2(src: List[int], xdim: int, ydim: int) -> list:
    dst = [0] * (xdim * ydim)
    map_ = [0] * ((xdim - 2) * (ydim - 1))
    init_value = src[0]

    m = 0
    for i in range(1, ydim - 1):
        for j in range(1, xdim - 1):
            k = i * xdim + j
            map_[m] = 0 if (abs(src[k] - src[k - 1]) <= abs(src[k] - src[k - xdim])) else 1
            m += 1

    k = 0
    m = 0
    total_sum = 0
    for i in range(ydim):
        if i == 0:
            dst[k] = 0
            k += 1
            for j in range(1, xdim):
                delta = src[k] - src[k - 1]
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
        else:
            delta = src[k] - src[k - xdim]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)
            for j in range(1, xdim - 1):
                n = map_[m]
                m += 1
                delta = (src[k] - src[k - 1]) if n == 0 else (src[k] - src[k - xdim])
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
            delta = src[k] - src[k - 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

    return [total_sum, dst, map_, init_value]


def get_values_from_ideal_deltas2(src: List[int], xdim: int, ydim: int, init_value: int, map_: List[int]) -> List[int]:
    dst = [0] * (xdim * ydim)
    k = 0
    dst[k] = init_value
    k += 1
    for i in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
        k += 1

    m = 0
    for i in range(1, ydim):
        init_value += src[k]
        dst[k] = init_value
        k += 1
        for j in range(1, xdim - 1):
            n = map_[m]
            m += 1
            if n == 0:
                dst[k] = dst[k - 1] + src[k]
            elif n == 1:
                dst[k] = dst[k - xdim] + src[k]
            else:
                dst[k] = jdiv(dst[k - 1] + dst[k - xdim], 2) + src[k]
            k += 1
        dst[k] = dst[k - 1] + src[k]
        k += 1
    return dst


def get_ideal_deltas_from_values3(src: List[int], xdim: int, ydim: int) -> list:
    dst = [0] * (xdim * ydim)
    map_ = [0] * ((xdim - 2) * (ydim - 1))
    init_value = src[0]

    m = 0
    for i in range(1, ydim - 1):
        for j in range(1, xdim - 1):
            k = i * xdim + j
            delta = [
                src[k] - src[k - 1],
                src[k] - src[k - xdim],
                src[k] - jdiv(src[k - 1] + src[k - xdim], 2),
            ]
            value = abs(delta[0])
            index = 0
            for n in range(1, 3):
                if abs(delta[n]) < value:
                    value = abs(delta[n])
                    index = n
            map_[m] = index
            m += 1

    k = 0
    m = 0
    total_sum = 0
    for i in range(ydim):
        if i == 0:
            dst[k] = 0
            k += 1
            for j in range(1, xdim):
                delta = src[k] - src[k - 1]
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
        else:
            delta = src[k] - src[k - xdim]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)
            for j in range(1, xdim - 1):
                n = map_[m]
                m += 1
                if n == 0:
                    delta = src[k] - src[k - 1]
                elif n == 1:
                    delta = src[k] - src[k - xdim]
                else:
                    delta = src[k] - jdiv(src[k - 1] + src[k - xdim], 2)
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
            delta = src[k] - src[k - 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

    return [total_sum, dst, map_, init_value]


def get_values_from_ideal_deltas3(src: List[int], xdim: int, ydim: int, init_value: int, map_: List[int]) -> List[int]:
    dst = [0] * (xdim * ydim)
    k = 0
    dst[k] = init_value
    k += 1
    for i in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
        k += 1

    m = 0
    for i in range(1, ydim):
        init_value += src[k]
        dst[k] = init_value
        k += 1
        for j in range(1, xdim - 1):
            n = map_[m]
            m += 1
            if n == 0:
                dst[k] = dst[k - 1] + src[k]
            elif n == 1:
                dst[k] = dst[k - xdim] + src[k]
            else:
                dst[k] = jdiv(dst[k - 1] + dst[k - xdim], 2) + src[k]
            k += 1
        dst[k] = dst[k - 1] + src[k]
        k += 1
    return dst


def get_ideal_deltas_from_values4(src: List[int], xdim: int, ydim: int) -> list:
    dst = [0] * (xdim * ydim)
    map_ = [0] * ((xdim - 2) * (ydim - 1))
    init_value = src[0]

    m = 0
    for i in range(1, ydim - 1):
        for j in range(1, xdim - 1):
            k = i * xdim + j
            delta = [
                src[k] - src[k - 1],
                src[k] - src[k - xdim],
                src[k] - jdiv(src[k - 1] + src[k - xdim], 2),
                src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2),
            ]
            value = abs(delta[0])
            index = 0
            for n in range(1, 4):
                if abs(delta[n]) < value:
                    value = abs(delta[n])
                    index = n
            map_[m] = index
            m += 1

    k = 0
    m = 0
    total_sum = 0
    for i in range(ydim):
        if i == 0:
            dst[k] = 0
            k += 1
            for j in range(1, xdim):
                delta = src[k] - src[k - 1]
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
        else:
            delta = src[k] - src[k - xdim]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)
            for j in range(1, xdim - 1):
                n = map_[m]
                m += 1
                if n == 0:
                    delta = src[k] - src[k - 1]
                elif n == 1:
                    delta = src[k] - src[k - xdim]
                elif n == 2:
                    delta = src[k] - jdiv(src[k - 1] + src[k - xdim], 2)
                else:
                    delta = src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2)
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
            delta = src[k] - src[k - 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

    return [total_sum, dst, map_, init_value]


def get_values_from_ideal_deltas4(src: List[int], xdim: int, ydim: int, init_value: int, map_: List[int]) -> List[int]:
    dst = [0] * (xdim * ydim)
    k = 0
    dst[k] = init_value
    k += 1
    for i in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
        k += 1

    m = 0
    for i in range(1, ydim):
        init_value += src[k]
        dst[k] = init_value
        k += 1
        for j in range(1, xdim - 1):
            n = map_[m]
            m += 1
            if n == 0:
                dst[k] = dst[k - 1] + src[k]
            elif n == 1:
                dst[k] = dst[k - xdim] + src[k]
            elif n == 2:
                dst[k] = jdiv(dst[k - 1] + dst[k - xdim], 2) + src[k]
            else:
                dst[k] = jdiv(dst[k - 1] + dst[k - xdim + 1], 2) + src[k]
            k += 1
        dst[k] = dst[k - 1] + src[k]
        k += 1
    return dst


def get_ideal_deltas_from_values5(src: List[int], xdim: int, ydim: int) -> list:
    dst = [0] * (xdim * ydim)
    map_ = [0] * ((xdim - 2) * (ydim - 1))
    init_value = src[0]

    m = 0
    for i in range(1, ydim - 1):
        for j in range(1, xdim - 1):
            k = i * xdim + j
            delta = [
                src[k] - src[k - 1],
                src[k] - src[k - xdim],
                src[k] - src[k - xdim - 1],
                src[k] - jdiv(src[k - 1] + src[k - xdim], 2),
                src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2),
            ]
            value = abs(delta[0])
            index = 0
            for n in range(1, 5):
                if abs(delta[n]) < value:
                    value = abs(delta[n])
                    index = n
            map_[m] = index
            m += 1

    k = 0
    m = 0
    total_sum = 0
    for i in range(ydim):
        if i == 0:
            dst[k] = 0
            k += 1
            for j in range(1, xdim):
                delta = src[k] - src[k - 1]
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
        else:
            delta = src[k] - src[k - xdim]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)
            for j in range(1, xdim - 1):
                n = map_[m]
                m += 1
                if n == 0:
                    delta = src[k] - src[k - 1]
                elif n == 1:
                    delta = src[k] - src[k - xdim]
                elif n == 2:
                    delta = src[k] - src[k - xdim - 1]
                elif n == 3:
                    delta = src[k] - jdiv(src[k - 1] + src[k - xdim], 2)
                else:
                    delta = src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2)
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
            delta = src[k] - src[k - 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

    return [total_sum, dst, map_, init_value]


def get_values_from_ideal_deltas5(src: List[int], xdim: int, ydim: int, init_value: int, map_: List[int]) -> List[int]:
    dst = [0] * (xdim * ydim)
    k = 0
    dst[k] = init_value
    k += 1
    for i in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
        k += 1

    m = 0
    for i in range(1, ydim):
        init_value += src[k]
        dst[k] = init_value
        k += 1
        for j in range(1, xdim - 1):
            n = map_[m]
            m += 1
            if n == 0:
                dst[k] = dst[k - 1] + src[k]
            elif n == 1:
                dst[k] = dst[k - xdim] + src[k]
            elif n == 2:
                dst[k] = dst[k - xdim - 1] + src[k]
            elif n == 3:
                dst[k] = jdiv(dst[k - 1] + dst[k - xdim], 2) + src[k]
            else:
                dst[k] = jdiv(dst[k - 1] + dst[k - xdim + 1], 2) + src[k]
            k += 1
        dst[k] = dst[k - 1] + src[k]
        k += 1
    return dst


def get_ideal_deltas_from_values6(src: List[int], xdim: int, ydim: int) -> list:
    dst = [0] * (xdim * ydim)
    map_ = [0] * ((xdim - 2) * (ydim - 1))
    init_value = src[0]

    m = 0
    for i in range(1, ydim - 1):
        for j in range(1, xdim - 1):
            k = i * xdim + j
            delta = [
                float(abs(src[k] - src[k - 1])),
                float(abs(src[k] - jdiv(src[k - 1] + src[k - xdim - 1], 2))),
                float(abs(src[k] - src[k - xdim - 1])),
                float(abs(src[k] - jdiv(src[k - xdim - 1] + src[k - xdim + 1], 2))),
                float(abs(src[k] - src[k - xdim + 1])),
                float(abs(src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2))),
            ]

            addend = 0.00000001
            delta_table = {}
            key_list = []
            key_list.append(delta[0])
            delta_table[delta[0]] = 0
            for kk in range(1, 6):
                if delta[kk] in key_list:
                    delta[kk] += addend
                    addend *= 2.0
                key_list.append(delta[kk])
                delta_table[delta[kk]] = kk
            key_list.sort()
            map_[m] = delta_table[key_list[0]]
            m += 1

    k = 0
    m = 0
    total_sum = 0
    for i in range(ydim):
        if i == 0:
            dst[k] = 0
            k += 1
            for j in range(1, xdim):
                delta = src[k] - src[k - 1]
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
        else:
            delta = src[k] - src[k - xdim]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)
            for j in range(1, xdim - 1):
                n = map_[m]
                m += 1
                if n == 0:
                    delta = src[k] - src[k - 1]
                elif n == 1:
                    delta = src[k] - jdiv(src[k - 1] + src[k - xdim - 1], 2)
                elif n == 2:
                    delta = src[k] - src[k - xdim - 1]
                elif n == 3:
                    delta = src[k] - jdiv(src[k - xdim - 1] + src[k - xdim + 1], 2)
                elif n == 4:
                    delta = src[k] - src[k - xdim + 1]
                else:
                    delta = src[k] - jdiv(src[k - 1] + src[k - xdim + 1], 2)
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
            delta = src[k] - src[k - 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

    return [total_sum, dst, map_, init_value]


def get_values_from_ideal_deltas6(src: List[int], xdim: int, ydim: int, init_value: int, map_: List[int]) -> List[int]:
    dst = [0] * (xdim * ydim)
    k = 0
    dst[k] = init_value
    k += 1
    for i in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
        k += 1

    m = 0
    for i in range(1, ydim):
        init_value += src[k]
        dst[k] = init_value
        k += 1
        for j in range(1, xdim - 1):
            n = map_[m]
            m += 1
            if n == 0:
                dst[k] = dst[k - 1] + src[k]
            elif n == 1:
                dst[k] = jdiv(dst[k - 1] + dst[k - xdim - 1], 2) + src[k]
            elif n == 2:
                dst[k] = dst[k - xdim - 1] + src[k]
            elif n == 3:
                dst[k] = jdiv(dst[k - xdim - 1] + dst[k - xdim + 1], 2) + src[k]
            elif n == 4:
                dst[k] = dst[k - xdim + 1] + src[k]
            else:
                dst[k] = jdiv(dst[k - 1] + dst[k - xdim + 1], 2) + src[k]
            k += 1
        dst[k] = dst[k - 1] + src[k]
        k += 1
    return dst


def get_ideal_deltas_from_values8(src: List[int], xdim: int, ydim: int) -> list:
    dst = [0] * (xdim * ydim)
    map_ = [0] * ((xdim - 2) * (ydim - 1))
    init_value = src[0]

    m = 0
    for i in range(1, ydim - 1):
        for j in range(1, xdim - 1):
            k = i * xdim + j
            delta = [
                float(abs(src[k] - src[k - 1])),
                float(abs(src[k] - jdiv(src[k - 1] + src[k - xdim - 1], 2))),
                float(abs(src[k] - src[k - xdim - 1])),
                float(abs(src[k] - jdiv(src[k - xdim - 1] + src[k - xdim], 2))),
                float(abs(src[k] - src[k - xdim])),
                float(abs(src[k] - jdiv(src[k - xdim] + src[k - xdim + 1], 2))),
                float(abs(src[k] - src[k - xdim + 1])),
                float(abs(src[k] - jdiv(src[k - xdim + 1] + src[k - 1], 2))),
            ]

            addend = 0.00000001
            delta_table = {}
            key_list = []
            key_list.append(delta[0])
            delta_table[delta[0]] = 0
            for kk in range(1, 8):
                if delta[kk] in key_list:
                    delta[kk] += addend
                    addend *= 2.0
                key_list.append(delta[kk])
                delta_table[delta[kk]] = kk
            key_list.sort()
            map_[m] = delta_table[key_list[0]]
            m += 1

    k = 0
    m = 0
    total_sum = 0
    for i in range(ydim):
        if i == 0:
            dst[k] = 0
            k += 1
            for j in range(1, xdim):
                delta = src[k] - src[k - 1]
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
        else:
            delta = src[k] - src[k - xdim]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)
            for j in range(1, xdim - 1):
                n = map_[m]
                m += 1
                if n == 0:
                    delta = src[k] - src[k - 1]
                elif n == 1:
                    delta = src[k] - jdiv(src[k - 1] + src[k - xdim - 1], 2)
                elif n == 2:
                    delta = src[k] - src[k - xdim - 1]
                elif n == 3:
                    delta = src[k] - jdiv(src[k - xdim - 1] + src[k - xdim], 2)
                elif n == 4:
                    delta = src[k] - src[k - xdim]
                elif n == 5:
                    delta = src[k] - jdiv(src[k - xdim] + src[k - xdim + 1], 2)
                elif n == 6:
                    delta = src[k] - src[k - xdim + 1]
                else:
                    delta = src[k] - jdiv(src[k - xdim + 1] + src[k - 1], 2)
                dst[k] = delta
                k += 1
                total_sum += abs(delta)
            delta = src[k] - src[k - 1]
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

    return [total_sum, dst, map_, init_value]


def get_values_from_ideal_deltas8(src: List[int], xdim: int, ydim: int, init_value: int, map_: List[int]) -> List[int]:
    dst = [0] * (xdim * ydim)
    k = 0
    dst[k] = init_value
    k += 1
    for i in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
        k += 1

    m = 0
    for i in range(1, ydim):
        init_value += src[k]
        dst[k] = init_value
        k += 1
        for j in range(1, xdim - 1):
            n = map_[m]
            m += 1
            if n == 0:
                dst[k] = dst[k - 1] + src[k]
            elif n == 1:
                dst[k] = jdiv(dst[k - 1] + dst[k - xdim - 1], 2) + src[k]
            elif n == 2:
                dst[k] = dst[k - xdim - 1] + src[k]
            elif n == 3:
                dst[k] = jdiv(dst[k - xdim - 1] + dst[k - xdim], 2) + src[k]
            elif n == 4:
                dst[k] = dst[k - xdim] + src[k]
            elif n == 5:
                dst[k] = jdiv(dst[k - xdim] + dst[k - xdim + 1], 2) + src[k]
            elif n == 6:
                dst[k] = dst[k - xdim + 1] + src[k]
            else:
                dst[k] = jdiv(dst[k - xdim + 1] + dst[k - 1], 2) + src[k]
            k += 1
        dst[k] = dst[k - 1] + src[k]
        k += 1
    return dst


# ---------------------------------------------------------------------------
# 16-option ideal delta encoder/decoder -- all predictors are causal.
#
# Predictor set (a=left, b=above, c=above-left, d=above-right):
#
#   0  a                    8  (a+b)>>1
#   1  c                    9  (c+d)>>1
#   2  b                   10  (a+b+c+d)>>2
#   3  d                   11  MED(a,b,c)
#   4  (a+c)>>1            12  (a+b+c)>>2
#   5  (c+b)>>1            13  (a+b+d)>>2
#   6  (b+d)>>1            14  (a+c+d)>>2
#   7  (d+a)>>1            15  (b+c+d)>>2
#
# Map covers rows 1..ydim-1, cols 1..xdim-2.
# Map entries stored as raw ints (value 0-15).
# ---------------------------------------------------------------------------

def get_ideal_deltas_from_values16(src: List[int], xdim: int, ydim: int) -> list:
    dst = [0] * (xdim * ydim)
    map_ = [0] * ((xdim - 2) * (ydim - 1))
    init_value = src[0]
    total_sum = 0
    m = 0

    # Pass 1: choose best of 16 causal predictors for each map pixel.
    for i in range(1, ydim):
        for j in range(1, xdim - 1):
            k = i * xdim + j
            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]
            d = src[k - xdim + 1]
            e = src[k]

            if c >= max(a, b):
                med = min(a, b)
            elif c <= min(a, b):
                med = max(a, b)
            else:
                med = a + b - c

            pred = [
                a, c, b, d,
                (a + c) >> 1, (c + b) >> 1, (b + d) >> 1, (d + a) >> 1,
                (a + b) >> 1, (c + d) >> 1,
                (a + b + c + d) >> 2, med,
                (a + b + c) >> 2, (a + b + d) >> 2, (a + c + d) >> 2, (b + c + d) >> 2,
            ]

            best_abs = None
            best_idx = 0
            for n in range(16):
                abs_delta = abs(e - pred[n])
                if best_abs is None or abs_delta < best_abs:
                    best_abs = abs_delta
                    best_idx = n
            map_[m] = best_idx
            m += 1

    # Pass 2: compute deltas.
    k = 0
    m = 0

    # Row 0: horizontal deltas.
    dst[k] = 0
    k += 1
    for j in range(1, xdim):
        delta = src[k] - src[k - 1]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

    # Rows 1..ydim-1: col 0 vertical, interior map-driven, last col horizontal.
    for i in range(1, ydim):
        delta = src[k] - src[k - xdim]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

        for j in range(1, xdim - 1):
            n = map_[m] & 0xFF
            m += 1
            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]
            d = src[k - xdim + 1]

            if c >= max(a, b):
                med = min(a, b)
            elif c <= min(a, b):
                med = max(a, b)
            else:
                med = a + b - c

            if n == 0:
                pred_val = a
            elif n == 1:
                pred_val = c
            elif n == 2:
                pred_val = b
            elif n == 3:
                pred_val = d
            elif n == 4:
                pred_val = (a + c) >> 1
            elif n == 5:
                pred_val = (c + b) >> 1
            elif n == 6:
                pred_val = (b + d) >> 1
            elif n == 7:
                pred_val = (d + a) >> 1
            elif n == 8:
                pred_val = (a + b) >> 1
            elif n == 9:
                pred_val = (c + d) >> 1
            elif n == 10:
                pred_val = (a + b + c + d) >> 2
            elif n == 11:
                pred_val = med
            elif n == 12:
                pred_val = (a + b + c) >> 2
            elif n == 13:
                pred_val = (a + b + d) >> 2
            elif n == 14:
                pred_val = (a + c + d) >> 2
            else:
                pred_val = (b + c + d) >> 2  # 15

            delta = src[k] - pred_val
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

        delta2 = src[k] - src[k - 1]
        dst[k] = delta2
        k += 1
        total_sum += abs(delta2)

    return [total_sum, dst, map_, init_value]


def get_values_from_ideal_deltas16(src: List[int], xdim: int, ydim: int, init_value: int, map_: List[int]) -> List[int]:
    dst = [0] * (xdim * ydim)
    k = 0
    m = 0

    dst[k] = init_value
    k += 1
    for j in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
        k += 1

    for i in range(1, ydim):
        dst[k] = dst[k - xdim] + src[k]
        k += 1

        for j in range(1, xdim - 1):
            n = map_[m] & 0xFF
            m += 1
            a = dst[k - 1]
            b = dst[k - xdim]
            c = dst[k - xdim - 1]
            d = dst[k - xdim + 1]

            if c >= max(a, b):
                med = min(a, b)
            elif c <= min(a, b):
                med = max(a, b)
            else:
                med = a + b - c

            if n == 0:
                pred_val = a
            elif n == 1:
                pred_val = c
            elif n == 2:
                pred_val = b
            elif n == 3:
                pred_val = d
            elif n == 4:
                pred_val = (a + c) >> 1
            elif n == 5:
                pred_val = (c + b) >> 1
            elif n == 6:
                pred_val = (b + d) >> 1
            elif n == 7:
                pred_val = (d + a) >> 1
            elif n == 8:
                pred_val = (a + b) >> 1
            elif n == 9:
                pred_val = (c + d) >> 1
            elif n == 10:
                pred_val = (a + b + c + d) >> 2
            elif n == 11:
                pred_val = med
            elif n == 12:
                pred_val = (a + b + c) >> 2
            elif n == 13:
                pred_val = (a + b + d) >> 2
            elif n == 14:
                pred_val = (a + c + d) >> 2
            else:
                pred_val = (b + c + d) >> 2  # 15

            dst[k] = pred_val + src[k]
            k += 1

        dst[k] = dst[k - 1] + src[k]
        k += 1

    return dst


# ---------------------------------------------------------------------------
# Adaptive predictor -- no map, deterministic from causal neighbors.
# ---------------------------------------------------------------------------

@njit
def _adaptive_pred(a: int, b: int, c: int, d: int) -> int:
    pa = abs(b - c)  # vertical gradient at above-left corner
    pb = abs(a - c)  # horizontal gradient at above-left corner
    if pb > pa * 2:
        return a  # strong horizontal edge -> left
    if pa > pb * 2:
        return b  # strong vertical edge -> above
    if c >= max(a, b):
        return min(a, b)
    if c <= min(a, b):
        return max(a, b)
    return a + b - c


@njit
def _adaptive_deltas_from_values_core(src, xdim, ydim):
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    total_sum = 0
    k = 0

    dst[k] = 0
    k += 1
    for j in range(1, xdim):
        delta = src[k] - src[k - 1]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

    for i in range(1, ydim):
        delta = src[k] - src[k - xdim]
        dst[k] = delta
        k += 1
        total_sum += abs(delta)

        for j in range(1, xdim - 1):
            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]
            d = src[k - xdim + 1]
            delta = src[k] - _adaptive_pred(a, b, c, d)
            dst[k] = delta
            k += 1
            total_sum += abs(delta)

        delta2 = src[k] - src[k - 1]
        dst[k] = delta2
        k += 1
        total_sum += abs(delta2)

    return total_sum, dst


def get_adaptive_deltas_from_values(src: List[int], xdim: int, ydim: int) -> list:
    src_arr = np.asarray(src, dtype=np.int64)
    total_sum, dst = _adaptive_deltas_from_values_core(src_arr, xdim, ydim)
    return [int(total_sum), dst.tolist(), int(src_arr[0])]


@njit
def get_values_from_adaptive_deltas(src, xdim: int, ydim: int, init_value: int) -> List[int]:
    src = np.asarray(src, dtype=np.int64)
    dst = np.zeros(xdim * ydim, dtype=np.int64)
    k = 0

    dst[k] = init_value
    k += 1
    for j in range(1, xdim):
        dst[k] = dst[k - 1] + src[k]
        k += 1

    for i in range(1, ydim):
        dst[k] = dst[k - xdim] + src[k]
        k += 1

        for j in range(1, xdim - 1):
            a = dst[k - 1]
            b = dst[k - xdim]
            c = dst[k - xdim - 1]
            d = dst[k - xdim + 1]
            dst[k] = _adaptive_pred(a, b, c, d) + src[k]
            k += 1

        dst[k] = dst[k - 1] + src[k]
        k += 1

    return dst


@njit
def _adaptive_frequency_core(src, xdim, ydim):
    n = xdim * ydim - 1
    delta_list = np.empty(n, dtype=np.int64)
    idx = 0
    k = 1  # skip pixel 0 (delta = 0)

    for j in range(1, xdim):
        delta_list[idx] = src[k] - src[k - 1]
        idx += 1
        k += 1

    for i in range(1, ydim):
        delta_list[idx] = src[k] - src[k - xdim]
        idx += 1
        k += 1
        for j in range(1, xdim - 1):
            a = src[k - 1]
            b = src[k - xdim]
            c = src[k - xdim - 1]
            d = src[k - xdim + 1]
            delta_list[idx] = src[k] - _adaptive_pred(a, b, c, d)
            idx += 1
            k += 1
        delta_list[idx] = src[k] - src[k - 1]
        idx += 1
        k += 1

    mn = delta_list.min()
    mx = delta_list.max()
    freq = np.zeros(mx - mn + 1, dtype=np.int64)
    for i in range(n):
        freq[delta_list[i] - mn] += 1
    return freq


def get_adaptive_frequency(src: List[int], xdim: int, ydim: int) -> List[int]:
    src_arr = np.asarray(src, dtype=np.int64)
    return _adaptive_frequency_core(src_arr, xdim, ydim).tolist()


# ---------------------------------------------------------------------------
# Delta list utilities
# ---------------------------------------------------------------------------

def get_delta_list_from_values(src: List[int], xdim: int, ydim: int) -> list:
    delta_list = []

    k = 0
    for i in range(ydim):
        for j in range(xdim):
            if i == 0 or i == ydim - 1:
                size = 3 if (j == 0 or j == xdim - 1) else 5
            else:
                size = 5 if (j == 0 or j == xdim - 1) else 8

            value = [0] * size
            location = [0] * size

            if i == 0:
                if j == 0:
                    value[0] = src[k] - src[k + 1]; location[0] = 4
                    value[1] = src[k + xdim];       location[1] = 6
                    value[2] = src[k + xdim + 1];   location[2] = 7
                elif j == xdim - 1:
                    value[0] = src[k] - src[k - 1];     location[0] = 3
                    value[1] = src[k + xdim - 1];       location[1] = 5
                    value[2] = src[k + xdim];           location[2] = 6
                else:
                    value[0] = src[k] - src[k - 1];       location[0] = 3
                    value[1] = src[k] - src[k + 1];       location[1] = 4
                    value[2] = src[k] - src[k + xdim - 1]; location[2] = 5
                    value[3] = src[k] - src[k + xdim];     location[3] = 6
                    value[4] = src[k] - src[k + xdim + 1]; location[4] = 7
            elif i == ydim - 1:
                if j == 0:
                    value[0] = src[k] - src[k - xdim];   location[0] = 1
                    value[1] = src[k - xdim + 1];        location[1] = 2
                    value[2] = src[k + 1];                location[2] = 4
                elif j == xdim - 1:
                    value[0] = src[k] - src[k - xdim - 1]; location[0] = 0
                    value[1] = src[k - xdim];              location[1] = 1
                    value[2] = src[k - 1];                  location[2] = 3
                else:
                    value[0] = src[k] - src[k - xdim - 1]; location[0] = 0
                    value[1] = src[k] - src[k - xdim];     location[1] = 1
                    value[2] = src[k] - src[k - xdim + 1]; location[2] = 2
                    value[3] = src[k] - src[k - 1];        location[3] = 3
                    value[4] = src[k] - src[k + 1];        location[4] = 4
            else:
                if j == 0:
                    value[0] = src[k] - src[k - xdim];      location[0] = 1
                    value[1] = src[k] - src[k - xdim + 1];  location[1] = 2
                    value[2] = src[k] - src[k + 1];         location[2] = 4
                    value[3] = src[k] - src[k + xdim];      location[3] = 6
                    value[4] = src[k] - src[k + xdim + 1];  location[4] = 7
                elif j == xdim - 1:
                    value[0] = src[k] - src[k - xdim - 1];  location[0] = 0
                    value[1] = src[k] - src[k - xdim];      location[1] = 1
                    value[2] = src[k] - src[k - 1];         location[2] = 3
                    value[3] = src[k] - src[k + xdim - 1];  location[3] = 5
                    value[4] = src[k] - src[k + xdim];      location[4] = 6
                else:
                    value[0] = src[k] - src[k - xdim - 1];  location[0] = 0
                    value[1] = src[k] - src[k - xdim];      location[1] = 1
                    value[2] = src[k] - src[k - xdim + 1];  location[2] = 2
                    value[3] = src[k] - src[k - 1];         location[3] = 3
                    value[4] = src[k] - src[k + 1];         location[4] = 4
                    value[5] = src[k] - src[k + xdim - 1];  location[5] = 5
                    value[6] = src[k] - src[k + xdim];      location[6] = 6
                    value[7] = src[k] - src[k + xdim + 1];  location[7] = 7

            delta = [float(abs(value[m])) for m in range(size)]

            addend = 0.00000001
            delta_table = {}
            key_list = []

            for m in range(size):
                if delta[m] in key_list:
                    delta[m] += addend
                    addend *= 2
                key_list.append(delta[m])
                delta_table[delta[m]] = [value[m], location[m]]

            key_list.sort()
            table = [[0, 0] for _ in range(size)]
            for m in range(size):
                key = key_list[m]
                current = delta_table[key]
                table[m][0] = current[0]
                table[m][1] = current[1]

            delta_list.append(table)
            k += 1

    return delta_list


def get_ideal_deltas_from_list(delta_list: list) -> List[int]:
    ideal_delta = [0] * len(delta_list)
    for i in range(len(delta_list)):
        table = delta_list[i]
        ideal_delta[i] = table[0][0]
    return ideal_delta


def get_ideal_delta_sum(delta_list: list) -> int:
    total = 0
    for i in range(len(delta_list)):
        table = delta_list[i]
        total += abs(table[0][0])
    return total


def get_worstl_delta_sum(delta_list: list) -> int:
    total = 0
    for i in range(len(delta_list)):
        table = delta_list[i]
        total += abs(table[len(table) - 1][0])
    return total


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def get_neighbors(src: List[int], x: int, y: int, xdim: int, ydim: int) -> List[int]:
    neighbors = []
    if y > 0:
        if x > 0:
            neighbors.append(src[(y - 1) * xdim + x - 1])
        neighbors.append(src[(y - 1) * xdim + x])
        if x < xdim - 1:
            neighbors.append(src[(y - 1) * xdim + x + 1])
    if x > 0:
        neighbors.append(src[y * xdim + x - 1])
    if x < xdim - 1:
        neighbors.append(src[y * xdim + x + 1])
    if y < ydim - 1:
        if x > 0:
            neighbors.append(src[(y + 1) * xdim + x - 1])
        neighbors.append(src[(y + 1) * xdim + x])
        if x < xdim - 1:
            neighbors.append(src[(y + 1) * xdim + x + 1])
    return neighbors


def get_location_type(x: int, y: int, xdim: int, ydim: int) -> int:
    if y == 0:
        if x == 0:
            return 1
        if x < xdim - 1:
            return 2
        return 3
    if y < ydim - 1:
        if x == 0:
            return 4
        if x < xdim - 1:
            return 5
        return 6
    if x == 0:
        return 7
    if x < xdim - 1:
        return 8
    return 9


def get_location_index(location_type: int, location: int) -> int:
    if location_type == 1:
        if location == 4: return 0
        if location == 6: return 1
        if location == 7: return 2
    elif location_type == 2:
        if location == 3: return 0
        if location == 4: return 1
        if location == 5: return 2
        if location == 6: return 3
        if location == 7: return 4
    elif location_type == 3:
        if location == 3: return 0
        if location == 5: return 1
        if location == 6: return 2
    elif location_type == 4:
        if location == 1: return 0
        if location == 2: return 1
        if location == 4: return 2
        if location == 6: return 3
        if location == 7: return 4
    elif location_type == 5:
        if location == 0: return 0
        if location == 1: return 1
        if location == 2: return 2
        if location == 3: return 3
        if location == 4: return 4
        if location == 5: return 5
        if location == 6: return 6
        if location == 7: return 7
    elif location_type == 6:
        if location == 0: return 0
        if location == 1: return 1
        if location == 3: return 2
        if location == 5: return 3
        if location == 6: return 4
    elif location_type == 7:
        if location == 1: return 0
        if location == 2: return 1
        if location == 4: return 2
    elif location_type == 8:
        if location == 0: return 0
        if location == 1: return 1
        if location == 2: return 2
        if location == 3: return 3
        if location == 4: return 4
    elif location_type == 9:
        if location == 0: return 0
        if location == 1: return 1
        if location == 3: return 2
    return -1


def get_neighbor_index(x: int, y: int, xdim: int, location: int) -> int:
    k = y * xdim + x
    if location == 0: return k - xdim - 1
    if location == 1: return k - xdim
    if location == 2: return k - xdim + 1
    if location == 3: return k - 1
    if location == 4: return k + 1
    if location == 5: return k + xdim - 1
    if location == 6: return k + xdim
    if location == 7: return k + xdim + 1
    return k


def get_inverse_location(location: int) -> int:
    return 7 - location


def get_channels(set_id: int) -> List[int]:
    channel = [0, 0, 0]
    if set_id == 0:
        channel[0], channel[1], channel[2] = 0, 1, 2
    elif set_id == 1:
        channel[0], channel[1], channel[2] = 0, 2, 4
    elif set_id == 2:
        channel[0], channel[1], channel[2] = 0, 2, 3
    elif set_id == 3:
        channel[0], channel[1], channel[2] = 0, 3, 4
    elif set_id == 4:
        channel[0], channel[1], channel[2] = 0, 3, 5
    elif set_id == 5:
        channel[0], channel[1], channel[2] = 1, 2, 3
    elif set_id == 6:
        channel[0], channel[1], channel[2] = 2, 3, 4
    elif set_id == 7:
        channel[0], channel[1], channel[2] = 1, 3, 4
    elif set_id == 8:
        channel[0], channel[1], channel[2] = 1, 4, 5
    elif set_id == 9:
        channel[0], channel[1], channel[2] = 2, 4, 5
    return channel
