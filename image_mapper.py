"""
image_mapper.py -- ImageMapper.java (version 2.0): the pixel-pyramid
operators (shrink / gradient expand / sign-bit refinement / padding) that
DeltaWriter and DeltaReader use for "Average" quantization, plus the older
dilation, area-resampling and registration utilities merged into the same
Java class.

Arrays: flat images are row-major int64 numpy arrays plus a width, as in
the rest of the port; Java's int[][] / double[][] are 2-D numpy arrays
(int64 / float64); boolean[] / boolean[][] are np.bool_ arrays.
Numba-compiled where it matters (the flat pyramid operators are the hot
path); everything also runs without Numba, identically.

Java semantics kept on purpose:
  - Math.round(double) is round-half-up (floor(x + 0.5), computed exactly):
    _jround below. Python's round() is round-half-even and would differ.
  - (int) casts of doubles truncate toward zero.
  - The area-resampling transforms do their arithmetic in 32-bit Java int,
    which can overflow for large shrink factors; _i32 reproduces the wrap
    and jdiv the truncating division, so results match Java bit for bit.
  - Methods that mutate their arguments in Java (smooth's dst, the
    dilate_image* family's is_interpolated and dst) mutate them here too.

Overloads get distinct names:
  shrinkAvg(int[][])                  -> shrink_avg
  shrinkAvg(int[], xdim)              -> shrink_avg_flat
  shrinkAvg(double[][])               -> shrink_avg_double
  expandGradient / expandGradientSaddle / buildGeqBits / refineWithSignBits
  / padEdgeReplicate / crop: the int[][] versions keep the plain name, the
  flat (int[] + width) versions end in _flat.
  avgAreaXTransform / avgAreaYTransform / avgAreaTransform: flat versions
  keep the plain name, the int[][] versions end in _2d.
  getGradient(int[][]) and getGradient(double[][]) -> get_gradient (either
  dtype). Java returns ArrayList[][] of {xgradient, ygradient}; here it is
  a float64 array of shape (ydim, xdim, 2), NaN where Java stores NaN.
  shift(int[][]) and shift(double[][]) -> shift (either dtype).
  expandX(int[][], expand) -> expand_x_int; expandX(double[][]) ->
  expand_x; expandX(double[][], iterations) -> expand_x_iterated.

Behaviour carried over unchanged from the Java (flagged, not fixed, so the
two stay interchangeable):
  - avgAreaYTransform(int[][]) and avgAreaTransform(int[][]) reset their
    copy index at the start of every row, so only the first row of the
    working array is ever filled (with the LAST source row), and the output
    rows are all copies of the first working row. avgAreaTransform(int[][])
    also copies xdim (not new_xdim) columns per output row, so it raises
    IndexError, as Java throws, when new_xdim < xdim.
  - expandX(int[][], expand) steps from start toward start + (start - end),
    i.e. away from the next pixel rather than toward it, and truncates to
    int after every step (Java's compound += on an int).
  - translate() multiplies the interpolated value by .5 before rounding, so
    its output is about half the brightness of its input.
  - getImageDilation prints progress messages, as the Java does.
"""

import math

import numpy as np

from numba_support import njit, jdiv


# ---------------------------------------------------------------------------
# Java numeric helpers
# ---------------------------------------------------------------------------

@njit
def _jround(x):
    """Java Math.round(double) for finite x: nearest long, halves up."""
    r = math.floor(x)
    if x - r >= 0.5:
        r += 1.0
    return int(r)


@njit
def _clamp255(v):
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v


@njit
def _clamp(v, max_value):
    if v < 0:
        return 0
    if v > max_value:
        return max_value
    return v


@njit
def _i32(x):
    """Wrap to a 32-bit signed Java int."""
    return ((x + 2147483648) & 0xFFFFFFFF) - 2147483648


@njit
def _jtrunc(x):
    """Java (int) cast of a finite double: truncate toward zero."""
    return int(x)


# ===========================================================================
# Pyramid demo operators, int[][] versions (used by ShrinkExpander.java)
# ===========================================================================

@njit
def shrink_avg(src):
    """Rounded 2x2 block average of a 2-D int array ((sum + 2) / 4)."""
    ydim, xdim = src.shape
    _xdim = xdim // 2
    _ydim = ydim // 2
    dst = np.zeros((_ydim, _xdim), dtype=np.int64)
    for i in range(0, ydim - 1, 2):
        k = i // 2
        for j in range(0, xdim - 1, 2):
            m = j // 2
            dst[k, m] = (src[i, j] + src[i, j + 1] + src[i + 1, j] + src[i + 1, j + 1] + 2) // 4
    return dst


@njit
def _horiz_grad(avg, k, m):
    _xdim = avg.shape[1]
    m0 = max(m - 1, 0)
    m1 = min(m + 1, _xdim - 1)
    if m0 == m1:
        return 0.0
    return (avg[k, m1] - avg[k, m0]) / (2.0 * (m1 - m0))


@njit
def _vert_grad(avg, k, m):
    _ydim = avg.shape[0]
    k0 = max(k - 1, 0)
    k1 = min(k + 1, _ydim - 1)
    if k0 == k1:
        return 0.0
    return (avg[k1, m] - avg[k0, m]) / (2.0 * (k1 - k0))


@njit
def _cross_grad(avg, k, m):
    _ydim, _xdim = avg.shape
    k0 = max(k - 1, 0)
    k1 = min(k + 1, _ydim - 1)
    m0 = max(m - 1, 0)
    m1 = min(m + 1, _xdim - 1)
    if k0 == k1 or m0 == m1:
        return 0.0
    num = float(avg[k1, m1] - avg[k1, m0] - avg[k0, m1] + avg[k0, m0])
    denom = (2.0 * (k1 - k0)) * (2.0 * (m1 - m0))
    return num / denom


@njit
def expand_gradient(avg):
    """Plane-fit 2x expansion of a 2-D block-average array (clamped to 0..255)."""
    _ydim, _xdim = avg.shape
    dst = np.zeros((_ydim * 2, _xdim * 2), dtype=np.int64)
    for k in range(_ydim):
        for m in range(_xdim):
            a = float(avg[k, m])
            gx = _horiz_grad(avg, k, m)
            gy = _vert_grad(avg, k, m)
            i = 2 * k
            j = 2 * m
            dst[i, j] = _clamp255(_jround(a - 0.5 * gy - 0.5 * gx))
            dst[i, j + 1] = _clamp255(_jround(a - 0.5 * gy + 0.5 * gx))
            dst[i + 1, j] = _clamp255(_jround(a + 0.5 * gy - 0.5 * gx))
            dst[i + 1, j + 1] = _clamp255(_jround(a + 0.5 * gy + 0.5 * gx))
    return dst


@njit
def expand_gradient_saddle(avg):
    """expand_gradient plus the mixed-partial (saddle) term."""
    _ydim, _xdim = avg.shape
    dst = np.zeros((_ydim * 2, _xdim * 2), dtype=np.int64)
    for k in range(_ydim):
        for m in range(_xdim):
            a = float(avg[k, m])
            gx = _horiz_grad(avg, k, m)
            gy = _vert_grad(avg, k, m)
            gxy = _cross_grad(avg, k, m)
            i = 2 * k
            j = 2 * m
            dst[i, j] = _clamp255(_jround(a - 0.5 * gy - 0.5 * gx + 0.25 * gxy))
            dst[i, j + 1] = _clamp255(_jround(a - 0.5 * gy + 0.5 * gx - 0.25 * gxy))
            dst[i + 1, j] = _clamp255(_jround(a + 0.5 * gy - 0.5 * gx - 0.25 * gxy))
            dst[i + 1, j + 1] = _clamp255(_jround(a + 0.5 * gy + 0.5 * gx + 0.25 * gxy))
    return dst


@njit
def refine_with_sign_bits(avg, predicted, geq):
    """POCS refinement of a prediction using one >=-average bit per pixel."""
    ydim, xdim = predicted.shape
    dst = np.zeros((ydim, xdim), dtype=np.int64)
    p = np.zeros(4)
    bit = np.zeros(4, dtype=np.bool_)
    for i in range(0, ydim - 1, 2):
        k = i // 2
        for j in range(0, xdim - 1, 2):
            m = j // 2
            a = float(avg[k, m])
            p[0] = predicted[i, j]
            p[1] = predicted[i, j + 1]
            p[2] = predicted[i + 1, j]
            p[3] = predicted[i + 1, j + 1]
            bit[0] = geq[i, j]
            bit[1] = geq[i, j + 1]
            bit[2] = geq[i + 1, j]
            bit[3] = geq[i + 1, j + 1]
            for _ in range(10):
                for t in range(4):
                    if (p[t] >= a) != bit[t]:
                        p[t] = a if bit[t] else a - 1
                total = p[0] + p[1] + p[2] + p[3]
                corr = (total - 4 * a) / 4.0
                for t in range(4):
                    p[t] -= corr
            dst[i, j] = _clamp255(_jround(p[0]))
            dst[i, j + 1] = _clamp255(_jround(p[1]))
            dst[i + 1, j] = _clamp255(_jround(p[2]))
            dst[i + 1, j + 1] = _clamp255(_jround(p[3]))
    return dst


@njit
def _damping_factor(center, raw_offsets, max_value):
    """Largest s in [0,1] keeping center + s*offset inside [0, max_value]."""
    s = 1.0
    for off in raw_offsets:
        if off > 0:
            s = min(s, (max_value - center) / off)
        elif off < 0:
            s = min(s, center / (-off))
    return max(0.0, s)


@njit
def build_geq_bits(orig, avg):
    """geq[i, j] = orig[i, j] >= avg[i // 2, j // 2] (2-D)."""
    h, w = orig.shape
    geq = np.zeros((h, w), dtype=np.bool_)
    for i in range(h):
        k = i // 2
        for j in range(w):
            geq[i, j] = orig[i, j] >= avg[k, j // 2]
    return geq


# ---- boundary handling for dimensions not divisible by a pyramid's PAD_MULTIPLE ----

def pad_to(dim, multiple):
    """Next multiple of `multiple` at or above dim."""
    rem = dim % multiple
    return dim if rem == 0 else dim + (multiple - rem)


def pad_edge_replicate(src, multiple):
    """Pad a 2-D array at the bottom/right to a multiple, replicating the edge.
    Returns src itself when no padding is needed (as the Java does)."""
    h, w = src.shape
    new_h = pad_to(h, multiple)
    new_w = pad_to(w, multiple)
    if new_h == h and new_w == w:
        return src
    rows = np.minimum(np.arange(new_h), h - 1)
    cols = np.minimum(np.arange(new_w), w - 1)
    return src[rows][:, cols].copy()


def crop(src, h, w):
    """Top-left h x w of a 2-D array (a copy)."""
    return src[:h, :w].copy()


# ---- error measurement ----

def error_stats(orig, recon):
    """(mean signed error, mean absolute error, pixel count) for one channel."""
    h, w = orig.shape
    n = h * w
    e = recon.astype(np.int64) - orig.astype(np.int64)
    return [float(e.sum()) / n, float(np.abs(e).sum()) / n, float(n)]


# ===========================================================================
# Pyramid operators, flat-array versions (version 2.0) -- what DeltaWriter /
# DeltaReader use. xdim is always the width of the FIRST array argument
# unless the name says otherwise.
# ===========================================================================

@njit
def shrink_avg_flat(src, xdim):
    ydim = src.shape[0] // xdim
    _xdim = xdim // 2
    _ydim = ydim // 2
    dst = np.zeros(_xdim * _ydim, dtype=np.int64)
    for i in range(0, ydim - 1, 2):
        k = i // 2
        for j in range(0, xdim - 1, 2):
            m = j // 2
            dst[k * _xdim + m] = (src[i * xdim + j] + src[i * xdim + j + 1]
                                  + src[(i + 1) * xdim + j] + src[(i + 1) * xdim + j + 1] + 2) // 4
    return dst


@njit
def _horiz_grad_flat(avg, xdim, k, m):
    m0 = max(m - 1, 0)
    m1 = min(m + 1, xdim - 1)
    if m0 == m1:
        return 0.0
    return (avg[k * xdim + m1] - avg[k * xdim + m0]) / (2.0 * (m1 - m0))


@njit
def _vert_grad_flat(avg, xdim, ydim, k, m):
    k0 = max(k - 1, 0)
    k1 = min(k + 1, ydim - 1)
    if k0 == k1:
        return 0.0
    return (avg[k1 * xdim + m] - avg[k0 * xdim + m]) / (2.0 * (k1 - k0))


@njit
def _cross_grad_flat(avg, xdim, ydim, k, m):
    k0 = max(k - 1, 0)
    k1 = min(k + 1, ydim - 1)
    m0 = max(m - 1, 0)
    m1 = min(m + 1, xdim - 1)
    if k0 == k1 or m0 == m1:
        return 0.0
    num = float(avg[k1 * xdim + m1] - avg[k1 * xdim + m0] - avg[k0 * xdim + m1] + avg[k0 * xdim + m0])
    denom = (2.0 * (k1 - k0)) * (2.0 * (m1 - m0))
    return num / denom


@njit
def expand_gradient_flat(avg, xdim, max_value):
    """Plane-fit 2x expansion; xdim is avg's (smaller) width. Corners are
    damped together toward the block average rather than clamped one by one,
    which keeps the block mean exact; max_value is 255 for an RGB channel,
    up to 510 for a shifted difference channel."""
    _xdim = xdim
    _ydim = avg.shape[0] // xdim
    new_xdim = _xdim * 2
    dst = np.zeros(new_xdim * _ydim * 2, dtype=np.int64)
    raw = np.zeros(4)
    for k in range(_ydim):
        for m in range(_xdim):
            a = float(avg[k * _xdim + m])
            gx = _horiz_grad_flat(avg, _xdim, k, m)
            gy = _vert_grad_flat(avg, _xdim, _ydim, k, m)
            raw[0] = -0.5 * gy - 0.5 * gx
            raw[1] = -0.5 * gy + 0.5 * gx
            raw[2] = 0.5 * gy - 0.5 * gx
            raw[3] = 0.5 * gy + 0.5 * gx
            s = _damping_factor(a, raw, max_value)
            i = 2 * k
            j = 2 * m
            dst[i * new_xdim + j] = _clamp(_jround(a + s * raw[0]), max_value)
            dst[i * new_xdim + j + 1] = _clamp(_jround(a + s * raw[1]), max_value)
            dst[(i + 1) * new_xdim + j] = _clamp(_jround(a + s * raw[2]), max_value)
            dst[(i + 1) * new_xdim + j + 1] = _clamp(_jround(a + s * raw[3]), max_value)
    return dst


@njit
def expand_gradient_saddle_flat(avg, xdim, max_value):
    """expand_gradient_flat plus the mixed-partial (saddle) term."""
    _xdim = xdim
    _ydim = avg.shape[0] // xdim
    new_xdim = _xdim * 2
    dst = np.zeros(new_xdim * _ydim * 2, dtype=np.int64)
    raw = np.zeros(4)
    for k in range(_ydim):
        for m in range(_xdim):
            a = float(avg[k * _xdim + m])
            gx = _horiz_grad_flat(avg, _xdim, k, m)
            gy = _vert_grad_flat(avg, _xdim, _ydim, k, m)
            gxy = _cross_grad_flat(avg, _xdim, _ydim, k, m)
            raw[0] = -0.5 * gy - 0.5 * gx + 0.25 * gxy
            raw[1] = -0.5 * gy + 0.5 * gx - 0.25 * gxy
            raw[2] = 0.5 * gy - 0.5 * gx - 0.25 * gxy
            raw[3] = 0.5 * gy + 0.5 * gx + 0.25 * gxy
            s = _damping_factor(a, raw, max_value)
            i = 2 * k
            j = 2 * m
            dst[i * new_xdim + j] = _clamp(_jround(a + s * raw[0]), max_value)
            dst[i * new_xdim + j + 1] = _clamp(_jround(a + s * raw[1]), max_value)
            dst[(i + 1) * new_xdim + j] = _clamp(_jround(a + s * raw[2]), max_value)
            dst[(i + 1) * new_xdim + j + 1] = _clamp(_jround(a + s * raw[3]), max_value)
    return dst


@njit
def build_geq_bits_flat(orig, avg, orig_xdim):
    """orig_xdim is orig's (the larger array's) width; avg is half that width."""
    h = orig.shape[0] // orig_xdim
    w = orig_xdim
    avg_xdim = w // 2
    geq = np.zeros(h * w, dtype=np.bool_)
    for i in range(h):
        k = i // 2
        for j in range(w):
            geq[i * w + j] = orig[i * w + j] >= avg[k * avg_xdim + j // 2]
    return geq


@njit
def refine_with_sign_bits_flat(avg, predicted, geq, predicted_xdim, max_value):
    """predicted_xdim is predicted's (the larger array's) width. A block whose
    prediction disagrees with any of its four bits restarts from flat; the
    result is damped toward the block average, never clamped corner by corner."""
    xdim = predicted_xdim
    ydim = predicted.shape[0] // predicted_xdim
    avg_xdim = xdim // 2
    dst = np.zeros(predicted.shape[0], dtype=np.int64)
    p = np.zeros(4)
    bit = np.zeros(4, dtype=np.bool_)
    dev = np.zeros(4)
    for i in range(0, ydim - 1, 2):
        k = i // 2
        for j in range(0, xdim - 1, 2):
            m = j // 2
            a = float(avg[k * avg_xdim + m])
            p[0] = predicted[i * xdim + j]
            p[1] = predicted[i * xdim + j + 1]
            p[2] = predicted[(i + 1) * xdim + j]
            p[3] = predicted[(i + 1) * xdim + j + 1]
            bit[0] = geq[i * xdim + j]
            bit[1] = geq[i * xdim + j + 1]
            bit[2] = geq[(i + 1) * xdim + j]
            bit[3] = geq[(i + 1) * xdim + j + 1]

            any_mismatch = False
            for t in range(4):
                if (p[t] >= a) != bit[t]:
                    any_mismatch = True
            if any_mismatch:
                for t in range(4):
                    p[t] = a

            for _ in range(10):
                for t in range(4):
                    if (p[t] >= a) != bit[t]:
                        p[t] = a if bit[t] else a - 1
                total = p[0] + p[1] + p[2] + p[3]
                corr = (total - 4 * a) / 4.0
                for t in range(4):
                    p[t] -= corr

            for t in range(4):
                dev[t] = p[t] - a
            s = _damping_factor(a, dev, max_value)
            dst[i * xdim + j] = _clamp(_jround(a + s * dev[0]), max_value)
            dst[i * xdim + j + 1] = _clamp(_jround(a + s * dev[1]), max_value)
            dst[(i + 1) * xdim + j] = _clamp(_jround(a + s * dev[2]), max_value)
            dst[(i + 1) * xdim + j + 1] = _clamp(_jround(a + s * dev[3]), max_value)
    return dst


def pad_edge_replicate_flat(src, xdim, ydim, new_xdim, new_ydim):
    """Pad a flat image at the bottom/right, replicating the last row/column.
    Returns src itself when the size doesn't change (as the Java does)."""
    if new_xdim == xdim and new_ydim == ydim:
        return src
    rows = np.minimum(np.arange(new_ydim), ydim - 1)
    cols = np.minimum(np.arange(new_xdim), xdim - 1)
    return np.asarray(src).reshape(-1)[:xdim * ydim].reshape(ydim, xdim)[rows][:, cols].reshape(-1).copy()


def crop_flat(src, xdim, ydim, new_xdim, new_ydim):
    """Top-left new_xdim x new_ydim of a flat image (a copy)."""
    src = np.asarray(src).reshape(-1)
    dst = np.zeros(new_xdim * new_ydim, dtype=src.dtype)
    for y in range(new_ydim):
        dst[y * new_xdim:(y + 1) * new_xdim] = src[y * xdim:y * xdim + new_xdim]
    return dst


# ===========================================================================
# Image dilation / area-resampling / registration utility methods
# ===========================================================================

@njit
def smooth(src, xdim, ydim, smooth_factor, number_of_iterations, dst):
    """Edge-preserving smoothing (weights exp(-|grad|^2 / 2 sigma^2), 3x3).
    Writes the (truncated) result into dst, like the Java."""
    size = xdim * ydim
    even = np.zeros(size)
    odd = np.zeros(size)
    weight = np.zeros(size)
    product = np.zeros(size)
    factor = 1.0 / (2 * smooth_factor * smooth_factor)
    current_src = odd
    current_dst = even
    for i in range(size):
        current_src[i] = float(src[i])
        current_dst[i] = float(src[i])

    for i in range(number_of_iterations):
        if i % 2 == 0:
            current_src = even
            current_dst = odd
        else:
            current_src = odd
            current_dst = even

        for j in range(1, ydim - 1):
            index = j * xdim
            for _ in range(1, xdim - 1):
                index += 1
                dx = (current_src[index - 1] - current_src[index + 1]) / 2.
                dy = (current_src[index - xdim] - current_src[index + xdim]) / 2.
                dxy = dx * dx + dy * dy
                weight[index] = math.exp(-dxy * factor)
                product[index] = weight[index] * current_src[index]

        for j in range(2, ydim - 2):
            index = j * xdim + 2
            total_weights = (weight[index - xdim - 1] + weight[index - xdim] + weight[index - xdim + 1]
                             + weight[index - 1] + weight[index] + weight[index + 1] + weight[index + xdim - 1]
                             + weight[index + xdim] + weight[index + xdim + 1])
            total = (product[index - xdim - 1] + product[index - xdim] + product[index - xdim + 1] + product[index - 1]
                     + product[index] + product[index + 1] + product[index + xdim - 1] + product[index + xdim]
                     + product[index + xdim + 1])
            for _ in range(2, xdim - 2):
                current_dst[index] = total / total_weights
                total_weights += (weight[index + xdim + 2] + weight[index + 2] + weight[index - xdim + 2]
                                  - weight[index - xdim - 1] - weight[index - 1] - weight[index + xdim - 1])
                total += (product[index - xdim + 2] + product[index + 2] + product[index + xdim + 2]
                          - product[index - xdim - 1] - product[index - 1] - product[index + xdim - 1])
                index += 1

    for i in range(size):
        dst[i] = _jtrunc(current_dst[i])


@njit
def get_location_type(xindex, yindex, xdim, ydim):
    """1..9: corner / edge / interior position, row-major from top-left."""
    if yindex == 0:
        if xindex == 0:
            return 1
        elif xindex % xdim != xdim - 1:
            return 2
        return 3
    elif yindex % ydim != ydim - 1:
        if xindex == 0:
            return 4
        elif xindex % xdim != xdim - 1:
            return 5
        return 6
    else:
        if xindex == 0:
            return 7
        elif xindex % xdim != xdim - 1:
            return 8
        return 9


# Neighbour offsets (dy, dx) each location type may read, in the Java's order.
# Orthogonal neighbours weigh 1, diagonal ones 0.7071 in dilate_image.
_ORTHO = {
    1: ((0, 1), (1, 0)),
    2: ((0, -1), (0, 1), (1, 0)),
    3: ((0, -1), (1, 0)),
    4: ((-1, 0), (1, 0), (0, 1)),
    5: ((-1, 0), (1, 0), (0, -1), (0, 1)),
    6: ((-1, 0), (1, 0), (0, -1)),
    7: ((-1, 0), (0, 1)),
    8: ((-1, 0), (0, -1), (0, 1)),
    9: ((-1, 0), (0, -1)),
}
_DIAG = {
    1: ((1, 1),),
    2: ((1, -1), (1, 1)),
    3: ((1, -1),),
    4: ((-1, 1), (1, 1)),
    5: ((-1, -1), (1, -1), (-1, 1), (1, 1)),
    6: ((-1, -1), (1, -1)),
    7: ((-1, 1),),
    8: ((-1, -1), (-1, 1)),
    9: ((-1, -1),),
}
_VERT = {1: ((1, 0),), 2: ((1, 0),), 3: ((1, 0),),
         4: ((-1, 0), (1, 0)), 5: ((-1, 0), (1, 0)), 6: ((-1, 0), (1, 0)),
         7: ((-1, 0),), 8: ((-1, 0),), 9: ((-1, 0),)}


def _offset_table(ortho, diag):
    """Per location type, the (dy, dx, is_diagonal) neighbours to read, in
    the Java's order (the order matters: it fixes the floating-point sums)."""
    table = np.zeros((10, 8, 3), dtype=np.int64)
    counts = np.zeros(10, dtype=np.int64)
    for t in range(1, 10):
        n = 0
        for dy, dx in ortho.get(t, ()):
            table[t, n] = (dy, dx, 0)
            n += 1
        for dy, dx in diag.get(t, ()):
            table[t, n] = (dy, dx, 1)
            n += 1
        counts[t] = n
    return table, counts


_DILATE_TABLE = _offset_table(_ORTHO, _DIAG)
_DILATE_V_TABLE = _offset_table(_VERT, {})
_DILATE_D_TABLE = _offset_table({}, _DIAG)


@njit
def _dilate(src, is_interpolated, xdim, ydim, neighbor_threshold, dst, table, counts, diagonal_weight):
    was_interpolated = np.zeros(xdim * ydim, dtype=np.bool_)
    for i in range(ydim):
        for j in range(xdim):
            k = i * xdim + j
            if is_interpolated[k]:
                dst[k] = src[k]
                was_interpolated[k] = True
            else:
                total_weight = 0.0
                value = 0.0
                number_of_neighbors = 0
                t = get_location_type(j, i, xdim, ydim)
                for q in range(counts[t]):
                    n = k + table[t, q, 0] * xdim + table[t, q, 1]
                    if is_interpolated[n]:
                        number_of_neighbors += 1
                        if table[t, q, 2] == 1:
                            total_weight += diagonal_weight
                            value += diagonal_weight * src[n]
                        else:
                            total_weight += 1.
                            value += src[n]
                if number_of_neighbors > neighbor_threshold:
                    value /= total_weight
                    dst[k] = _jtrunc(value)
                    was_interpolated[k] = True
                else:
                    dst[k] = 0
                    was_interpolated[k] = False
    for i in range(xdim * ydim):
        is_interpolated[i] = was_interpolated[i]


def dilate_image(src, is_interpolated, xdim, ydim, neighbor_threshold, dst):
    """One dilation pass over the 8-neighbourhood (diagonals weigh 0.7071).
    Fills dst and updates is_interpolated in place; call repeatedly until
    every cell is interpolated."""
    _dilate(src, is_interpolated, xdim, ydim, neighbor_threshold, dst,
            _DILATE_TABLE[0], _DILATE_TABLE[1], 0.7071)


def dilate_image_vertical(src, is_interpolated, xdim, ydim, neighbor_threshold, dst):
    """One dilation pass using only the pixels above and below."""
    _dilate(src, is_interpolated, xdim, ydim, neighbor_threshold, dst,
            _DILATE_V_TABLE[0], _DILATE_V_TABLE[1], 1.0)


def dilate_image_diagonal(src, is_interpolated, xdim, ydim, neighbor_threshold, dst):
    """One dilation pass using only the four diagonal neighbours (weight 1)."""
    _dilate(src, is_interpolated, xdim, ydim, neighbor_threshold, dst,
            _DILATE_D_TABLE[0], _DILATE_D_TABLE[1], 1.0)


def get_image_dilation(src, is_interpolated):
    """Fill the uninterpolated cells of a 2-D float array: vertical dilation
    until it stalls, then diagonal, then regular. Assumes the sample density
    is greater in y than x. Prints progress, as the Java does."""
    ydim, xdim = src.shape
    gray1 = np.asarray(src, dtype=np.float64).reshape(-1).copy()
    gray2 = np.zeros(xdim * ydim)
    is_assigned = np.asarray(is_interpolated, dtype=np.bool_).reshape(-1).copy()
    number_of_uninterpolated_cells = int(np.count_nonzero(~is_assigned))
    number_of_iterations = 0
    even = True

    def next_pair():
        nonlocal even
        if even:
            even = False
            return gray1, gray2
        even = True
        return gray2, gray1

    previous = 0
    while number_of_uninterpolated_cells != 0 and number_of_uninterpolated_cells != previous:
        previous = number_of_uninterpolated_cells
        number_of_iterations += 1
        source, dest = next_pair()
        dilate_image_vertical(source, is_assigned, xdim, ydim, 0, dest)
        number_of_uninterpolated_cells = int(np.count_nonzero(~is_assigned))

    previous = 0
    while number_of_uninterpolated_cells != 0 and previous != number_of_uninterpolated_cells:
        print("Vertical dilation did not complete.")
        source, dest = next_pair()
        dilate_image_diagonal(source, is_assigned, xdim, ydim, 0, dest)
        previous = number_of_uninterpolated_cells
        number_of_uninterpolated_cells = int(np.count_nonzero(~is_assigned))

    previous = 0
    while number_of_uninterpolated_cells != 0 and previous != number_of_uninterpolated_cells:
        print("Diagonal dilation did not complete.")
        source, dest = next_pair()
        dilate_image(source, is_assigned, xdim, ydim, 0, dest)
        previous = number_of_uninterpolated_cells
        number_of_uninterpolated_cells = int(np.count_nonzero(~is_assigned))

    print("The final number of uninterpolated cells is " + str(number_of_uninterpolated_cells))
    result = (gray1 if even else gray2).reshape(ydim, xdim).copy()
    print("The number of iterations was " + str(number_of_iterations))
    return result


# ---- area-average resampling (32-bit Java int arithmetic) ----

@njit
def _area_tables(dim, new_dim):
    differential = dim / new_dim
    real_position = 0.
    current_whole_number = 0
    start_fraction = np.zeros(new_dim, dtype=np.int64)
    end_fraction = np.zeros(new_dim, dtype=np.int64)
    number_of_pixels = np.zeros(new_dim, dtype=np.int64)
    for i in range(new_dim):
        previous_position = real_position
        previous_whole_number = current_whole_number
        real_position += differential
        current_whole_number = _jtrunc(real_position)
        number_of_pixels[i] = current_whole_number - previous_whole_number
        start_fraction[i] = _jtrunc(1000. * (1. - (previous_position - previous_whole_number)))
        end_fraction[i] = _jtrunc(1000. * (real_position - current_whole_number))
    weight = _i32(_i32(_jtrunc(differential * dim)) * 1000)
    factor = _i32(dim * 1000)
    return start_fraction, end_fraction, number_of_pixels, weight, factor


@njit
def _area_line(source, dest, j0, i0, n_out, stride_in, stride_out, dim,
               start_fraction, end_fraction, number_of_pixels, weight, factor):
    """Resample one row (strides 1) or column (strides xdim); the input
    starts at j0, the output at i0."""
    i = i0
    j = j0
    for x in range(n_out - 1):
        if number_of_pixels[x] == 0:
            dest[i] = source[j]
            i += stride_out
        else:
            total = _i32(_i32(start_fraction[x] * dim) * source[j])
            j += stride_in
            k = number_of_pixels[x] - 1
            while k > 0:
                total = _i32(total + _i32(factor * source[j]))
                j += stride_in
                k -= 1
            total = _i32(total + _i32(_i32(end_fraction[x] * dim) * source[j]))
            total = _i32(jdiv(total, weight))
            dest[i] = total
            i += stride_out
    x = n_out - 1
    if number_of_pixels[x] == 0:
        dest[i] = source[j]
    else:
        total = _i32(_i32(start_fraction[x] * dim) * source[j])
        j += stride_in
        k = number_of_pixels[x] - 1
        while k > 0:
            total = _i32(total + _i32(factor * source[j]))
            j += stride_in
            k -= 1
        total = _i32(jdiv(total, _i32(weight - _i32(end_fraction[x] * dim))))
        dest[i] = total


@njit
def avg_area_x_transform(source, xdim, ydim, new_xdim):
    """Area-average resample of a flat image to width new_xdim."""
    sf, ef, npx, weight, factor = _area_tables(xdim, new_xdim)
    dest = np.zeros(ydim * new_xdim, dtype=np.int64)
    for y in range(ydim):
        _area_line(source, dest, y * xdim, y * new_xdim, new_xdim, 1, 1, xdim,
                   sf, ef, npx, weight, factor)
    return dest


@njit
def avg_area_y_transform(src, xdim, ydim, new_ydim):
    """Area-average resample of a flat image to height new_ydim."""
    sf, ef, npx, weight, factor = _area_tables(ydim, new_ydim)
    dst = np.zeros(xdim * new_ydim, dtype=np.int64)
    for x in range(xdim):
        _area_line(src, dst, x, x, new_ydim, xdim, xdim, ydim,
                   sf, ef, npx, weight, factor)
    return dst


def avg_area_transform(src, xdim, ydim, new_xdim, new_ydim):
    """Area-average resample of a flat image to new_xdim x new_ydim."""
    intermediate = avg_area_x_transform(src, xdim, ydim, new_xdim)
    return avg_area_y_transform(intermediate, new_xdim, ydim, new_ydim)


def avg_area_x_transform_2d(src, new_xdim):
    """2-D version of avg_area_x_transform (this overload is correct in Java)."""
    ydim, xdim = src.shape
    flat = np.ascontiguousarray(src, dtype=np.int64).reshape(-1)
    return avg_area_x_transform(flat, xdim, ydim, new_xdim).reshape(ydim, new_xdim)


def _java_row_reset_copy(src):
    """Reproduces the Java's `k = 0` inside the row loop: only the first xdim
    cells of the working array are written, ending up as the last row."""
    ydim, xdim = src.shape
    source = np.zeros(xdim * ydim, dtype=np.int64)
    source[:xdim] = src[ydim - 1]
    return source


def avg_area_y_transform_2d(src, new_ydim):
    """int[][] avgAreaYTransform, bug for bug (see the module docstring):
    every output row is the first row of the transformed working array."""
    ydim, xdim = src.shape
    dest = avg_area_y_transform(_java_row_reset_copy(src), xdim, ydim, new_ydim)
    return np.tile(dest[:xdim], (new_ydim, 1))


def avg_area_transform_2d(src, new_xdim, new_ydim):
    """int[][] avgAreaTransform, bug for bug (see the module docstring)."""
    ydim, xdim = src.shape
    dest = avg_area_transform(_java_row_reset_copy(src), xdim, ydim, new_xdim, new_ydim)
    if xdim > new_xdim:
        raise IndexError("Index %d out of bounds for length %d" % (new_xdim, new_xdim))
    dst = np.zeros((new_ydim, new_xdim), dtype=np.int64)
    dst[:, :xdim] = dest[:xdim]
    return dst


# ---- gradients, variance and registration ----

@njit
def get_gradient(src):
    """Per-pixel (xgradient, ygradient) of a 2-D array, shape (ydim, xdim, 2).
    Corners are NaN/NaN; horizontal edges have only x, vertical only y."""
    ydim, xdim = src.shape
    dst = np.full((ydim, xdim, 2), np.nan)
    for i in range(ydim):
        for j in range(xdim):
            t = get_location_type(j, i, xdim, ydim)
            if t == 2:
                dst[i, j, 0] = ((src[i, j + 1] - src[i, j - 1]) + (src[i + 1, j + 1] - src[i + 1, j - 1])) / 2
            elif t == 4:
                dst[i, j, 1] = ((src[i + 1, j] - src[i - 1, j]) + (src[i + 1, j + 1] - src[i - 1, j + 1])) / 2
            elif t == 5:
                dst[i, j, 0] = (src[i - 1, j + 1] - src[i - 1, j - 1] + src[i, j + 1] - src[i, j - 1]
                                + src[i + 1, j + 1] - src[i + 1, j - 1]) / 3
                dst[i, j, 1] = (src[i + 1, j - 1] - src[i - 1, j - 1] + src[i + 1, j] - src[i - 1, j]
                                + src[i + 1, j + 1] - src[i - 1, j + 1]) / 3
            elif t == 6:
                dst[i, j, 1] = (src[i + 1, j - 1] - src[i - 1, j - 1] + src[i + 1, j] - src[i - 1, j]) / 2
            elif t == 8:
                dst[i, j, 0] = ((src[i - 1, j + 1] - src[i - 1, j - 1]) + (src[i, j + 1] - src[i, j - 1])) / 2
    return dst


@njit
def get_smooth_gradient(src):
    """get_gradient, but an interior pixel's gradient is NaN along an axis
    where the pixel isn't strictly between its two neighbours."""
    ydim, xdim = src.shape
    dst = get_gradient(src)
    for i in range(1, ydim - 1):
        for j in range(1, xdim - 1):
            if not ((src[i, j + 1] < src[i, j] and src[i, j] < src[i, j - 1])
                    or (src[i, j + 1] > src[i, j] and src[i, j] > src[i, j - 1])):
                dst[i, j, 0] = np.nan
            if not ((src[i + 1, j] < src[i, j] and src[i, j] < src[i - 1, j])
                    or (src[i + 1, j] > src[i, j] and src[i, j] > src[i - 1, j])):
                dst[i, j, 1] = np.nan
    return dst


@njit
def get_variance(src):
    """Sum of absolute differences from each pixel to its in-bounds neighbours."""
    ydim, xdim = src.shape
    dst = np.zeros((ydim, xdim), dtype=np.int64)
    for i in range(ydim):
        for j in range(xdim):
            v = 0
            for di in range(-1, 2):
                ii = i + di
                if ii < 0 or ii >= ydim:
                    continue
                for dj in range(-1, 2):
                    jj = j + dj
                    if (di == 0 and dj == 0) or jj < 0 or jj >= xdim:
                        continue
                    v += abs(src[i, j] - src[ii, jj])
            dst[i, j] = v
    return dst


def extract(source, xoffset, yoffset, xdim, ydim):
    """The xdim x ydim window of a 2-D array starting at (xoffset, yoffset)."""
    if (xoffset < 0 or yoffset < 0 or yoffset + ydim > source.shape[0]
            or xoffset + xdim > source.shape[1]):
        raise IndexError("extract window out of bounds")
    return source[yoffset:yoffset + ydim, xoffset:xoffset + xdim].copy()


def shift(source, x, y):
    """Crop |x| columns and |y| rows: from the left/top when positive, from
    the right/bottom when negative (works for int or float arrays)."""
    ydim, xdim = source.shape
    _xdim = xdim - abs(x)
    _ydim = ydim - abs(y)
    k = y if y > 0 else 0
    m = x if x > 0 else 0
    return source[k:k + _ydim, m:m + _xdim].copy()


def contract(source):
    """Average each 2x2 neighbourhood (truncated): one pixel smaller each way."""
    s = source.astype(np.float64)
    avg = (s[:-1, :-1] + s[:-1, 1:] + s[1:, :-1] + s[1:, 1:]) * .25
    return np.trunc(avg).astype(np.int64)


@njit
def translate(source, x, y):
    """Bilinear sub-pixel shift, x and y in [-1, 1]; one pixel smaller each way.
    (Keeps the Java's *.5 on the result -- see the module docstring.)"""
    ydim, xdim = source.shape
    dest = np.zeros((ydim - 1, xdim - 1), dtype=np.int64)
    x = (x + 1.) * .5
    y = (y + 1.) * .5
    for i in range(ydim - 1):
        for j in range(xdim - 1):
            a = float(source[i, j]) * (1. - x) + float(source[i, j + 1]) * x
            b = float(source[i + 1, j]) * (1. - x) + float(source[i + 1, j + 1]) * x
            dest[i, j] = _jtrunc((a * (1. - y) + b * y) * .5 + .5)
    return dest


def _fdiv(a, b):
    """Java double division (IEEE: x/0 -> +-Inf or NaN, never an exception)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(a) / np.float64(b))


def _normal_equations(gradient, source, estimate, ydim, xdim):
    w = x = z = b1 = b2 = 0.0
    for i in range(1, ydim - 1):
        for j in range(1, xdim - 1):
            xg = gradient[i, j, 0]
            yg = gradient[i, j, 1]
            if not math.isnan(xg) and not math.isnan(yg):
                delta = float(source[i, j] - estimate[i, j])
                w += xg * xg
                x += xg * yg
                z += yg * yg
                b1 += xg * delta
                b2 += yg * delta
    return w, x, z, b1, b2


def get_translation(source1, source2):
    """Sub-pixel translation of source2 relative to source1 by iterated
    least squares on the gradients. Returns [status, xtranslation,
    ytranslation]; status 1 converged, 2 oscillating, 3 hit the iteration
    limit, 4 translation reached 1 pixel, 0 no motion."""
    ydim, xdim = source1.shape
    estimate = np.array(source2, dtype=np.int64, copy=True)

    w, x, z, b1, b2 = _normal_equations(get_gradient(estimate), source1, estimate, ydim, xdim)
    xincrement = _fdiv(b1 - _fdiv(x * b2, z), w - _fdiv(x * x, z))
    yincrement = _fdiv(b2 - _fdiv(x * b1, w), z - _fdiv(x * x, w))

    if xincrement == 0. and yincrement == 0.:
        return [0.0, 0.0, 0.0]

    xincrement_min = abs(xincrement) / 100.
    yincrement_min = abs(yincrement) / 100.
    xtranslation = xincrement
    ytranslation = yincrement

    current_source = contract(source1)
    estimate = translate(source2, xtranslation, ytranslation)
    current_number_of_estimates = 1
    maximum_number_of_estimates = 10

    while current_number_of_estimates < maximum_number_of_estimates:
        _ydim, _xdim = estimate.shape
        w, x, z, b1, b2 = _normal_equations(get_gradient(estimate), current_source, estimate, _ydim, _xdim)

        # As in the Java, "previous" is assigned the new increment before the
        # comparison, so the oscillation test (status 2) never fires.
        xincrement = _fdiv(b1 - _fdiv(x * b2, z), w - _fdiv(x * x, z))
        xtranslation += xincrement
        previous_xincrement = xincrement
        yincrement = _fdiv(b2 - _fdiv(x * b1, w), z - _fdiv(x * x, w))
        ytranslation += yincrement
        previous_yincrement = yincrement

        if abs(xincrement) < xincrement_min or abs(yincrement) < yincrement_min:
            return [1.0, xtranslation, ytranslation]
        elif ((xincrement < 0 and previous_xincrement > 0) or (xincrement > 0 and previous_xincrement < 0)
              or (yincrement < 0 and previous_yincrement > 0) or (yincrement > 0 and previous_yincrement < 0)):
            return [2.0, xtranslation, ytranslation]
        elif xtranslation >= 1. or ytranslation >= 1.:
            return [4.0, xtranslation, ytranslation]
        else:
            estimate = translate(source2, xtranslation, ytranslation)
            current_number_of_estimates += 1
    return [3.0, xtranslation, ytranslation]


@njit
def expand_x_int(src, expand):
    """Insert `expand` stepped values between horizontal neighbours (int[][]
    expandX, bug for bug -- see the module docstring)."""
    ydim, xdim = src.shape
    _xdim = (xdim - 1) * expand + xdim
    dst = np.zeros((ydim, _xdim), dtype=np.int64)
    for i in range(ydim):
        k = 0
        end_value = 0
        for j in range(xdim - 1):
            start_value = src[i, j]
            end_value = src[i, j + 1]
            dst[i, k] = start_value
            k += 1
            delta = float(start_value - end_value)
            increment = delta / (expand + 1)
            for _ in range(expand):
                start_value = _jtrunc(start_value + increment)
                dst[i, k] = start_value
                k += 1
        dst[i, k] = end_value
    return dst


def expand_x(src):
    """Double the width of a 2-D float array minus one, inserting midpoints."""
    ydim, xdim = src.shape
    dst = np.zeros((ydim, 2 * xdim - 1))
    dst[:, 0::2] = src
    dst[:, 1::2] = (src[:, :-1] + src[:, 1:]) / 2
    return dst


def expand_x_iterated(src, iterations):
    """expand_x applied `iterations` times (src itself when iterations <= 0)."""
    result = src
    for _ in range(iterations):
        result = expand_x(result)
    return result


def shrink_avg_double(src):
    """Unrounded 2x2 block average of a 2-D float array."""
    ydim, xdim = src.shape
    s = np.asarray(src, dtype=np.float64)[:ydim // 2 * 2, :xdim // 2 * 2]
    return (s[0::2, 0::2] + s[0::2, 1::2] + s[1::2, 0::2] + s[1::2, 1::2]) / 4.
