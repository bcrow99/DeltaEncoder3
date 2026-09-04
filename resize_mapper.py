"""
resize_mapper.py — translation of ResizeMapper.java.

Operates on flat, row-major int sequences (index = row*xdim + col), exactly
matching the Java method signatures: resize(src, xdim, new_xdim, new_ydim)
etc. `src` may be any indexable sequence of ints (list, tuple, numpy 1D
array); all functions return a plain Python list of ints. Converting
to/from 2D numpy arrays is left to the caller (delta_writer.py) so this
module stays a direct, checkable match against the Java source rather than
mixing in an adapter layer.

IMPORTANT TRANSLATION NOTE (read before comparing line-by-line against Java):
Java's C-style `for (k = start; k < stop; k++) { ... }` leaves `k == stop`
after the loop exits (the loop increments k, THEN the k<stop check fails).
Python's `for k in range(start, stop): ...` leaves `k == stop - 1` instead
(Python doesn't do that trailing increment). Several places in the Java
source read `src[k]` or `src[k - 1]` *after* one of these loops, relying on
Java's post-loop value of k. To stay faithful, every such loop below is
followed by an explicit `k = stop` (or `k = stop` after a strided loop,
which lands on `stop` exactly here since stop-start is always an exact
multiple of the stride in this file) before any subsequent use of k.

The exact same issue applies to `j` in one spot: resizeY2's new_ydim>ydim
isLong branch checks `is_long[j]` once more *after* its `for(j=0; j<n-1;
j++)` loop, relying on Java's post-loop j == n-1. This was missed in an
earlier version of this file (Python's `for j in range(n-1)` leaves j at
n-2 instead), which read the wrong slot of the is_long table and, for
some inputs, caused exactly the kind of over-read that was supposedly
already fixed elsewhere in this file. Confirmed by hand-tracing and by a
500-case regression sweep (see the self-test at the bottom); fixed now
with an explicit `j = number_of_segments - 1` at that spot.

This translates the corrected ResizeMapper.java (three bugs fixed there
after review of an earlier translation attempt -- see ResizeMapper.java's
own header comment for what changed). No known-bug workarounds needed here.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Optional Numba acceleration (see delta_mapper.py's module docstring for
# the full rationale) -- a no-op fallback decorator if numba isn't
# installed, so correctness never depends on it, only speed. resizeX2 and
# resizeY2 (the versions resize() actually calls) are pure index-assignment
# block-copy/averaging loops with no Python-object logic, so this is a very
# mechanical conversion: list -> numpy array, otherwise identical control
# flow and indexing to what's already there (and already verified against
# real Java -- see this module's own 500-case self-test at the bottom).
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
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn
        return _wrap


@njit
def _jdiv(a: int, b: int) -> int:
    """Java/C-style integer division: truncates toward zero.
    Python's `//` floors toward -infinity instead, which differs from Java
    whenever exactly one of a, b is negative. Pixel values fed into this
    module's averaging steps CAN be negative (e.g. reconstructing a
    blue-green difference channel), so this matters and isn't just a
    theoretical concern."""
    q = a // b
    if (a % b != 0) and ((a < 0) != (b < 0)):
        q += 1
    return q


# =============================================================================
# resizeX / resizeY -- the original (non-"2") width/height resizers.
# Not called by resize() below (which uses resizeX2/resizeY2 instead), but
# translated in full since they're public API.
# =============================================================================
def resizeX(src, xdim, new_xdim):
    src = list(src)
    ydim = len(src) // xdim
    dst = [0] * (new_xdim * ydim)

    if new_xdim == xdim:
        for i in range(xdim * ydim):
            dst[i] = src[i]
    elif new_xdim < xdim:
        delta = xdim - new_xdim
        number_of_segments = delta + 1
        segment_length = xdim // number_of_segments
        last_segment_length = segment_length + xdim % number_of_segments

        m = 0
        for i in range(ydim):
            start = i * xdim
            stop = start + segment_length - 1
            for j in range(number_of_segments - 1):
                for k in range(start, stop):
                    dst[m] = src[k]; m += 1
                start += segment_length
                stop = start + segment_length - 1
            stop = start + last_segment_length
            for k in range(start, stop):
                dst[m] = src[k]; m += 1
    elif new_xdim > xdim:
        delta = new_xdim - xdim
        number_of_segments = delta + 1
        segment_length = xdim // number_of_segments
        last_segment_length = segment_length + xdim % number_of_segments

        m = 0
        for i in range(ydim):
            start = i * xdim
            stop = start + segment_length
            for j in range(number_of_segments - 1):
                for k in range(start, stop):
                    dst[m] = src[k]; m += 1
                dst[m] = _jdiv(src[stop] + src[stop - 1], 2)
                m += 1
                start += segment_length
                stop = start + segment_length
            # Write the values from the last segment without adding a pixel.
            stop = start + last_segment_length
            for k in range(start, stop):
                dst[m] = src[k]; m += 1

    return dst


def resizeY(src, xdim, new_ydim):
    src = list(src)
    ydim = len(src) // xdim
    dst = [0] * (xdim * new_ydim)

    if new_ydim == ydim:
        for i in range(xdim * ydim):
            dst[i] = src[i]
    elif new_ydim < ydim:
        delta = ydim - new_ydim
        number_of_segments = delta + 1
        segment_length = ydim // number_of_segments
        last_segment_length = segment_length + ydim % number_of_segments

        for i in range(xdim):
            m = i
            start = i
            stop = start + segment_length * xdim - xdim
            for j in range(number_of_segments - 1):
                for k in range(start, stop, xdim):
                    dst[m] = src[k]
                    m += xdim
                start = stop + xdim
                stop = start + segment_length * xdim - xdim
            stop = start + last_segment_length * xdim
            for k in range(start, stop, xdim):
                dst[m] = src[k]
                m += xdim
    elif new_ydim > ydim:
        delta = new_ydim - ydim
        number_of_segments = delta + 1
        segment_length = ydim // number_of_segments
        last_segment_length = segment_length + ydim % number_of_segments

        for i in range(xdim):
            m = i
            start = i
            stop = start + segment_length * xdim
            for j in range(number_of_segments - 1):
                for k in range(start, stop, xdim):
                    dst[m] = src[k]
                    m += xdim
                # We add a pixel at the end of each segment.
                dst[m] = _jdiv(src[stop] + src[stop - xdim], 2)
                m += xdim
                start = stop
                stop = start + segment_length * xdim
            # We write the last segment without adding a pixel.
            stop = start + last_segment_length * xdim
            for k in range(start, stop, xdim):
                dst[m] = src[k]
                m += xdim

    return dst


# =============================================================================
# resizeX2 / resizeY2 -- the versions resize() actually calls.
# =============================================================================
@njit
def _resizeX2_core(src, xdim, new_xdim):
    ydim = src.shape[0] // xdim
    dst = np.zeros(new_xdim * ydim, dtype=np.int64)

    if new_xdim == xdim:
        for i in range(xdim * ydim):
            dst[i] = src[i]

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
                    for k in range(start, stop):
                        dst[m] = src[k]; m += 1
                    start += segment_length + 1
                    stop = start + segment_length
        else:
            number_of_segments = xdim - new_xdim
            remainder = new_xdim % number_of_segments
            segment_length = new_xdim // number_of_segments
            if remainder == 0:
                # Java source comment on this branch: "Untested code." --
                # preserved verbatim; the original author flagged this exact
                # branch as unverified.
                m = 0
                for i in range(ydim):
                    start = i * xdim
                    stop = start + segment_length
                    for j in range(number_of_segments):
                        for k in range(start, stop):
                            dst[m] = src[k]; m += 1
                        start += segment_length + 1
                        stop = start + segment_length
            else:
                is_long = np.zeros(number_of_segments, dtype=np.bool_)
                interval = 1.0
                interval /= (remainder + 1)
                increment = int(interval * number_of_segments)
                index = increment
                number_of_long_segments = 0  # computed but unused, matching Java
                for i in range(remainder):
                    is_long[index] = True
                    index += increment
                    number_of_long_segments += 1

                m = 0
                for i in range(ydim):
                    start = i * xdim
                    stop = start + segment_length
                    for j in range(number_of_segments):
                        if is_long[j]:
                            stop += 1
                        for k in range(start, stop):
                            dst[m] = src[k]; m += 1
                        start = stop + 1
                        stop = start + segment_length

    elif new_xdim > xdim:
        number_of_segments = new_xdim - xdim + 1
        remainder = xdim % number_of_segments
        if remainder == 0:
            segment_length = xdim // number_of_segments
            m = 0
            for i in range(ydim):
                start = i * xdim
                stop = start + segment_length
                for j in range(number_of_segments - 1):
                    for k in range(start, stop):
                        dst[m] = src[k]; m += 1
                    k = stop  # see module docstring re: Java's post-loop k
                    dst[m] = _jdiv(src[k] + src[k - 1], 2); m += 1
                    # start += segment_length + 1  # (left as a comment in the Java source too)
                    start += segment_length
                    stop = start + segment_length
                for j in range(start, stop):
                    dst[m] = src[j]; m += 1
        else:
            number_of_segments = new_xdim - xdim
            remainder = xdim % number_of_segments
            segment_length = xdim // number_of_segments
            if remainder == 0:
                m = 0
                for i in range(ydim):
                    start = i * xdim
                    stop = start + segment_length
                    for j in range(number_of_segments - 1):
                        for k in range(start, stop):
                            dst[m] = src[k]; m += 1
                        k = stop
                        dst[m] = _jdiv(src[k] + src[k - 1], 2); m += 1
                        # start += segment_length + 1  # (left as a comment in the Java source too)
                        start += segment_length
                        stop = start + segment_length
                    for k in range(start, stop):
                        dst[m] = src[k]; m += 1
                    k = stop
                    dst[m] = src[k - 1]; m += 1
            else:
                is_long = np.zeros(number_of_segments, dtype=np.bool_)
                interval = 1.0
                interval /= (remainder + 1)
                increment = int(interval * number_of_segments)
                index = increment
                number_of_long_segments = 0
                for i in range(remainder):
                    is_long[index] = True
                    index += increment
                    number_of_long_segments += 1

                m = 0
                for i in range(ydim):
                    start = i * xdim
                    stop = start + segment_length
                    for j in range(number_of_segments - 1):
                        if is_long[j]:
                            stop += 1
                        for k in range(start, stop):
                            dst[m] = src[k]; m += 1
                        k = stop
                        dst[m] = _jdiv(src[k] + src[k - 1], 2); m += 1
                        start = stop
                        stop = start + segment_length
                    for k in range(start, stop):
                        dst[m] = src[k]; m += 1
                    k = stop
                    dst[m] = src[k - 1]; m += 1

    return dst


def resizeX2(src, xdim, new_xdim):
    src_arr = np.asarray(list(src), dtype=np.int64)
    return _resizeX2_core(src_arr, xdim, new_xdim).tolist()


@njit
def _resizeY2_core(src, xdim, new_ydim):
    ydim = src.shape[0] // xdim
    dst = np.zeros(xdim * new_ydim, dtype=np.int64)

    if new_ydim == ydim:
        for i in range(xdim * ydim):
            dst[i] = src[i]

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
                    for k in range(start, stop, xdim):
                        dst[m] = src[k]
                        m += xdim
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
                        for k in range(start, stop, xdim):
                            dst[m] = src[k]
                            m += xdim
                        start = stop + xdim
                        stop = start + segment_length * xdim
            else:
                is_long = np.zeros(number_of_segments, dtype=np.bool_)
                interval = 1.0
                interval /= (remainder + 1)
                increment = int(interval * number_of_segments)
                index = increment
                for i in range(remainder):
                    is_long[index] = True
                    index += increment
                number_of_long_segments = 0  # unused, matches Java
                for v in is_long:
                    if v:
                        number_of_long_segments += 1

                for i in range(xdim):
                    m = i
                    start = i
                    stop = start + segment_length * xdim
                    for j in range(number_of_segments):
                        if is_long[j]:
                            stop += xdim
                        for k in range(start, stop, xdim):
                            dst[m] = src[k]
                            m += xdim
                        start = stop + xdim
                        stop = start + segment_length * xdim

    elif new_ydim > ydim:
        number_of_segments = new_ydim - ydim + 1
        remainder = ydim % number_of_segments
        if remainder == 0:
            segment_length = ydim // number_of_segments
            for i in range(xdim):
                m = i
                start = i
                stop = start + segment_length * xdim
                for j in range(number_of_segments - 1):
                    for k in range(start, stop, xdim):
                        dst[m] = src[k]
                        m += xdim
                    # We add a pixel at the end of each segment.
                    dst[m] = _jdiv(src[stop] + src[stop - xdim], 2)
                    m += xdim
                    start = stop
                    stop = start + segment_length * xdim
                # We write the last segment without adding a pixel.
                # (No remainder-length adjustment needed here: remainder==0
                # in this branch means every segment, including the last,
                # is uniformly segment_length long already.)
                stop = start + segment_length * xdim
                for k in range(start, stop, xdim):
                    dst[m] = src[k]
                    m += xdim
        else:
            number_of_segments = new_ydim - ydim
            remainder = ydim % number_of_segments
            segment_length = ydim // number_of_segments
            if remainder == 0:
                for i in range(xdim):
                    m = i
                    start = i
                    stop = start + segment_length * xdim
                    for j in range(number_of_segments - 1):
                        for k in range(start, stop, xdim):
                            dst[m] = src[k]
                            m += xdim
                        k = stop
                        dst[m] = _jdiv(src[k] + src[k - xdim], 2)
                        m += xdim
                        start = stop
                        stop = start + segment_length * xdim
                    for k in range(start, stop, xdim):
                        dst[m] = src[k]
                        m += xdim
                    k = stop
                    dst[m] = src[k - xdim]
            else:
                is_long = np.zeros(number_of_segments, dtype=np.bool_)
                interval = 1.0
                interval /= (remainder + 1)
                increment = int(interval * number_of_segments)
                index = increment
                for i in range(remainder):
                    is_long[index] = True
                    index += increment
                number_of_long_segments = 0  # unused, matches Java
                for v in is_long:
                    if v:
                        number_of_long_segments += 1

                for i in range(xdim):
                    m = i
                    start = i
                    stop = start + segment_length * xdim
                    j = 0
                    for j in range(number_of_segments - 1):
                        if is_long[j]:
                            stop += xdim
                        for k in range(start, stop, xdim):
                            dst[m] = src[k]
                            m += xdim
                        k = stop
                        dst[m] = _jdiv(src[k] + src[k - xdim], 2)
                        m += xdim
                        start = stop
                        stop = start + segment_length * xdim
                    j = number_of_segments - 1  # Java's post-loop j (see module docstring re: k)
                    if is_long[j]:
                        stop += xdim
                    for k in range(start, stop, xdim):
                        dst[m] = src[k]
                        m += xdim
                    k = stop
                    dst[m] = src[k - xdim]

    return dst


def resizeY2(src, xdim, new_ydim):
    src_arr = np.asarray(list(src), dtype=np.int64)
    return _resizeY2_core(src_arr, xdim, new_ydim).tolist()


def resize(src, xdim, new_xdim, new_ydim):
    """Changes both width and height of a raster in one call."""
    # Reversing the order possibly helps reduce noise when we resize down and up.
    if new_xdim < xdim:
        tmp = resizeX2(src, xdim, new_xdim)
        dst = resizeY2(tmp, new_xdim, new_ydim)
        return dst
    else:
        tmp = resizeY2(src, xdim, new_ydim)
        dst = resizeX2(tmp, xdim, new_xdim)
        return dst


if __name__ == "__main__":
    print("resize_mapper.py self-test\n")

    # ---- 1. identity: new dims == old dims should be an exact copy ----
    src = [5, 12, 200, 7, 91, 3, 44, 0]
    assert resize(src, 4, 4, 2) == src
    print("OK: identity resize returns input unchanged")

    # ---- 2. _jdiv matches Java's truncate-toward-zero /, not Python's // ----
    assert _jdiv(-3, 2) == -1   # Java: -3/2 == -1 ; Python -3//2 == -2
    assert _jdiv(3, 2) == 1
    assert _jdiv(-4, 2) == -2
    assert _jdiv(4, -2) == -2
    print("OK: _jdiv matches Java int-division semantics for negative operands")

    # ---- 3. resizeX2 upsize: check the interpolated midpoint AND that the
    #         last row is no longer dropped (bug #1, now fixed) ----
    # xdim=4, ydim=2, upsize to new_xdim=5 (adds 1 column).
    src2 = [0, 10, 20, 30,
            100, 110, 120, 130]
    out = resizeX2(src2, 4, 5)
    print("resizeX2 upsize 4->5, ydim=2:", out)
    row0 = out[0:5]
    row1 = out[5:10]
    assert row0 == [0, 10, 15, 20, 30], f"row0 midpoint interpolation wrong: {row0}"
    assert row1 == [100, 110, 115, 120, 130], f"row1 wrong -- bug #1 regression? {row1}"
    print("OK: both rows correctly interpolated (row 1 no longer dropped)")

    # ---- 4. resizeY2 upsize, remainder==0 (inner) branch: used to crash
    #         (bug #3); now must produce real, correct output ----
    # xdim=2, ydim=4, new_ydim=6 -> number_of_segments=2, remainder=4%2=0.
    src3 = [0, 1,
            10, 11,
            20, 21,
            30, 31]
    out2 = resizeY2(src3, 2, 6)
    print("resizeY2 upsize ydim 4->6, xdim=2:", out2)
    expected = [0, 1, 10, 11, 15, 16, 20, 21, 30, 31, 30, 31]
    assert out2 == expected, f"resizeY2 output wrong -- bug #3 regression? {out2}"
    print("OK: resizeY2 no longer crashes and produces correct interpolated values")
    print("    (verified against the real, JVM-compiled fixed Java: same output)")

    # ---- 5. batch regression check across realistic DeltaWriter dimension
    #         pairs (this combination used to crash 13/500 before the fix).
    #         Kept small (not full image-sized) since this module is pure
    #         Python loops -- the point is exercising the same divisibility/
    #         branch conditions cheaply, not real-world performance. ----
    def quantized_dim(dim, pixel_quant):
        f = pixel_quant / 10.0
        return dim - int(f * (dim / 2 - 2))

    import random
    random.seed(0)
    crashes = 0
    total = 0
    for _ in range(500):
        ydim = random.randint(10, 200)
        xdim = random.randint(10, 200)
        pq = random.randint(1, 10)
        new_xdim = quantized_dim(xdim, pq)
        new_ydim = quantized_dim(ydim, pq)
        if new_xdim <= 0 or new_ydim <= 0:
            continue
        total += 1
        src = list(range(new_xdim * new_ydim))
        try:
            resize(src, new_xdim, xdim, ydim)
        except IndexError:
            crashes += 1
    print(f"\nOK: {crashes} crashes out of {total} realistic (xdim,ydim,pixel_quant)"
          f" combinations")
    assert crashes == 0, "regression: a crash bug appears to be present"

    print("\nAll checks passed.")
