"""
neighbour_mapper.py -- NeighbourMapper.java for Python: the neighbour Huffman
coding of a channel's deltas (HuffmanWriter2's, and GenericWriter's and
NativeWriter's "Neighbour Huffman"), the same bytes from the same deltas.

Each delta's code table is chosen by how busy its neighbourhood is: the size
of the deltas already coded to its left and above (_activity, a weighted sum
of six of them), in one of CONTEXTS - 1 steps. The decoder knows the same
neighbours, so it makes the same choice: the coding says nothing about which
table is used where. For the quiet parts, where most deltas are 0 and a
Huffman code's one bit a delta at the least would be too much: a run mode
(where the neighbourhood is all zeros, the number of zero deltas that follow,
to the row's end at most, coded as one symbol: its length's size class, then
the length's bits below its top bit as they are, then the delta that stopped
the run), and pairs (in contexts 1 to PAIR_CONTEXTS, two deltas as one
symbol, each -PAIR_RANGE to PAIR_RANGE or an escape, then coded from a table
of its own). A single delta is coded with its sign flipped where its
neighbours' (left + above) is negative (version 2: NeighbourMapper.flipped),
so a delta going the way its neighbours go is always coded as positive, and a
table can give its short codes to one side. Version 3 adds a second opinion
(NeighbourMapper.opinion: how far the median edge detector, from the pixel
values around, is from the rebuilding's prediction, 0 to 4 either way): it
picks the tables too, flips a delta (pairs too) when it's negative, and breaks
the run mode when it isn't 0. Its decoder needs the values and predictions as
it goes, so it decodes a pixel at a time inside the rebuilding (_nb_next;
neighbour_bands); versions 1 and 2 decode a channel's deltas first
(neighbour_decode). Each channel has its own tables, made from its own deltas; their
code lengths are coded compactly (length_coding._length_symbols), with no
Deflate, so the coding doesn't depend on a zlib (Python's own is zlib-ng on
Windows).

Layout (neighbour_code): int lo, the smallest delta either way round (-max of
|smallest|, |largest|, so that a flipped delta is in range too); int the number
of delta values (-2 lo + 1); int the number of bytes that follow; those bytes, a bit
stream (most significant bit first): the tables' code lengths, then the codes.
Delta 0 isn't coded (the band's first value is written instead).

As in NativeEncoder's C (native/NeighbourMapper.c), the coding walks the
deltas once, counting and recording the symbols, and then codes the records;
the decoding is one pass. Both are compiled by Numba (numba_support.njit,
releasing the GIL).
"""

import threading

import numpy as np

from numba_support import njit, njit_no_refs, jdiv
import delta_mapper as dm
from length_coding import (MAX_CODE_LENGTH, LENGTH_SYMBOLS, LOOKUP_BITS, _code_lengths, _canonical_codes, _length_symbols, _put, _make_table, _read_lengths)   # the table machinery the table codings share

CONTEXTS = 16
ACTIVITY_STEPS = (2, 3, 4, 6, 8, 12, 16, 23, 32, 46, 64, 91, 128, 181)
PAIR_CONTEXTS = 6
PAIR_RANGE = 3
PAIR_ESCAPE = 2 * PAIR_RANGE + 1
PAIR_WIDTH = PAIR_ESCAPE + 1
RUN_SYMBOLS = 18
RUN_TABLE = 0
OPINION_LEVELS = 5   # version 3's opinion levels, the size of a second opinion (NeighbourMapper.OPINION_LEVELS)
# The coding's versions, those of the files that hold it (NeighbourMapper.VERSION,
# VERSION_FLIPPED, VERSION_UNFLIPPED): 3, with the second opinion, which
# neighbour_code writes and _nb_next reads; 2, the single deltas' signs flipped
# where their neighbours' (left + above) is negative, and 1, the first,
# without, which neighbour_decode still reads.
VERSION, VERSION_FLIPPED, VERSION_UNFLIPPED = 3, 2, 1


# The tables (NeighbourMapper.singleTable, pairTable, stopTable, escapeTable,
# tables): the run table, the singles' (context c, opinion level l), the
# pairs', the stop table, the escape table; levels the number of opinion
# levels the tables are split by (1 in versions 1 and 2, OPINION_LEVELS in 3).
@njit
def _single_table(c, l):
    return 1 + l * (CONTEXTS - 1) + c - 1


@njit
def _pair_table(c, l, levels):
    return 1 + levels * (CONTEXTS - 1) + l * PAIR_CONTEXTS + c - 1


@njit
def _stop_table(levels):
    return 1 + levels * (CONTEXTS - 1 + PAIR_CONTEXTS)


@njit
def _tables(levels):
    return _stop_table(levels) + 2


def levels(version):
    """The number of opinion levels a version's tables are split by (NeighbourMapper.levels)."""
    return OPINION_LEVELS if version >= 3 else 1


def _context_table():
    """context(a) for a up to the last step (NeighbourMapper.context)."""
    table = np.zeros(ACTIVITY_STEPS[-1] + 1, dtype=np.int64)
    for a in range(1, len(table)):
        c = 1
        while c - 1 < len(ACTIVITY_STEPS) and a >= ACTIVITY_STEPS[c - 1]:
            c += 1
        table[a] = c
    return table


CONTEXT_OF = _context_table()
LAST_STEP = ACTIVITY_STEPS[-1]


@njit
def _activity(d, k, x, y, xdim):
    """NeighbourMapper.activity: 2 |left| + 2 |above| + |above left| + |above
    right| + |two left| + |two above| (on the second row, above again for two
    above); on the first row 8 |left|, in the first column 8 |above|, in the
    second and last columns 4 (|left| + |above|)."""
    if y == 0:
        return 8 * abs(d[k - 1])
    if x == 0:
        return 8 * abs(d[k - xdim])
    if x == 1 or x == xdim - 1:
        return 4 * (abs(d[k - 1]) + abs(d[k - xdim]))
    two_above = d[k - 2 * xdim] if y >= 2 else d[k - xdim]
    return (2 * abs(d[k - 1]) + 2 * abs(d[k - xdim]) + abs(d[k - xdim - 1]) + abs(d[k - xdim + 1])
            + abs(d[k - 2]) + abs(two_above))


@njit
def _context(a):
    """NeighbourMapper.context: 0 for 0, else 1 plus the steps a reaches."""
    return CONTEXTS - 1 if a >= LAST_STEP else CONTEXT_OF[a]


@njit
def _table_size(t, values, levels):
    if t == RUN_TABLE:
        return RUN_SYMBOLS
    if _pair_table(1, 0, levels) <= t < _stop_table(levels):
        return PAIR_WIDTH * PAIR_WIDTH
    return values


@njit
def _opinion(v, at, x, y, xdim, p):
    """A pixel's second opinion (version 3; NeighbourMapper.opinion): the
    median edge detector (left, above, above-left) less the rebuilding's
    prediction p, as a JPEG-LS gradient level (-4 to 4); only at rows 1 on,
    columns 1 to xdim - 2, else 0. at: the pixel's index in v."""
    if y < 1 or x < 1 or x > xdim - 2:
        return 0
    return dm._gradient_level(dm._med(v[at - 1], v[at - xdim], v[at - xdim - 1]) - p)


@njit
def _pair_value(delta):
    return delta + PAIR_RANGE if -PAIR_RANGE <= delta <= PAIR_RANGE else PAIR_ESCAPE


# ---- The coding ------------------------------------------------------------

@njit
def _walk(d, opinion, n, xdim, lo, frequency, record, extra):
    """The walk through the deltas (NeighbourMapper.walk; version 3, with
    each pixel's second opinion), once: each symbol counted into
    frequency[table] and recorded, table * 2^24 + symbol, in order; a run
    length's bits below its top bit into extra. Returns (the number of
    records, the number of extras)."""
    m = 0
    e = 0
    ydim = n // xdim
    stop_table = _stop_table(OPINION_LEVELS)
    escape_table = stop_table + 1
    for y in range(ydim):
        row = y * xdim
        end = row + xdim
        k = 1 if y == 0 else row
        while k < end:
            x = k - row
            a = _activity(d, k, x, y, xdim)
            o = opinion[k]
            if a == 0 and o == 0:
                r = 0
                while k + r < end and d[k + r] == 0:
                    r += 1
                size = 0
                v = r
                while v > 0:
                    size += 1
                    v >>= 1
                frequency[RUN_TABLE, size] += 1
                record[m] = (RUN_TABLE << 24) | size
                m += 1
                if size > 1:
                    extra[e] = r & ((1 << (size - 1)) - 1)
                    e += 1
                k += r
                if k < end:
                    s = d[k] - lo
                    frequency[stop_table, s] += 1
                    record[m] = (stop_table << 24) | s
                    m += 1
                    k += 1
                continue
            c = 1 if a == 0 else _context(a)
            l = abs(o)
            if c <= PAIR_CONTEXTS and k + 1 < end:
                # Two deltas as one symbol (flipped if the opinion is negative).
                d0 = -d[k] if o < 0 else d[k]
                d1 = -d[k + 1] if o < 0 else d[k + 1]
                u = _pair_value(d0)
                w = _pair_value(d1)
                t = _pair_table(c, l, OPINION_LEVELS)
                s = u * PAIR_WIDTH + w
                frequency[t, s] += 1
                record[m] = (t << 24) | s
                m += 1
                if u == PAIR_ESCAPE:
                    s = d0 - lo
                    frequency[escape_table, s] += 1
                    record[m] = (escape_table << 24) | s
                    m += 1
                if w == PAIR_ESCAPE:
                    s = d1 - lo
                    frequency[escape_table, s] += 1
                    record[m] = (escape_table << 24) | s
                    m += 1
                k += 2
                continue
            single = d[k]
            if o < 0 or (o == 0 and (d[k - 1] if x > 0 else 0) + (d[k - xdim] if y > 0 else 0) < 0):
                single = -single   # flipped (by the opinion's sign, or, if it's 0, the neighbours': NeighbourMapper.flipped)
            s = single - lo
            t = _single_table(c, l)
            frequency[t, s] += 1
            record[m] = (t << 24) | s
            m += 1
            k += 1
    return m, e


@njit
def _code(delta, value, offset, xdim):
    """neighbour_code's work: (lo, the number of delta values, the bit
    stream)."""
    n = len(delta)
    d = delta.copy()
    d[0] = 0
    opinion = np.zeros(n, dtype=np.int64)
    for k in range(xdim, n):
        opinion[k] = _opinion(value, offset + k, k % xdim, k // xdim, xdim, value[offset + k] - delta[k])
    lo = 0
    hi = 0
    if n > 1:
        lo = d[1]
        hi = d[1]
        for k in range(2, n):
            if d[k] < lo:
                lo = d[k]
            elif d[k] > hi:
                hi = d[k]
    hi = max(-lo, hi)   # either way round: a flipped delta is in range too
    lo = -hi
    values = hi - lo + 1
    width = max(values, PAIR_WIDTH * PAIR_WIDTH, RUN_SYMBOLS)
    tables = _tables(OPINION_LEVELS)
    frequency = np.zeros((tables, width), dtype=np.int64)
    record = np.empty(2 * n + 1, dtype=np.int64)
    extra = np.empty(n + 1, dtype=np.int64)
    m, e = _walk(d, opinion, n, xdim, lo, frequency, record, extra)

    # The tables, their lengths all together (for _length_symbols).
    length = np.zeros((tables, width), dtype=np.int64)
    code = np.zeros((tables, width), dtype=np.int64)
    total = 0
    for t in range(tables):
        total += _table_size(t, values, OPINION_LEVELS)
    all_lengths = np.empty(total, dtype=np.int64)
    at = 0
    for t in range(tables):
        size = _table_size(t, values, OPINION_LEVELS)
        lt = _code_lengths(frequency[t], size, MAX_CODE_LENGTH)
        ct = _canonical_codes(lt, size)
        length[t, :size] = lt
        code[t, :size] = ct
        all_lengths[at:at + size] = lt
        at += size
    pre_vals, pre_bits = _length_symbols(all_lengths, MAX_CODE_LENGTH)

    # The bits counted, then written.
    nbits = 0
    for j in range(len(pre_bits)):
        nbits += pre_bits[j]
    for j in range(m):
        t = record[j] >> 24
        s = record[j] & 0xffffff
        nbits += length[t, s]
        if t == RUN_TABLE and s > 1:
            nbits += s - 1
    out = np.zeros((nbits + 7) // 8, dtype=np.uint8)
    size = 0
    bits = 0
    count = 0
    for j in range(len(pre_bits)):
        size, bits, count = _put(out, size, bits, count, pre_vals[j], pre_bits[j])
    # The records (the writing written out here: as a call to _put, as for
    # the tables' lengths above, Numba made it several times slower).
    ei = 0
    for j in range(m):
        t = record[j] >> 24
        s = record[j] & 0xffffff
        nbits = length[t, s]
        bits = (bits << nbits) | code[t, s]
        count += nbits
        if t == RUN_TABLE and s > 1:
            bits = (bits << (s - 1)) | extra[ei]
            count += s - 1
            ei += 1
        while count >= 8:
            count -= 8
            out[size] = (bits >> count) & 0xff
            size += 1
        bits &= (1 << count) - 1
    if count > 0:
        out[size] = (bits << (8 - count)) & 0xff
        size += 1
    return lo, values, out[:size]


def neighbour_code(delta, value, offset, xdim):
    """A channel's deltas neighbour Huffman coded, version 3
    (NeighbourMapper.neighbourCode; see the top of this module for the
    layout): value the channel's values as the decoder rebuilds them, the
    deltas' first at offset (for the second opinions, with the predictions
    value - delta). delta isn't changed."""
    lo, values, code = _code(np.ascontiguousarray(delta, dtype=np.int64), np.ascontiguousarray(value, dtype=np.int64), int(offset), int(xdim))
    from java_io import DataOutput
    out = DataOutput()
    out.write_int(int(lo))
    out.write_int(int(values))
    out.write_int(len(code))
    out.write(code.tobytes())
    return out.to_bytes()


# ---- The decoding ----------------------------------------------------------

class NeighbourChannel:
    """A channel's coding as neighbour_code wrote it (neighbour_read)."""
    __slots__ = ("lo", "values", "code", "version")


def neighbour_read(inp, version=VERSION):
    """Reads a channel's coding of a version (the file's: VERSION, which
    neighbour_code writes, VERSION_FLIPPED or VERSION_UNFLIPPED)
    (NeighbourMapper.neighbourRead)."""
    h = NeighbourChannel()
    h.version = version
    h.lo = inp.read_int()
    h.values = inp.read_int()
    length = inp.read_int()
    if h.values < 1 or length < 0:
        raise IOError("bad neighbour Huffman coding")
    h.code = inp.read_fully(length)
    return h


@njit
def _decode(buf, lo, values, xdim, ydim, flip):
    """neighbour_decode's work: (0 and the deltas, or a code saying what's
    wrong and None-like zeros). flip: the single deltas' signs are flipped
    (version 2)."""
    n = xdim * ydim
    delta = np.zeros(n, dtype=np.int64)
    st = np.zeros(4, dtype=np.int64)
    TABLES = _tables(1)
    STOP_TABLE = _stop_table(1)
    ESCAPE_TABLE = STOP_TABLE + 1
    width = max(values, PAIR_WIDTH * PAIR_WIDTH, RUN_SYMBOLS, LENGTH_SYMBOLS)
    longest = np.zeros(TABLES + 1, dtype=np.int64)
    first = np.zeros((TABLES + 1, MAX_CODE_LENGTH + 2), dtype=np.int64)
    count = np.zeros((TABLES + 1, MAX_CODE_LENGTH + 2), dtype=np.int64)
    start = np.zeros((TABLES + 1, MAX_CODE_LENGTH + 2), dtype=np.int64)
    sorted_ = np.zeros((TABLES + 1, width), dtype=np.int64)
    lookup = np.zeros((TABLES + 1, 1 << LOOKUP_BITS), dtype=np.int64)

    # The tables' lengths (with the code lengths' own code as table TABLES).
    total = 0
    for t in range(TABLES):
        total += _table_size(t, values, 1)
    ok, all_lengths = _read_lengths(buf, st, total, MAX_CODE_LENGTH, TABLES, longest, first, count, start, sorted_, lookup)
    if not ok:
        return 1, delta
    at = 0
    for t in range(TABLES):
        size = _table_size(t, values, 1)
        if not _make_table(all_lengths[at:at + size], size, t, longest, first, count, start, sorted_, lookup):
            return 1, delta
        at += size

    # The deltas. One symbol is read each time round (the reading written
    # out here, once: as a function, Numba made it ten times slower); what
    # it is depends on step: NORMAL, a delta's own symbol (its table chosen
    # by its activity: run, pair or single); STOP, the delta that stopped a
    # run; FIRST and SECOND, a pair's escaped deltas.
    NORMAL, STOP, FIRST, SECOND = 0, 1, 2, 3
    bits, nb, at, past = st[0], st[1], st[2], st[3]
    nbuf = len(buf)
    for y in range(ydim):
        row = y * xdim
        end = row + xdim
        k = 1 if y == 0 else row
        step = NORMAL
        mode = 0          # NORMAL's symbol: 0 run, 1 pair, 2 single
        second = False    # the pair's second delta escaped too
        while k < end:
            if step == NORMAL:
                a = _activity(delta, k, k - row, y, xdim)
                if a == 0:
                    t = RUN_TABLE
                    mode = 0
                else:
                    c = _context(a)
                    if c <= PAIR_CONTEXTS and k + 1 < end:
                        t = _pair_table(c, 0, 1)
                        mode = 1
                    else:
                        t = c
                        mode = 2
            elif step == STOP:
                t = STOP_TABLE
            else:
                t = ESCAPE_TABLE

            # The symbol, from table t.
            if nb < MAX_CODE_LENGTH:
                while nb <= 48:
                    if at < nbuf:
                        bits = (bits << 8) | np.int64(buf[at])
                        at += 1
                    else:
                        bits <<= 8
                        past += 1
                    nb += 8
            e = lookup[t, (bits >> (nb - LOOKUP_BITS)) & ((1 << LOOKUP_BITS) - 1)]
            if e != 0:
                nb -= e & 31
                s = e >> 5
            else:
                s = -1
                every = (bits >> (nb - MAX_CODE_LENGTH)) & ((1 << MAX_CODE_LENGTH) - 1)
                for l in range(LOOKUP_BITS + 1, longest[t] + 1):
                    i = (every >> (MAX_CODE_LENGTH - l)) - first[t, l]
                    if 0 <= i < count[t, l]:
                        nb -= l
                        s = sorted_[t, start[t, l] + i]
                        break
                if s < 0:
                    return 2, delta
            bits &= (1 << nb) - 1

            if step == NORMAL:
                if mode == 0:
                    # A run: its length's other bits, then (before the row's
                    # end) the delta that stopped it.
                    r = s
                    if s > 1:
                        n = s - 1
                        if nb < n:
                            while nb <= 48:
                                if at < nbuf:
                                    bits = (bits << 8) | np.int64(buf[at])
                                    at += 1
                                else:
                                    bits <<= 8
                                    past += 1
                                nb += 8
                        nb -= n
                        r = (1 << n) | ((bits >> nb) & ((1 << n) - 1))
                        bits &= (1 << nb) - 1
                    if r > end - k:
                        return 3, delta
                    k += r
                    step = STOP
                elif mode == 1:
                    u = s // PAIR_WIDTH
                    w = s - u * PAIR_WIDTH
                    second = w == PAIR_ESCAPE
                    if u != PAIR_ESCAPE:
                        delta[k] = u - PAIR_RANGE
                    if not second:
                        delta[k + 1] = w - PAIR_RANGE
                    if u == PAIR_ESCAPE:
                        step = FIRST
                    elif second:
                        step = SECOND
                    else:
                        k += 2
                else:
                    single = s + lo
                    if flip and (delta[k - 1] if k > row else 0) + (delta[k - xdim] if y > 0 else 0) < 0:
                        single = -single   # flipped back (NeighbourMapper.flipped)
                    delta[k] = single
                    k += 1
            elif step == STOP:
                delta[k] = s + lo
                k += 1
                step = NORMAL
            elif step == FIRST:
                delta[k] = s + lo
                if second:
                    step = SECOND
                else:
                    k += 2
                    step = NORMAL
            else:
                delta[k + 1] = s + lo
                k += 2
                step = NORMAL
    if 8 * past > nb:
        return 4, delta
    return 0, delta


_DECODE_ERRORS = {1: "bad code lengths", 2: "no such code", 3: "a run past the row's end", 4: "it ends too soon"}


def neighbour_decode(h, xdim, ydim):
    """A channel's deltas from its neighbour Huffman coding of version 1 or 2
    (NeighbourMapper.neighbourDecode; version 3 decodes inside the
    rebuilding: neighbour_bands): xdim * ydim deltas, delta 0 being 0.
    Raises IOError if the coding isn't what neighbour_code wrote."""
    if h.version >= 3:
        raise IOError("neighbour Huffman version %d decodes with the values (neighbour_bands)" % h.version)
    if h.values < 1:
        raise IOError("bad neighbour Huffman coding")
    status, delta = _decode(np.frombuffer(h.code, dtype=np.uint8), int(h.lo), int(h.values), int(xdim), int(ydim), h.version >= VERSION_FLIPPED)
    if status != 0:
        raise IOError("bad neighbour Huffman coding: " + _DECODE_ERRORS[status])
    return delta


# ---- Version 3: a pixel at a time, inside the rebuilding -------------------
#
# A decoder's state: the bit reader's (st, as _peek's), its own (ns: the
# zeros still to come of a run, the pixel where the run's stop comes or -1,
# the pixel where a pair's second delta comes or -1, that delta, and 1 once a
# code was no code of its table), and the tables' (NeighbourMapper.Table:
# longest, first, count, start, sorted_, lookup), with lo.

@njit
def _nb_tables(buf, values):
    """A version 3 decoder's tables, its lengths read: (0, st, the tables'
    arrays), or a code (1: bad code lengths) and the same arrays, empty."""
    tables = _tables(OPINION_LEVELS)
    st = np.zeros(4, dtype=np.int64)
    width = max(values, PAIR_WIDTH * PAIR_WIDTH, RUN_SYMBOLS, LENGTH_SYMBOLS)
    longest = np.zeros(tables + 1, dtype=np.int64)
    first = np.zeros((tables + 1, MAX_CODE_LENGTH + 2), dtype=np.int64)
    count = np.zeros((tables + 1, MAX_CODE_LENGTH + 2), dtype=np.int64)
    start = np.zeros((tables + 1, MAX_CODE_LENGTH + 2), dtype=np.int64)
    sorted_ = np.zeros((tables + 1, width), dtype=np.int64)
    lookup = np.zeros((tables + 1, 1 << LOOKUP_BITS), dtype=np.int64)
    total = 0
    for t in range(tables):
        total += _table_size(t, values, OPINION_LEVELS)
    ok, all_lengths = _read_lengths(buf, st, total, MAX_CODE_LENGTH, tables, longest, first, count, start, sorted_, lookup)
    if not ok:
        return 1, st, longest, first, count, start, sorted_, lookup
    at = 0
    for t in range(tables):
        size = _table_size(t, values, OPINION_LEVELS)
        if not _make_table(all_lengths[at:at + size], size, t, longest, first, count, start, sorted_, lookup):
            return 1, st, longest, first, count, start, sorted_, lookup
        at += size
    return 0, st, longest, first, count, start, sorted_, lookup


@njit(inline="always")   # called per symbol: a call passing arrays is slow in Numba
def _nb_symbol(buf, st, ns, t, longest, first, count, start, sorted_, lookup):
    """A symbol of table t (_read_symbol, inline); 0 and ns[4] set if the
    bits are no code of its."""
    while st[1] < MAX_CODE_LENGTH:
        if st[2] < buf.shape[0]:
            st[0] = (st[0] << 8) | np.int64(buf[st[2]])
            st[2] += 1
        else:
            st[0] <<= 8
            st[3] += 1
        st[1] += 8
    e = lookup[t, (st[0] >> (st[1] - LOOKUP_BITS)) & ((1 << LOOKUP_BITS) - 1)]
    if e != 0:
        st[1] -= e & 31
        st[0] &= (1 << st[1]) - 1
        return e >> 5
    every = (st[0] >> (st[1] - MAX_CODE_LENGTH)) & ((1 << MAX_CODE_LENGTH) - 1)
    for l in range(LOOKUP_BITS + 1, longest[t] + 1):
        i = (every >> (MAX_CODE_LENGTH - l)) - first[t, l]
        if 0 <= i < count[t, l]:
            st[1] -= l
            st[0] &= (1 << st[1]) - 1
            return sorted_[t, start[t, l] + i]
    ns[4] = 1
    return 0


@njit(inline="always")   # called per pixel: a call passing arrays is slow in Numba
def _nb_next(buf, st, ns, longest, first, count, start, sorted_, lookup, lo, xdim, k, d, v, offset, predicted, p):
    """Decodes delta k into d[k] (NeighbourMapper.NeighbourDecoder.next): v
    the values (filled below offset + k), p the rebuilding's prediction at k
    if predicted."""
    if k == 0:
        d[0] = 0
        return
    if k == ns[2]:
        d[k] = ns[3]
        ns[2] = -1
        return
    if ns[0] > 0:
        d[k] = 0
        ns[0] -= 1
        return
    stop_table = _stop_table(OPINION_LEVELS)
    if k == ns[1]:
        d[k] = _nb_symbol(buf, st, ns, stop_table, longest, first, count, start, sorted_, lookup) + lo
        ns[1] = -1
        return
    y = k // xdim
    x = k - y * xdim
    end = (y + 1) * xdim
    a = _activity(d, k, x, y, xdim)
    o = _opinion(v, offset + k, x, y, xdim, p) if predicted else 0
    if a == 0 and o == 0:
        size = _nb_symbol(buf, st, ns, RUN_TABLE, longest, first, count, start, sorted_, lookup)
        r = size
        if size > 1:
            n = size - 1
            while st[1] < n:
                if st[2] < buf.shape[0]:
                    st[0] = (st[0] << 8) | np.int64(buf[st[2]])
                    st[2] += 1
                else:
                    st[0] <<= 8
                    st[3] += 1
                st[1] += 8
            st[1] -= n
            r = (1 << n) | ((st[0] >> st[1]) & ((1 << n) - 1))
            st[0] &= (1 << st[1]) - 1
        if r > end - k:
            r = end - k
            ns[4] = 1
        if r == 0:
            d[k] = _nb_symbol(buf, st, ns, stop_table, longest, first, count, start, sorted_, lookup) + lo
        else:
            d[k] = 0
            ns[0] = r - 1
            ns[1] = k + r if k + r < end else -1
        return
    c = 1 if a == 0 else _context(a)
    l = abs(o)
    if c <= PAIR_CONTEXTS and k + 1 < end:
        s = _nb_symbol(buf, st, ns, _pair_table(c, l, OPINION_LEVELS), longest, first, count, start, sorted_, lookup)
        u = s // PAIR_WIDTH
        w = s - u * PAIR_WIDTH
        d0 = _nb_symbol(buf, st, ns, stop_table + 1, longest, first, count, start, sorted_, lookup) + lo if u == PAIR_ESCAPE else u - PAIR_RANGE
        d1 = _nb_symbol(buf, st, ns, stop_table + 1, longest, first, count, start, sorted_, lookup) + lo if w == PAIR_ESCAPE else w - PAIR_RANGE
        d[k] = -d0 if o < 0 else d0
        ns[3] = -d1 if o < 0 else d1
        ns[2] = k + 1
        return
    single = _nb_symbol(buf, st, ns, _single_table(c, l), longest, first, count, start, sorted_, lookup) + lo
    if o < 0 or (o == 0 and (d[k - 1] if x > 0 else 0) + (d[k - xdim] if y > 0 else 0) < 0):
        single = -single   # flipped back
    d[k] = single


def _nb_decoder(h):
    """A version 3 decoder's arrays: (buf, st, ns, the tables' arrays);
    raises IOError if the tables aren't what neighbour_code writes."""
    if h.values < 1:
        raise IOError("bad neighbour Huffman coding")
    buf = np.frombuffer(h.code, dtype=np.uint8)
    status, st, longest, first, count, start, sorted_, lookup = _nb_tables(buf, int(h.values))
    if status != 0:
        raise IOError("bad neighbour Huffman coding: " + _DECODE_ERRORS[status])
    ns = np.array([0, -1, -1, 0, 0], dtype=np.int64)
    return buf, st, ns, longest, first, count, start, sorted_, lookup


def _nb_finish(dec):
    """After the last delta: raises IOError if a code was bad or the coding ran out."""
    st, ns = dec[1], dec[2]
    if ns[4] != 0:
        raise IOError("bad neighbour Huffman coding: " + _DECODE_ERRORS[2])
    if 8 * st[3] > st[1]:
        raise IOError("bad neighbour Huffman coding: " + _DECODE_ERRORS[4])


@njit_no_refs   # (no arrays made: see numba_support)
def _nb_average(buf, st, ns, longest, first, count, start, sorted_, lookup, lo, xdim, ydim, init, d, dst, offset):
    """A band's channel decoded and rebuilt from its average deltas, into dst
    from offset on (NeighbourMapper.neighbourAverage)."""
    k = 0
    for y in range(ydim):
        for x in range(xdim):
            if k == 0:
                p = init
            elif y == 0:
                p = dst[offset + k - 1]
            elif x == 0:
                p = dst[offset + k - xdim]
            else:
                p = jdiv(dst[offset + k - 1] + dst[offset + k - xdim], 2)
            _nb_next(buf, st, ns, longest, first, count, start, sorted_, lookup, lo, xdim, k, d, dst, offset, True, p)
            dst[offset + k] = init if k == 0 else p + d[k]
            k += 1


def neighbour_average(h, xdim, ydim, init, dst, offset):
    """A band's channel (version 3) decoded and rebuilt from its average
    deltas into dst from offset on (NeighbourMapper.neighbourAverage)."""
    dec = _nb_decoder(h)
    _nb_average(*dec, int(h.lo), int(xdim), int(ydim), int(init), np.zeros(xdim * ydim, dtype=np.int64), dst, int(offset))
    _nb_finish(dec)


@njit_no_refs   # (no arrays made: see numba_support)
def _nb_known(buf, st, ns, longest, first, count, start, sorted_, lookup, lo, xdim, d, value, offset, delta):
    for k in range(d.shape[0]):
        _nb_next(buf, st, ns, longest, first, count, start, sorted_, lookup, lo, xdim, k, d, value, offset, True,
                 value[offset + k] - delta[k])


def neighbour_decode_known(h, value, offset, delta, xdim):
    """A band's channel's deltas decoded (version 3) with its values and the
    predictions (value - delta) known: the writers' check
    (NeighbourMapper.neighbourDecodeKnown). Delta 0 is 0."""
    dec = _nb_decoder(h)
    d = np.zeros(len(delta), dtype=np.int64)
    _nb_known(*dec, int(h.lo), int(xdim), d, np.ascontiguousarray(value, dtype=np.int64), int(offset), np.ascontiguousarray(delta, dtype=np.int64))
    _nb_finish(dec)
    return d


@njit_no_refs   # (no arrays made: see numba_support)
def _nb_row_fit_row(ref, has_ref, xdim, init, every, coef, valid, z, delta, dst, y, r, col, points, win, a, b, dg, za, zd,
                    buf, st, ns, longest, first, count, start, sorted_, lookup, lo):
    """Row y of a band's channel decoded (version 3) and rebuilt from its
    RowFit deltas, the row's fits solved first (as
    delta_mapper._row_fit_row_inline, with this decoder): the fit's
    prediction at the interior pixels, given to the decoder for the second
    opinion."""
    nf = dm.RF_SPATIAL + 3 if has_ref else dm.RF_SPATIAL
    if y >= 1:
        if y >= 2:
            dm._rf_advance(dst, ref, has_ref, xdim, y, r, col, 1, xdim - 1, za, zd)
        dm._rf_solve(col, xdim, r, nf, every, 0, points, coef, valid, win, a, b, dg)
    for x in range(xdim):
        k = y * xdim + x
        predicted = False
        if y == 0:
            p = init if x == 0 else dst[k - 1]
        elif x == 0:
            p = dst[k - xdim]
        elif x == xdim - 1:
            p = dst[k - 1]
        else:
            p = dm._rf_predict(dst, ref, has_ref, xdim, nf, y, x, every, coef, valid, z)
            predicted = True
        _nb_next(buf, st, ns, longest, first, count, start, sorted_, lookup, lo, xdim, k, delta, dst, 0, predicted, p)
        dst[k] = init if k == 0 else delta[k] + p


def neighbour_row_fit_channels(coded, xdim, ydim, init, radius, every):
    """A band's channels (version 3) decoded and rebuilt from their RowFit
    deltas: a thread per channel, a row at a time, once the channel before
    has rebuilt the row (the fit uses it), each row's fits solved in the
    same compiled call that decodes it (_nb_row_fit_row; as
    delta_mapper.get_values_from_row_fit_channels). Returns the values, per
    channel."""
    nch, n = len(coded), xdim * ydim
    value = np.zeros((nch, n), dtype=np.int64)
    done = [0] * nch
    cond = threading.Condition()
    failed = []

    def run(i):
        try:
            dec = _nb_decoder(coded[i])
            has_ref = i > 0
            nf = dm.RF_SPATIAL + 3 if has_ref else dm.RF_SPATIAL
            col, coef, valid, z, _, _, _, _, _, _, _, points = dm._rf_state(xdim, has_ref, every)
            ref = value[i - 1] if i > 0 else np.zeros(1, dtype=np.int64)
            w = dm._rf_work(nf)
            delta = np.zeros(n, dtype=np.int64)
            for y in range(ydim):
                if i > 0:
                    need = min(ydim, y + 1)
                    with cond:
                        while done[i - 1] < need and not failed:
                            cond.wait()
                        if failed:
                            return
                _nb_row_fit_row(ref, has_ref, xdim, int(init[i]), every, coef, valid, z, delta, value[i], y,
                                radius[i], col, points, w[2], w[3], w[4], w[5], w[6], w[7], *dec, int(coded[i].lo))
                with cond:
                    done[i] = y + 1
                    cond.notify_all()
            _nb_finish(dec)
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
    return list(value)


def neighbour_bands(coded, init, w, h, average, radius, every, raw):
    """Every band's channels decoded and rebuilt into the channels raw[i]
    (GenericReader.neighbourBands): with the average deltas, each band's
    channel on its own, all at the same time (version 3: neighbour_average;
    1 and 2: neighbour_decode, then the average rebuilt); with the fit's,
    each band's channels rebuilt, viewer_support.ls_bands_at_once bands at a
    time (version 3: decoded inside the rebuilding,
    neighbour_row_fit_channels; 1 and 2: every band's channels' deltas first,
    all at the same time, then delta_mapper.get_values_from_row_fit_deltas)."""
    import viewer_support as vs
    from simple_writer import band_start
    n = len(coded)
    v3 = coded[0][0].version >= 3
    delta = [[None] * 3 for _ in range(n)]

    def decode_one(j):
        b, i = j // 3, j % 3
        y0, y1 = band_start(b, n, h), band_start(b + 1, n, h)
        if average and v3:
            neighbour_average(coded[b][i], w, y1 - y0, init[b][i], raw[i], y0 * w)   # (straight into the channel)
        elif average:
            dm.average_into(neighbour_decode(coded[b][i], w, y1 - y0), w, y1 - y0, init[b][i], raw[i][y0 * w:y1 * w])
        else:
            delta[b][i] = neighbour_decode(coded[b][i], w, y1 - y0)
    if average or not v3:
        vs.parallel(3 * n, decode_one)
    if average:
        return

    def rebuild(b):
        y0, y1 = band_start(b, n, h), band_start(b + 1, n, h)
        if v3:
            v = neighbour_row_fit_channels(coded[b], w, y1 - y0, init[b], radius, every)
        else:
            v = dm.get_values_from_row_fit_deltas(delta[b], w, y1 - y0, init[b], radius, every)
        for i in range(3):
            raw[i][y0 * w:y1 * w] = v[i]
    vs.for_bands(n, vs.ls_bands_at_once(n), rebuild)
