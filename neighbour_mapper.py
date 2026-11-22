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
of its own). Each channel has its own tables, made from its own deltas; their
code lengths are coded compactly (_length_symbols), with no Deflate, so the
coding doesn't depend on a zlib (Python's own is zlib-ng on Windows).

Layout (neighbour_code): int lo, the smallest delta; int the number of delta
values (hi - lo + 1); int the number of bytes that follow; those bytes, a bit
stream (most significant bit first): the tables' code lengths, then the codes.
Delta 0 isn't coded (the band's first value is written instead).

As in NativeEncoder's C (native/NeighbourMapper.c), the coding walks the
deltas once, counting and recording the symbols, and then codes the records;
the decoding is one pass. Both are compiled by Numba (numba_support.njit,
releasing the GIL).
"""

import numpy as np

from numba_support import njit

CONTEXTS = 16
ACTIVITY_STEPS = (2, 3, 4, 6, 8, 12, 16, 23, 32, 46, 64, 91, 128, 181)
PAIR_CONTEXTS = 6
PAIR_RANGE = 3
PAIR_ESCAPE = 2 * PAIR_RANGE + 1
PAIR_WIDTH = PAIR_ESCAPE + 1
RUN_SYMBOLS = 18
RUN_TABLE = 0
PAIR_TABLE = CONTEXTS - 1
STOP_TABLE = PAIR_TABLE + PAIR_CONTEXTS + 1
ESCAPE_TABLE = STOP_TABLE + 1
TABLES = ESCAPE_TABLE + 1
MAX_CODE_LENGTH = 20
LENGTH_REPEAT = MAX_CODE_LENGTH + 1
LENGTH_ZEROS = LENGTH_REPEAT + 1
LENGTH_MANY_ZEROS = LENGTH_ZEROS + 1
LENGTH_SYMBOLS = LENGTH_MANY_ZEROS + 1
MAX_LENGTH_CODE_LENGTH = 15
LOOKUP_BITS = 10


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
def _table_size(t, values):
    if t == RUN_TABLE:
        return RUN_SYMBOLS
    if PAIR_TABLE < t <= PAIR_TABLE + PAIR_CONTEXTS:
        return PAIR_WIDTH * PAIR_WIDTH
    return values


@njit
def _pair_value(delta):
    return delta + PAIR_RANGE if -PAIR_RANGE <= delta <= PAIR_RANGE else PAIR_ESCAPE


# ---- Code lengths ----------------------------------------------------------

@njit
def _code_lengths(frequency, size, limit):
    """Huffman code lengths for frequency[:size], none longer than limit, as
    NeighbourMapper.codeLengths makes them: the two least frequent merged
    again and again (ties to the symbols first, then the merged nodes in the
    order made; symbols in order of count, then value), then, if any code is
    too long, the lengths brought within limit as JPEG does it (Annex K.3)
    and given out again, shortest first, in the order of the first lengths
    (then of value)."""
    length = np.zeros(size, dtype=np.int64)
    used = 0
    for s in range(size):
        if frequency[s] > 0:
            used += 1
    if used == 0:
        return length
    if used == 1:
        for s in range(size):
            if frequency[s] > 0:
                length[s] = 1
        return length
    key = np.empty(used, dtype=np.int64)
    j = 0
    for s in range(size):
        if frequency[s] > 0:
            key[j] = (frequency[s] << 32) | s
            j += 1
    key = np.sort(key)
    nodes = 2 * used - 1
    weight = np.zeros(nodes, dtype=np.int64)
    parent = np.zeros(nodes, dtype=np.int64)
    for j in range(used):
        weight[j] = key[j] >> 32
    leaf = 0
    merged = used
    for nxt in range(used, nodes):
        a = 0
        b = 0
        for m in range(2):
            if leaf < used and (merged >= nxt or weight[leaf] <= weight[merged]):
                pick = leaf
                leaf += 1
            else:
                pick = merged
                merged += 1
            if m == 0:
                a = pick
            else:
                b = pick
        weight[nxt] = weight[a] + weight[b]
        parent[a] = nxt
        parent[b] = nxt
    depth = np.zeros(nodes, dtype=np.int64)
    longest = 0
    for v in range(nodes - 2, -1, -1):
        depth[v] = depth[parent[v]] + 1
        if v < used and depth[v] > longest:
            longest = depth[v]
    for j in range(used):
        length[key[j] & 0xffffffff] = depth[j]
    if longest <= limit:
        return length
    count = np.zeros(longest + 1, dtype=np.int64)
    for s in range(size):
        if length[s] > 0:
            count[length[s]] += 1
    for i in range(longest, limit, -1):
        while count[i] > 0:
            j = i - 2
            while count[j] == 0:
                j -= 1
            count[i] -= 2
            count[i - 1] += 1
            count[j + 1] += 2
            count[j] -= 1
    order = np.empty(used, dtype=np.int64)
    j = 0
    for s in range(size):
        if length[s] > 0:
            order[j] = (length[s] << 32) | s
            j += 1
    order = np.sort(order)
    l = 1
    for j in range(used):
        while count[l] == 0:
            l += 1
        length[order[j] & 0xffffffff] = l
        count[l] -= 1
    return length


@njit
def _canonical_codes(length, size):
    """The canonical codes for these lengths (NeighbourMapper.Table's: in
    order of length, then of symbol)."""
    longest = 0
    for s in range(size):
        if length[s] > longest:
            longest = length[s]
    count = np.zeros(longest + 2, dtype=np.int64)
    for s in range(size):
        if length[s] > 0:
            count[length[s]] += 1
    fill = np.zeros(longest + 2, dtype=np.int64)
    nxt = 0
    for l in range(1, longest + 1):
        nxt = (nxt + count[l - 1]) << 1
        fill[l] = nxt
    code = np.zeros(size, dtype=np.int64)
    for s in range(size):
        if length[s] > 0:
            code[s] = fill[length[s]]
            fill[length[s]] += 1
    return code


@njit
def _length_symbols(all_lengths):
    """NeighbourMapper.writeLengths, as (value, bits) pairs to write: the
    code lengths' own code's lengths (4 bits each), then the tables' lengths
    as that code's symbols (a length, the last length again 3 to 6 times, or
    a run of zeros), each followed by its extra bits."""
    total = len(all_lengths)
    symbol = np.empty(total + 1, dtype=np.int64)
    extra = np.empty(total + 1, dtype=np.int64)
    m = 0
    i = 0
    while i < total:
        v = all_lengths[i]
        run = 1
        while i + run < total and all_lengths[i + run] == v:
            run += 1
        i += run
        if v == 0:
            while run >= 11:
                r = min(run, 138)
                symbol[m] = LENGTH_MANY_ZEROS
                extra[m] = r - 11
                m += 1
                run -= r
            if run >= 3:
                symbol[m] = LENGTH_ZEROS
                extra[m] = run - 3
                m += 1
                run = 0
            while run > 0:
                symbol[m] = 0
                m += 1
                run -= 1
        else:
            symbol[m] = v
            m += 1
            run -= 1
            while run >= 3:
                r = min(run, 6)
                symbol[m] = LENGTH_REPEAT
                extra[m] = r - 3
                m += 1
                run -= r
            while run > 0:
                symbol[m] = v
                m += 1
                run -= 1
    frequency = np.zeros(LENGTH_SYMBOLS, dtype=np.int64)
    for j in range(m):
        frequency[symbol[j]] += 1
    length = _code_lengths(frequency, LENGTH_SYMBOLS, MAX_LENGTH_CODE_LENGTH)
    code = _canonical_codes(length, LENGTH_SYMBOLS)
    vals = np.empty(LENGTH_SYMBOLS + 2 * m, dtype=np.int64)
    bits = np.empty(LENGTH_SYMBOLS + 2 * m, dtype=np.int64)
    p = 0
    for s in range(LENGTH_SYMBOLS):
        vals[p] = length[s]
        bits[p] = 4
        p += 1
    for j in range(m):
        s = symbol[j]
        vals[p] = code[s]
        bits[p] = length[s]
        p += 1
        if s == LENGTH_REPEAT:
            vals[p] = extra[j]
            bits[p] = 2
            p += 1
        elif s == LENGTH_ZEROS:
            vals[p] = extra[j]
            bits[p] = 3
            p += 1
        elif s == LENGTH_MANY_ZEROS:
            vals[p] = extra[j]
            bits[p] = 7
            p += 1
    return vals[:p], bits[:p]


# ---- The coding ------------------------------------------------------------

@njit
def _walk(d, n, xdim, lo, frequency, record, extra):
    """The walk through the deltas (NeighbourMapper.walk), once: each symbol
    counted into frequency[table] and recorded, table * 2^24 + symbol, in
    order; a run length's bits below its top bit into extra. Returns (the
    number of records, the number of extras)."""
    m = 0
    e = 0
    ydim = n // xdim
    for y in range(ydim):
        row = y * xdim
        end = row + xdim
        k = 1 if y == 0 else row
        while k < end:
            x = k - row
            a = _activity(d, k, x, y, xdim)
            if a == 0:
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
                    frequency[STOP_TABLE, s] += 1
                    record[m] = (STOP_TABLE << 24) | s
                    m += 1
                    k += 1
                continue
            c = _context(a)
            if c <= PAIR_CONTEXTS and k + 1 < end:
                u = _pair_value(d[k])
                w = _pair_value(d[k + 1])
                t = PAIR_TABLE + c
                s = u * PAIR_WIDTH + w
                frequency[t, s] += 1
                record[m] = (t << 24) | s
                m += 1
                if u == PAIR_ESCAPE:
                    s = d[k] - lo
                    frequency[ESCAPE_TABLE, s] += 1
                    record[m] = (ESCAPE_TABLE << 24) | s
                    m += 1
                if w == PAIR_ESCAPE:
                    s = d[k + 1] - lo
                    frequency[ESCAPE_TABLE, s] += 1
                    record[m] = (ESCAPE_TABLE << 24) | s
                    m += 1
                k += 2
                continue
            s = d[k] - lo
            frequency[c, s] += 1
            record[m] = (c << 24) | s
            m += 1
            k += 1
    return m, e


@njit
def _put(out, size, bits, count, value, nbits):
    """Writes value's low nbits into out (most significant first): (the
    bytes written, the bits not yet written out, how many)."""
    bits = (bits << nbits) | (value & ((1 << nbits) - 1))
    count += nbits
    while count >= 8:
        count -= 8
        out[size] = (bits >> count) & 0xff
        size += 1
    bits &= (1 << count) - 1
    return size, bits, count


@njit
def _code(delta, xdim):
    """neighbour_code's work: (lo, the number of delta values, the bit
    stream)."""
    n = len(delta)
    d = delta.copy()
    d[0] = 0
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
    values = hi - lo + 1
    width = max(values, PAIR_WIDTH * PAIR_WIDTH, RUN_SYMBOLS)
    frequency = np.zeros((TABLES, width), dtype=np.int64)
    record = np.empty(2 * n + 1, dtype=np.int64)
    extra = np.empty(n + 1, dtype=np.int64)
    m, e = _walk(d, n, xdim, lo, frequency, record, extra)

    # The tables, their lengths all together (for _length_symbols).
    length = np.zeros((TABLES, width), dtype=np.int64)
    code = np.zeros((TABLES, width), dtype=np.int64)
    total = 0
    for t in range(TABLES):
        total += _table_size(t, values)
    all_lengths = np.empty(total, dtype=np.int64)
    at = 0
    for t in range(TABLES):
        size = _table_size(t, values)
        lt = _code_lengths(frequency[t], size, MAX_CODE_LENGTH)
        ct = _canonical_codes(lt, size)
        length[t, :size] = lt
        code[t, :size] = ct
        all_lengths[at:at + size] = lt
        at += size
    pre_vals, pre_bits = _length_symbols(all_lengths)

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


def neighbour_code(delta, xdim):
    """A channel's deltas neighbour Huffman coded (NeighbourMapper.neighbourCode;
    see the top of this module for the layout). delta isn't changed."""
    lo, values, code = _code(np.ascontiguousarray(delta, dtype=np.int64), int(xdim))
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
    __slots__ = ("lo", "values", "code")


def neighbour_read(inp):
    """Reads a channel's coding neighbour_code wrote (NeighbourMapper.neighbourRead)."""
    h = NeighbourChannel()
    h.lo = inp.read_int()
    h.values = inp.read_int()
    length = inp.read_int()
    if h.values < 1 or length < 0:
        raise IOError("bad neighbour Huffman coding")
    h.code = inp.read_fully(length)
    return h


@njit
def _peek(buf, st, n):
    """The next n bits (st: bits read in, how many, the next byte, the 0
    bytes made up past the end)."""
    while st[1] < n:
        if st[2] < len(buf):
            x = np.int64(buf[st[2]])
            st[2] += 1
        else:
            x = 0
            st[3] += 1
        st[0] = (st[0] << 8) | x
        st[1] += 8
    return (st[0] >> (st[1] - n)) & ((1 << n) - 1)


@njit
def _skip(st, n):
    st[1] -= n
    st[0] &= (1 << st[1]) - 1


@njit
def _read_bits(buf, st, n):
    if n == 0:
        return 0
    v = _peek(buf, st, n)
    _skip(st, n)
    return v


@njit
def _make_table(length, size, t, longest, first, count, start, sorted_, lookup):
    """Table t from its code lengths (NeighbourMapper.Table). Returns False if
    the lengths aren't a code's."""
    for l in range(MAX_CODE_LENGTH + 2):
        count[t, l] = 0
    mx = 0
    for s in range(size):
        if length[s] > MAX_CODE_LENGTH or length[s] < 0:
            return False
        if length[s] > mx:
            mx = length[s]
        if length[s] > 0:
            count[t, length[s]] += 1
    longest[t] = mx
    nxt = 0
    at = 0
    for l in range(1, mx + 1):
        nxt = (nxt + count[t, l - 1]) << 1
        first[t, l] = nxt
        start[t, l] = at
        at += count[t, l]
        if nxt + count[t, l] > (1 << l):
            return False
    fill = first[t].copy()
    place = start[t].copy()
    for i in range(1 << LOOKUP_BITS):
        lookup[t, i] = 0
    for s in range(size):
        l = length[s]
        if l == 0:
            continue
        c = fill[l]
        fill[l] += 1
        sorted_[t, place[l]] = s
        place[l] += 1
        if l <= LOOKUP_BITS:
            lo = c << (LOOKUP_BITS - l)
            for i in range(lo, lo + (1 << (LOOKUP_BITS - l))):
                lookup[t, i] = (s << 5) | l
    return True


@njit
def _read_symbol(buf, st, t, longest, first, count, start, sorted_, lookup):
    """Reads a symbol of table t; -1 if the bits are no code of its."""
    e = lookup[t, _peek(buf, st, LOOKUP_BITS)]
    if e != 0:
        _skip(st, e & 31)
        return e >> 5
    every = _peek(buf, st, MAX_CODE_LENGTH)
    for l in range(LOOKUP_BITS + 1, longest[t] + 1):
        c = every >> (MAX_CODE_LENGTH - l)
        i = c - first[t, l]
        if 0 <= i < count[t, l]:
            _skip(st, l)
            return sorted_[t, start[t, l] + i]
    return -1


@njit
def _decode(buf, lo, values, xdim, ydim):
    """neighbour_decode's work: (0 and the deltas, or a code saying what's
    wrong and None-like zeros)."""
    n = xdim * ydim
    delta = np.zeros(n, dtype=np.int64)
    st = np.zeros(4, dtype=np.int64)
    width = max(values, PAIR_WIDTH * PAIR_WIDTH, RUN_SYMBOLS, LENGTH_SYMBOLS)
    longest = np.zeros(TABLES + 1, dtype=np.int64)
    first = np.zeros((TABLES + 1, MAX_CODE_LENGTH + 2), dtype=np.int64)
    count = np.zeros((TABLES + 1, MAX_CODE_LENGTH + 2), dtype=np.int64)
    start = np.zeros((TABLES + 1, MAX_CODE_LENGTH + 2), dtype=np.int64)
    sorted_ = np.zeros((TABLES + 1, width), dtype=np.int64)
    lookup = np.zeros((TABLES + 1, 1 << LOOKUP_BITS), dtype=np.int64)

    # The code lengths' own code (table TABLES), then the tables' lengths.
    cl = np.empty(LENGTH_SYMBOLS, dtype=np.int64)
    for s in range(LENGTH_SYMBOLS):
        cl[s] = _read_bits(buf, st, 4)
    L = TABLES
    if not _make_table(cl, LENGTH_SYMBOLS, L, longest, first, count, start, sorted_, lookup):
        return 1, delta
    total = 0
    for t in range(TABLES):
        total += _table_size(t, values)
    all_lengths = np.zeros(total, dtype=np.int64)
    i = 0
    last = -1
    while i < total:
        s = _read_symbol(buf, st, L, longest, first, count, start, sorted_, lookup)
        if s < 0:
            return 1, delta
        if s <= MAX_CODE_LENGTH:
            all_lengths[i] = s
            i += 1
            last = s
            continue
        if s == LENGTH_REPEAT:
            if last < 0:
                return 1, delta
            run = 3 + _read_bits(buf, st, 2)
            v = last
        elif s == LENGTH_ZEROS:
            run = 3 + _read_bits(buf, st, 3)
            v = 0
            last = 0
        else:
            run = 11 + _read_bits(buf, st, 7)
            v = 0
            last = 0
        if i + run > total or 8 * st[3] > st[1]:
            return 1, delta
        for j in range(run):
            all_lengths[i + j] = v
        i += run
    at = 0
    for t in range(TABLES):
        size = _table_size(t, values)
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
                        t = PAIR_TABLE + c
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
                    delta[k] = s + lo
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
    """A channel's deltas from its neighbour Huffman coding
    (NeighbourMapper.neighbourDecode): xdim * ydim deltas, delta 0 being 0.
    Raises IOError if the coding isn't what neighbour_code writes."""
    if h.values < 1:
        raise IOError("bad neighbour Huffman coding")
    status, delta = _decode(np.frombuffer(h.code, dtype=np.uint8), int(h.lo), int(h.values), int(xdim), int(ydim))
    if status != 0:
        raise IOError("bad neighbour Huffman coding: " + _DECODE_ERRORS[status])
    return delta
