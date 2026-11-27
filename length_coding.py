"""
length_coding.py -- LengthCoding.java for Python: the table machinery the
table codings share. Neighbour Huffman (neighbour_mapper), Context tables
(context_tables) and Huffman (huffman_writer's code lengths) all write a model
as code lengths, or, for Context tables, levels, which are code lengths in
steps of a bit or half a bit: -log2 of a probability, rounded. They're written
compactly the same way (_length_symbols: a length, a repeat of the last, or a
run of zeros, each Huffman coded, with no Deflate), with the same bit streams
(most significant bit first: _put, _pack_bits; _peek, _skip, _read_bits), the
same canonical Huffman code (_canonical_codes, _make_table, _read_symbol:
codes in order of length, then of symbol, decoded LOOKUP_BITS at a time) and
the same length-limited code lengths (_code_lengths). A canonical Huffman
table is the special case of a Context tables (rANS) table whose frequencies
are all powers of 2: a code of length L takes 2^(precision - L) of its slots.
As native/LengthCoding.c in C. Compiled by Numba.
"""
import numpy as np

from numba_support import njit

MAX_CODE_LENGTH = 20          # the longest code a table reads (the neighbour Huffman tables' longest)
LENGTH_REPEAT = MAX_CODE_LENGTH + 1
LENGTH_ZEROS = LENGTH_REPEAT + 1
LENGTH_MANY_ZEROS = LENGTH_ZEROS + 1
LENGTH_SYMBOLS = LENGTH_MANY_ZEROS + 1
MAX_LENGTH_CODE_LENGTH = 15   # the longest code in the code lengths' own code (its lengths written in 4 bits each)
LOOKUP_BITS = 10              # how many bits a table's decoder looks up at once


# ---- Code lengths ----------------------------------------------------------

@njit
def _code_lengths(frequency, size, limit):
    """Huffman code lengths for frequency[:size], none longer than limit, as
    LengthCoding.codeLengths makes them: the two least frequent merged
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
    """The canonical codes for these lengths (LengthCoding.Table's: in
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
def _length_symbols(all_lengths, max_length):
    """LengthCoding.writeLengthSymbols, as (value, bits) pairs to write:
    the code lengths' own code's lengths (4 bits each), then the lengths (0
    to max_length) as that code's symbols (a length, the last length again 3
    to 6 times, or a run of zeros), each followed by its extra bits. With
    max_length MAX_CODE_LENGTH, the neighbour Huffman tables'; with
    code_mapper.REGULAR_MAX_CODE_LENGTH, the Huffman coding's code lengths
    (pack_lengths)."""
    repeat = max_length + 1
    zeros = repeat + 1
    many_zeros = zeros + 1
    symbols = many_zeros + 1
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
                symbol[m] = many_zeros
                extra[m] = r - 11
                m += 1
                run -= r
            if run >= 3:
                symbol[m] = zeros
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
                symbol[m] = repeat
                extra[m] = r - 3
                m += 1
                run -= r
            while run > 0:
                symbol[m] = v
                m += 1
                run -= 1
    frequency = np.zeros(symbols, dtype=np.int64)
    for j in range(m):
        frequency[symbol[j]] += 1
    length = _code_lengths(frequency, symbols, MAX_LENGTH_CODE_LENGTH)
    code = _canonical_codes(length, symbols)
    vals = np.empty(symbols + 2 * m, dtype=np.int64)
    bits = np.empty(symbols + 2 * m, dtype=np.int64)
    p = 0
    for s in range(symbols):
        vals[p] = length[s]
        bits[p] = 4
        p += 1
    for j in range(m):
        s = symbol[j]
        vals[p] = code[s]
        bits[p] = length[s]
        p += 1
        if s == repeat:
            vals[p] = extra[j]
            bits[p] = 2
            p += 1
        elif s == zeros:
            vals[p] = extra[j]
            bits[p] = 3
            p += 1
        elif s == many_zeros:
            vals[p] = extra[j]
            bits[p] = 7
            p += 1
    return vals[:p], bits[:p]


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
    """Table t from its code lengths (LengthCoding.Table). Returns False if
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
def _read_lengths(buf, st, total, max_length, L, longest, first, count, start, sorted_, lookup):
    """LengthCoding.readLengthSymbols: total code lengths (0 to
    max_length) written by _length_symbols, the code lengths' own code made
    as table L of the tables' arrays. Returns (False if they aren't what
    _length_symbols writes, the lengths)."""
    repeat = max_length + 1
    zeros = repeat + 1
    symbols = zeros + 2
    all_lengths = np.zeros(total, dtype=np.int64)
    cl = np.empty(symbols, dtype=np.int64)
    for s in range(symbols):
        cl[s] = _read_bits(buf, st, 4)
    if not _make_table(cl, symbols, L, longest, first, count, start, sorted_, lookup):
        return False, all_lengths
    i = 0
    last = -1
    while i < total:
        s = _read_symbol(buf, st, L, longest, first, count, start, sorted_, lookup)
        if s < 0:
            return False, all_lengths
        if s <= max_length:
            all_lengths[i] = s
            i += 1
            last = s
            continue
        if s == repeat:
            if last < 0:
                return False, all_lengths
            run = 3 + _read_bits(buf, st, 2)
            v = last
        elif s == zeros:
            run = 3 + _read_bits(buf, st, 3)
            v = 0
            last = 0
        else:
            run = 11 + _read_bits(buf, st, 7)
            v = 0
            last = 0
        if i + run > total or 8 * st[3] > st[1]:
            return False, all_lengths
        for j in range(run):
            all_lengths[i + j] = v
        i += run
    return True, all_lengths


# ---- A code length table on its own (the Huffman coding's) ---------------------

@njit
def _pack_bits(vals, nbits):
    """The (value, bits) pairs written most significant first, the last byte
    filled out with 0 bits."""
    total = 0
    for j in range(len(nbits)):
        total += nbits[j]
    out = np.zeros((total + 7) // 8, dtype=np.uint8)
    size = 0
    bits = 0
    count = 0
    for j in range(len(nbits)):
        bits = (bits << nbits[j]) | (vals[j] & ((1 << nbits[j]) - 1))
        count += nbits[j]
        while count >= 8:
            count -= 8
            out[size] = (bits >> count) & 0xff
            size += 1
        bits &= (1 << count) - 1
    if count > 0:
        out[size] = (bits << (8 - count)) & 0xff
    return out


@njit
def _unpack_lengths(buf, size, max_length):
    """_read_lengths for one table on its own: (0, the lengths), or a code
    saying what's wrong."""
    st = np.zeros(4, dtype=np.int64)
    symbols = max_length + 4
    longest = np.zeros(1, dtype=np.int64)
    first = np.zeros((1, MAX_CODE_LENGTH + 2), dtype=np.int64)
    count = np.zeros((1, MAX_CODE_LENGTH + 2), dtype=np.int64)
    start = np.zeros((1, MAX_CODE_LENGTH + 2), dtype=np.int64)
    sorted_ = np.zeros((1, symbols), dtype=np.int64)
    lookup = np.zeros((1, 1 << LOOKUP_BITS), dtype=np.int64)
    ok, lengths = _read_lengths(buf, st, size, max_length, 0, longest, first, count, start, sorted_, lookup)
    if not ok:
        return 1, lengths
    if 8 * st[3] > st[1]:
        return 4, lengths
    return 0, lengths


def pack_lengths(lengths, max_length):
    """A code length table written compactly, on its own
    (LengthCoding.packLengths: no Deflate, so it doesn't depend on a
    zlib): the Huffman coding's code lengths (huffman_writer.huffman_code)."""
    vals, nbits = _length_symbols(np.ascontiguousarray(lengths, dtype=np.int64), int(max_length))
    return _pack_bits(vals, nbits).tobytes()


def unpack_lengths(packed, size, max_length):
    """Reads a code length table pack_lengths wrote
    (LengthCoding.unpackLengths): size lengths, as uint8. Raises IOError
    if they aren't what pack_lengths writes."""
    status, lengths = _unpack_lengths(np.frombuffer(packed, dtype=np.uint8), int(size), int(max_length))
    if status != 0:
        raise IOError("bad code lengths" + (": they end too soon" if status == 4 else ""))
    return lengths.astype(np.uint8)
