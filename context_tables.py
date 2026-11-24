"""
context_tables.py -- ContextTables.java for Python: the Context tables coding
of a channel's deltas (GenericWriter's and NativeWriter's "Context tables"),
the same bytes from the same deltas, and the same deltas back.

The context aware coding's contexts (delta_mapper._mix_contexts: the
activity, the earlier channels' deltas and a second opinion), with fixed
tables in place of its adaptive models: the deltas are counted once, by
context, the counts written as tables, and the deltas coded with them by rANS
(range asymmetric numeral systems), which decodes a symbol with a table
lookup and a multiply, and no division. Smaller than the Huffman codings, a
little bigger than context aware (whose models adapt, and mix three
opinions), and several times faster to decode.

The symbols are the deltas' ranks (as the context aware coding's: the most
frequent value is 0). One model's contexts are used: model 0, 1 or, after the
first channel, 2 (MIX_OPINION_LEVELS times CONTEXTS of them), or the same
with the earlier channels' cross buckets merged (fewer tables, for a small
image). Each context's table is its symbols' probabilities to a step of a
factor of 2 (steps 1) or of the square root of 2 (steps 2), as levels 1 to
LEVEL_RANGE times steps (0: a symbol that isn't there), from which coder and
decoder make the same frequencies, out of 2^precision (_normalize). The
model, the merging and the steps are chosen by the coded size.

Layout (code): int the smallest delta, the rank table (delta_mapper.
write_table), byte the model, byte 1 if merged (else 0), byte the steps, byte
the precision, int the tables' length, the tables, int the coded length, and
the coded bytes: rANS's final state (4 bytes, the low first), then the bytes
it gave out, in the order the decoder reads them. The tables: a bit stream
(most significant bit first) of each context's number of levels (its last
symbol + 1; 0 for a context not used) plus 1, in Elias's gamma code, then the
used contexts' levels, one table after another, as _chain writes them
(neighbour_mapper._length_symbols, up to twice the top level).
"""

import threading

import numpy as np

from numba_support import njit, njit_no_refs, jdiv
import delta_mapper as dm
import neighbour_mapper as nm

VERSION = 1   # the coding's version, the Context tables format's ('T')
CONTEXTS = dm.CONTEXTS * dm.MIX_OPINION_LEVELS
CROSS = len(dm.CROSS_LIMIT)
LEVELS = dm.MIX_OPINION_LEVELS
MERGED_CONTEXTS = CONTEXTS // CROSS
MIN_PRECISION, MAX_PRECISION = 12, 16
LEVEL_RANGE = 12
# Each level's weight (0 to LEVEL_RANGE steps), for steps 1 and 2: 16 times
# 2^((level - 1) / steps), rounded (0 for level 0).
WEIGHT = np.zeros((3, 2 * LEVEL_RANGE + 1), dtype=np.int64)
WEIGHT[1, :LEVEL_RANGE + 1] = [0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
WEIGHT[2] = [0, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024, 1448, 2048, 2896, 4096, 5793, 8192, 11585, 16384, 23170, 32768, 46341]
RANS_LOW = 1 << 23   # rANS's state is kept from RANS_LOW to 256 times it, renormalized a byte at a time


@njit(inline="always")
def _merged(c):
    """A context, merged: its cross bucket left out."""
    return c // (CROSS * LEVELS) * LEVELS + c % LEVELS


def precision(symbols):
    """The precision, in bits: MIN_PRECISION, or more, so that the
    frequencies are out of at least twice as many as there are symbols."""
    bits = MIN_PRECISION
    while bits < MAX_PRECISION and (1 << bits) < 2 * symbols:
        bits += 1
    return bits


@njit
def _normalize(level, n, steps, prec, weight, f):
    """A context's frequencies (into f[:n]) from its levels (ContextTables.normalize)."""
    w = weight[steps]
    s_sum = 0
    for s in range(n):
        s_sum += w[level[s]]
    total = 1 << prec
    given = 0
    big = 0
    for s in range(n):
        f[s] = 0
        if level[s] == 0:
            continue
        f[s] = max(1, w[level[s]] * total // s_sum)
        given += f[s]
        if f[s] > f[big]:
            big = s
    left = total - given
    if f[big] + left >= 1:
        f[big] += left
    else:
        while left < 0:
            for s in range(n):
                if left >= 0:
                    break
                if f[s] > 1:
                    f[s] -= 1
                    left += 1


@njit
def _levels(count, last, steps, weight, level):
    """A context's levels (into level[:last + 1]) from its counts (ContextTables.levels)."""
    w = weight[steps]
    top = LEVEL_RANGE * steps
    most = 0
    for s in range(last + 1):
        most = max(most, count[s])
    for s in range(last + 1):
        level[s] = 0
        if count[s] == 0:
            continue
        e = top
        while e > 1 and 2 * count[s] * w[top] < (w[e - 1] + w[e]) * most:
            e -= 1
        level[s] = e


@njit
def _log2_fixed(f):
    """log2(f) times 2^16, rounded down (ContextTables.log2Fixed)."""
    whole = 0
    while (f >> (whole + 1)) > 0:
        whole += 1
    y = (f >> (whole - 30)) if whole >= 30 else (f << (30 - whole))
    r = whole << 16
    for i in range(15, -1, -1):
        y = (y * y) >> 30
        if y >= (1 << 31):
            y >>= 1
            r |= 1 << i
    return r


@njit
def _tables(count, steps, prec, weight):
    """The tables for these counts (ContextTables.Tables): (each context's
    frequencies, each context's length (0: not used), the size in bits times
    2^16 without the tables' bytes, and the (value, bits) pairs of the
    tables as written)."""
    contexts, symbols = count.shape
    top = LEVEL_RANGE * steps
    freq = np.zeros((contexts, symbols), dtype=np.int64)
    length = np.zeros(contexts, dtype=np.int64)
    level = np.zeros((contexts, symbols), dtype=np.int64)
    gamma_v = np.zeros(contexts, dtype=np.int64)
    gamma_n = np.zeros(contexts, dtype=np.int64)
    total = 0
    for c in range(contexts):
        last = -1
        for s in range(symbols):
            if count[c, s] > 0:
                last = s
        v = last + 2
        nb = 0
        while (v >> nb) > 0:
            nb += 1
        gamma_v[c] = v
        gamma_n[c] = nb
        if last < 0:
            continue
        length[c] = last + 1
        _levels(count[c], last, steps, weight, level[c])
        _normalize(level[c], last + 1, steps, prec, weight, freq[c])
        total += last + 1
    # The levels, chained (ContextTables.chain).
    out = np.zeros(total, dtype=np.int64)
    at = 0
    for c in range(contexts):
        before = top
        for s in range(length[c]):
            e = level[c, s]
            if e != 0:
                diff = e - before
                out[at] = 1 + (2 * diff if diff >= 0 else -2 * diff - 1)
                before = e
            at += 1
    vals, nbits = nm._length_symbols(out, 2 * top)
    pv = np.zeros(2 * contexts + len(vals), dtype=np.int64)
    pb = np.zeros(2 * contexts + len(vals), dtype=np.int64)
    p = 0
    for c in range(contexts):
        if gamma_n[c] > 1:
            pv[p] = 0
            pb[p] = gamma_n[c] - 1
            p += 1
        pv[p] = gamma_v[c]
        pb[p] = gamma_n[c]
        p += 1
    for j in range(len(vals)):
        pv[p] = vals[j]
        pb[p] = nbits[j]
        p += 1
    whole = prec << 16
    size = 0
    for c in range(contexts):
        for s in range(length[c]):
            if count[c, s] > 0:
                size += count[c, s] * (whole - _log2_fixed(freq[c, s]))
    return freq, length, size, pv[:p], pb[:p]


@njit
def _count(sym, d, v, xdim, cb, fwd, c9b, m, act_table, count):
    """Each model's counts of the symbols by context (count[j, 0]) and merged
    (count[j, 1][:MERGED_CONTEXTS])."""
    ctx = np.zeros(3, dtype=np.int64)
    for k in range(sym.shape[0]):
        dm._mix_contexts(k, d, v, xdim, cb, fwd, c9b, m, act_table, ctx, True, v[k] - d[k])
        for j in range(m):
            count[j, 0, ctx[j], sym[k]] += 1
            count[j, 1, _merged(ctx[j]), sym[k]] += 1


@njit
def _rans_encode(sym, d, v, xdim, cb, fwd, c9b, m, act_table, model, merge, freq, prec):
    """The chosen model's contexts, then the ranks rANS coded from the last
    (ContextTables.code): the coded bytes, in the order the decoder reads them."""
    n = sym.shape[0]
    ctx = np.zeros(3, dtype=np.int64)
    cc = np.zeros(n, dtype=np.int64)
    for k in range(n):
        dm._mix_contexts(k, d, v, xdim, cb, fwd, c9b, m, act_table, ctx, True, v[k] - d[k])
        cc[k] = _merged(ctx[model]) if merge else ctx[model]
    contexts, symbols = freq.shape
    cum = np.zeros((contexts, symbols), dtype=np.int64)
    for c in range(contexts):
        for s in range(1, symbols):
            cum[c, s] = cum[c, s - 1] + freq[c, s - 1]
    buf = np.zeros(2 * n + 16, dtype=np.uint8)
    p = buf.shape[0]
    x = RANS_LOW
    for k in range(n - 1, -1, -1):
        c = cc[k]
        s = sym[k]
        f = freq[c, s]
        x_max = ((RANS_LOW >> prec) << 8) * f
        while x >= x_max:
            p -= 1
            buf[p] = x & 0xff
            x >>= 8
        x = ((x // f) << prec) + (x % f) + cum[c, s]
    for b in range(3, -1, -1):
        p -= 1
        buf[p] = (x >> (8 * b)) & 0xff
    return buf[p:]


def code(delta, value, previous, xdim):
    """A channel's deltas Context tables coded (ContextTables.code; see the
    top of this module): value the band's values as the decoder rebuilds
    them (for the second opinions, the predictions value - delta); previous
    the deltas of the channels coded before it."""
    from java_io import DataOutput
    d = np.ascontiguousarray(delta, dtype=np.int64)
    v = np.ascontiguousarray(value, dtype=np.int64)
    n = len(d)
    sym, lo, rank = dm._rank_symbols(d)
    symbols = len(rank)
    prec = precision(symbols)
    cb, fwd, c9b, m = dm._mix_setup(previous, xdim, n)
    count = np.zeros((m, 2, CONTEXTS, symbols), dtype=np.int64)
    _count(sym, d, v, int(xdim), cb, fwd, c9b, m, dm.MIX_ACTIVITY_TABLE, count)
    best = None
    for j in range(m):
        for g in range(2):
            cnt = count[j, g, :MERGED_CONTEXTS] if g else count[j, g]
            for st in (1, 2):
                freq, length, size, pv, pb = _tables(cnt, st, prec, WEIGHT)
                tb = nm._pack_bits(pv, pb)
                size += (8 * len(tb)) << 16
                if best is None or size < best[0]:
                    best = (size, j, g, st, freq, tb)
    _, model, merge, steps, freq, tb = best
    coded = _rans_encode(sym, d, v, int(xdim), cb, fwd, c9b, m, dm.MIX_ACTIVITY_TABLE, model, merge, freq, prec)
    out = DataOutput()
    out.write_int(lo)
    dm.write_table(out, rank)
    out.write_byte(model)
    out.write_byte(merge)
    out.write_byte(steps)
    out.write_byte(prec)
    out.write_int(len(tb))
    out.write(tb.tobytes())
    out.write_int(len(coded))
    out.write(coded.tobytes())
    return out.to_bytes()


# ---- The decoding ----------------------------------------------------------

class TablesChannel:
    """A channel's coding as code wrote it (read)."""
    __slots__ = ("value", "model", "merged", "steps", "precision", "tables", "code")


def read(inp):
    """Reads a channel's coding code wrote (ContextTables.read)."""
    h = TablesChannel()
    lo = inp.read_int()
    rank = dm.read_table(inp)
    h.value = np.zeros(len(rank), dtype=np.int64)
    h.value[rank] = np.arange(len(rank), dtype=np.int64) + lo
    h.model = inp.read_unsigned_byte()
    merge = inp.read_unsigned_byte()
    h.steps = inp.read_unsigned_byte()
    h.precision = inp.read_unsigned_byte()
    if h.model > 2 or merge > 1 or not 1 <= h.steps <= 2 or not MIN_PRECISION <= h.precision <= MAX_PRECISION:
        raise IOError("bad Context tables coding: model %d, merged %d, steps %d, precision %d" % (h.model, merge, h.steps, h.precision))
    h.merged = merge == 1
    length = inp.read_int()
    if length < 0:
        raise IOError("bad Context tables coding")
    h.tables = np.frombuffer(inp.read_fully(length), dtype=np.uint8)
    length = inp.read_int()
    if length < 4:
        raise IOError("bad Context tables coding")
    h.code = np.frombuffer(inp.read_fully(length), dtype=np.uint8)
    return h


@njit
def _read_tables(buf, contexts, symbols, steps, prec, weight):
    """The tables read (ContextTables.TablesDecoder's constructor): (0, each
    context's row of freq, cum and lookup or -1, freq, cum, lookup), or an
    error code (1: a table's length, 2: the levels, 3: a level, 4: they end
    too soon, 5: a table ending with a symbol that isn't there)."""
    st = np.zeros(4, dtype=np.int64)
    top = LEVEL_RANGE * steps
    length = np.zeros(contexts, dtype=np.int64)
    slot = np.full(contexts, -1, dtype=np.int64)
    total = 0
    used = 0
    e1 = np.zeros(1, dtype=np.int64)
    e2 = np.zeros((1, 1), dtype=np.int64)
    e3 = np.zeros((1, 1), dtype=np.uint16)
    for c in range(contexts):
        nb = 1
        while _read_zero(buf, st):
            nb += 1
            if nb > 24:
                return 1, e1, e2, e2, e3
        v = 1 if nb == 1 else (1 << (nb - 1)) | nm._read_bits(buf, st, nb - 1)
        length[c] = v - 1
        if length[c] > symbols:
            return 1, e1, e2, e2, e3
        if length[c] > 0:
            slot[c] = used
            used += 1
        total += length[c]
    longest = np.zeros(1, dtype=np.int64)
    first = np.zeros((1, nm.MAX_CODE_LENGTH + 2), dtype=np.int64)
    count = np.zeros((1, nm.MAX_CODE_LENGTH + 2), dtype=np.int64)
    start = np.zeros((1, nm.MAX_CODE_LENGTH + 2), dtype=np.int64)
    sorted_ = np.zeros((1, 2 * top + 4), dtype=np.int64)
    lookup = np.zeros((1, 1 << nm.LOOKUP_BITS), dtype=np.int64)
    ok, out = nm._read_lengths(buf, st, total, 2 * top, 0, longest, first, count, start, sorted_, lookup)
    if not ok:
        return 2, e1, e2, e2, e3
    if 8 * st[3] > st[1]:
        return 4, e1, e2, e2, e3
    freq = np.zeros((max(used, 1), symbols), dtype=np.int64)
    cum = np.zeros((max(used, 1), symbols), dtype=np.int64)
    look = np.zeros((max(used, 1), 1 << prec), dtype=np.uint16)
    level = np.zeros(symbols, dtype=np.int64)
    at = 0
    for c in range(contexts):
        if length[c] == 0:
            continue
        before = top
        for s in range(length[c]):
            z = out[at + s]
            level[s] = 0
            if z == 0:
                continue
            z -= 1
            e = before + (z // 2 if (z & 1) == 0 else -((z + 1) // 2))
            if e < 1 or e > top:
                return 3, e1, e2, e2, e3
            level[s] = e
            before = e
        if level[length[c] - 1] == 0:
            return 5, e1, e2, e2, e3
        r = slot[c]
        _normalize(level, length[c], steps, prec, weight, freq[r])
        t = 0
        for s in range(length[c]):
            cum[r, s] = t
            for i in range(freq[r, s]):
                look[r, t + i] = s
            t += freq[r, s]
        at += length[c]
    return 0, slot, freq, cum, look


@njit(inline="always")
def _read_zero(buf, st):
    """Reads a bit; whether it's 0 (for the gamma code's prefix)."""
    return nm._read_bits(buf, st, 1) == 0


_TABLE_ERRORS = {1: "a table's length", 2: "the levels", 3: "a level", 4: "the tables end too soon", 5: "a table ends with a symbol that isn't there"}


def decoder(h, m):
    """The state of a decoder at the start of a channel's coding (with m
    models' contexts: 3, or 2 for the first channel): (slot, freq, cum,
    lookup, code, state: rANS's x, the next byte, an error)."""
    if h.model >= m:
        raise IOError("bad Context tables coding: model %d of %d" % (h.model, m))
    contexts = MERGED_CONTEXTS if h.merged else CONTEXTS
    status, slot, freq, cum, look = _read_tables(h.tables, contexts, len(h.value), h.steps, h.precision, WEIGHT)
    if status != 0:
        raise IOError("bad Context tables coding: " + _TABLE_ERRORS[status])
    c = h.code
    st = np.array([int(c[0]) | (int(c[1]) << 8) | (int(c[2]) << 16) | (int(c[3]) << 24), 4, 0], dtype=np.int64)
    return slot, freq, cum, look, c, st


def finish(dec, h):
    """After the last delta: raises IOError if the coding was bad, or didn't end where it should."""
    st = dec[5]
    if st[2] != 0:
        raise IOError("bad Context tables coding: a context with no table")
    if st[1] != len(dec[4]) or st[0] != RANS_LOW:
        raise IOError("bad Context tables coding: it doesn't end where it should")


@njit(inline="always")   # called per pixel: a call passing arrays is slow in Numba
def _context(k, d, v, xdim, cb, fwd, c9b, m, act_table, j, pred):
    """Model j's context at pixel k alone (DeltaMapper.MixedModels.context):
    delta_mapper._mix_contexts' ctx[j], with the second opinions, without
    the other models' terms (models 1 and 2 need none of gap's)."""
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
    top = act_table.shape[0] - 1
    mid = dm.MIX_OPINION_LEVELS // 2
    inside = k >= xdim and x >= 1 and x < xdim - 1
    if j == 0:
        g = 0
        if x >= 2 and k >= 2 * xdim and x < xdim - 1:
            W = v[k - 1]
            N = v[k - xdim]
            NW = v[k - xdim - 1]
            NE = v[k - xdim + 1]
            g = (abs(W - v[k - 2]) + abs(N - NW) + abs(N - NE)
                 + abs(W - NW) + abs(N - v[k - 2 * xdim]) + abs(NE - v[k - 2 * xdim + 1]))
        o = mid
        if inside:
            a = v[k - 1]
            b = v[k - xdim]
            cc = v[k - xdim - 1]
            dd = v[k - xdim + 1]
            up2 = k >= 2 * xdim
            aa = v[k - 2] if x >= 2 else a
            bb = v[k - 2 * xdim] if up2 else b
            nne = v[k - 2 * xdim + 1] if up2 else dd
            o = dm._gradient_level(dm._gap(a, aa, b, bb, cc, dd, nne, 10, 4, 1) - pred) + mid
        return (act_table[min((act + g) >> 1, top)] * CROSS + cb[k]) * LEVELS + o
    if j == 1:
        a1 = act + (fwd[k] >> 1) if m == 3 else act
        o = mid
        if inside:
            o = dm._gradient_level(dm._block_predictor(11, v[k - 1], v[k - xdim], v[k - xdim - 1], v[k - xdim + 1]) - pred) + mid
        return (act_table[min(a1, top)] * CROSS + cb[k]) * LEVELS + o
    o = mid
    if inside:
        o = dm._gradient_level(v[k - 1] - pred) + mid
    return (act_table[min(act, top)] * CROSS + c9b[k]) * LEVELS + o


@njit(inline="always")   # called per pixel: a call passing arrays is slow in Numba
def _ct_next(slot, freq, cum, look, code, st, value, model, merge, prec, k, d, v, xdim, cb, fwd, c9b, m, act_table, ctx, pred):
    """Decodes delta k into d[k] (ContextTables.TablesDecoder.decode): its
    context, its symbol from the context's lookup, rANS's state moved on."""
    c = _context(k, d, v, xdim, cb, fwd, c9b, m, act_table, model, pred)
    if merge:
        c = _merged(c)
    r = slot[c]
    if r < 0:
        st[2] = 1
        d[k] = 0
        return
    x = st[0]
    low = x & ((1 << prec) - 1)
    s = look[r, low]
    x = freq[r, s] * (x >> prec) + low - cum[r, s]
    while x < RANS_LOW:
        b = 0
        if st[1] < code.shape[0]:
            b = code[st[1]]
            st[1] += 1
        x = (x << 8) | b
    st[0] = x
    d[k] = value[s]


@njit_no_refs   # (no arrays made: see numba_support)
def _ct_known(slot, freq, cum, look, code, st, value, model, merge, prec, d, v, delta, xdim, cb, fwd, c9b, m, act_table, ctx):
    """The writers' check (decode_known): the deltas decoded with the values known."""
    for k in range(d.shape[0]):
        _ct_next(slot, freq, cum, look, code, st, value, model, merge, prec, k, d, v, xdim, cb, fwd, c9b, m, act_table, ctx, v[k] - delta[k])


def decode_known(h, value, delta, previous, xdim):
    """A band's channel's deltas decoded with its values and predictions
    known (ContextTables.decodeKnown: the writers' check)."""
    v = np.ascontiguousarray(value, dtype=np.int64)
    dl = np.ascontiguousarray(delta, dtype=np.int64)
    cb, fwd, c9b, m = dm._mix_setup(previous, xdim, len(dl))
    dec = decoder(h, m)
    d = np.zeros(len(dl), dtype=np.int64)
    _ct_known(*dec, h.value, h.model, h.merged, h.precision, d, v, dl, int(xdim), cb, fwd, c9b, m, dm.MIX_ACTIVITY_TABLE, np.zeros(3, dtype=np.int64))
    finish(dec, h)
    return d


@njit_no_refs   # (no arrays made: see numba_support)
def _ct_row_fit_row(ref, has_ref, xdim, ydim, init, every, coef, valid, z, delta, dst, y, r, col, points, win, a, b, dg, za, zd,
                    previous, cross, cb, fwd, c9b, earlier, m, act_table, ctx,
                    slot, freq, cum, look, code, st, value, model, merge, prec):
    """Row y of a band's channel decoded and rebuilt from its RowFit deltas,
    the row's fits solved first (as delta_mapper._row_fit_row_inline, with
    this decoder); the earlier channels' terms filled first as far as row
    y + 1 (they must be decoded to row y + 1)."""
    nf = dm.RF_SPATIAL + 3 if has_ref else dm.RF_SPATIAL
    if y >= 1:
        if y >= 2:
            dm._rf_advance(dst, ref, has_ref, xdim, y, r, col, 1, xdim - 1, za, zd)
        dm._rf_solve(col, xdim, r, nf, every, 0, points, coef, valid, win, a, b, dg)
    if m == 3:
        dm._mix_earlier_rows(previous, xdim, ydim, dm.MIX_CROSS_TABLE, cross, cb, fwd, c9b, earlier, min(ydim, y + 1))
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
            p = dm._rf_predict(dst, ref, has_ref, xdim, nf, y, x, every, coef, valid, z)
            pred = p
        _ct_next(slot, freq, cum, look, code, st, value, model, merge, prec, k, delta, dst, xdim, cb, fwd, c9b, m, act_table, ctx, pred)
        dst[k] = init if k == 0 else delta[k] + p


@njit_no_refs   # (no arrays made: see numba_support)
def _ct_average_row(xdim, ydim, init, delta, dst, y, previous, cross, cb, fwd, c9b, earlier, m, act_table, ctx,
                    slot, freq, cum, look, code, st, value, model, merge, prec):
    """Row y of a band's channel decoded and rebuilt from its average deltas
    (as delta_mapper._average_row, with this decoder)."""
    if m == 3:
        dm._mix_earlier_rows(previous, xdim, ydim, dm.MIX_CROSS_TABLE, cross, cb, fwd, c9b, earlier, min(ydim, y + 1))
    for x in range(xdim):
        k = y * xdim + x
        if k == 0:
            p = init
        elif y == 0:
            p = dst[k - 1]
        elif x == 0:
            p = dst[k - xdim]
        else:
            p = jdiv(dst[k - 1] + dst[k - xdim], 2)
        _ct_next(slot, freq, cum, look, code, st, value, model, merge, prec, k, delta, dst, xdim, cb, fwd, c9b, m, act_table, ctx, p)
        dst[k] = init if k == 0 else p + delta[k]


def channels(coded, xdim, ydim, init, average, radius, every):
    """A band's channels decoded and rebuilt (GenericReader.tablesBands' for
    a band): a thread per channel, a row at a time, once the channel before
    has decoded two rows further (this row's contexts use its rows y and
    y + 1), each row a single compiled call (_ct_average_row or
    _ct_row_fit_row, its fits solved in the same call). Returns the values,
    per channel."""
    nch, n = len(coded), xdim * ydim
    delta = np.zeros((nch, n), dtype=np.int64)     # rows of one array: a channel's previous is delta[:i], no copy
    value = np.zeros((nch, n), dtype=np.int64)
    done = [0] * nch
    cond = threading.Condition()
    failed = []

    def run(i):
        try:
            h = coded[i]
            m = 3 if i > 0 else 2
            dec = decoder(h, m)
            args = (*dec, h.value, h.model, h.merged, h.precision)
            terms = dm._mix_terms(n)   # cross, cb, fwd, c9b (zero for the first channel)
            earlier = np.zeros(2, dtype=np.int64)
            ctx = np.zeros(3, dtype=np.int64)
            if not average:
                has_ref = i > 0
                nf = dm.RF_SPATIAL + 3 if has_ref else dm.RF_SPATIAL
                col, coef, valid, z, _, _, _, _, _, _, _, points = dm._rf_state(xdim, has_ref, every)
                ref = value[i - 1] if i > 0 else np.zeros(1, dtype=np.int64)
                w = dm._rf_work(nf)
            for y in range(ydim):
                if i > 0:
                    need = min(ydim, y + 2)
                    with cond:
                        while done[i - 1] < need and not failed:
                            cond.wait()
                        if failed:
                            return
                if average:
                    _ct_average_row(xdim, ydim, int(init[i]), delta[i], value[i], y, delta[:i], terms[0], terms[1], terms[2], terms[3],
                                    earlier, m, dm.MIX_ACTIVITY_TABLE, ctx, *args)
                else:
                    _ct_row_fit_row(ref, has_ref, xdim, ydim, int(init[i]), every, coef, valid, z, delta[i], value[i], y,
                                    radius[i], col, points, w[2], w[3], w[4], w[5], w[6], w[7],
                                    delta[:i], terms[0], terms[1], terms[2], terms[3], earlier, m, dm.MIX_ACTIVITY_TABLE, ctx, *args)
                with cond:
                    done[i] = y + 1
                    cond.notify_all()
            finish(dec, h)
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


def tables_bands(coded, init, w, h, average, radius, every, raw):
    """Every band's channels decoded and rebuilt into the channels raw[i]
    (GenericReader.tablesBands): each band's channels decoded inside the
    rebuilding (channels); with the average deltas all the bands at the
    same time, with the fit's viewer_support.ls_bands_at_once at a time."""
    import viewer_support as vs
    from simple_writer import band_start
    n = len(coded)

    def rebuild(b):
        y0, y1 = band_start(b, n, h), band_start(b + 1, n, h)
        v = channels(coded[b], w, y1 - y0, init[b], average, radius, every)
        for i in range(3):
            raw[i][y0 * w:y1 * w] = v[i]
    vs.for_bands(n, n if average else vs.ls_bands_at_once(n), rebuild)
