"""
probabilistic_mapper.py -- ProbabilisticMapper.java: steers an exact
arithmetic-coding interval onto a target fraction by choosing a per-step
symbol ordering (steer_encode / steer_decode), the combinadic and
c-sequence coders that serialize those orderings (compress / decompress),
the simplest-fraction-in-interval search, and the order-table utilities
moved here from ArithmeticMapper.

Representations:
  - FractionMapper.BigFraction -> fractions.Fraction (both exact and always
    in lowest terms). BigInteger -> Python int.
  - A Java BitSet of possible subset sums -> a Python int used as a bit set
    (bit s set <=> sum s is achievable).
  - byte[] -> bytearray of UNSIGNED 0..255 values, the same convention as
    string_mapper.py and segment_mapper.py. Every place the Java reads a
    byte it immediately maps it to 0..255 (j < 0 ? j + 256 : j, or & 0xFF),
    so this gives identical results. Inputs may be bytes, bytearray, or a
    list of ints in either the signed or unsigned range.
  - ArrayList<byte[]> -> list of bytearray.
  - steer_encode's deadline is in the units of time.monotonic_ns() (Java's
    System.nanoTime()): pass time.monotonic_ns() + budget.

Java semantics kept on purpose:
  - binomial / combinadic_encode / combinadic_decode use 64-bit Java long
    arithmetic, INCLUDING its silent overflow (C(n, k) past 2^63 wraps), so
    compress() writes the same bits as the Java for any alphabet size.
  - java.util.Random is reproduced exactly (JavaRandom), so
    get_random_table(frequency, seed) gives the Java's table for the seed.
  - Sorts are stable, like Java's List.sort / Collections.sort.

Overloads get distinct names:
  getRandomTable(freq) / getRandomTable(freq, long seed)
    -> get_random_table(freq, seed=None)
  getRandomOrderTable(freq, long | byte | short seed)
    -> get_random_order_table(freq, seed) /
       get_random_order_table_byte_seed(freq, seed) /
       get_random_order_table_short_seed(freq, seed)

Speed: steer_encode is a pure-Python search over exact fractions, so it is
much slower than the Java (and the Java's multi-threaded callers, such as
SegmentSteerDemo, rely on threads the GIL won't run in parallel -- use
processes instead). Within a time budget it will more often return None.
"""

import math
import time
from fractions import Fraction


# ---------------------------------------------------------------------------
# Java numeric helpers
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1


def _jlong(x):
    """Wrap to a 64-bit signed Java long."""
    x &= _MASK64
    return x - (1 << 64) if x >= (1 << 63) else x


def _jdiv(a, b):
    """Java integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _unsigned(values):
    return bytearray(v & 0xFF for v in values)


class JavaRandom:
    """java.util.Random: the same 48-bit LCG, so a given seed gives the
    same sequence as in Java."""

    _MULT = 0x5DEECE66D
    _MASK = (1 << 48) - 1

    def __init__(self, seed=None):
        if seed is None:
            seed = time.monotonic_ns() ^ id(self)
        self._seed = (seed ^ self._MULT) & self._MASK

    def _next(self, bits):
        self._seed = (self._seed * self._MULT + 0xB) & self._MASK
        r = self._seed >> (48 - bits)
        return r - (1 << 32) if r >= (1 << 31) else r   # (int) cast

    def next_int(self, bound):
        if bound <= 0:
            raise ValueError("bound must be positive")
        r = self._next(31)
        m = bound - 1
        if (bound & m) == 0:
            return (bound * r) >> 31
        u = r
        while True:
            r = u % bound
            if ((u - r + m + (1 << 31)) & 0xFFFFFFFF) - (1 << 31) >= 0:
                return r
            u = self._next(31)


# ---------------------------------------------------------------------------
# Subset sums
# ---------------------------------------------------------------------------

def possible_subset_sum(frequency):
    """Bit set (as an int) of every sum some subset of `frequency` reaches,
    including 0. Bit s is set iff s is achievable."""
    total = sum(frequency)
    mask = (1 << (total + 1)) - 1
    possible = 1
    for k in frequency:
        if k == 0:
            continue
        possible |= (possible << k) & mask
    return possible


def _iter_bits(bits):
    """Indices of the set bits, ascending (BitSet.nextSetBit order)."""
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low


def _reachable_rows(counts, sJ):
    """rows[i] = bit set of sums <= sJ reachable from counts[0..i-1]."""
    mask = (1 << (sJ + 1)) - 1
    rows = [1]
    for c in counts:
        r = rows[-1]
        rows.append(r | ((r << c) & mask))
    return rows


def _choose_subset(counts, sJ, rows):
    """Backtrack through the DP rows, preferring to take item i (as the Java
    does). Returns the used flags."""
    used = [False] * len(counts)
    s = sJ
    for i in range(len(counts) - 1, -1, -1):
        c = counts[i]
        if s - c >= 0 and (rows[i] >> (s - c)) & 1:
            used[i] = True
            s -= c
    return used


def ordering_for_choice(real_symbol, other_syms, f, sJ):
    """One ordering consistent with a chosen s_j: an exact subset of the other
    symbols' counts summing to sJ is placed before real_symbol, the rest after.
    (orderingForChoicePublic and orderingForChoice in the Java.)"""
    counts = [f[s] for s in other_syms]
    rows = _reachable_rows(counts, sJ)
    if not (rows[-1] >> sJ) & 1:
        raise RuntimeError("subset-sum reconstruction failed: sJ=%d not reachable from counts=%s"
                           " (candidate generation bug)" % (sJ, counts))
    used = _choose_subset(counts, sJ, rows)
    result = [s for s, u in zip(other_syms, used) if u]
    result.append(real_symbol)
    result.extend(s for s, u in zip(other_syms, used) if not u)
    return result


ordering_for_choice_public = ordering_for_choice


def reconstruct_before_count(counts, sJ):
    """How many items ordering_for_choice would put before the real symbol
    (0 if sJ isn't reachable)."""
    rows = _reachable_rows(counts, sJ)
    if not (rows[-1] >> sJ) & 1:
        return 0
    return sum(_choose_subset(counts, sJ, rows))


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------

class SteerResult:
    """orderings (list of lists), off and rng (Fractions), backtracks."""

    __slots__ = ("orderings", "off", "rng", "backtracks")

    def __init__(self, orderings, off, rng, backtracks):
        self.orderings = orderings
        self.off = off
        self.rng = rng
        self.backtracks = backtracks


def steer_decode(orderings, freq, target, n):
    """Replay decode: the stored orderings say which slice the target falls
    into at every step."""
    f = list(freq)
    m = sum(f)
    residual = Fraction(target)
    decoded = [0] * n
    for i in range(n):
        cum = 0
        chosen = -1
        sJ = -1
        target_pos = residual * m
        for sym in orderings[i]:
            f_sym = f[sym]
            if cum <= target_pos < cum + f_sym:
                chosen = sym
                sJ = cum
                break
            cum += f_sym
        if chosen < 0:
            raise RuntimeError("decode failed at step %d" % i)
        fJ = f[chosen]
        lo = Fraction(sJ, m)
        hi = Fraction(sJ + fJ, m)
        residual = (residual - lo) / (hi - lo)
        f[chosen] -= 1
        m -= 1
        decoded[i] = chosen
    return decoded


def _log2(x):
    return math.log(x) / math.log(2)


def _log2_factorial(n):
    b = 0.0
    for k in range(2, n + 1):
        b += _log2(k)
    return b


def serialized_ordering_bits(orderings):
    """Sum over steps of log2(k!) for a step with k symbols."""
    total = 0.0
    for perm in orderings:
        bits = 0.0
        for v in range(2, len(perm) + 1):
            bits += math.log(v) / math.log(2)
        total += bits
    return total


def log2_binomial_coeff(n, k):
    if k < 0 or k > n:
        return 0.0
    return _log2_factorial(n) - _log2_factorial(k) - _log2_factorial(n - k)


class _Candidate:
    __slots__ = ("symbol", "sum", "other_symbol")

    def __init__(self, symbol, sums, other_symbol):
        self.symbol = symbol
        self.sum = sums
        self.other_symbol = other_symbol


def _get_candidate(real_symbol, f, m, residual):
    other_syms = [s for s in range(len(f)) if f[s] > 0 and s != real_symbol]
    other_counts = [f[s] for s in other_syms]
    k_minus_1 = len(other_syms)
    fJ = f[real_symbol]

    cand = []
    for s in _iter_bits(possible_subset_sum(other_counts)):
        if Fraction(s, m) <= residual < Fraction(s + fJ, m):
            cand.append(s)

    # Cheapest to encode first: log2 C(k-1, before-count).
    cand.sort(key=lambda s: log2_binomial_coeff(k_minus_1, reconstruct_before_count(other_counts, s)))
    return _Candidate(real_symbol, cand, other_syms)


class _Frame:
    __slots__ = ("f", "m", "off", "rng", "residual", "cand", "idx")

    def __init__(self, f, m, off, rng, residual, cand):
        self.f = f
        self.m = m
        self.off = off
        self.rng = rng
        self.residual = residual
        self.cand = cand
        self.idx = 0


def steer_encode(src, freq, target, max_backtracks, deadline_nanos):
    """Backtracking search for per-step orderings that steer the final
    interval onto `target` exactly. Returns a SteerResult, or None when no
    path was found within max_backtracks or before deadline_nanos (a
    time.monotonic_ns() value) -- "not in budget", not necessarily impossible."""
    target = Fraction(target)
    n = len(src)
    orderings = [None] * n

    stack = []
    f0 = list(freq)
    m0 = sum(f0)
    i = 0
    stack.append(_Frame(f0, m0, Fraction(0), Fraction(1), target, _get_candidate(src[0], f0, m0, target)))
    backtracks = 0

    while True:
        if i == n:
            break
        if not stack or backtracks > max_backtracks or time.monotonic_ns() > deadline_nanos:
            return None

        top = stack[-1]
        if top.idx >= len(top.cand.sum):
            stack.pop()
            i -= 1
            backtracks += 1
            if not stack:
                return None
            continue

        sJ = top.cand.sum[top.idx]
        top.idx += 1
        j = src[i]
        fJ = top.f[j]
        orderings[i] = ordering_for_choice(j, top.cand.other_symbol, top.f, sJ)

        lo = Fraction(sJ, top.m)
        hi = Fraction(sJ + fJ, top.m)
        new_off = top.off + top.rng * lo
        new_rng = top.rng * Fraction(fJ, top.m)
        new_residual = (top.residual - lo) / (hi - lo)
        f2 = list(top.f)
        f2[j] -= 1
        i += 1
        if i == n:
            return SteerResult(orderings, new_off, new_rng, backtracks)
        stack.append(_Frame(f2, top.m - 1, new_off, new_rng, new_residual,
                            _get_candidate(src[i], f2, top.m - 1, new_residual)))
    return SteerResult(orderings, Fraction(0), Fraction(1), backtracks)


# ---------------------------------------------------------------------------
# Combinadic (Java long arithmetic, overflow included)
# ---------------------------------------------------------------------------

def binomial(n, k):
    """C(n, k) computed as the Java does, in wrapping 64-bit long arithmetic."""
    if k < 0 or k > n:
        return 0
    if k > n - k:
        k = n - k
    result = 1
    for i in range(k):
        result = _jdiv(_jlong(result * (n - i)), i + 1)
    return result


def combinadic_encode(sorted_chosen):
    rank = 0
    for i, c in enumerate(sorted_chosen):
        rank = _jlong(rank + binomial(c, i + 1))
    return rank


def combinadic_decode(rank, k):
    result = [0] * k
    r = rank
    for pos in range(k, 0, -1):
        c = pos - 1
        while binomial(c + 1, pos) <= r:
            c += 1
        result[pos - 1] = c
        r = _jlong(r - binomial(c, pos))
    return result


# ---------------------------------------------------------------------------
# Simplest fraction in an interval
# ---------------------------------------------------------------------------

def _floor_div(n, d):
    """Floor division for d > 0 (Java's floorDiv helper)."""
    q = _jdiv(n, d)
    if n - q * d != 0 and n < 0:
        return q - 1
    return q


def simplest_fraction_in_interval(lo_n, lo_d, hi_n, hi_d):
    """[p, q]: the fraction with the smallest denominator in [lo, hi) --
    the search runs strictly inside (lo, hi), and lo itself (reduced) wins
    when its denominator is no larger."""
    orig_lo_n, orig_lo_d = lo_n, lo_d
    floors = []
    while True:
        flo = _floor_div(lo_n, lo_d)
        candidate = flo + 1
        if candidate * hi_d < hi_n:
            p, q = candidate, 1
            break
        lo_frac_n = lo_n - flo * lo_d
        hi_frac_n = hi_n - flo * hi_d
        if lo_frac_n == 0:
            k = _jdiv(hi_d, hi_frac_n) + 1
            p, q = flo * k + 1, k
            break
        floors.append(flo)
        lo_n, lo_d, hi_n, hi_d = hi_d, hi_frac_n, lo_d, lo_frac_n

    for flo in reversed(floors):
        p, q = flo * p + q, p

    g = math.gcd(p, q)
    p, q = _jdiv(p, g), _jdiv(q, g)

    lo_g = math.gcd(orig_lo_n, orig_lo_d)
    lo_reduced_n, lo_reduced_d = _jdiv(orig_lo_n, lo_g), _jdiv(orig_lo_d, lo_g)
    if lo_reduced_d <= q:
        return [lo_reduced_n, lo_reduced_d]
    return [p, q]


# ---------------------------------------------------------------------------
# C-sequence coder: plain exact arithmetic coding (sampling without
# replacement) of which symbol comes at each step
# ---------------------------------------------------------------------------

def _reduce(n, d):
    g = math.gcd(n, d)
    if g != 0 and g != 1:
        return _jdiv(n, g), _jdiv(d, g)
    return n, d


def encode_c_sequence(src, freq):
    """[numerator, denominator] of the simplest code for src."""
    f = list(freq)
    m = sum(f)
    off_n, off_d, rng_n, rng_d = 0, 1, 1, 1
    for j in src:
        s = sum(f[:j])
        fJ = f[j]
        new_off_n = off_n * rng_d * m + rng_n * s * off_d
        new_off_d = off_d * rng_d * m
        new_rng_n = rng_n * fJ
        new_rng_d = rng_d * m
        off_n, off_d = _reduce(new_off_n, new_off_d)
        rng_n, rng_d = _reduce(new_rng_n, new_rng_d)
        f[j] -= 1
        m -= 1
    hi_n = off_n * rng_d + rng_n * off_d
    hi_d = off_d * rng_d
    return simplest_fraction_in_interval(off_n, off_d, hi_n, hi_d)


def decode_c_sequence(code_n, code_d, freq, n):
    f = list(freq)
    m = sum(f)
    decoded = [0] * n
    for i in range(n):
        scaled_n = code_n * m
        cum = 0
        chosen = -1
        for j in range(len(f)):
            if f[j] == 0:
                continue
            if cum * code_d <= scaled_n < (cum + f[j]) * code_d:
                chosen = j
                break
            cum += f[j]
        if chosen < 0:
            raise RuntimeError("c-sequence decode failed at step %d" % i)
        new_n = code_n * m - cum * code_d
        new_d = code_d * f[chosen]
        code_n, code_d = _reduce(new_n, new_d)
        f[chosen] -= 1
        m -= 1
        decoded[i] = chosen
    return decoded


# ---------------------------------------------------------------------------
# Compressed format: c-sequence + combinadic "before" subsets
# ---------------------------------------------------------------------------

class _BitWriter:
    def __init__(self):
        self.buf = bytearray()
        self.bit_len = 0

    def write_bits(self, value, num_bits):
        value &= _MASK64                      # Java >>> on a long
        for b in range(num_bits - 1, -1, -1):
            if self.bit_len % 8 == 0:
                self.buf.append(0)
            if (value >> b) & 1:
                self.buf[self.bit_len // 8] |= 1 << (7 - self.bit_len % 8)
            self.bit_len += 1

    def to_bytes(self):
        return bytes(self.buf)


class _BitReader:
    def __init__(self, buf):
        self.buf = buf
        self.bit_pos = 0

    def read_bits(self, num_bits):
        value = 0
        for _ in range(num_bits):
            bit = (self.buf[self.bit_pos // 8] >> (7 - self.bit_pos % 8)) & 1
            value = _jlong((value << 1) | bit)
            self.bit_pos += 1
        return value


def _bits_needed(num_values):
    if num_values <= 1:
        return 0
    return (num_values - 1).bit_length()


def _int_min_total(total):
    """(int) Math.min(total, Integer.MAX_VALUE) for a Java long total."""
    v = min(total, 2147483647)
    return ((v + 2147483648) & 0xFFFFFFFF) - 2147483648


class Compressed:
    """c_seq_n / c_seq_d (the c-sequence code), subset_data (bytes), n."""

    __slots__ = ("c_seq_n", "c_seq_d", "subset_data", "n")

    def __init__(self, c_seq_n, c_seq_d, subset_data, n):
        self.c_seq_n = c_seq_n
        self.c_seq_d = c_seq_d
        self.subset_data = subset_data
        self.n = n

    def total_bits(self):
        def nbytes(v):
            return 1 if v == 0 else (v.bit_length() + 1 + 7) // 8
        return (nbytes(self.c_seq_n) + nbytes(self.c_seq_d)) * 8 + len(self.subset_data) * 8


def compress(orderings, src, freq):
    """Orderings (from steer_encode) plus the data -> Compressed."""
    c_code = encode_c_sequence(src, freq)
    w = _BitWriter()
    n = len(orderings)
    for i in range(n):
        table = orderings[i]
        k = len(table)
        if k <= 1:
            continue
        real_symbol = src[i]
        other_syms = sorted(v for v in table if v != real_symbol)
        pos = table.index(real_symbol)
        index_of = {v: t for t, v in enumerate(other_syms)}
        before_idx = sorted(index_of[v] for v in table[:pos])
        w.write_bits(pos, _bits_needed(k))
        rank = combinadic_encode(before_idx)
        total = binomial(k - 1, pos)
        w.write_bits(rank, _bits_needed(_int_min_total(total)))
    return Compressed(c_code[0], c_code[1], w.to_bytes(), n)


def decompress(c, freq):
    """Compressed -> the orderings steer_decode expects."""
    src = decode_c_sequence(c.c_seq_n, c.c_seq_d, freq, c.n)
    r = _BitReader(c.subset_data)
    f = list(freq)
    orderings = [None] * c.n
    for i in range(c.n):
        remaining = [s for s in range(len(f)) if f[s] > 0]
        k = len(remaining)
        real_symbol = src[i]
        if k <= 1:
            orderings[i] = remaining
            if k == 1:
                f[remaining[0]] -= 1
            continue
        other_syms = [v for v in remaining if v != real_symbol]
        pos = r.read_bits(_bits_needed(k))
        total = binomial(k - 1, pos)
        rank = r.read_bits(_bits_needed(_int_min_total(total)))
        before_vals = {other_syms[idx] for idx in combinadic_decode(rank, pos)}
        table = [v for v in other_syms if v in before_vals]
        table.append(real_symbol)
        table.extend(v for v in other_syms if v not in before_vals)
        orderings[i] = table
        f[real_symbol] -= 1
    return orderings


# ===========================================================================
# Order-table methods (moved here from ArithmeticMapper)
# ===========================================================================

def _keyed(frequency):
    """The Java's Hashtable<Double,Integer> trick: frequency as a double key,
    bumped by .001 until unique. Returns (keys in index order, key -> index)."""
    table = {}
    keys = []
    for i, fr in enumerate(frequency):
        key = float(fr)
        while key in table:
            key += .001
        table[key] = i
        keys.append(key)
    return keys, table


def get_ascending_table(frequency):
    """Symbol indices by ascending frequency, greatest last."""
    keys, table = _keyed(frequency)
    return bytearray(table[key] & 0xFF for key in sorted(keys))


def get_descending_table(frequency):
    """For each symbol, its rank by descending frequency (greatest first)."""
    keys, table = _keyed(frequency)
    descending_table = bytearray(len(frequency))
    for i, key in enumerate(sorted(keys, reverse=True)):
        descending_table[table[key]] = i & 0xFF
    return descending_table


def _exhausted_list(src, frequency):
    exhausted = [i for i, fr in enumerate(frequency) if fr == 0]
    f = list(frequency)
    for v in src:
        j = v & 0xFF
        f[j] -= 1
        if f[j] == 0:
            exhausted.append(j)
    return exhausted


def get_first_table(src, frequency):
    """Symbol indices in the order each is used up in src."""
    exhausted = _exhausted_list(src, frequency)
    return bytearray(exhausted[i] & 0xFF for i in range(len(frequency)))


def get_last_table(src, frequency):
    """Symbol indices in the reverse of the order each is used up in src."""
    exhausted = _exhausted_list(src, frequency)
    return bytearray(exhausted[i] & 0xFF for i in range(len(frequency) - 1, -1, -1))


def _table_series(src, frequency, pick_first, move_down):
    """Shared body of getTableSeries 1-4: start from the last table and walk
    one entry to the front (move_down) or the back, recording each step.
    The entry walked is descending_table[0] (pick_first) or its last value
    -- a rank, compared against symbol values, exactly as in the Java."""
    descending_table = get_descending_table(frequency)
    last_table = get_last_table(src, frequency)
    target = descending_table[0] if pick_first else descending_table[-1]

    place = 0
    for i, v in enumerate(last_table):
        if v == target:
            place = i
            break

    result = [bytearray(last_table)]
    end = 0 if move_down else len(last_table) - 1
    while place != end:
        other = place - 1 if move_down else place + 1
        last_table[place] = last_table[other]
        last_table[other] = target
        place = other
        result.append(bytearray(last_table))
    return result


def get_table_series(src, frequency):
    return _table_series(src, frequency, pick_first=False, move_down=True)


def get_table_series2(src, frequency):
    return _table_series(src, frequency, pick_first=False, move_down=False)


def get_table_series3(src, frequency):
    return _table_series(src, frequency, pick_first=True, move_down=True)


def get_table_series4(src, frequency):
    return _table_series(src, frequency, pick_first=True, move_down=False)


def get_random_table(frequency, seed=None):
    """A Fisher-Yates shuffle of 0..n-1 (as unsigned bytes). With a seed it
    matches java.util.Random(seed) exactly; without one it is unseeded."""
    n = len(frequency)
    table = bytearray(i & 0xFF for i in range(n))
    rand = JavaRandom(seed)
    for i in range(n - 1, 0, -1):
        j = rand.next_int(i + 1)
        table[i], table[j] = table[j], table[i]
    return table


def get_random_order_table(frequency, seed):
    """get_random_table(frequency, seed) inverted to symbol -> rank, the shape
    the order argument of get_interval_value / get_arithmetic_values takes."""
    rank_to_symbol = get_random_table(frequency, seed)
    symbol_to_rank = bytearray(len(rank_to_symbol))
    for rank, symbol in enumerate(rank_to_symbol):
        symbol_to_rank[symbol] = rank & 0xFF
    return symbol_to_rank


def get_random_order_table_byte_seed(frequency, seed):
    """Byte seed, read as unsigned 0..255 (seed & 0xFF)."""
    return get_random_order_table(frequency, seed & 0xFF)


def get_random_order_table_short_seed(frequency, seed):
    """Short seed, sign-extended from 16 bits (as readShort gives it back)."""
    seed &= 0xFFFF
    return get_random_order_table(frequency, seed - 0x10000 if seed >= 0x8000 else seed)


# ---------------------------------------------------------------------------
# Exact arithmetic coding with an order table
# ---------------------------------------------------------------------------

def get_interval_value(src, frequency, order):
    """[numerator, denominator] of the simplest code for src, with the
    symbols' positions in the interval permuted by `order`."""
    f = [0] * len(frequency)
    for i, o in enumerate(order):
        f[o & 0xFF] = frequency[i]
    s = []
    m = 0
    for v in f:
        s.append(m)
        m += v

    off = Fraction(0)
    rng = Fraction(1)
    for v in src:
        j = order[v & 0xFF] & 0xFF
        off += rng * Fraction(s[j], m)
        rng *= Fraction(f[j], m)
        f[j] -= 1
        m -= 1
        for k in range(j + 1, len(s)):
            s[k] -= 1

    hi = off + rng
    return simplest_fraction_in_interval(off.numerator, off.denominator, hi.numerator, hi.denominator)


def get_arithmetic_values(v, frequency, n, order):
    """Decode n symbols (as a bytearray) from the code v = [num, den]."""
    frequency2 = [0] * len(frequency)
    inverse_order = [0] * len(order)
    for i, o in enumerate(order):
        j = o & 0xFF
        frequency2[j] = frequency[i]
        inverse_order[j] = i & 0xFF

    target = Fraction(v[0], v[1])
    value = bytearray(n)

    arithmetic_list = []                         # [symbol, frequency, start]
    m = 0
    for i in range(len(frequency)):
        if frequency2[i] != 0:
            arithmetic_list.append([i, frequency2[i], m])
            m += frequency2[i]

    offset = Fraction(0)
    range_ = Fraction(1)
    for i in range(n):
        w = target - offset
        j = len(arithmetic_list) // 2
        entry = arithmetic_list[j]
        f, s = entry[1], entry[2]
        a = range_ * Fraction(s, m)
        c = range_ * Fraction(s + f, m)

        if a > w:
            k = j // 2
            while a > w:
                j -= k
                entry = arithmetic_list[j]
                f, s = entry[1], entry[2]
                a = range_ * Fraction(s, m)
                k //= 2
                if k == 0:
                    k = 1
            c = range_ * Fraction(s + f, m)
            while c <= w:
                j += 1
                entry = arithmetic_list[j]
                f, s = entry[1], entry[2]
                c = range_ * Fraction(s + f, m)
        elif c <= w:
            k = (len(arithmetic_list) - j) // 2
            while c <= w:
                j += k
                entry = arithmetic_list[j]
                f, s = entry[1], entry[2]
                c = range_ * Fraction(s + f, m)
                k //= 2
                if k == 0:
                    k = 1
            a = range_ * Fraction(s, m)
            while a > w:
                j -= 1
                entry = arithmetic_list[j]
                f, s = entry[1], entry[2]
                a = range_ * Fraction(s, m)

        offset += range_ * Fraction(s, m)
        range_ *= Fraction(f, m)

        for p in range(j + 1, len(arithmetic_list)):
            arithmetic_list[p][2] -= 1

        f -= 1
        m -= 1
        if f != 0:
            entry[1] = f
        else:
            del arithmetic_list[j]

        value[i] = inverse_order[entry[0]] & 0xFF
    return value


# ---------------------------------------------------------------------------
# Fast approximate offset (for order-table search)
# ---------------------------------------------------------------------------

def get_approx_offset_fast_ordered(src, frequency, order):
    """The leading ~52 bits of the order-table code as a float in [0, 1),
    using 32-bit renormalizing integer coding. A scorer for hill climbing,
    not a codec."""
    f = [0] * len(frequency)
    for i, o in enumerate(order):
        f[o & 0xFF] = frequency[i]
    s = []
    m = 0
    for v in f:
        s.append(m)
        m += v

    TOP = 0x100000000
    HALF = 0x80000000
    QTR = 0x40000000
    TQTR = 0xC0000000
    MAX_BITS = 52

    low, high = 0, TOP
    pending = 0
    accum = 0
    count = 0

    def append(bit):
        nonlocal accum, count
        if count < MAX_BITS:
            accum = (accum << 1) | bit
            count += 1

    for v in src:
        j = order[v & 0xFF] & 0xFF
        rng = high - low
        new_low = low + (rng * s[j]) // m
        new_high = high if s[j] + f[j] == m else low + (rng * (s[j] + f[j])) // m
        low, high = new_low, new_high

        while True:
            if high <= HALF:
                append(0)
                for _ in range(pending):
                    append(1)
                pending = 0
                low <<= 1
                high <<= 1
            elif low >= HALF:
                append(1)
                for _ in range(pending):
                    append(0)
                pending = 0
                low = (low - HALF) << 1
                high = (high - HALF) << 1
            elif low >= QTR and high <= TQTR:
                pending += 1
                low = (low - QTR) << 1
                high = (high - QTR) << 1
            else:
                break

        f[j] -= 1
        m -= 1
        for k in range(j + 1, len(s)):
            s[k] -= 1

    pending += 1
    if low < QTR:
        append(0)
        for _ in range(pending):
            append(1)
    else:
        append(1)
        for _ in range(pending):
            append(0)

    return 0.0 if count == 0 else float(accum) / float(1 << count)
