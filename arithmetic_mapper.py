"""
arithmetic_mapper.py — translation of the fixed ArithmeticMapper.java.

This targets the CORRECTED Java (two real bugs found and fixed there,
verified against real compiled Java before this translation was written --
see ArithmeticMapper.java's own header comment for full detail):
  1. simplestFractionInInterval() previously used a continued-fraction
     expansion with zero-padding that could return a fraction outside the
     target interval entirely. Replaced with an iterative floor/reciprocal
     (Stern-Brocot) descent.
  2. getArithmeticValuesFast()/getArithmeticValuesFastFenwick() previously
     had a decoder-side symbol-selection bug at exact interval boundaries
     (truncating-division asymmetry between encoder and decoder). Fixed by
     verifying the candidate symbol against the encoder's exact formula and
     nudging forward/backward as needed.

BYTE REPRESENTATION: same convention as the other translated modules --
byte arrays are bytearray/list objects of unsigned 0..255 ints. Java's
`(byte) x` truncating cast becomes `x & 0xFF`; `if(j<0) j+=256` sign-fixups
on array-derived values are no-ops here and omitted, since inputs are
already unsigned by this convention.

BIGINTEGER -> PYTHON INT. Python's int is already arbitrary-precision, so
every BigInteger[2] {numerator, denominator} pair in the Java becomes a
plain 2-element Python list/tuple of ints, and BigInteger arithmetic
becomes ordinary +, -, *, //, math.gcd. This is a large simplification but
a mechanical one -- the actual fraction arithmetic is translated literally,
term for term, not re-derived.

FLOOR DIVISION. Java's BigInteger.divide() truncates toward zero, which is
why the Java added a small floorDiv() helper for simplestFractionInInterval
(see that method's own comment there). Python's `//` operator already does
true floor division for any sign combination, so no equivalent helper is
needed here -- plain `//` is used directly throughout.

JAVA-COMPATIBLE RANDOM. get_random_order_table()'s docstring (preserved
from the Java) guarantees that encoder and decoder derive the *identical*
table from a given (frequency, seed) pair. To honor that exactly -- rather
than silently redefining what a "seed" means -- this file ports
java.util.Random's actual 48-bit LCG algorithm (_JavaRandom below) instead
of using Python's own `random` module, which is a different, incompatible
generator (Mersenne Twister). Verified bit-for-bit against real
java.util.Random output across multiple seeds and bounds, including the
exact Fisher-Yates shuffle pattern get_random_table() uses.

JAVA OVERLOAD COLLAPSING. Java has three get_random_order_table overloads
(long/byte/short seed) that all normalize to the same long-seed call --
collapsed to one Python function taking a plain int, since Python has no
overload resolution to preserve and int already covers the full range.
"""

import math


# =============================================================================
# java.util.Random port (see module docstring)
# =============================================================================
def _to_signed_32(x):
    x &= 0xFFFFFFFF
    if x >= 0x80000000:
        x -= 0x100000000
    return x


class _JavaRandom:
    """Port of java.util.Random's 48-bit linear congruential generator.
    Only next_int(bound) is implemented, since that's all this module needs."""
    _MULTIPLIER = 0x5DEECE66D
    _ADDEND = 0xB
    _MASK = (1 << 48) - 1

    def __init__(self, seed):
        self._seed = (seed ^ self._MULTIPLIER) & self._MASK

    def _next(self, bits):
        self._seed = (self._seed * self._MULTIPLIER + self._ADDEND) & self._MASK
        result = self._seed >> (48 - bits)
        return _to_signed_32(result)

    def next_int(self, bound):
        if bound <= 0:
            raise ValueError("bound must be positive")
        if (bound & -bound) == bound:  # bound is a power of 2
            return (bound * self._next(31)) >> 31
        while True:
            bits = self._next(31)
            val = bits % bound
            if _to_signed_32(bits - val + (bound - 1)) >= 0:
                return val


# =============================================================================
# getSerialOffset / getSerialValues
# =============================================================================
def get_serial_offset(src, frequency, n):
    """Java: getSerialOffset(byte[] src, int[] frequency, int n) -> ArrayList [offset, f]
    offset is a [numerator, denominator] pair; f is the depleted frequency table."""
    f = list(frequency)

    s = [0] * 256
    m = 0
    for i in range(256):
        s[i] = m
        m += f[i]

    offset = [0, 1]
    range_ = [1, 1]

    for i in range(n):
        j = src[i]

        addend = [range_[0] * s[j], range_[1] * m]
        g = math.gcd(addend[0], addend[1])
        if g > 1:
            addend[0] //= g
            addend[1] //= g

        offset[0] = offset[0] * addend[1]
        addend[0] = addend[0] * offset[1]
        offset[1] = offset[1] * addend[1]
        offset[0] = offset[0] + addend[0]

        g = math.gcd(offset[0], offset[1])
        if g > 1:
            offset[0] //= g
            offset[1] //= g

        range_[0] = range_[0] * f[j]
        range_[1] = range_[1] * m
        g = math.gcd(range_[0], range_[1])
        if g > 1:
            range_[0] //= g
            range_[1] //= g

        f[j] -= 1
        m -= 1
        for k in range(j + 1, len(s)):
            s[k] -= 1

    return [offset, f]


def get_serial_values(v, frequency, n):
    """Java: getSerialValues(BigInteger[] v, int[] frequency, int n) -> ArrayList [value, frequency2]"""
    value = bytearray(n)

    arithmetic_list = []
    m = 0
    for i in range(len(frequency)):
        if frequency[i] != 0:
            arithmetic_list.append([i, frequency[i], m])
            m += frequency[i]

    frequency2 = list(frequency)

    offset = [0, 1]
    range_ = [1, 1]
    w = [v[0], v[1]]

    for i in range(n):
        if offset[0] != 0:
            w[0] = v[0]
            w[1] = v[1]
            w[0] = w[0] * offset[1]
            w[0] = w[0] - offset[0] * w[1]
            w[1] = w[1] * offset[1]

            g = math.gcd(w[0], w[1])
            if g > 1:
                w[0] //= g
                w[1] //= g

        j = len(arithmetic_list) // 2
        lst = arithmetic_list[j]
        f = lst[1]
        s = lst[2]

        a = range_[0] * s
        b = w[0]
        c = range_[0] * (s + f)
        d = range_[1] * m

        a = a * w[1]
        b = b * d
        c = c * w[1]

        if a > b:
            k = j // 2
            while a > b:
                j -= k
                lst = arithmetic_list[j]
                f = lst[1]
                s = lst[2]
                a = range_[0] * s
                a = a * w[1]
                k //= 2
                if k == 0:
                    k = 1

            # Check if we passed value.
            c = range_[0] * (s + f)
            c = c * w[1]
            if c <= b:
                while c <= b:
                    j += 1
                    lst = arithmetic_list[j]
                    f = lst[1]
                    s = lst[2]
                    c = range_[0] * (s + f)
                    c = c * w[1]
        elif c <= b:
            size = len(arithmetic_list)
            k = (size - j) // 2

            while c <= b:
                j += k
                lst = arithmetic_list[j]
                f = lst[1]
                s = lst[2]
                c = range_[0] * (s + f)
                c = c * w[1]
                k //= 2
                if k == 0:
                    k = 1

            # Check if we passed value.
            a = range_[0] * s
            a = a * w[1]
            if a > b:
                while a > b:
                    j -= 1
                    lst = arithmetic_list[j]
                    f = lst[1]
                    s = lst[2]
                    a = range_[0] * s
                    a = a * w[1]

        addend = [range_[0] * s, range_[1] * m]

        offset[0] = offset[0] * addend[1]
        offset[0] = offset[0] + addend[0] * offset[1]
        offset[1] = offset[1] * addend[1]

        g = math.gcd(offset[0], offset[1])
        if g > 1:
            offset[0] //= g
            offset[1] //= g

        range_[0] = range_[0] * f
        range_[1] = range_[1] * m

        g = math.gcd(range_[0], range_[1])
        if g > 1:
            range_[0] //= g
            range_[1] //= g

        for p in range(j + 1, len(arithmetic_list)):
            arithmetic_list[p][2] -= 1

        f -= 1
        m -= 1
        if f != 0:
            lst[1] = f
            arithmetic_list[j] = lst
        else:
            del arithmetic_list[j]

        k = lst[0]
        frequency2[k] -= 1
        value[i] = k & 0xFF

    return [value, frequency2]


def gcd(a, b):
    """Java: gcd(long, long). Equivalent to math.gcd for the non-negative
    inputs used throughout this module (kept as a named 1:1 mapping back
    to the Java source; get_normal_range_quotient and the Fenwick methods
    call this by name)."""
    return math.gcd(a, b)


# =============================================================================
# getNormalRangeQuotient
# =============================================================================
def get_normal_range_quotient(src, table, frequency):
    """Java: getNormalRangeQuotient(byte[] src, Hashtable<Integer,Integer> table, int[] frequency) -> ArrayList [bits, bitlength]
    `table` maps raw byte value -> compact index, matching frequency's shape.
    Self-described by the original author as having precision limitations
    (see the Java source comment above this method) -- translated as-is,
    not reworked."""
    f = list(frequency)
    s = [0] * len(f)
    m = 0
    for i in range(len(f)):
        s[i] = m
        m += f[i]

    bit_buffer = bytearray(len(src) * 2)

    bit_offset = 0
    byte_offset = 0
    bits_outstanding = 0

    offset = [1, 4]
    range_ = [1, 2]

    n = len(src)

    for i in range(n):
        j = table[src[i]]

        addend = [range_[0] * s[j], range_[1] * m]
        g = gcd(addend[0], addend[1])
        if g > 1:
            addend[0] //= g
            addend[1] //= g

        offset[0] = offset[0] * addend[1]
        addend[0] = addend[0] * offset[1]
        offset[1] = offset[1] * addend[1]
        offset[0] = offset[0] + addend[0]

        g = gcd(offset[0], offset[1])
        if g > 1:
            offset[0] //= g
            offset[1] //= g

        range_[0] = range_[0] * f[j]
        range_[1] = range_[1] * m
        g = gcd(range_[0], range_[1])
        if g > 1:
            range_[0] //= g
            range_[1] //= g

        p = offset[0] / offset[1]
        r = range_[0] / range_[1]
        while r <= 0.25:
            if p + r <= 0.5:
                bit_offset += 1
                if bit_offset == 8:
                    byte_offset += 1
                    bit_offset = 0
                value = 1
                while bits_outstanding > 0:
                    bit_buffer[byte_offset] = (bit_buffer[byte_offset] | (value << bit_offset)) & 0xFF
                    bit_offset += 1
                    if bit_offset == 8:
                        byte_offset += 1
                        bit_offset = 0
                    bits_outstanding -= 1
            elif p >= 0.5:
                value = 1
                bit_buffer[byte_offset] = (bit_buffer[byte_offset] | (value << bit_offset)) & 0xFF
                while bits_outstanding > 0:
                    bit_offset += 1
                    if bit_offset == 8:
                        byte_offset += 1
                        bit_offset = 0
                    bits_outstanding -= 1
                p = p - 0.5
            else:
                bits_outstanding += 1
                p = p - 0.25
            p *= 2
            r *= 2

        f[j] -= 1
        m -= 1
        for k in range(j + 1, len(s)):
            s[k] -= 1

        offset[0] = 1
        offset[1] = 4
        range_[0] = 1
        range_[1] = 2

    bits = bytearray(byte_offset + 1)
    for i in range(len(bits)):
        bits[i] = bit_buffer[i]
    extra_bits = 0
    if bit_offset != 0:
        extra_bits = 8 - bit_offset
    bitlength = len(bits) * 8 - extra_bits

    return [bits, bitlength]


# =============================================================================
# simplestFractionInInterval (FIXED -- see module docstring)
# =============================================================================
def simplest_fraction_in_interval(lo_n, lo_d, hi_n, hi_d):
    """Return (p, q) -- the fraction with smallest denominator strictly
    inside the open interval (lo_n/lo_d, hi_n/hi_d). Requires lo_d > 0,
    hi_d > 0, lo_n/lo_d < hi_n/hi_d.

    Iterative floor/reciprocal (Stern-Brocot) descent -- see module
    docstring for why this replaced the original continued-fraction
    implementation, and ArithmeticMapper.java's header for the full bug
    writeup. Validated against a brute-force reference (enumerate
    candidate denominators) across thousands of randomized cases before
    being ported here, including lo=0 exactly, hi landing on exact "nice"
    fractions at various depths, very large close-together denominators,
    and a deliberately pathological Fibonacci-ratio interval forcing
    maximal continued-fraction depth."""
    floors = []

    while True:
        flo = lo_n // lo_d
        candidate = flo + 1

        if candidate * hi_d < hi_n:
            p, q = candidate, 1
            break

        lo_frac_n = lo_n - flo * lo_d
        hi_frac_n = hi_n - flo * hi_d

        if lo_frac_n == 0:
            k = hi_d // hi_frac_n + 1
            p, q = flo * k + 1, k
            break

        floors.append(flo)
        lo_n, lo_d, hi_n, hi_d = hi_d, hi_frac_n, lo_d, lo_frac_n

    for flo in reversed(floors):
        p, q = flo * p + q, p

    g = math.gcd(p, q)
    return p // g, q // g


# =============================================================================
# getIntervalValue (core encoder) / getArithmeticValues (core decoder)
# =============================================================================
def get_interval_value(src, frequency):
    """Java: getIntervalValue(byte[] src, int[] frequency) -> BigInteger[2] {num, den}"""
    f = list(frequency)
    n = len(src)

    s = [0] * len(f)
    m = 0
    for i in range(len(f)):
        s[i] = m
        m += f[i]

    off_n, off_d = 0, 1
    rng_n, rng_d = 1, 1

    for i in range(n):
        j = src[i]

        add_n = rng_n * s[j]
        add_d = rng_d * m
        g = math.gcd(add_n, add_d)
        if g > 1:
            add_n //= g
            add_d //= g

        off_n = off_n * add_d + add_n * off_d
        off_d = off_d * add_d
        g = math.gcd(off_n, off_d)
        if g > 1:
            off_n //= g
            off_d //= g

        rng_n = rng_n * f[j]
        rng_d = rng_d * m
        g = math.gcd(rng_n, rng_d)
        if g > 1:
            rng_n //= g
            rng_d //= g

        f[j] -= 1
        m -= 1
        for k in range(j + 1, len(s)):
            s[k] -= 1

    if off_d != rng_d:
        off_n = off_n * rng_d
        rng_n = rng_n * off_d
        common_d = off_d * rng_d
        off_d = common_d
        rng_d = common_d

    hi_n = off_n + rng_n
    hi_d = off_d

    return simplest_fraction_in_interval(off_n, off_d, hi_n, hi_d)


def get_arithmetic_values(v, frequency, n):
    """Java: getArithmeticValues(BigInteger[] v, int[] frequency, int n) -> byte[]
    Binary/jump-search decoder matching get_interval_value()'s encoding.
    Translated with literal fidelity to Java's control flow (see module
    docstring re: this style of search)."""
    value = bytearray(n)

    arithmetic_list = []
    m = 0
    for i in range(len(frequency)):
        if frequency[i] != 0:
            arithmetic_list.append([i, frequency[i], m])
            m += frequency[i]

    offset = [0, 1]
    range_ = [1, 1]
    w = [v[0], v[1]]

    for i in range(n):
        if offset[0] != 0:
            w[0] = v[0]
            w[1] = v[1]
            w[0] = w[0] * offset[1]
            w[0] = w[0] - offset[0] * w[1]
            w[1] = w[1] * offset[1]

            g = math.gcd(w[0], w[1])
            if g > 1:
                w[0] //= g
                w[1] //= g

        j = len(arithmetic_list) // 2
        lst = arithmetic_list[j]
        f = lst[1]
        s = lst[2]

        a = range_[0] * s
        b = w[0]
        c = range_[0] * (s + f)
        d = range_[1] * m

        a = a * w[1]
        b = b * d
        c = c * w[1]

        if a > b:
            k = j // 2
            while a > b:
                j -= k
                lst = arithmetic_list[j]
                f = lst[1]
                s = lst[2]
                a = range_[0] * s
                a = a * w[1]
                k //= 2
                if k == 0:
                    k = 1

            # Check if we passed value.
            c = range_[0] * (s + f)
            c = c * w[1]
            if c <= b:
                while c <= b:
                    j += 1
                    lst = arithmetic_list[j]
                    f = lst[1]
                    s = lst[2]
                    c = range_[0] * (s + f)
                    c = c * w[1]
        elif c <= b:
            size = len(arithmetic_list)
            k = (size - j) // 2

            while c <= b:
                j += k
                lst = arithmetic_list[j]
                f = lst[1]
                s = lst[2]
                c = range_[0] * (s + f)
                c = c * w[1]
                k //= 2
                if k == 0:
                    k = 1

            # Check if we passed value.
            a = range_[0] * s
            a = a * w[1]
            if a > b:
                while a > b:
                    j -= 1
                    lst = arithmetic_list[j]
                    f = lst[1]
                    s = lst[2]
                    a = range_[0] * s
                    a = a * w[1]

        addend = [range_[0] * s, range_[1] * m]

        offset[0] = offset[0] * addend[1]
        offset[0] = offset[0] + addend[0] * offset[1]
        offset[1] = offset[1] * addend[1]

        g = math.gcd(offset[0], offset[1])
        if g > 1:
            offset[0] //= g
            offset[1] //= g

        range_[0] = range_[0] * f
        range_[1] = range_[1] * m

        g = math.gcd(range_[0], range_[1])
        if g > 1:
            range_[0] //= g
            range_[1] //= g

        for p in range(j + 1, len(arithmetic_list)):
            arithmetic_list[p][2] -= 1

        f -= 1
        m -= 1
        if f != 0:
            lst[1] = f
            arithmetic_list[j] = lst
        else:
            del arithmetic_list[j]

        k = lst[0]
        value[i] = k & 0xFF

    return value


# =============================================================================
# Table-ordering utility methods
# =============================================================================
def get_ascending_table(frequency):
    """Java: getAscendingTable(int[]) -> byte[]
    Table of frequency-table indices in ascending frequency order (greatest
    last). Ties broken by nudging the key by .001 repeatedly -- translated
    exactly as the Java does it (a somewhat fragile but faithfully
    preserved technique), not reworked to use a stable sort."""
    n = len(frequency)
    table = {}
    keys = []
    for i in range(n):
        key = float(frequency[i])
        while key in table:
            key += 0.001
        table[key] = i
        keys.append(key)

    keys_sorted = sorted(keys)

    ascending_table = bytearray(n)
    for i in range(n):
        key = keys_sorted[i]
        j = table[key]
        ascending_table[i] = j & 0xFF
    return ascending_table


def get_descending_table(frequency):
    """Java: getDescendingTable(int[]) -> byte[]"""
    n = len(frequency)
    table = {}
    keys = []
    for i in range(n):
        key = float(frequency[i])
        while key in table:
            key += 0.001
        table[key] = i
        keys.append(key)

    keys_sorted = sorted(keys, reverse=True)

    descending_table = bytearray(n)
    for i in range(n):
        key = keys_sorted[i]
        j = table[key]
        descending_table[j] = i & 0xFF
    return descending_table


def get_first_table(src, frequency):
    """Java: getFirstTable(byte[], int[]) -> byte[]
    Table of indices in the order each value's frequency is exhausted
    first (values that start at frequency 0 are already "exhausted")."""
    exhausted_list = [i for i in range(len(frequency)) if frequency[i] == 0]

    f = list(frequency)
    for i in range(len(src)):
        j = src[i]
        f[j] -= 1
        if f[j] == 0:
            exhausted_list.append(j)

    first_table = bytearray(len(frequency))
    for i in range(len(frequency)):
        first_table[i] = exhausted_list[i] & 0xFF
    return first_table


def get_last_table(src, frequency):
    """Java: getLastTable(byte[], int[]) -> byte[]"""
    exhausted_list = [i for i in range(len(frequency)) if frequency[i] == 0]

    f = list(frequency)
    for i in range(len(src)):
        j = src[i]
        f[j] -= 1
        if f[j] == 0:
            exhausted_list.append(j)

    last_table = bytearray(len(frequency))
    k = 0
    for i in range(len(frequency) - 1, -1, -1):
        j = exhausted_list[i]
        last_table[k] = j & 0xFF
        k += 1
    return last_table


def _descending_table_and_last_table(src, frequency):
    """Shared setup used by get_table_series/2/3/4 -- builds the same
    descending_table and last_table each of those four Java methods builds
    independently (identical code duplicated four times in the original)."""
    n = len(frequency)
    table = {}
    keys = []
    for i in range(n):
        key = float(frequency[i])
        while key in table:
            key += 0.001
        table[key] = i
        keys.append(key)

    keys_sorted = sorted(keys, reverse=True)
    descending_table = bytearray(n)
    for i in range(n):
        key = keys_sorted[i]
        j = table[key]
        descending_table[j] = i & 0xFF

    exhausted_list = [i for i in range(len(frequency)) if frequency[i] == 0]
    f = list(frequency)
    for i in range(len(src)):
        j = src[i]
        f[j] -= 1
        if f[j] == 0:
            exhausted_list.append(j)

    last_table = bytearray(len(frequency))
    k = 0
    for i in range(len(frequency) - 1, -1, -1):
        j = exhausted_list[i]
        last_table[k] = j & 0xFF
        k += 1

    return descending_table, last_table


def get_table_series(src, frequency):
    """Java: getTableSeries(byte[], int[]) -> ArrayList<byte[]>
    Starting from last_table, repeatedly swaps the least-frequent symbol
    one step toward the front, recording each intermediate table."""
    descending_table, last_table = _descending_table_and_last_table(src, frequency)

    length = len(descending_table)
    least = descending_table[length - 1]

    least_place = 0
    for i in range(len(last_table)):
        if last_table[i] == least:
            least_place = i
            break

    result = [bytearray(last_table)]

    done = False
    while not done:
        if least_place == 0:
            done = True
        else:
            down = last_table[least_place - 1]
            last_table[least_place] = down
            last_table[least_place - 1] = least
            least_place -= 1
            result.append(bytearray(last_table))
            if least_place == 0:
                done = True

    return result


def get_table_series2(src, frequency):
    """Java: getTableSeries2(byte[], int[]) -> ArrayList<byte[]>
    Same as get_table_series but swaps the least-frequent symbol toward
    the back instead of the front."""
    descending_table, last_table = _descending_table_and_last_table(src, frequency)

    length = len(descending_table)
    least = descending_table[length - 1]

    least_place = 0
    for i in range(len(last_table)):
        if last_table[i] == least:
            least_place = i
            break

    result = [bytearray(last_table)]

    done = False
    while not done:
        if least_place == len(last_table) - 1:
            done = True
        else:
            up = last_table[least_place + 1]
            last_table[least_place] = up
            last_table[least_place + 1] = least
            least_place += 1
            result.append(bytearray(last_table))
            if least_place == len(last_table) - 1:
                done = True

    return result


def get_table_series3(src, frequency):
    """Java: getTableSeries3(byte[], int[]) -> ArrayList<byte[]>
    Same idea as get_table_series but tracks the MOST-frequent symbol
    (descending_table[0]) instead of the least, swapping it toward front."""
    descending_table, last_table = _descending_table_and_last_table(src, frequency)

    greatest = descending_table[0]

    greatest_place = 0
    for i in range(len(last_table)):
        if last_table[i] == greatest:
            greatest_place = i
            break

    result = [bytearray(last_table)]

    done = False
    while not done:
        if greatest_place == 0:
            done = True
        else:
            down = last_table[greatest_place - 1]
            last_table[greatest_place] = down
            last_table[greatest_place - 1] = greatest
            greatest_place -= 1
            result.append(bytearray(last_table))
            if greatest_place == 0:
                done = True

    return result


def get_table_series4(src, frequency):
    """Java: getTableSeries4(byte[], int[]) -> ArrayList<byte[]>
    Same as get_table_series3 but swaps the most-frequent symbol toward
    the back instead of the front."""
    descending_table, last_table = _descending_table_and_last_table(src, frequency)

    greatest = descending_table[0]

    greatest_place = 0
    for i in range(len(last_table)):
        if last_table[i] == greatest:
            greatest_place = i
            break

    result = [bytearray(last_table)]

    done = False
    while not done:
        if greatest_place == len(last_table) - 1:
            done = True
        else:
            up = last_table[greatest_place + 1]
            last_table[greatest_place] = up
            last_table[greatest_place + 1] = greatest
            greatest_place += 1
            result.append(bytearray(last_table))
            if greatest_place == len(last_table) - 1:
                done = True

    return result


def get_random_table(frequency, seed=None):
    """Java: getRandomTable(int[]) [no seed, system randomness] and
    getRandomTable(int[], long) [seeded] -- collapsed into one function
    with an optional seed, since Python doesn't need Java's separate
    overloads. When seed is None, uses system time (matching Java's own
    no-seed constructor, which seeds from nanoTime()-derived entropy) via
    Python's own random module -- NOT java.util.Random, since there's no
    reproducibility guarantee to preserve in the unseeded case. When seed
    is given, uses the Java-compatible _JavaRandom (see module docstring),
    since get_random_order_table's contract requires that."""
    n = len(frequency)
    table = bytearray(i & 0xFF for i in range(n))

    if seed is None:
        import random as _random
        rand = _random.Random()
        for i in range(n - 1, 0, -1):
            j = rand.randint(0, i)
            table[i], table[j] = table[j], table[i]
    else:
        rand = _JavaRandom(seed)
        for i in range(n - 1, 0, -1):
            j = rand.next_int(i + 1)
            table[i], table[j] = table[j], table[i]

    return table


def get_random_order_table(frequency, seed):
    """Java: getRandomOrderTable(int[], long/byte/short) -- all three Java
    overloads normalize their seed to a long before doing the same work;
    collapsed to one Python function taking a plain int (Python has no
    overload resolution to preserve, and int already covers the full
    range those three Java types spanned).

    Builds the random rank->symbol table via get_random_table(frequency,
    seed) and inverts it to the symbol->rank shape expected by the order
    parameter of get_interval_value/get_arithmetic_values. Both the
    encoder (search) and decoder (reconstruction from a stored seed) call
    this same helper rather than each inverting get_random_table's output
    separately, so they are guaranteed to derive the identical order table
    from a given (frequency, seed) pair -- see module docstring re: why
    this specifically requires the Java-compatible RNG."""
    rank_to_symbol = get_random_table(frequency, seed)
    symbol_to_rank = bytearray(len(rank_to_symbol))
    for rank in range(len(rank_to_symbol)):
        symbol = rank_to_symbol[rank]
        symbol_to_rank[symbol] = rank & 0xFF
    return symbol_to_rank


# =============================================================================
# Order-table variants
# =============================================================================
def get_arithmetic_offset_and_range(src, frequency, order):
    """Java: getArithmeticOffsetAndRange(byte[], int[], byte[]) -> BigInteger[4] {offN, offD, rngN, rngD}
    Same computation as get_interval_value_ordered() below but returns the
    raw offset/range pair directly instead of collapsing it through
    simplest_fraction_in_interval -- a diagnostic/intermediate-value
    variant, not itself used for encoding."""
    f = [0] * len(frequency)
    n = len(src)

    for i in range(len(order)):
        j = order[i]
        f[j] = frequency[i]

    s = [0] * len(f)
    m = 0
    for i in range(len(f)):
        s[i] = m
        m += f[i]

    off_n, off_d = 0, 1
    rng_n, rng_d = 1, 1

    for i in range(n):
        j = src[i]
        j = order[j]

        add_n = rng_n * s[j]
        add_d = rng_d * m
        g = math.gcd(add_n, add_d)
        if g > 1:
            add_n //= g
            add_d //= g

        off_n = off_n * add_d + add_n * off_d
        off_d = off_d * add_d
        g = math.gcd(off_n, off_d)
        if g > 1:
            off_n //= g
            off_d //= g

        rng_n = rng_n * f[j]
        rng_d = rng_d * m
        g = math.gcd(rng_n, rng_d)
        if g > 1:
            rng_n //= g
            rng_d //= g

        f[j] -= 1
        m -= 1
        for k in range(j + 1, len(s)):
            s[k] -= 1

    if off_d != rng_d:
        off_n = off_n * rng_d
        rng_n = rng_n * off_d
        common_d = off_d * rng_d
        off_d = common_d
        rng_d = common_d

    return [off_n, off_d, rng_n, rng_d]


def get_interval_value_ordered(src, frequency, order):
    """Java: getIntervalValue(byte[], int[], byte[]) -> BigInteger[2] {num, den}
    Order-table variant of get_interval_value()."""
    f = [0] * len(frequency)
    n = len(src)

    for i in range(len(order)):
        j = order[i]
        f[j] = frequency[i]

    s = [0] * len(f)
    m = 0
    for i in range(len(f)):
        s[i] = m
        m += f[i]

    off_n, off_d = 0, 1
    rng_n, rng_d = 1, 1

    for i in range(n):
        j = src[i]
        j = order[j]

        add_n = rng_n * s[j]
        add_d = rng_d * m
        g = math.gcd(add_n, add_d)
        if g > 1:
            add_n //= g
            add_d //= g

        off_n = off_n * add_d + add_n * off_d
        off_d = off_d * add_d
        g = math.gcd(off_n, off_d)
        if g > 1:
            off_n //= g
            off_d //= g

        rng_n = rng_n * f[j]
        rng_d = rng_d * m
        g = math.gcd(rng_n, rng_d)
        if g > 1:
            rng_n //= g
            rng_d //= g

        f[j] -= 1
        m -= 1
        for k in range(j + 1, len(s)):
            s[k] -= 1

    if off_d != rng_d:
        off_n = off_n * rng_d
        rng_n = rng_n * off_d
        common_d = off_d * rng_d
        off_d = common_d
        rng_d = common_d

    hi_n = off_n + rng_n
    hi_d = off_d

    return simplest_fraction_in_interval(off_n, off_d, hi_n, hi_d)


def get_arithmetic_values_ordered(v, frequency, n, order):
    """Java: getArithmeticValues(BigInteger[], int[], int, byte[]) -> byte[]
    Order-table variant of get_arithmetic_values()."""
    frequency2 = [0] * len(frequency)
    inverse_order = bytearray(len(order))
    for i in range(len(order)):
        j = order[i]
        frequency2[j] = frequency[i]
        inverse_order[j] = i & 0xFF

    value = bytearray(n)

    arithmetic_list = []
    m = 0
    for i in range(len(frequency)):
        if frequency2[i] != 0:
            arithmetic_list.append([i, frequency2[i], m])
            m += frequency2[i]

    offset = [0, 1]
    range_ = [1, 1]
    w = [v[0], v[1]]

    for i in range(n):
        if offset[0] != 0:
            w[0] = v[0]
            w[1] = v[1]
            w[0] = w[0] * offset[1]
            w[0] = w[0] - offset[0] * w[1]
            w[1] = w[1] * offset[1]

            g = math.gcd(w[0], w[1])
            if g > 1:
                w[0] //= g
                w[1] //= g

        j = len(arithmetic_list) // 2
        lst = arithmetic_list[j]
        f = lst[1]
        s = lst[2]

        a = range_[0] * s
        b = w[0]
        c = range_[0] * (s + f)
        d = range_[1] * m

        a = a * w[1]
        b = b * d
        c = c * w[1]

        if a > b:
            k = j // 2
            while a > b:
                j -= k
                lst = arithmetic_list[j]
                f = lst[1]
                s = lst[2]
                a = range_[0] * s
                a = a * w[1]
                k //= 2
                if k == 0:
                    k = 1

            c = range_[0] * (s + f)
            c = c * w[1]
            if c <= b:
                while c <= b:
                    j += 1
                    lst = arithmetic_list[j]
                    f = lst[1]
                    s = lst[2]
                    c = range_[0] * (s + f)
                    c = c * w[1]
        elif c <= b:
            size = len(arithmetic_list)
            k = (size - j) // 2

            while c <= b:
                j += k
                lst = arithmetic_list[j]
                f = lst[1]
                s = lst[2]
                c = range_[0] * (s + f)
                c = c * w[1]
                k //= 2
                if k == 0:
                    k = 1

            a = range_[0] * s
            a = a * w[1]
            if a > b:
                while a > b:
                    j -= 1
                    lst = arithmetic_list[j]
                    f = lst[1]
                    s = lst[2]
                    a = range_[0] * s
                    a = a * w[1]

        addend = [range_[0] * s, range_[1] * m]

        offset[0] = offset[0] * addend[1]
        offset[0] = offset[0] + addend[0] * offset[1]
        offset[1] = offset[1] * addend[1]

        g = math.gcd(offset[0], offset[1])
        if g > 1:
            offset[0] //= g
            offset[1] //= g

        range_[0] = range_[0] * f
        range_[1] = range_[1] * m

        g = math.gcd(range_[0], range_[1])
        if g > 1:
            range_[0] //= g
            range_[1] //= g

        for p in range(j + 1, len(arithmetic_list)):
            arithmetic_list[p][2] -= 1

        f -= 1
        m -= 1
        if f != 0:
            lst[1] = f
            arithmetic_list[j] = lst
        else:
            del arithmetic_list[j]

        k = lst[0]
        k = inverse_order[k]
        value[i] = k & 0xFF

    return value


# =============================================================================
# getPrimeFactors / getIntervalValue2 / getArithmeticValues2
# =============================================================================
def get_prime_factors(n):
    """Java: getPrimeFactors(BigInteger) -> ArrayList<BigInteger>
    Used by get_interval_value2() below."""
    factors = []

    if n == 1:
        return factors
    if _is_probable_prime(n):
        factors.append(n)
        return factors

    divisor = 2
    while n % divisor == 0:
        factors.append(divisor)
        n //= divisor

    divisor = 3
    while divisor * divisor <= n:
        if n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        else:
            divisor = _next_probable_prime(divisor)

    if n > 1:
        factors.append(n)

    return factors


def _is_probable_prime(n, k=40):
    """Miller-Rabin primality test, matching the confidence level of Java's
    BigInteger.isProbablePrime(100) closely enough for this module's use
    (a helper for get_interval_value2's factorization, not a
    cryptographic primitive)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    import random as _random
    for _ in range(k):
        a = _random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _next_probable_prime(n):
    """Java: BigInteger.nextProbablePrime() -- smallest probable prime > n."""
    candidate = n + 1
    if candidate % 2 == 0:
        candidate += 1
    while not _is_probable_prime(candidate):
        candidate += 2
    return candidate


def get_interval_value2(src, frequency):
    """Java: getIntervalValue2(byte[], int[]) -> BigInteger[2] {num, den}
    "A slower version that produces sub-optimal results, but is easier to
    understand. It uses a search mechanism instead of continued fraction
    expansion to find a simpler fraction than the offset." -- translated
    as-is; this method does NOT share simplest_fraction_in_interval's bug
    (confirmed: 100/100 round trip against real Java), since it uses a
    completely different simplification approach (prime-factor reduction
    and a linear search)."""
    f = list(frequency)
    s = [0] * len(f)
    m = 0
    for i in range(len(f)):
        s[i] = m
        m += f[i]

    offset = [0, 1]
    range_ = [1, 1]

    n = len(src)

    for i in range(n):
        j = src[i]

        addend = [range_[0], range_[1]]
        factor = s[j]
        addend[0] = addend[0] * factor
        factor = m
        addend[1] = addend[1] * factor

        g = math.gcd(addend[0], addend[1])
        if g > 1:
            addend[0] //= g
            addend[1] //= g

        offset[0] = offset[0] * addend[1]
        addend[0] = addend[0] * offset[1]
        offset[1] = offset[1] * addend[1]
        offset[0] = offset[0] + addend[0]

        g = math.gcd(offset[0], offset[1])
        if g > 1:
            offset[0] //= g
            offset[1] //= g

        factor = f[j]
        range_[0] = range_[0] * factor
        factor = m
        range_[1] = range_[1] * factor

        g = math.gcd(range_[0], range_[1])
        if g > 1:
            range_[0] //= g
            range_[1] //= g

        f[j] -= 1
        m -= 1
        for k in range(j + 1, len(s)):
            s[k] -= 1

    if offset[1] != range_[1]:
        range_factor = offset[1]
        offset_factor = range_[1]
        offset[0] = offset[0] * offset_factor
        offset[1] = offset[1] * offset_factor
        range_[0] = range_[0] * range_factor
        range_[1] = range_[1] * range_factor

    gcd_val = math.gcd(offset[1], offset[0])

    factor_list = get_prime_factors(gcd_val)

    factor = 1

    maximum_range = 10000
    minimum_range = 512
    j = len(factor_list) - 1
    while range_[0] // factor > maximum_range and j >= 0:
        next_factor = factor_list[j]
        factor = factor * next_factor
        j -= 1

    if factor != 1:
        offset[0] = offset[0] // factor
        offset[1] = offset[1] // factor
        range_[0] = range_[0] // factor
        range_[1] = offset[1]

    # If the offset pair had no common divisor,
    # the range numerator is 1.
    if range_[0] < minimum_range:
        factor = 2
        while range_[0] * factor < minimum_range:
            factor = factor * 2

        offset[0] = offset[0] * factor
        offset[1] = offset[1] * factor
        range_[0] = range_[0] * factor
        range_[1] = offset[1]

    gcd_val = math.gcd(offset[0], offset[1])
    max_gcd = gcd_val

    value = [offset[0], offset[1]]

    # NOTE: range_[0] is used as a plain int here (and the loop below is
    # O(range_[0]), a linear scan) -- the maximum_range/minimum_range
    # adjustment above is what keeps range_[0] in a manageable few-
    # thousand ballpark before reaching this point. Python ints don't
    # truncate the way Java's BigInteger.intValue() does, so this can't
    # silently wrap the way the Java could in principle (see
    # ArithmeticMapper.java's own NOTE at this spot); range_[0] is used
    # directly.
    j = range_[0]
    k = 0
    for i in range(1, j):
        value[0] = value[0] + 1
        gcd_val = math.gcd(value[0], value[1])
        if gcd_val > max_gcd:
            max_gcd = gcd_val
            k = i

    largest_index = k

    value[0] = offset[0] + largest_index
    value[0] = value[0] // max_gcd
    value[1] = value[1] // max_gcd

    return value


def get_arithmetic_values2(v, frequency, n):
    """Java: getArithmeticValues2(BigInteger[], int[], int) -> byte[]
    "Slower version that uses a linear search" -- a simpler decoder than
    get_arithmetic_values()'s jump-search, stepping j by 1 at a time.
    Translated with the same literal fidelity as get_arithmetic_values."""
    value = bytearray(n)

    arithmetic_list = []
    m = 0
    for i in range(len(frequency)):
        if frequency[i] != 0:
            arithmetic_list.append([i, frequency[i], m])
            m += frequency[i]

    offset = [0, 1]
    range_ = [1, 1]
    w = [v[0], v[1]]

    for i in range(n):
        if offset[0] != 0:
            w[0] = v[0]
            w[1] = v[1]
            w[0] = w[0] * offset[1]
            w[0] = w[0] - offset[0] * w[1]
            w[1] = w[1] * offset[1]

            g = math.gcd(w[0], w[1])
            if g > 1:
                w[0] //= g
                w[1] //= g

        # Start j at middle of list.
        j = len(arithmetic_list) // 2
        lst = arithmetic_list[j]

        f = lst[1]
        s = lst[2]

        a = range_[0] * s
        b = w[0]
        c = range_[0] * (s + f)
        d = range_[1] * m

        a = a * w[1]
        b = b * d
        c = c * w[1]

        if a > b:
            while a > b:
                j -= 1
                lst = arithmetic_list[j]
                f = lst[1]
                s = lst[2]
                a = range_[0] * s
                a = a * w[1]
        elif c <= b:
            while c <= b:
                j += 1
                lst = arithmetic_list[j]
                f = lst[1]
                s = lst[2]
                c = range_[0] * (s + f)
                c = c * w[1]

        # Reset offset and range.
        addend = [range_[0] * s, range_[1] * m]

        offset[0] = offset[0] * addend[1]
        offset[0] = offset[0] + addend[0] * offset[1]
        offset[1] = offset[1] * addend[1]

        g = math.gcd(offset[0], offset[1])
        if g > 1:
            offset[0] //= g
            offset[1] //= g

        range_[0] = range_[0] * f
        range_[1] = range_[1] * m

        g = math.gcd(range_[0], range_[1])
        if g > 1:
            range_[0] //= g
            range_[1] //= g

        # Reset sums.
        for p in range(j + 1, len(arithmetic_list)):
            arithmetic_list[p][2] -= 1

        f -= 1
        m -= 1
        if f != 0:
            lst[1] = f
            arithmetic_list[j] = lst
        else:
            del arithmetic_list[j]

        k = lst[0]
        value[i] = k & 0xFF

    return value


# =============================================================================
# Fast renormalization-based arithmetic coder (no BigInteger).
#
# Uses standard E1/E2/E3 (Witten-Neal-Cleary) interval rescaling with a
# 32-bit fixed-point interval. After renormalization the range is always
# >= 2^30, so every symbol with frequency >= 1 gets a non-zero interval:
# no precision loss for the segment sizes used in DeltaWriter.
#
# Output format (get_interval_value_fast):
#   bytes [0..3] : bit-stream length in bits, big-endian int
#   bytes [4..]  : compressed bit stream, LSB-first within each byte
#
# Input format (get_arithmetic_values_fast):
#   same byte array produced by get_interval_value_fast
# =============================================================================

_TOP = 0x100000000
_HALF = 0x80000000
_QTR = 0x40000000
_TQTR = 0xC0000000
_MASK32 = 0xFFFFFFFF


def _fast_write_bit(buf, pos, bit):
    if bit != 0:
        buf[pos >> 3] = (buf[pos >> 3] | (1 << (pos & 7))) & 0xFF


def _fast_read_bit(buf, data_byte_offset, pos):
    abs_pos = data_byte_offset * 8 + pos
    return (buf[abs_pos >> 3] >> (abs_pos & 7)) & 1


def _find_fast_symbol(s, target):
    """Binary search on cumulative-frequency table s[]. Returns the
    largest index j such that s[j] <= target. s[] is non-decreasing."""
    lo, hi = 0, len(s) - 1
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if s[mid] <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo


def get_interval_value_fast(src, frequency):
    """Java: getIntervalValueFast(byte[], int[]) -> byte[]
    Fast arithmetic encoder using long-integer E1/E2/E3 renormalization.
    Drop-in replacement for the encode half of get_interval_value /
    get_arithmetic_values, but much faster because it avoids big-integer
    fraction arithmetic entirely."""
    f = list(frequency)
    n = len(src)

    s = [0] * len(f)
    m = 0
    for i in range(len(f)):
        s[i] = m
        m += f[i]

    low = 0
    high = _TOP
    pending = 0

    buf = bytearray(n * 2 + 16)
    bit_pos = 0

    for i in range(n):
        j = src[i]

        range_ = high - low
        new_low = low + (range_ * s[j]) // m
        new_high = high if (s[j] + f[j] == m) else low + (range_ * (s[j] + f[j])) // m
        low = new_low
        high = new_high

        while True:
            if high <= _HALF:
                _fast_write_bit(buf, bit_pos, 0)
                bit_pos += 1
                for _p in range(pending):
                    _fast_write_bit(buf, bit_pos, 1)
                    bit_pos += 1
                pending = 0
                low <<= 1
                high <<= 1
            elif low >= _HALF:
                _fast_write_bit(buf, bit_pos, 1)
                bit_pos += 1
                for _p in range(pending):
                    _fast_write_bit(buf, bit_pos, 0)
                    bit_pos += 1
                pending = 0
                low = (low - _HALF) << 1
                high = (high - _HALF) << 1
            elif low >= _QTR and high <= _TQTR:
                pending += 1
                low = (low - _QTR) << 1
                high = (high - _QTR) << 1
            else:
                break

        f[j] -= 1
        m -= 1
        for k in range(j + 1, len(s)):
            s[k] -= 1

    pending += 1
    if low < _QTR:
        _fast_write_bit(buf, bit_pos, 0)
        bit_pos += 1
        for _p in range(pending):
            _fast_write_bit(buf, bit_pos, 1)
            bit_pos += 1
    else:
        _fast_write_bit(buf, bit_pos, 1)
        bit_pos += 1
        for _p in range(pending):
            _fast_write_bit(buf, bit_pos, 0)
            bit_pos += 1

    bit_length = bit_pos
    byte_length = (bit_length + 7) // 8
    result = bytearray(4 + byte_length)
    result[0] = (bit_length >> 24) & 0xFF
    result[1] = (bit_length >> 16) & 0xFF
    result[2] = (bit_length >> 8) & 0xFF
    result[3] = bit_length & 0xFF
    result[4:4 + byte_length] = buf[0:byte_length]
    return result


def get_arithmetic_values_fast(encoded, frequency, n):
    """Java: getArithmeticValuesFast(byte[], int[], int) -> byte[]
    Fast arithmetic decoder, exact inverse of get_interval_value_fast.

    FIX (see module docstring / ArithmeticMapper.java header): includes
    the boundary-verification-and-nudge step. The initial `scaled`-based
    guess can land one symbol short of (or, in principle, past) the true
    one when `code` sits exactly at -- or extremely close to -- a symbol
    boundary, since `scaled` is computed by truncating-division inverting
    a value the encoder produced via its own truncating division; those
    two truncations don't perfectly cancel right at a boundary."""
    bit_length = ((encoded[0] & 0xFF) << 24) | ((encoded[1] & 0xFF) << 16) \
        | ((encoded[2] & 0xFF) << 8) | (encoded[3] & 0xFF)

    f = list(frequency)

    s = [0] * len(f)
    m = 0
    for i in range(len(f)):
        s[i] = m
        m += f[i]

    low = 0
    high = _TOP
    bit_ptr = 0

    code = 0
    for _b in range(32):
        bit = _fast_read_bit(encoded, 4, bit_ptr) if bit_ptr < bit_length else 0
        bit_ptr += 1
        code = (code << 1) | bit

    value = bytearray(n)

    for i in range(n):
        range_ = high - low
        scaled = (code - low) * m // range_
        if scaled < 0:
            scaled = 0
        if scaled >= m:
            scaled = m - 1

        j = _find_fast_symbol(s, scaled)
        while j < len(f) - 1 and f[j] == 0:
            j += 1

        new_low = low + (range_ * s[j]) // m
        new_high = high if (s[j] + f[j] == m) else low + (range_ * (s[j] + f[j])) // m

        while code >= new_high and j < len(f) - 1:
            j += 1
            while j < len(f) - 1 and f[j] == 0:
                j += 1
            new_low = low + (range_ * s[j]) // m
            new_high = high if (s[j] + f[j] == m) else low + (range_ * (s[j] + f[j])) // m
        while code < new_low and j > 0:
            j -= 1
            while j > 0 and f[j] == 0:
                j -= 1
            new_low = low + (range_ * s[j]) // m
            new_high = high if (s[j] + f[j] == m) else low + (range_ * (s[j] + f[j])) // m

        value[i] = j & 0xFF

        low = new_low
        high = new_high

        while True:
            if high <= _HALF:
                low <<= 1
                high <<= 1
                bit = _fast_read_bit(encoded, 4, bit_ptr) if bit_ptr < bit_length else 0
                bit_ptr += 1
                code = ((code << 1) | bit) & _MASK32
            elif low >= _HALF:
                low = (low - _HALF) << 1
                high = (high - _HALF) << 1
                bit = _fast_read_bit(encoded, 4, bit_ptr) if bit_ptr < bit_length else 0
                bit_ptr += 1
                code = (((code - _HALF) << 1) | bit) & _MASK32
            elif low >= _QTR and high <= _TQTR:
                low = (low - _QTR) << 1
                high = (high - _QTR) << 1
                bit = _fast_read_bit(encoded, 4, bit_ptr) if bit_ptr < bit_length else 0
                bit_ptr += 1
                code = (((code - _QTR) << 1) | bit) & _MASK32
            else:
                break

        f[j] -= 1
        m -= 1
        for k in range(j + 1, len(s)):
            s[k] -= 1

    return value


# =============================================================================
# Cheap approximate-offset scorer for order-table search (hill climbing /
# annealing).
# =============================================================================
class _LeadingBits:
    """Keeps only the leading MAX_BITS bits appended to it -- enough for
    full double precision -- and discards the rest. Used only by
    get_approx_offset_fast_ordered; not a general-purpose bit buffer."""
    MAX_BITS = 52  # matches double's mantissa precision

    def __init__(self):
        self.accum = 0
        self.count = 0

    def append(self, bit):
        if self.count < self.MAX_BITS:
            self.accum = (self.accum << 1) | bit
            self.count += 1

    def to_approx_offset(self):
        if self.count == 0:
            return 0.0
        return self.accum / float(1 << self.count)


def get_approx_offset_fast_ordered(src, frequency, order):
    """Java: getApproxOffsetFastOrdered(byte[], int[], byte[]) -> double
    Cheap approximate offset for order-table search. Runs the same
    renormalization as get_interval_value_fast, with an order-table remap
    like get_interval_value_ordered, but instead of packing bits into a
    byte stream for storage, captures the leading ~52 bits directly and
    returns them as a float in [0, 1). Not intended for round-trip
    encode/decode -- only as a fast scorer during hill-climbing/annealing."""
    f = [0] * len(frequency)
    for i in range(len(order)):
        j = order[i]
        f[j] = frequency[i]
    n = len(src)

    s = [0] * len(f)
    m = 0
    for i in range(len(f)):
        s[i] = m
        m += f[i]

    low = 0
    high = _TOP
    pending = 0

    bits = _LeadingBits()

    for i in range(n):
        j = src[i]
        j = order[j]

        range_ = high - low
        new_low = low + (range_ * s[j]) // m
        new_high = high if (s[j] + f[j] == m) else low + (range_ * (s[j] + f[j])) // m
        low = new_low
        high = new_high

        while True:
            if high <= _HALF:
                bits.append(0)
                for _p in range(pending):
                    bits.append(1)
                pending = 0
                low <<= 1
                high <<= 1
            elif low >= _HALF:
                bits.append(1)
                for _p in range(pending):
                    bits.append(0)
                pending = 0
                low = (low - _HALF) << 1
                high = (high - _HALF) << 1
            elif low >= _QTR and high <= _TQTR:
                pending += 1
                low = (low - _QTR) << 1
                high = (high - _QTR) << 1
            else:
                break

        f[j] -= 1
        m -= 1
        for k in range(j + 1, len(s)):
            s[k] -= 1

    pending += 1
    if low < _QTR:
        bits.append(0)
        for _p in range(pending):
            bits.append(1)
    else:
        bits.append(1)
        for _p in range(pending):
            bits.append(0)

    return bits.to_approx_offset()


# =============================================================================
# Fenwick-tree accelerated slow arithmetic coder.
#
# Same exact fraction arithmetic as get_interval_value/get_arithmetic_values,
# but replaces the O(256) cumulative-frequency update loop with a Fenwick
# (Binary Indexed) tree giving O(log 256) = 8 operations per symbol for
# both prefix-sum queries and updates.
# =============================================================================
def _fenwick_build(frequency):
    bit = [0] * 257
    for i in range(256):
        if frequency[i] > 0:
            _fenwick_update(bit, i, frequency[i])
    return bit


def _fenwick_update(bit, i, delta):
    i += 1
    while i <= 256:
        bit[i] += delta
        i += i & (-i)


def _fenwick_query(bit, i):
    total = 0
    i += 1
    while i > 0:
        total += bit[i]
        i -= i & (-i)
    return total


def _fenwick_find(bit, target):
    """Find 0-indexed symbol j: prefix_sum[0..j-1] <= target < prefix_sum[0..j]"""
    pos = 0
    for b in range(8, -1, -1):
        nxt = pos + (1 << b)
        if nxt <= 256 and bit[nxt] <= target:
            target -= bit[nxt]
            pos = nxt
    return pos


def get_interval_value_fenwick(src, frequency):
    """Java: getIntervalValueFenwick(byte[], int[]) -> BigInteger[2] {num, den}
    Encoder: same as get_interval_value but O(log 256) adaptive updates
    via Fenwick tree."""
    f = list(frequency)
    n = len(src)
    bit = _fenwick_build(f)
    m = sum(f)

    off_n, off_d = 0, 1
    rng_n, rng_d = 1, 1

    for i in range(n):
        j = src[i]
        sj = _fenwick_query(bit, j - 1) if j > 0 else 0

        ig = gcd(sj, m) if sj > 0 else 1
        add_n = rng_n * (sj // ig)
        add_d = rng_d * (m // ig)

        off_n = off_n * add_d + add_n * off_d
        off_d = off_d * add_d
        g = math.gcd(off_n, off_d)
        if g > 1:
            off_n //= g
            off_d //= g

        ig = gcd(f[j], m)
        rng_n = rng_n * (f[j] // ig)
        rng_d = rng_d * (m // ig)
        g = math.gcd(rng_n, rng_d)
        if g > 1:
            rng_n //= g
            rng_d //= g

        _fenwick_update(bit, j, -1)
        f[j] -= 1
        m -= 1

    if off_d != rng_d:
        off_n = off_n * rng_d
        rng_n = rng_n * off_d
        common_d = off_d * rng_d
        off_d = common_d
        rng_d = common_d

    hi_n = off_n + rng_n
    hi_d = off_d
    return simplest_fraction_in_interval(off_n, off_d, hi_n, hi_d)


def get_arithmetic_values_fenwick(v, frequency, n):
    """Java: getArithmeticValuesFenwick(BigInteger[], int[], int) -> byte[]
    Decoder: same as get_arithmetic_values but O(log 256) symbol search
    and updates via Fenwick tree."""
    f = list(frequency)
    bit = _fenwick_build(f)
    m = sum(f)

    offset = [0, 1]
    range_ = [1, 1]
    w = [v[0], v[1]]

    value = bytearray(n)

    for i in range(n):
        if offset[0] != 0:
            w[0] = v[0] * offset[1] - offset[0] * v[1]
            w[1] = v[1] * offset[1]
            g2 = math.gcd(w[0], w[1])
            if g2 > 1:
                w[0] //= g2
                w[1] //= g2

        scaled = (w[0] * range_[1] * m) // (range_[0] * w[1])
        target = min(max(scaled, 0), m - 1)
        j = _fenwick_find(bit, target)
        while j < 255 and f[j] == 0:
            j += 1

        value[i] = j & 0xFF
        sj = _fenwick_query(bit, j - 1) if j > 0 else 0

        ig = gcd(sj, m) if sj > 0 else 1
        add_n = range_[0] * (sj // ig)
        add_d = range_[1] * (m // ig)
        offset[0] = offset[0] * add_d + add_n * offset[1]
        offset[1] = offset[1] * add_d
        g = math.gcd(offset[0], offset[1])
        if g > 1:
            offset[0] //= g
            offset[1] //= g

        ig = gcd(f[j], m)
        range_[0] = range_[0] * (f[j] // ig)
        range_[1] = range_[1] * (m // ig)
        g = math.gcd(range_[0], range_[1])
        if g > 1:
            range_[0] //= g
            range_[1] //= g

        _fenwick_update(bit, j, -1)
        f[j] -= 1
        m -= 1

    return value


# =============================================================================
# Fenwick-tree accelerated fast arithmetic coder.
# Same 32-bit renormalization as get_interval_value_fast/
# get_arithmetic_values_fast but O(log 256) cumulative frequency updates
# instead of O(256).
# =============================================================================
def get_interval_value_fast_fenwick(src, frequency):
    """Java: getIntervalValueFastFenwick(byte[], int[]) -> byte[]"""
    f = list(frequency)
    n = len(src)
    bit = _fenwick_build(f)
    m = sum(f)

    low = 0
    high = _TOP
    pending = 0

    buf = bytearray(n * 2 + 16)
    bit_pos = 0

    for i in range(n):
        j = src[i]

        sj = _fenwick_query(bit, j - 1) if j > 0 else 0
        sj_fj = _fenwick_query(bit, j)

        range_ = high - low
        new_low = low + (range_ * sj) // m
        new_high = high if (sj_fj == m) else low + (range_ * sj_fj) // m
        low = new_low
        high = new_high

        while True:
            if high <= _HALF:
                _fast_write_bit(buf, bit_pos, 0)
                bit_pos += 1
                for _p in range(pending):
                    _fast_write_bit(buf, bit_pos, 1)
                    bit_pos += 1
                pending = 0
                low <<= 1
                high <<= 1
            elif low >= _HALF:
                _fast_write_bit(buf, bit_pos, 1)
                bit_pos += 1
                for _p in range(pending):
                    _fast_write_bit(buf, bit_pos, 0)
                    bit_pos += 1
                pending = 0
                low = (low - _HALF) << 1
                high = (high - _HALF) << 1
            elif low >= _QTR and high <= _TQTR:
                pending += 1
                low = (low - _QTR) << 1
                high = (high - _QTR) << 1
            else:
                break

        _fenwick_update(bit, j, -1)
        f[j] -= 1
        m -= 1

    pending += 1
    if low < _QTR:
        _fast_write_bit(buf, bit_pos, 0)
        bit_pos += 1
        for _p in range(pending):
            _fast_write_bit(buf, bit_pos, 1)
            bit_pos += 1
    else:
        _fast_write_bit(buf, bit_pos, 1)
        bit_pos += 1
        for _p in range(pending):
            _fast_write_bit(buf, bit_pos, 0)
            bit_pos += 1

    bit_length = bit_pos
    byte_length = (bit_length + 7) // 8
    result = bytearray(4 + byte_length)
    result[0] = (bit_length >> 24) & 0xFF
    result[1] = (bit_length >> 16) & 0xFF
    result[2] = (bit_length >> 8) & 0xFF
    result[3] = bit_length & 0xFF
    result[4:4 + byte_length] = buf[0:byte_length]
    return result


def get_arithmetic_values_fast_fenwick(encoded, frequency, n):
    """Java: getArithmeticValuesFastFenwick(byte[], int[], int) -> byte[]

    FIX: same boundary issue as get_arithmetic_values_fast -- see that
    function's docstring for the full explanation. Verify and nudge j
    using Fenwick queries instead of direct array access."""
    bit_length = ((encoded[0] & 0xFF) << 24) | ((encoded[1] & 0xFF) << 16) \
        | ((encoded[2] & 0xFF) << 8) | (encoded[3] & 0xFF)

    f = list(frequency)
    bit = _fenwick_build(f)
    m = sum(f)

    low = 0
    high = _TOP
    bit_ptr = 0

    code = 0
    for _b in range(32):
        bt = _fast_read_bit(encoded, 4, bit_ptr) if bit_ptr < bit_length else 0
        bit_ptr += 1
        code = (code << 1) | bt

    value = bytearray(n)

    for i in range(n):
        range_ = high - low
        scaled = (code - low) * m // range_
        if scaled < 0:
            scaled = 0
        if scaled >= m:
            scaled = m - 1

        j = _fenwick_find(bit, scaled)
        while j < len(f) - 1 and f[j] == 0:
            j += 1

        sj = _fenwick_query(bit, j - 1) if j > 0 else 0
        sj_fj = _fenwick_query(bit, j)

        new_low = low + (range_ * sj) // m
        new_high = high if (sj_fj == m) else low + (range_ * sj_fj) // m

        while code >= new_high and j < len(f) - 1:
            j += 1
            while j < len(f) - 1 and f[j] == 0:
                j += 1
            sj = _fenwick_query(bit, j - 1) if j > 0 else 0
            sj_fj = _fenwick_query(bit, j)
            new_low = low + (range_ * sj) // m
            new_high = high if (sj_fj == m) else low + (range_ * sj_fj) // m
        while code < new_low and j > 0:
            j -= 1
            while j > 0 and f[j] == 0:
                j -= 1
            sj = _fenwick_query(bit, j - 1) if j > 0 else 0
            sj_fj = _fenwick_query(bit, j)
            new_low = low + (range_ * sj) // m
            new_high = high if (sj_fj == m) else low + (range_ * sj_fj) // m

        value[i] = j & 0xFF

        low = new_low
        high = new_high

        while True:
            if high <= _HALF:
                low <<= 1
                high <<= 1
                bt = _fast_read_bit(encoded, 4, bit_ptr) if bit_ptr < bit_length else 0
                bit_ptr += 1
                code = ((code << 1) | bt) & _MASK32
            elif low >= _HALF:
                low = (low - _HALF) << 1
                high = (high - _HALF) << 1
                bt = _fast_read_bit(encoded, 4, bit_ptr) if bit_ptr < bit_length else 0
                bit_ptr += 1
                code = (((code - _HALF) << 1) | bt) & _MASK32
            elif low >= _QTR and high <= _TQTR:
                low = (low - _QTR) << 1
                high = (high - _QTR) << 1
                bt = _fast_read_bit(encoded, 4, bit_ptr) if bit_ptr < bit_length else 0
                bit_ptr += 1
                code = (((code - _QTR) << 1) | bt) & _MASK32
            else:
                break

        _fenwick_update(bit, j, -1)
        f[j] -= 1
        m -= 1

    return value
