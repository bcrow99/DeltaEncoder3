"""
fraction_mapper.py -- FractionMapper.java (version 2.1): exact rational
arithmetic on a "wheel" (BigFraction), plus conversion to and from
repeating decimal and binary digit strings.

BigFraction is a wheel rather than plain rationals: besides the ordinary
fractions there is a single infinity (1/0, no separate +inf and -inf) and an
indeterminate "bottom" (0/0, written as the up tack). Python's
fractions.Fraction can't hold either, so this module has its own class.
Every operation follows the Java exactly: infinity - infinity, 0 * infinity
and infinity / infinity all come out as bottom; ordering anything against
infinity or bottom raises ArithmeticError (Java's ArithmeticException).

BigInteger is a Python int. Java's overloads get one name where the
argument type tells them apart, and separate names where the Java's
arithmetic width changes the answer:
  multiply(BigFraction) / multiply(long)         -> multiply(x), either
  getRationalNumber(String, String) /
  getRationalNumber(BinaryDigits, BinaryDigits)  -> get_rational_number, either
  getBinaryDigits(BigInteger, BigInteger)        -> get_binary_digits
  getBinaryDigits(long, long)                    -> get_binary_digits_long
A BinaryDigits' bits are an int used as a bit set, like the Java BitSet:
bit i is the block's i-th digit, most significant digit first.

get_decimal_digits takes Java ints and get_binary_digits_long Java longs,
as their Java versions do (anything outside that range is a ValueError).
Version 2.1 fixed the integer overflow those two used to have, so both now
give the exact digits for every input, and get_binary_digits_long the same
as get_binary_digits.

get_rational_number accepts only what Java's BigInteger(String) does: an
optional sign and decimal digits (Python's int() would also take spaces and
underscores).
"""

import re
from fractions import Fraction

_INT_MIN, _INT_MAX = -(1 << 31), (1 << 31) - 1
_LONG_MIN, _LONG_MAX = -(1 << 63), (1 << 63) - 1


def _gcd(a, b):
    from math import gcd
    return gcd(a, b)


# =============================================================================
# BigFraction
# =============================================================================

class BigFraction:
    """An exact fraction n/d, always in lowest terms with d > 0, or one of
    the wheel's two extra elements: infinity (stored as 1/0) and bottom
    (0/0). Immutable."""

    __slots__ = ("n", "d")

    def __init__(self, numerator, denominator):
        numerator, denominator = int(numerator), int(denominator)
        if numerator == 0 and denominator == 0:
            # Bottom: its own equivalence class, equal only to itself.
            n, d = 0, 0
        elif denominator == 0:
            # Infinity: every a/0 with a != 0 is the same element, so the
            # numerator's sign and size are dropped.
            n, d = 1, 0
        else:
            if denominator < 0:
                numerator, denominator = -numerator, -denominator
            if numerator == 0:
                denominator = 1
            else:
                g = _gcd(denominator, numerator)
                if g != 1:
                    numerator //= g
                    denominator //= g
            n, d = numerator, denominator
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "d", d)

    def __setattr__(self, name, value):
        raise AttributeError("BigFraction is immutable")

    @classmethod
    def of(cls, n, d):
        return cls(n, d)

    # ---- arithmetic (the Java's formulas, so the wheel cases fall out the same) ----

    def add(self, f):
        return BigFraction(self.n * f.d + f.n * self.d, self.d * f.d)

    def subtract(self, f):
        return BigFraction(self.n * f.d - f.n * self.d, self.d * f.d)

    def multiply(self, f):
        """multiply(BigFraction), or multiply(long) for an int."""
        if isinstance(f, BigFraction):
            return BigFraction(self.n * f.n, self.d * f.d)
        return BigFraction(self.n * int(f), self.d)

    def divide(self, f):
        return BigFraction(self.n * f.d, self.d * f.n)

    def negate(self):
        return BigFraction(-self.n, self.d)

    def abs(self):
        return self.negate() if self.n < 0 else self

    def is_infinite(self):
        """True for the single infinity (1/0)."""
        return self.d == 0 and self.n != 0

    def is_bottom(self):
        """True for bottom (0/0), the indeterminate element."""
        return self.d == 0 and self.n == 0

    # ---- comparison ----

    def compare_to(self, f):
        """-1, 0 or 1. Raises ArithmeticError if either side is infinity or
        bottom: a wheel has no order once either is involved."""
        if self.is_infinite() or self.is_bottom() or f.is_infinite() or f.is_bottom():
            raise ArithmeticError("cannot order infinity or bottom (0/0) against anything -- "
                                  "a wheel has no total order once either value is infinite or indeterminate")
        a, b = self.n * f.d, f.n * self.d
        return (a > b) - (a < b)

    def lt(self, f):
        return self.compare_to(f) < 0

    def le(self, f):
        return self.compare_to(f) <= 0

    def gt(self, f):
        return self.compare_to(f) > 0

    def eq(self, f):
        return self.equals(f)

    def equals(self, other):
        """Field comparison, which is value equality since every value has
        one canonical (n, d). Bottom equals only bottom."""
        return isinstance(other, BigFraction) and self.n == other.n and self.d == other.d

    # ---- conversion ----

    def __str__(self):
        if self.is_bottom():
            return "⊥"     # bottom
        if self.is_infinite():
            return "∞"     # infinity
        return "%d/%d" % (self.n, self.d)

    def __repr__(self):
        return "BigFraction(%s)" % self

    def to_double(self):
        """inf for infinity, nan for bottom; otherwise n/d rounded half-even
        to 40 decimal places (Java's BigDecimal divide), then to the nearest
        double."""
        if self.is_bottom():
            return float("nan")
        if self.is_infinite():
            return float("inf")
        num, den = abs(self.n) * 10 ** 40, self.d
        q, r = divmod(num, den)
        if 2 * r > den or (2 * r == den and q % 2 == 1):
            q += 1
        if self.n < 0:
            q = -q
        return float(Fraction(q, 10 ** 40))

    def to_fraction(self):
        """The value as a fractions.Fraction (finite values only)."""
        if self.d == 0:
            raise ArithmeticError("%s has no Fraction equivalent" % self)
        return Fraction(self.n, self.d)

    # ---- Python operators, for convenience: the same methods underneath ----

    def __add__(self, f):
        return self.add(_as_big(f))

    def __radd__(self, f):
        return _as_big(f).add(self)

    def __sub__(self, f):
        return self.subtract(_as_big(f))

    def __rsub__(self, f):
        return _as_big(f).subtract(self)

    def __mul__(self, f):
        return self.multiply(_as_big(f))

    def __rmul__(self, f):
        return _as_big(f).multiply(self)

    def __truediv__(self, f):
        return self.divide(_as_big(f))

    def __rtruediv__(self, f):
        return _as_big(f).divide(self)

    def __neg__(self):
        return self.negate()

    def __abs__(self):
        return self.abs()

    def __eq__(self, other):
        return self.equals(other)

    def __hash__(self):
        return hash((self.n, self.d))

    def __lt__(self, f):
        return self.lt(_as_big(f))

    def __le__(self, f):
        return self.le(_as_big(f))

    def __gt__(self, f):
        return self.gt(_as_big(f))

    def __ge__(self, f):
        return self.compare_to(_as_big(f)) >= 0

    def __float__(self):
        return self.to_double()


def _as_big(x):
    if isinstance(x, BigFraction):
        return x
    if isinstance(x, int):
        return BigFraction(x, 1)
    if isinstance(x, Fraction):
        return BigFraction(x.numerator, x.denominator)
    raise TypeError("can't use %r as a BigFraction" % (x,))


BigFraction.ZERO = BigFraction(0, 1)
BigFraction.ONE = BigFraction(1, 1)
BigFraction.HALF = BigFraction(1, 2)


# =============================================================================
# Decimal digits
# =============================================================================

def get_decimal_digits(a, b):
    """The decimal expansion of a/b's FRACTIONAL PART (integer part and sign
    dropped), as [static digits, repeating digits]. Both empty: a/b is a
    whole number. Only the second empty: the decimal terminates. a and b
    must fit in a Java int. Raises ArithmeticError if b == 0."""
    if not (_INT_MIN <= a <= _INT_MAX and _INT_MIN <= b <= _INT_MAX):
        raise ValueError("a and b must fit in a Java int")
    if b == 0:
        raise ArithmeticError("division by zero")
    a, b = abs(a), abs(b)
    remainder = a % b
    if remainder == 0:
        return ["", ""]

    digits = []
    seen_at = {}           # remainder -> the digit position where it was first seen
    while remainder != 0 and remainder not in seen_at:
        seen_at[remainder] = len(digits)
        digit, remainder = divmod(remainder * 10, b)
        digits.append(chr(ord("0") + digit))

    s = "".join(digits)
    if remainder == 0:
        return [s, ""]
    start = seen_at[remainder]
    return [s[:start], s[start:]]


_BIGINTEGER = re.compile(r"[+-]?[0-9]+\Z")


def _big_integer(s):
    """Java's new BigInteger(String), for decimal digit strings."""
    if not _BIGINTEGER.match(s):
        raise ValueError("For input string: \"%s\"" % s)
    return int(s)


def get_rational_number(static_digits, repeating_digits):
    """Inverse of get_decimal_digits (two strings) or get_binary_digits (two
    BinaryDigits): the value 0.static(repeating repeating) as a BigFraction.

    No repeating part:  S / base^s
    Repeating part:     (S * (base^r - 1) + R) / (base^s * (base^r - 1))
    where s and r are the block lengths (leading zeros count), so 0.1(6) in
    decimal gives (1*9 + 6) / (10*9) = 1/6."""
    if isinstance(static_digits, BinaryDigits):
        return _get_rational_number_binary(static_digits, repeating_digits)
    s, r = len(static_digits), len(repeating_digits)
    S = 0 if s == 0 else _big_integer(static_digits)
    if r == 0:
        return BigFraction(S, 10 ** s)
    R = _big_integer(repeating_digits)
    nines = 10 ** r - 1
    return BigFraction(S * nines + R, 10 ** s * nines)


# =============================================================================
# Binary digits
# =============================================================================

class BinaryDigits:
    """A block of binary digits and its length (the length is needed because
    "100" and "1" set the same bits). bits is an int used as a bit set: bit i
    is the i-th digit of the block, most significant digit first."""

    __slots__ = ("bits", "length")

    def __init__(self, bits, length):
        self.bits = bits
        self.length = length

    def __str__(self):
        return "".join("1" if (self.bits >> i) & 1 else "0" for i in range(self.length))

    def __repr__(self):
        return "BinaryDigits(%r)" % str(self)

    def __eq__(self, other):
        return isinstance(other, BinaryDigits) and self.length == other.length and self.bits == other.bits


def _split(digits, pos, remainder, seen_at):
    if remainder == 0:
        return [BinaryDigits(digits, pos), BinaryDigits(0, 0)]
    start = seen_at[remainder]
    static = digits & ((1 << start) - 1)
    repeating = (digits >> start) & ((1 << (pos - start)) - 1)
    return [BinaryDigits(static, start), BinaryDigits(repeating, pos - start)]


def get_binary_digits(a, b):
    """Binary analogue of get_decimal_digits, with exact (BigInteger)
    arithmetic: [static, repeating] BinaryDigits of a/b's fractional part,
    integer part and sign dropped. A block of length 0 is empty. Raises
    ArithmeticError if b == 0."""
    if b == 0:
        raise ArithmeticError("division by zero")
    a, b = abs(a), abs(b)
    remainder = a % b
    if remainder == 0:
        return [BinaryDigits(0, 0), BinaryDigits(0, 0)]
    digits, seen_at, pos = 0, {}, 0
    while remainder != 0 and remainder not in seen_at:
        seen_at[remainder] = pos
        remainder *= 2
        if remainder >= b:
            digits |= 1 << pos
            remainder -= b
        pos += 1
    return _split(digits, pos, remainder, seen_at)


def get_binary_digits_long(a, b):
    """getBinaryDigits(long, long): the same as get_binary_digits, for a and b
    that fit in a Java long."""
    if not (_LONG_MIN <= a <= _LONG_MAX and _LONG_MIN <= b <= _LONG_MAX):
        raise ValueError("a and b must fit in a Java long")
    return get_binary_digits(a, b)


def _bits_to_value(bits, length):
    """Digit i (bit i of the set) is the value's bit length-1-i."""
    value = 0
    for i in range(length):
        if (bits >> i) & 1:
            value |= 1 << (length - 1 - i)
    return value


def _get_rational_number_binary(static_digits, repeating_digits):
    s, r = static_digits.length, repeating_digits.length
    S = _bits_to_value(static_digits.bits, s)
    if r == 0:
        return BigFraction(S, 1 << s)
    R = _bits_to_value(repeating_digits.bits, r)
    ones = (1 << r) - 1
    return BigFraction(S * ones + R, (1 << s) * ones)
