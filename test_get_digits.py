#!/usr/bin/env python3
"""
test_get_digits.py -- TestGetDigits.java for Python.

Interactive demo: type two integers a and b, and this shows the decimal
digits of a/b's fractional part (static and repeating blocks), then rebuilds
a fraction from those digits and checks that it equals (|a| % |b|) / |b| --
get_decimal_digits only encodes the fractional part (magnitude only, sign and
integer part dropped), so that reduced value is the fair comparison.

    python3 test_get_digits.py

a and b are Java ints, as in the Java version.
"""

import fraction_mapper as fm


def parse_int(text):
    """Integer.parseInt: an optional sign and decimal digits, in int range."""
    s = text.strip()
    if not fm._BIGINTEGER.match(s):
        raise ValueError("For input string: \"%s\"" % s)
    v = int(s)
    if not (fm._INT_MIN <= v <= fm._INT_MAX):
        raise ValueError("For input string: \"%s\" (out of int range)" % s)
    return v


def main():
    a = parse_int(input("Enter integer a: "))
    b = parse_int(input("Enter integer b: "))

    static_digits, repeating_digits = fm.get_decimal_digits(a, b)

    print()
    print("Static digits:    \"" + static_digits + "\"" + ("  (none)" if static_digits == "" else ""))
    print("Repeating digits: \"" + repeating_digits + "\"" + ("  (none -- terminates)" if repeating_digits == "" else ""))

    reconstructed = fm.get_rational_number(static_digits, repeating_digits)
    print("\nReconstructed fraction: " + str(reconstructed))

    expected = fm.BigFraction(abs(a) % abs(b), abs(b))

    print("Expected (|a| % |b|) / |b|: " + str(expected))
    print("Got the integers back: " + ("true" if reconstructed.equals(expected) else "false"))


if __name__ == "__main__":
    main()
