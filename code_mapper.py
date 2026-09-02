"""
code_mapper.py — translation of CodeMapper.java.

BYTE REPRESENTATION: same convention as string_mapper.py / segment_mapper.py
-- byte arrays are bytearray/list objects of unsigned 0..255 ints. Java's
`(byte) x` truncating cast becomes `x & 0xFF` throughout.

PYTHON'S UNIFIED INT COLLAPSES SEVERAL JAVA OVERLOADS. Java distinguishes
int/long/BigInteger code words (three separate, nearly-identical packCode/
unpackCode/getCanonicalCode overloads each) because those types have
different fixed widths (or, for BigInteger, arbitrary width via a
different API). Python's int is already arbitrary-precision, so:
  - get_canonical_code() (below) implements the byte[]-length AND
    int[]-length Java overloads with one function (the arithmetic is
    identical; Java only split them because of static typing).
  - get_canonical_code2() and get_big_canonical_code() are aliases of
    get_canonical_code() -- Java's long[]/BigInteger[] versions of the
    same algorithm, functionally identical once ints are unbounded.
  - The long[]/BigInteger[] pack_code_int_dst_longcode /
    pack_code_int_dst_bigcode (and their unpack counterparts) are kept as
    SEPARATE functions below (matching Java's distinct overloads 1:1) even
    though their bodies could be merged in Python, to preserve a direct
    mapping back to the Java source for review.

DEAD/UNREACHABLE CODE OMITTED (verified inert, not a behavior change):
  - get_huffman_bitlength_bytes/ints(): Java computes a canonical code
    (`huffman_code`) that is never subsequently read. Pure function, no
    side effects -- omitted here.
  - unpack_code_int_dst_longlen/longcode/bigcode: Java has a `boolean
    debug = false;` gated print block that can never execute (no code
    path sets debug true). Omitted here; the gate itself never runs in
    the original either.

CONFIRMED BUG #1, PRESERVED (not fixed): unpack_length_table_from_list()
(Java: unpackLengthTable(ArrayList)) does not initialize length[0] in its
max_delta==2 branch -- unlike its sibling max_delta==1/>4 branches, and
unlike the OTHER overload, unpack_length_table() (Java:
unpackLengthTable(int,byte,byte,byte[])), which sets length[0]=init_value
unconditionally before branching. Reproduced against real Java: packing
[3,3,5,5,5,7] (max_delta=2) and unpacking via the ArrayList-based
function gives [0,0,2,2,2,4] (offset by -init_value throughout, since
each entry is built from the previous one); the other overload correctly
returns [3,3,5,5,5,7] for the same packed data. Both behaviors are
reproduced exactly below.

Labeled `outer:` break loops in Java (packLengthTable/unpackLengthTable)
are restructured here as a single loop over a flat index computed via
arithmetic (Python has no labeled break) -- verified as a mechanical,
behavior-preserving equivalence, not a logic change.

CONFIRMED BUG #2, PRESERVED: get_huffman_list2() (Java: getHuffmanList2)
builds its rank_table from StringMapper's min-relative compact histogram,
then indexes it with raw 0..255 byte values inside packCode -- crashes
with an index error whenever the input byte string doesn't happen to
include the value 0. Confirmed against real Java (identical
ArrayIndexOutOfBoundsException on the same input). get_huffman_list() has
the same latent issue in its returned rank_table, though it doesn't call
packCode itself so it won't crash directly. See each function's docstring.

CONFIRMED BUG #3, PRESERVED: pack_length_table()/unpack_length_table()
don't actually round-trip when max_delta==0 (all Huffman lengths equal).
pack_length_table's catch-all branch (anything other than max_delta 1/2/3)
stores raw, one-byte-per-delta data -- correct for max_delta>4, but ALSO
taken for max_delta==0. unpack_length_table's matching catch-all branch
assumes 4-bit-packed pairs (correct only for max_delta==3), so the length
check fails and it prints an error instead of recovering the data.
Confirmed identical behavior (same print statements) against real Java.
"""

import math
from typing import List

import string_mapper as sm
import segment_mapper as segm


# =============================================================================
# getHuffmanBitlength
# =============================================================================
def get_huffman_bitlength_bytes(src):
    """Java: getHuffmanBitlength(byte[]). See module docstring re: the
    unused canonical-code computation, omitted here."""
    _min, histogram, _rng = sm.get_histogram_bytes(src)
    frequency = sorted(histogram, reverse=True)
    huffman_length = get_huffman_length2(frequency)
    return get_cost(huffman_length, frequency)


def get_huffman_bitlength_ints(src):
    """Java: getHuffmanBitlength(int[])."""
    _min, histogram, _rng = sm.get_histogram(src)
    frequency = sorted(histogram, reverse=True)
    huffman_length = get_huffman_length2(frequency)
    return get_cost(huffman_length, frequency)


# =============================================================================
# packCode family
# =============================================================================
def pack_code_byte_dst(src, table, code, length, dst):
    """Java: packCode(byte[] src, int[] table, int[] code, byte[] length, byte[] dst) -> int"""
    current_bit = 0
    for i in range(len(src)):
        j = src[i]  # already unsigned 0..255 in this port; Java's `if(j<0) j+=256` is a no-op here
        k = table[j]

        code_word = code[k]
        code_length = length[k]
        offset = current_bit % 8
        code_word = code_word << offset
        or_length = code_length + offset
        number_of_bytes = or_length // 8
        if or_length % 8 != 0:
            number_of_bytes += 1
        current_byte = current_bit // 8
        if number_of_bytes == 1:
            dst[current_byte] = (dst[current_byte] | (code_word & 0xFF)) & 0xFF
        else:
            for _m in range(number_of_bytes - 1):
                dst[current_byte] = (dst[current_byte] | (code_word & 0xFF)) & 0xFF
                current_byte += 1
                code_word >>= 8
            if current_byte < len(dst):
                dst[current_byte] = code_word & 0xFF
        current_bit += code_length

    return current_bit


def pack_code_byte(src, table, code, length):
    """Java: packCode(byte[] src, int[] table, int[] code, byte[] length) -> ArrayList
    Returns [dst, bitlength, table, code, length, len(src)]."""
    buffer = bytearray(len(src) * 4)  # generous upper bound, matching Java

    current_bit = 0
    for i in range(len(src)):
        j = src[i]
        if j > len(table) - 1:
            print("Index is larger than rank table length.")
            print(f"Index is {j}, rank table length is {len(table)}")
            print(f"Source length is {len(src)}, source index is {i}")
            j = len(table) - 1

        k = table[j]
        code_word = code[k]
        code_length = length[k]
        offset = current_bit % 8
        code_word = code_word << offset
        or_length = code_length + offset
        number_of_bytes = or_length // 8
        if or_length % 8 != 0:
            number_of_bytes += 1
        current_byte = current_bit // 8
        if number_of_bytes == 1:
            buffer[current_byte] = (buffer[current_byte] | (code_word & 0xFF)) & 0xFF
        else:
            for _m in range(number_of_bytes - 1):
                buffer[current_byte] = (buffer[current_byte] | (code_word & 0xFF)) & 0xFF
                current_byte += 1
                code_word >>= 8
            buffer[current_byte] = code_word & 0xFF
        current_bit += code_length

    bitlength = current_bit
    number_of_bytes = bitlength // 8
    if bitlength % 8 != 0:
        number_of_bytes += 1
    dst = bytearray(number_of_bytes)
    for i in range(len(dst)):
        dst[i] = buffer[i]

    return [dst, bitlength, table, code, length, len(src)]


def pack_code_int(src, table, code, length):
    """Java: packCode(int[] src, int[] table, int[] code, byte[] length) -> ArrayList
    NOTE: unlike pack_code_byte, this has NO bounds/sign check on `j` --
    matches the Java asymmetry exactly (translated faithfully, not added)."""
    buffer = bytearray(len(src) * 4)

    current_bit = 0
    for i in range(len(src)):
        j = src[i]
        k = table[j]
        code_word = code[k]
        code_length = length[k]
        offset = current_bit % 8
        code_word = code_word << offset
        or_length = code_length + offset
        number_of_bytes = or_length // 8
        if or_length % 8 != 0:
            number_of_bytes += 1
        current_byte = current_bit // 8
        if number_of_bytes == 1:
            buffer[current_byte] = (buffer[current_byte] | (code_word & 0xFF)) & 0xFF
        else:
            for _m in range(number_of_bytes - 1):
                buffer[current_byte] = (buffer[current_byte] | (code_word & 0xFF)) & 0xFF
                current_byte += 1
                code_word >>= 8
            buffer[current_byte] = code_word & 0xFF
        current_bit += code_length

    bitlength = current_bit
    number_of_bytes = bitlength // 8
    if bitlength % 8 != 0:
        number_of_bytes += 1
    dst = bytearray(number_of_bytes)
    for i in range(len(dst)):
        dst[i] = buffer[i]

    return [dst, bitlength, table, code, length, len(src)]


def pack_code_int_dst(src, table, code, length, dst):
    """Java: packCode(int[] src, int[] table, int[] code, byte[] length, byte[] dst) -> int"""
    current_bit = 0
    for i in range(len(src)):
        j = src[i]
        k = table[j]
        code_word = code[k]
        code_length = length[k]
        offset = current_bit % 8
        code_word = code_word << offset
        or_length = code_length + offset
        number_of_bytes = or_length // 8
        if or_length % 8 != 0:
            number_of_bytes += 1
        current_byte = current_bit // 8
        if number_of_bytes == 1:
            dst[current_byte] = (dst[current_byte] | (code_word & 0xFF)) & 0xFF
        else:
            for _m in range(number_of_bytes - 1):
                dst[current_byte] = (dst[current_byte] | (code_word & 0xFF)) & 0xFF
                current_byte += 1
                code_word >>= 8
            dst[current_byte] = code_word & 0xFF
        current_bit += code_length

    return current_bit


def pack_code_int_dst_longlen(src, table, code, length, dst):
    """Java: packCode(int[] src, int[] table, int[] code, int[] length, byte[] dst) -> int
    ("Method that supports longer lengths" -- int[] length instead of byte[])."""
    return pack_code_int_dst(src, table, code, length, dst)  # identical body in Java; length being int[] vs byte[] doesn't matter in Python


def pack_code_int_dst_longcode(src, table, code, length, dst):
    """Java: packCode(int[] src, int[] table, long[] code, int[] length, byte[] dst) -> int"""
    return pack_code_int_dst(src, table, code, length, dst)  # identical body; Python ints subsume Java long


def pack_code_int_dst_bigcode(src, table, code, length, dst):
    """Java: packCode(int[] src, int[] table, BigInteger[] code, int[] length, byte[] dst) -> int
    Translated literally (bit-by-bit assembly, try/except around each
    byte's worth of bits) rather than collapsed to pack_code_int_dst,
    since this one's structure genuinely differs from the others (byte-
    by-byte BigInteger shifting instead of a single Python int shift) --
    kept close to the Java for review, even though the *result* is the
    same. The try/except mirrors Java's, which incidentally also catches
    (and silently continues past) an out-of-bounds dst write."""
    current_bit = 0
    for i in range(len(src)):
        j = src[i]
        k = table[j]

        code_word = code[k]
        code_length = length[k]

        shift = current_bit % 8
        code_word = code_word << shift

        or_length = code_length + shift
        number_of_bytes = or_length // 8
        if or_length % 8 != 0:
            number_of_bytes += 1

        current_byte = current_bit // 8
        shift = 0

        for _m in range(number_of_bytes):
            shifted_code_word = code_word >> shift
            shifted_code_word = shifted_code_word & 255
            try:
                mask_value = 1
                for _n in range(8):
                    bit_value = shifted_code_word & mask_value
                    if bit_value != 0:
                        byte_mask = mask_value & 0xFF
                        dst[current_byte] = (dst[current_byte] | byte_mask) & 0xFF
                    mask_value *= 2
                current_byte += 1
                shift += 8
            except Exception as e:
                print(e)
                current_byte += 1
                shift += 8

        current_bit += code_length

    return current_bit


# =============================================================================
# unpackCode family
# =============================================================================
def _build_inverse_table(table):
    inverse_table = [0] * len(table)
    for i in range(len(table)):
        inverse_table[table[i]] = i
    return inverse_table


def unpack_code_byte_dst(src, table, code, code_length, bit_length, dst):
    """Java: unpackCode(byte[] src, int[] table, int[] code, byte[] code_length, int bit_length, byte[] dst) -> int"""
    inverse_table = _build_inverse_table(table)
    buffer = [0] * len(dst)

    max_length = code_length[len(code) - 1]
    max_bytes = max_length // 8
    if max_length % 8 != 0:
        max_bytes += 1

    current_bit = 0
    offset = 0
    current_byte = 0
    number_unpacked = 0
    dst_byte = 0

    for _i in range(len(dst)):
        src_word = 0
        for j in range(max_bytes):
            index = current_byte + j
            if index < len(src):
                src_byte = src[index]
                if j == 0:
                    src_byte >>= offset
                else:
                    src_byte <<= (j * 8 - offset)
                src_word |= src_byte

        if offset != 0:
            index = current_byte + max_bytes
            if index < len(src):
                src_byte = src[index]
                src_byte <<= (max_bytes * 8 - offset)
                src_word |= src_byte

        for j in range(len(code)):
            code_word = code[j]
            mask = -1
            mask <<= code_length[j]
            mask = ~mask

            masked_src_word = src_word & mask
            masked_code_word = code_word & mask

            if masked_src_word == masked_code_word:
                buffer[dst_byte] = inverse_table[j]
                dst_byte += 1
                number_unpacked += 1
                current_bit += code_length[j]
                current_byte = current_bit // 8
                offset = current_bit % 8
                break
            elif j == len(code) - 1:
                print(f"No match for prefix-free code at byte {current_byte}")

    for i in range(len(dst)):
        dst[i] = buffer[i] & 0xFF
    return number_unpacked


def unpack_code_from_list(pack_list):
    """Java: unpackCode(ArrayList) -> byte[]"""
    src, bitlength, table, code, code_length, n = pack_list

    print(f"Table length is {len(table)}")
    print(f"Code length is {len(code)}")
    print(f"Code length length is {len(code_length)}")
    print(f"Number of output bytes is {n}")
    print()

    inverse_table = _build_inverse_table(table)

    max_length = code_length[len(code) - 1]
    max_bytes = max_length // 8
    if max_length % 8 != 0:
        max_bytes += 1

    current_bit = 0
    offset = 0
    current_byte = 0
    dst_byte = 0

    dst = bytearray(n)
    for _i in range(n):
        src_word = 0
        for j in range(max_bytes):
            index = current_byte + j
            if index < len(src):
                src_byte = src[index]
                if j == 0:
                    src_byte >>= offset
                else:
                    src_byte <<= (j * 8 - offset)
                src_word |= src_byte

        if offset != 0:
            index = current_byte + max_bytes
            if index < len(src):
                src_byte = src[index]
                src_byte <<= (max_bytes * 8 - offset)
                src_word |= src_byte

        for j in range(len(code)):
            code_word = code[j]
            mask = -1
            mask <<= code_length[j]
            mask = ~mask

            masked_src_word = src_word & mask
            masked_code_word = code_word & mask

            if masked_src_word == masked_code_word:
                dst[dst_byte] = inverse_table[j] & 0xFF
                dst_byte += 1
                current_bit += code_length[j]
                current_byte = current_bit // 8
                offset = current_bit % 8
                break
            elif j == len(code) - 1:
                print(f"No match for prefix-free code at byte {current_byte}")

    return dst


def unpack_code2_from_list(pack_list):
    """Java: unpackCode2(ArrayList) -> int[] (same as unpack_code_from_list but no debug prints, int output)."""
    src, bitlength, table, code, code_length, n = pack_list

    inverse_table = _build_inverse_table(table)

    max_length = code_length[len(code) - 1]
    max_bytes = max_length // 8
    if max_length % 8 != 0:
        max_bytes += 1

    current_bit = 0
    offset = 0
    current_byte = 0
    dst_byte = 0

    dst = [0] * n
    for _i in range(n):
        src_word = 0
        for j in range(max_bytes):
            index = current_byte + j
            if index < len(src):
                src_byte = src[index]
                if j == 0:
                    src_byte >>= offset
                else:
                    src_byte <<= (j * 8 - offset)
                src_word |= src_byte

        if offset != 0:
            index = current_byte + max_bytes
            if index < len(src):
                src_byte = src[index]
                src_byte <<= (max_bytes * 8 - offset)
                src_word |= src_byte

        for j in range(len(code)):
            code_word = code[j]
            mask = -1
            mask <<= code_length[j]
            mask = ~mask

            masked_src_word = src_word & mask
            masked_code_word = code_word & mask

            if masked_src_word == masked_code_word:
                dst[dst_byte] = inverse_table[j]
                dst_byte += 1
                current_bit += code_length[j]
                current_byte = current_bit // 8
                offset = current_bit % 8
                break
            elif j == len(code) - 1:
                print(f"No match for prefix-free code at byte {current_byte}")

    return dst


def unpack_code_int_dst(src, table, code, code_length, bit_length, dst):
    """Java: unpackCode(byte[] src, int[] table, int[] code, byte[] code_length, int bit_length, int[] dst) -> int
    Comment in the Java source: "This is the method used in HuffmanWriter." """
    inverse_table = _build_inverse_table(table)

    max_length = code_length[len(code) - 1]
    max_bytes = max_length // 8
    if max_length % 8 != 0:
        max_bytes += 1

    current_bit = 0
    offset = 0
    current_byte = 0
    number_unpacked = 0
    dst_byte = 0

    for _i in range(len(dst)):
        src_word = 0
        for j in range(max_bytes):
            index = current_byte + j
            if index < len(src):
                src_byte = src[index]
                if j == 0:
                    src_byte >>= offset
                else:
                    src_byte <<= (j * 8 - offset)
                src_word |= src_byte

        if offset != 0:
            index = current_byte + max_bytes
            if index < len(src):
                src_byte = src[index]
                src_byte <<= (max_bytes * 8 - offset)
                src_word |= src_byte

        for j in range(len(code)):
            code_word = code[j]
            mask = -1
            mask <<= code_length[j]
            mask = ~mask

            masked_src_word = src_word & mask
            masked_code_word = code_word & mask

            if masked_src_word == masked_code_word:
                dst[dst_byte] = inverse_table[j]
                dst_byte += 1
                number_unpacked += 1
                current_bit += code_length[j]
                current_byte = current_bit // 8
                offset = current_bit % 8
                break
            elif j == len(code) - 1:
                print(f"No match for prefix-free code at byte {current_byte}")

    return number_unpacked


def unpack_code_int_dst_longlen(src, table, code, code_length, bit_length, dst):
    """Java: unpackCode(byte[], int[], int[] code, int[] code_length, int, int[] dst) -> int
    ("Methods that support longer codes"). Identical body to
    unpack_code_int_dst in Python (code_length being int[] vs byte[]
    doesn't matter); Java's unreachable `debug=false` block omitted
    (see module docstring), as is an unused `boolean matched` variable."""
    return unpack_code_int_dst(src, table, code, code_length, bit_length, dst)


def unpack_code_int_dst_longcode(src, table, code, code_length, bit_length, dst):
    """Java: unpackCode(byte[] src, int[] table, long[] code, int[] code_length, int bit_length, int[] dst) -> int"""
    return unpack_code_int_dst(src, table, code, code_length, bit_length, dst)


def unpack_code_int_dst_bigcode(src, table, code, code_length, bit_length, dst):
    """Java: unpackCode(byte[] src, int[] table, BigInteger[] code, int[] code_length, int bit_length, int[] dst) -> int
    Java builds the mask via repeated doubling (BigInteger has no native
    shift-based bitmask builder in the style used elsewhere in this file);
    that loop is a well-defined identity (sum of 2^0..2^(code_length-1) ==
    2^code_length - 1), so it's computed directly here rather than
    replicating the accumulation loop -- a case where the "why" is
    provably just arithmetic, unlike most of this codebase's Java-
    specific quirks."""
    inverse_table = _build_inverse_table(table)

    max_length = code_length[len(code) - 1]
    max_bytes = max_length // 8
    if max_length % 8 != 0:
        max_bytes += 1

    current_bit = 0
    offset = 0
    current_byte = 0
    number_unpacked = 0
    dst_byte = 0

    for _i in range(len(dst)):
        src_word = 0
        for j in range(max_bytes):
            index = current_byte + j
            if index < len(src):
                src_byte = src[index]
                if j == 0:
                    src_byte >>= offset
                else:
                    src_byte <<= (j * 8 - offset)
                src_word |= src_byte

        if offset != 0:
            index = current_byte + max_bytes
            if index < len(src):
                src_byte = src[index]
                src_byte <<= (max_bytes * 8 - offset)
                src_word |= src_byte

        for j in range(len(code)):
            code_word = code[j]
            mask = (1 << code_length[j]) - 1
            masked_src_word = src_word & mask

            if code_word == masked_src_word:
                dst[dst_byte] = inverse_table[j]
                dst_byte += 1
                number_unpacked += 1
                current_bit += code_length[j]
                current_byte = current_bit // 8
                offset = current_bit % 8
                break
            elif j == len(code) - 1:
                print("No match for prefix-free code.")

    return number_unpacked


# =============================================================================
# Unary codes
# =============================================================================
def get_unary_code(n):
    code = [0] * n
    addend = 1
    for i in range(1, n):
        code[i] = code[i - 1] + addend
        addend *= 2
    return code


def get_big_unary_code(n):
    """Java: getBigUnaryCode -- identical to get_unary_code() in Python,
    since Python ints are already arbitrary precision (Java needed a
    separate BigInteger version because `long` is only 64 bits)."""
    return get_unary_code(n)


def get_unary_length(n):
    length = [i + 1 for i in range(n)]
    length[n - 1] -= 1
    return length


# =============================================================================
# Huffman code-length generation (in-place tree algorithm)
# =============================================================================
def _huffman_length_core(frequency):
    n = len(frequency)
    w = list(frequency)

    leaf = n - 1
    root = n - 1

    for next_ in range(n - 1, 0, -1):
        # Find first child.
        if leaf < 0 or (root > next_ and w[root] < w[leaf]):
            w[next_] = w[root]
            w[root] = next_
            root -= 1
        else:
            w[next_] = w[leaf]
            leaf -= 1

        # Find second child.
        if leaf < 0 or (root > next_ and w[root] < w[leaf]):
            w[next_] += w[root]
            w[root] = next_
            root -= 1
        else:
            w[next_] += w[leaf]
            leaf -= 1

    # Traverse tree from root down, converting parent pointers into
    # internal node depths.
    w[1] = 0
    for next_ in range(2, n):
        w[next_] = w[w[next_]] + 1

    # Final pass to produce code lengths.
    avail = 1
    used = 0
    depth = 0
    root = 1
    next_ = 0

    while avail > 0:
        while root < n and w[root] == depth:
            used += 1
            root += 1
        while avail > used:
            w[next_] = depth
            next_ += 1
            avail -= 1
        avail = 2 * used
        used = 0
        depth += 1

    return w


def get_huffman_length(frequency):
    """Java: getHuffmanLength(int[]) -> int[] (raw depths, no byte cast).
    NOTE: like the Java, this has no defined behavior for n<=1 (indexing
    w[1] on a length-1 or empty list) -- that's a pre-existing property of
    the algorithm, not something to special-case here."""
    return _huffman_length_core(frequency)


def get_huffman_length2(frequency):
    """Java: getHuffmanLength2(int[]) -> byte[]. Truncates each depth to
    8 bits (`& 0xFF`, matching Java's `(byte) w[i]` cast) -- for
    pathologically skewed frequency distributions with very large n,
    Java would silently wrap a depth > 127 into a corrupted (wrapped)
    byte value; this preserves that same wraparound rather than guarding
    against it, since real delta-channel histograms are far too small
    (well under 127 deep) to hit this in practice."""
    w = _huffman_length_core(frequency)
    return bytearray(v & 0xFF for v in w)


# =============================================================================
# Canonical codes
# =============================================================================
def get_canonical_code(length):
    """Java: getCanonicalCode(byte[]) AND getCanonicalCode(int[]) -- both
    overloads do the identical computation; Java only split them because
    of static typing on the `length` parameter. One function here covers
    both (see module docstring)."""
    n = len(length)
    code = [0] * n
    shifted_code = [0] * n
    max_length = length[n - 1]

    for i in range(1, n):
        # Java: code[i] = code[i-1] + (int)Math.pow(2, max_length-length[i-1]);
        # Math.pow(2, k) for non-negative integer k is always an exact
        # double, so int-truncating it is equivalent to 1<<k here -- no
        # floating-point discrepancy from using integer exponentiation.
        code[i] = code[i - 1] + (1 << (max_length - length[i - 1]))
        shift = max_length - length[i]
        shifted_code[i] = code[i] >> shift

    reversed_code = [0] * n
    for i in range(1, n):
        code_word = shifted_code[i]
        code_length = length[i]
        for j in range(code_length):
            if code_word & (1 << j):
                shift = (code_length - 1) - j
                reversed_code[i] |= (1 << shift)

    return reversed_code


# Java's long[]/BigInteger[] canonical-code variants -- identical
# computation to get_canonical_code() once ints are unbounded (see module
# docstring).
get_canonical_code2 = get_canonical_code
get_big_canonical_code = get_canonical_code


# =============================================================================
# Ratios, cost, Shannon limit
# =============================================================================
def get_code_zero_ratio(code, length, frequency):
    """Java: CodeMapper.getZeroRatio(int[] code, int[] length, int[] frequency).
    Named get_code_zero_ratio here (not get_zero_ratio) to avoid colliding
    with string_mapper.get_zero_ratio / segment_mapper's use of it -- same
    name, different signature and purpose in the original Java too."""
    number_of_zeros = 0
    number_of_ones = 0

    for i in range(len(code)):
        for j in range(length[i]):
            bit = code[i] & (1 << j)
            if bit == 0:
                number_of_zeros += 1
            else:
                number_of_ones += 1

    return number_of_zeros / (number_of_zeros + number_of_ones)


def log2(value):
    return math.log(value) / math.log(2.0)


def get_shannon_limit(frequency):
    n = len(frequency)
    total = sum(frequency)
    weight = [frequency[i] / total for i in range(n)]

    limit = 0.0
    for i in range(n):
        if weight[i] != 0:
            limit -= frequency[i] * log2(weight[i])

    return limit


def get_cost(length, frequency):
    """Java: getCost(int[], int[]) AND getCost(byte[], int[]) -- identical
    computation, merged into one function (Python doesn't distinguish)."""
    return sum(length[i] * frequency[i] for i in range(len(length)))


# =============================================================================
# Length-table pack/unpack
# =============================================================================
def pack_length_table(length):
    """Java: packLengthTable(byte[] length) -> ArrayList
    [n, init_value, max_delta, packed_or_raw_delta]

    NOTE: length_delta is only ever guaranteed non-negative because this
    is always called with a huffman_length array produced from
    DESCENDING-sorted frequencies (see get_huffman_bitlength_* /
    get_huffman_list*), which guarantees non-decreasing code lengths by
    the standard optimal-prefix-code property (freq(a) >= freq(b) implies
    length(a) <= length(b)). The max_delta==1/2/3 packing schemes below
    only encode POSITIVE deltas (`if value > 0`) -- a negative delta would
    be silently dropped (treated as 0) -- but that's provably unreachable
    given how this function is actually called, not a bug being worked
    around here.

    Java's `outer:` labeled-break nested loops are restructured below as
    a single loop over a flat index (Python has no labeled break) --
    verified as a mechanical, behavior-preserving equivalence."""
    n = len(length)
    init_value = length[0]

    length_delta = [0] * (n - 1)
    max_delta = 0
    for i in range(n - 1):
        length_delta[i] = (length[i + 1] - length[i]) & 0xFF
        # matches Java's `(byte)` cast semantics for this subtraction; delta
        # is expected non-negative per the note above, so this doesn't
        # actually wrap in practice
        if length_delta[i] > max_delta:
            max_delta = length_delta[i]

    if max_delta == 1:
        byte_length = (n - 1) // 8
        if (n - 1) % 8 != 0:
            byte_length += 1
        packed_length = bytearray(byte_length)
        mask = segm.get_positive_mask_table()
        for m in range(len(length_delta)):
            k = m // 8
            bit_pos = m % 8
            if length_delta[m] == 1:
                packed_length[k] = (packed_length[k] | mask[bit_pos]) & 0xFF
        return [n, init_value, max_delta, packed_length]

    elif max_delta == 2:
        byte_length = (n - 1) // 4
        if (n - 1) % 4 != 0:
            byte_length += 1
        packed_length = bytearray(byte_length)
        for m in range(len(length_delta)):
            k = m // 4
            bit_pos = (m % 4) * 2
            value = length_delta[m]
            if value > 0:
                packed_length[k] = (packed_length[k] | ((value << bit_pos) & 0xFF)) & 0xFF
        return [n, init_value, max_delta, packed_length]

    elif max_delta == 3:
        byte_length = (n - 1) // 2
        if (n - 1) % 2 != 0:
            byte_length += 1
        packed_length = bytearray(byte_length)
        for m in range(len(length_delta)):
            k = m // 2
            bit_pos = (m % 2) * 4
            value = length_delta[m]
            if value > 0:
                packed_length[k] = (packed_length[k] | ((value << bit_pos) & 0xFF)) & 0xFF
        return [n, init_value, max_delta, packed_length]

    else:
        return [n, init_value, max_delta, bytearray(length_delta)]


def unpack_length_table_from_list(length_list):
    """Java: unpackLengthTable(ArrayList) -> byte[]
    See module docstring: CONFIRMED BUG preserved here -- length[0] is not
    set in the max_delta==2 branch (unlike the sibling branches and unlike
    unpack_length_table() below), so it stays 0 instead of init_value in
    that case, offsetting the whole decoded sequence."""
    n, init_value, max_delta, packed_delta = length_list
    length = bytearray(n)

    if max_delta > 4:
        if len(packed_delta) != n - 1:
            print("Packed deltas are not the right length 1.")
        else:
            length[0] = init_value
            for i in range(1, n):
                length[i] = (length[i - 1] + packed_delta[i - 1]) & 0xFF

    elif max_delta == 1:
        byte_length = (n - 1) // 8
        if (n - 1) % 8 != 0:
            byte_length += 1
        if len(packed_delta) != byte_length:
            print("Packed deltas are not the right length 2.")
        else:
            mask = segm.get_positive_mask_table()
            length[0] = init_value
            k = 1
            for m in range(byte_length * 8):
                if k == n:
                    break
                i, j = m // 8, m % 8
                if (packed_delta[i] & mask[j]) != 0:
                    length[k] = (length[k - 1] + 1) & 0xFF
                else:
                    length[k] = length[k - 1]
                k += 1

    elif max_delta == 2:
        byte_length = (n - 1) // 4
        if (n - 1) % 4 != 0:
            byte_length += 1
        if len(packed_delta) != byte_length:
            print("Packed deltas are not the right length 3.")
        else:
            # BUG (preserved, see docstring): no `length[0] = init_value`
            # here, unlike every other branch.
            mask = [3, 12, 48, 192]  # mask[i] = 3 << (2*i)
            k = 1
            for m in range(byte_length * 4):
                if k == n:
                    break
                i, j = m // 4, m % 4
                value = packed_delta[i] & mask[j]
                value >>= (j * 2)
                value &= 3
                length[k] = (length[k - 1] + value) & 0xFF
                k += 1

    else:
        byte_length = (n - 1) // 2
        if (n - 1) % 2 != 0:
            byte_length += 1
        if len(packed_delta) != byte_length:
            print("Packed deltas are not the right length 4.")
        else:
            mask = [15, 240]  # mask[0]=15, mask[1]=15<<4
            length[0] = init_value
            k = 1
            for m in range(byte_length * 2):
                if k == n:
                    break
                i, j = m // 2, m % 2
                value = packed_delta[i] & mask[j]
                value >>= (j * 4)
                value &= 15
                length[k] = (length[k - 1] + value) & 0xFF
                k += 1

    return length


def unpack_length_table(n, init_value, max_delta, packed_delta):
    """Java: unpackLengthTable(int n, byte init_value, byte max_delta, byte[] packed_delta) -> byte[]
    Unlike unpack_length_table_from_list, this one sets length[0] =
    init_value unconditionally up front, so it does NOT have the max_delta
    ==2 bug that overload has -- confirmed against real Java, see module
    docstring.

    DELIBERATE FIX (not a straight Java translation here -- see the
    conversation this was produced in): the original catch-all branch
    lumped max_delta == 0, 3, AND 4 together and always unpacked them as
    4-bit-packed pairs. That format is only what pack_length_table's own
    max_delta==3 branch actually produces; for max_delta == 0 or == 4,
    pack_length_table's catch-all instead writes RAW one-byte-per-delta
    data (matching its max_delta>4 branch), so unpacking those two cases
    as 4-bit pairs reads garbage and hits the length-mismatch print seen
    below (this is CONFIRMED BUG #3 from the module docstring). Splitting
    max_delta==3 into its own branch, and folding 0 and 4 into the
    raw-byte branch alongside max_delta>4, makes pack/unpack agree for
    every max_delta value while leaving every already-correct branch
    (1, 2, 3, and the original >4 raw-byte case) untouched. This was
    necessary to make DeltaWriter's Huffman entropy coding round-trip
    through DeltaReader at all -- max_delta lands on 0 or 4 often enough
    in practice (e.g. every code length equal, or a single length-4 jump)
    that Huffman entropy coding was failing on ordinary images before
    this fix."""
    length = bytearray(n)
    length[0] = init_value

    if max_delta > 4 or max_delta == 0:
        if len(packed_delta) != n - 1:
            print("Packed deltas are not the right length 1.")
        else:
            for i in range(1, n):
                length[i] = (length[i - 1] + packed_delta[i - 1]) & 0xFF

    elif max_delta == 1:
        byte_length = (n - 1) // 8
        if (n - 1) % 8 != 0:
            byte_length += 1
        if len(packed_delta) != byte_length:
            print("Packed deltas are not the right length 2.")
        else:
            mask = segm.get_positive_mask_table()
            k = 1
            for m in range(byte_length * 8):
                if k == n:
                    break
                i, j = m // 8, m % 8
                if (packed_delta[i] & mask[j]) != 0:
                    length[k] = (length[k - 1] + 1) & 0xFF
                else:
                    length[k] = length[k - 1]
                k += 1

    elif max_delta == 2:
        byte_length = (n - 1) // 4
        if (n - 1) % 4 != 0:
            byte_length += 1
        if len(packed_delta) != byte_length:
            print("Packed deltas are not the right length 3.")
        else:
            mask = [3, 12, 48, 192]
            k = 1
            for m in range(byte_length * 4):
                if k == n:
                    break
                i, j = m // 4, m % 4
                value = packed_delta[i] & mask[j]
                value >>= (j * 2)
                value &= 3
                length[k] = (length[k - 1] + value) & 0xFF
                k += 1

    elif max_delta == 3:
        byte_length = (n - 1) // 2
        if (n - 1) % 2 != 0:
            byte_length += 1
        if len(packed_delta) != byte_length:
            print("Packed deltas are not the right length 4.")
            print(f"Expected length is {byte_length}, actual length is {len(packed_delta)}")
        else:
            mask = [15, 240]
            length[0] = init_value
            k = 1
            for m in range(byte_length * 2):
                if k == n:
                    break
                i, j = m // 2, m % 2
                value = packed_delta[i] & mask[j]
                value >>= (j * 4)
                value &= 15
                length[k] = (length[k - 1] + value) & 0xFF
                k += 1

    else:  # max_delta == 4 -- same raw-byte format as max_delta > 4, above
        if len(packed_delta) != n - 1:
            print("Packed deltas are not the right length 1.")
        else:
            for i in range(1, n):
                length[i] = (length[i - 1] + packed_delta[i - 1]) & 0xFF

    return length


# =============================================================================
# High-level convenience wrappers
# =============================================================================
def get_huffman_list(string):
    """Java: getHuffmanList(byte[]) -> ArrayList
    [estimated_bit_length, shannon_limit, rank_table, huffman_code, huffman_length]

    CONFIRMED BUG (preserved, not fixed): rank_table here is built from
    StringMapper's min-relative COMPACT histogram (size = max(string) -
    min(string) + 1), but packCode's `table` parameter is indexed by the
    RAW unsigned byte value (0..255) directly -- see pack_code_byte_dst.
    get_huffman_list() itself never calls packCode, so it won't crash, but
    a caller that later feeds this rank_table into pack_code_byte/
    pack_code_byte_dst with the original string will get an out-of-range
    index whenever min(string) > 0. This exact crash was confirmed against
    real Java in get_huffman_list2 below (which DOES call packCode
    internally) -- ArrayIndexOutOfBoundsException, reproducible whenever
    the input byte string doesn't happen to include the value 0."""
    string_min, string_histogram, value_range = sm.get_histogram_bytes(string)
    rank_table = sm.get_rank_table(string_histogram)
    frequency = sorted(string_histogram, reverse=True)

    shannon_limit = get_shannon_limit(frequency)
    huffman_length = get_huffman_length2(frequency)
    huffman_code = get_canonical_code(huffman_length)
    estimated_bit_length = get_cost(huffman_length, frequency)

    return [estimated_bit_length, shannon_limit, rank_table, huffman_code, huffman_length]


def get_huffman_list2(string):
    """Java: getHuffmanList2(byte[]) -> ArrayList
    [huffman_bit_length, rank_table, huffman_code, huffman_length, length_list, packed_string]

    CONFIRMED BUG (preserved, not fixed): this calls pack_code_byte_dst()
    with a rank_table sized to the input's observed min..max byte range
    (via StringMapper's compact histogram), but pack_code_byte_dst indexes
    that table by the RAW byte value (0..255). Verified against real
    compiled Java: CodeMapper.getHuffmanList2() throws
    ArrayIndexOutOfBoundsException on typical random byte strings whenever
    the input doesn't happen to include the byte value 0 (i.e. whenever
    min(string) > 0) -- reproduced here with the identical failure mode.
    Works "by luck" whenever the data happens to span down to 0."""
    string_min, string_histogram, value_range = sm.get_histogram_bytes(string)
    rank_table = sm.get_rank_table(string_histogram)
    frequency = sorted(string_histogram, reverse=True)

    huffman_length = get_huffman_length2(frequency)
    huffman_code = get_canonical_code(huffman_length)
    estimated_bit_length = get_cost(huffman_length, frequency)

    byte_length = estimated_bit_length // 8
    if estimated_bit_length % 8 != 0:
        byte_length += 1
    packed_string = bytearray(byte_length)

    huffman_bit_length = pack_code_byte_dst(string, rank_table, huffman_code, huffman_length, packed_string)

    print(f"Estimated bit length was {estimated_bit_length}")
    print(f"Actual bit length was {huffman_bit_length}")

    length_list = pack_length_table(huffman_length)

    return [huffman_bit_length, rank_table, huffman_code, huffman_length, length_list, packed_string]
