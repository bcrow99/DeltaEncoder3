#!/usr/bin/env python3
"""
segment_steer_demo.py -- SegmentSteerDemo.java for Python.

Compresses a square (dimension x dimension) segment of symbol data with
probabilistic_mapper's target-steering scheme, trying several candidate
target fractions at once (not every combination of histogram, sequence and
target can be steered exactly) and taking whichever succeeds first, then
decodes the result and checks it matches.

    python3 segment_steer_demo.py [dimension] [budget seconds] [max workers]
        (defaults 8, 60 and 16, as in the Java)

The Java races the targets on threads. Python threads can't run the pure
Python search in parallel (the GIL), so this races them on separate
processes instead; the result is the same. The segment and histogram are
built with the same java.util.Random sequence as the Java, so both programs
compress the same data.

Which target wins is a race, so it can differ from run to run (in the Java
too); each target's own search is deterministic.
"""

import math
import multiprocessing as mp
import os
import sys
import time
from fractions import Fraction

import fraction_mapper as fm
import probabilistic_mapper as pm


class CompressResult:
    __slots__ = ("success", "target", "orderings", "backtracks", "elapsed_millis", "cost_bits")

    def __init__(self, success, target, orderings, backtracks, elapsed_millis, cost_bits):
        self.success = success
        self.target = target
        self.orderings = orderings
        self.backtracks = backtracks
        self.elapsed_millis = elapsed_millis
        self.cost_bits = cost_bits


def _attempt(args):
    """One target's search (runs in a worker process). The target travels as
    (n, d); the deadline is a time.monotonic_ns() value, which is the same
    clock in every process."""
    segment, freq, (n, d), max_backtracks, deadline_nanos = args
    r = pm.steer_encode(segment, freq, Fraction(n, d), max_backtracks, deadline_nanos)
    if r is None:
        return (n, d), None
    return (n, d), (r.orderings, r.backtracks)


def compress_segment(segment, freq, candidate_targets, max_workers, budget_seconds, max_backtracks_per_attempt):
    """Races the candidate targets (BigFractions) across up to max_workers
    processes, each with its own backtracking search, stopping as soon as one
    succeeds or the time budget runs out."""
    n_workers = max(1, min(max_workers, os.cpu_count() or 1))
    n_workers = min(n_workers, max(1, len(candidate_targets)))

    start = time.monotonic_ns()
    deadline = start + int(budget_seconds * 1_000_000_000)
    winner = None
    pool = mp.Pool(n_workers)
    try:
        pending = [pool.apply_async(_attempt, ((segment, freq, (t.n, t.d), max_backtracks_per_attempt, deadline),))
                   for t in candidate_targets]
        while pending:
            for job in [j for j in pending if j.ready()]:
                pending.remove(job)
                try:
                    target, found = job.get()
                    if found is not None:
                        winner = (target, found)
                except Exception as e:
                    # A real failure (an unreachable target just returns None):
                    # report it and keep waiting on the others.
                    print("  [attempt threw unexpectedly] %r" % e)
            if winner is not None or time.monotonic_ns() > deadline:
                break
            time.sleep(0.02)
    finally:
        pool.terminate()
        pool.join()

    elapsed_millis = (time.monotonic_ns() - start) // 1_000_000
    if winner is None:
        return CompressResult(False, None, None, -1, elapsed_millis, -1)
    (n, d), (orderings, backtracks) = winner
    return CompressResult(True, fm.BigFraction(n, d), orderings, backtracks, elapsed_millis,
                          pm.serialized_ordering_bits(orderings))


def decompress_segment(orderings, freq, target, n):
    return pm.steer_decode(orderings, freq, target.to_fraction(), n)


def java_round(x):
    """Math.round(double): nearest, halves up."""
    return int(math.floor(x + 0.5))


def java_double(x):
    """Double.toString for the values this demo prints."""
    if x != x:
        return "NaN"
    if x in (float("inf"), float("-inf")):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0 or 1e-3 <= abs(x) < 1e7:
        s = repr(float(x))
        return s if ("." in s or "e" in s) else s + ".0"
    mantissa, exponent = ("%r" % x).split("e")
    if "." not in mantissa:
        mantissa += ".0"
    return "%sE%d" % (mantissa, int(exponent))


def build_segment(n, alphabet_size=16, seed=11):
    """A delta-like (geometric-ish) symbol sequence and its histogram, built
    exactly as the Java builds it, from java.util.Random(seed)."""
    rng = pm.JavaRandom(seed)
    weights = [max(1, java_round(200 * math.pow(0.6, k))) for k in range(alphabet_size)]
    total_weight = sum(weights)
    pool = []
    for k in range(alphabet_size):
        count = max(1, java_round(weights[k] / total_weight * n))
        pool.extend([k] * count)
    while len(pool) < n:
        pool.append(rng.next_int(alphabet_size))
    while len(pool) > n:
        pool.pop()
    for i in range(len(pool) - 1, 0, -1):          # Collections.shuffle(list, rng)
        j = rng.next_int(i + 1)
        pool[i], pool[j] = pool[j], pool[i]
    segment = pool[:n]
    freq = [0] * alphabet_size
    for v in segment:
        freq[v] += 1
    return segment, freq


def main(argv):
    dimension = int(argv[1]) if len(argv) >= 2 else 8
    budget_seconds = float(argv[2]) if len(argv) >= 3 else 60.0
    max_workers = int(argv[3]) if len(argv) >= 4 else 16

    n = dimension * dimension
    print("Segment dimension: %dx%d (%d symbols)" % (dimension, dimension, n))
    print("Time budget: %ss   max workers: %d   available cores: %d"
          % (java_double(budget_seconds), max_workers, os.cpu_count() or 1))

    segment, freq = build_segment(n)
    print("Alphabet size (used): %d" % sum(1 for f in freq if f > 0))
    print("Histogram: [" + ", ".join(str(f) for f in freq) + "]")

    # Candidate targets: simple fractions (denominators up to 20).
    numer = [1, 3, 1, 2, 1, 3, 2, 3, 4, 1]
    denom = [20, 20, 5, 5, 2, 5, 3, 4, 5, 4]
    candidates = [fm.BigFraction.of(a, b) for a, b in zip(numer, denom)]

    print("\nTrying %d candidate targets across up to %d workers..." % (len(candidates), min(max_workers, len(candidates))))

    t0 = time.monotonic_ns()
    result = compress_segment(segment, freq, candidates, max_workers, budget_seconds, 5_000_000)
    t1 = time.monotonic_ns()

    if not result.success:
        print("\nNo candidate target succeeded within the time budget (%d ms elapsed)." % ((t1 - t0) // 1_000_000))
        return

    print("\nSUCCESS")
    print("  target used:      %s  (%s)" % (result.target, java_double(result.target.to_double())))
    print("  backtracks:        %d" % result.backtracks)
    print("  elapsed:           %d ms" % result.elapsed_millis)
    print("  serialized cost:   %.1f bits (%.1f bytes)" % (result.cost_bits, result.cost_bits / 8))
    print("  original size:     %d bits (%d bytes, if 1 byte/symbol)" % (n * 8, n))
    print("  overhead ratio:    %.2fx" % (result.cost_bits / (n * 8)))

    decoded = decompress_segment(result.orderings, freq, result.target, n)
    matches = decoded == segment
    print("  decoded matches original exactly: " + ("true" if matches else "false"))
    if not matches:
        print("  MISMATCH -- first difference at index: ")
        for i in range(n):
            if decoded[i] != segment[i]:
                print("    index %d: expected %d got %d" % (i, segment[i], decoded[i]))
                break


if __name__ == "__main__":
    main(sys.argv)
