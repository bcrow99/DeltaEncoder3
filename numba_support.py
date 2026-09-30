"""
numba_support.py -- the optional Numba JIT used by every module here.

With Numba installed, @njit functions compile to machine code (cached on
disk after the first run, and releasing the GIL so the per-channel threads
really run in parallel). Without it, @njit is a no-op: everything still
works, identically, just much more slowly.

Java integer semantics the ports rely on:
  - Java's int division truncates toward zero; Python's // floors. jdiv()
    below truncates. Where both operands are known to be non-negative the
    code uses // directly.
  - >> is an arithmetic shift in both languages.
"""

try:
    from numba import njit as _njit
    NUMBA_AVAILABLE = True

    def njit(*args, **kwargs):
        kwargs.setdefault("cache", True)
        kwargs.setdefault("nogil", True)
        if len(args) == 1 and callable(args[0]) and len(kwargs) == 2:
            return _njit(**kwargs)(args[0])
        return _njit(*args, **kwargs)
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda fn: fn


@njit
def jdiv(a, b):
    """Java int division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q
