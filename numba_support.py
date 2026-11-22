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

    # Numba counts the references to an array each time one is passed to a
    # function, even an inlined one, with an atomic instruction: some 16 ns
    # per array per call, which in a loop calling helpers a pixel at a time
    # can be most of its time. njit_no_refs compiles a function without the
    # counting (Numba's _nrt option): it may not make arrays (or call a
    # function that does), only work in the ones it is given; the helpers it
    # calls with inline="always" are compiled into it, so are counted no
    # more either. Numba versions without the option get plain njit.
    try:
        from numba.core.cpu import CPUTargetOptions
        _NO_REFS = hasattr(CPUTargetOptions, "_nrt")
    except ImportError:
        _NO_REFS = False

    def njit_no_refs(fn):
        if _NO_REFS:
            return _njit(cache=True, nogil=True, _nrt=False)(fn)
        return njit(fn)
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda fn: fn

    def njit_no_refs(fn):
        return fn


@njit
def jdiv(a, b):
    """Java int division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q
