"""
Pin the BLAS to one thread, before numpy is imported.

This looks like a micro-optimisation and is not. Case D fits a 120-point
hyperparameter grid three times per round, on Gram matrices of at most 48 by 48.
Measured on the target interpreter with numpy's bundled OpenBLAS:

    two triangular solves, 48 x 48, default threading :  73929 us
    the same call with OPENBLAS_NUM_THREADS=1         :     27 us

That is a factor of about 2700, and it is entirely thread start-up and barrier
cost on matrices far too small to parallelise. Left alone it turns a one-minute
campaign into a two-hour one.

There is a correctness argument as well, and it is the more important of the two.
A threaded BLAS may split a reduction differently from run to run, so the last
bits of a dot product are not guaranteed to repeat. Case D requires byte-identical
reruns and asserts it in a test, so a single thread is required for correctness
and not only for speed.

``setdefault`` is used throughout: an operator who has deliberately set these
variables keeps their setting, and the run record reports what was actually in
force.
"""

from __future__ import annotations

import os
import sys

THREAD_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def pin_single_thread() -> dict[str, object]:
    """Request one BLAS thread and report how the request was made.

    Returns the state for the run record. ``numpy_already_imported`` matters:
    OpenBLAS reads these variables when the shared library loads, so if numpy
    came in first the environment alone will not take effect and the ctypes
    fallback below is what actually does the work.
    """
    already = "numpy" in sys.modules
    applied = {}
    for name in THREAD_VARS:
        os.environ.setdefault(name, "1")
        applied[name] = os.environ[name]

    fallback = None
    if already:
        fallback = _set_threads_on_loaded_openblas(1)

    return {
        "numpy_already_imported": already,
        "env": applied,
        "runtime_call": fallback,
    }


def _set_threads_on_loaded_openblas(n: int) -> str | None:
    """Best effort: call openblas_set_num_threads on the library numpy loaded."""
    try:
        import ctypes
        import ctypes.util
        import glob
        import os.path

        import numpy as np

        roots = [os.path.join(os.path.dirname(np.__file__), ".libs"),
                 os.path.join(os.path.dirname(os.path.dirname(np.__file__)), "numpy.libs")]
        for root in roots:
            for path in glob.glob(os.path.join(root, "*openblas*")):
                try:
                    lib = ctypes.CDLL(path)
                except OSError:
                    continue
                for symbol in ("openblas_set_num_threads", "openblas_set_num_threads64_"):
                    setter = getattr(lib, symbol, None)
                    if setter is not None:
                        setter(ctypes.c_int(n))
                        return f"{symbol}({n}) on {os.path.basename(path)}"
    except Exception:
        return None
    return None
