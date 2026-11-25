"""Sequential electrolyte optimisation with single-threaded BLAS."""

from ._blas import pin_single_thread

BLAS_STATE = pin_single_thread()
