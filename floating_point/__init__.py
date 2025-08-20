from .data_types import FloatingPoint
from .floating_point import inplace as cpp_inplace
from .floating_point import round as cpp_round
from .round import Round, StraightThroughEstimator


# Public API: Clear and intuitive function names
# Use the existing StraightThroughEstimator for autograd support
def round(input, exponent_bits, mantissa_bits, bias):
    dtype = FloatingPoint(1, exponent_bits, mantissa_bits, bias, exponent_bits + mantissa_bits + 1)
    return StraightThroughEstimator.apply(input, dtype, dtype.minimum, dtype.maximum)


inplace = cpp_inplace  # Inplace version without autograd

__all__ = ["FloatingPoint", "Round", "inplace", "round"]
