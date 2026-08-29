from .data_types import FloatingPoint

try:
    from .floating_point import inplace as cpp_inplace
    from .floating_point import round as cpp_round
except ImportError:  # pragma: no cover - extension not built yet
    cpp_inplace = None
    cpp_round = None

from .block_round import BlockFormat, BlockRound, block_round, sample_block_scaled
from .round import Round, StraightThroughEstimator

_EXTENSION_MISSING = "floating_point C++ extension is not built; install with pip install -e ."


def round(input, exponent_bits, mantissa_bits, bias, reserved_exponent=True, max_mantissa_at_max_exponent=None):
    dtype = FloatingPoint(
        1,
        exponent_bits,
        mantissa_bits,
        bias,
        exponent_bits + mantissa_bits + 1,
        max_mantissa_at_max_exponent=max_mantissa_at_max_exponent,
        reserved_exponent=reserved_exponent,
    )
    return StraightThroughEstimator.apply(input, dtype, dtype.minimum, dtype.maximum)


def inplace(input, exponent_bits, mantissa_bits, bias, reserved_exponent=True, max_mantissa_at_max_exponent=None):
    if cpp_inplace is None:
        raise RuntimeError(_EXTENSION_MISSING)
    if max_mantissa_at_max_exponent is None:
        max_mantissa_at_max_exponent = (1 << mantissa_bits) - 1
    return cpp_inplace(input, exponent_bits, mantissa_bits, bias, int(reserved_exponent), max_mantissa_at_max_exponent)


__all__ = [
    "BlockFormat",
    "BlockRound",
    "FloatingPoint",
    "Round",
    "block_round",
    "inplace",
    "round",
    "sample_block_scaled",
]
