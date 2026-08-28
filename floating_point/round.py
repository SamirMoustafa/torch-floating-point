from typing import Tuple

from torch import Tensor
from torch.autograd import Function

from floating_point import cpp_round
from floating_point.data_types import FloatingPoint

_EXTENSION_MISSING = "floating_point C++ extension is not built; install with pip install -e ."


class StraightThroughEstimator(Function):
    @staticmethod
    def forward(ctx: Function, x: Tensor, dtype: FloatingPoint, min: float, max: float) -> Tensor:
        if cpp_round is None:
            raise RuntimeError(_EXTENSION_MISSING)
        # Save pre-clamp x so backward can mask saturation. Do not mutate the
        # caller's tensor; cpp_round clones before rounding.
        ctx.min, ctx.max = min, max
        ctx.save_for_backward(x)
        rounded = cpp_round(
            x.clamp(min, max),
            dtype.exponent_bits,
            dtype.mantissa_bits,
            dtype.bias,
            int(dtype.reserved_exponent),
            dtype.max_mantissa_at_max_exponent,
        )
        return rounded

    @staticmethod
    def backward(ctx: Function, grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        (x,) = ctx.saved_tensors
        if x.grad_fn is not None and x.grad_fn.__class__.__name__ == ctx.__class__.__name__:
            raise RuntimeError("Double quantization detected.")
        # Clipped STE: identity through the staircase, zero outside [min, max].
        in_range = (x >= ctx.min) & (x <= ctx.max)
        grad_input = grad_output * in_range.to(dtype=grad_output.dtype)
        return grad_input, None, None, None


class Round:
    def __init__(self, data_type: FloatingPoint):
        self.data_type = data_type

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        return StraightThroughEstimator.apply(x, self.data_type, self.data_type.minimum, self.data_type.maximum)
