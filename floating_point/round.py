from typing import Tuple

from torch import Tensor
from torch.autograd import Function

from floating_point import cpp_round
from floating_point.data_types import FloatingPoint


class StraightThroughEstimator(Function):
    @staticmethod
    def forward(ctx: Function, x: Tensor, dtype: FloatingPoint, min: float, max: float) -> Tensor:
        x[x < min].fill_(min)
        x[x > max].fill_(max)
        rounded = cpp_round(x, dtype.exponent_bits, dtype.mantissa_bits, dtype.bias)
        ctx.min, ctx.max = min, max
        ctx.save_for_backward(x, rounded)
        return rounded

    @staticmethod
    def backward(ctx: Function, grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        x, rounded = ctx.saved_tensors
        if x.grad_fn.__class__.__name__ == ctx.__class__.__name__:
            raise RuntimeError("Double quantization detected.")
        grad_input = grad_output.clone()
        grad_input[grad_input < ctx.min].fill_(ctx.min)
        grad_input[grad_input > ctx.max].fill_(ctx.max)
        return grad_input, None, None, None


class Round:
    def __init__(self, data_type: FloatingPoint):
        self.data_type = data_type

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        return StraightThroughEstimator.apply(x, self.data_type, self.data_type.minimum, self.data_type.maximum)
