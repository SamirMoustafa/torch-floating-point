# Autograd

`Round` is a staircase, so the true derivative is \(0\) almost everywhere. Training uses a **clipped straight-through estimator** (STE): identity through the bins, zero in saturation. The C++/CUDA kernel is forward-only; `StraightThroughEstimator` is the autograd wrapper.

## Element-wise

\[
y = \operatorname{Round}(x),
\qquad
\frac{\partial y}{\partial x}
=
\mathbf{1}_{x \in [x_{\min},\, x_{\max}]}.
\]

\(x_{\min}\) and \(x_{\max}\) are `fp.minimum` and `fp.maximum`. Incoming \(\partial L/\partial y\) is **not** clipped to that range.

<div class="autograd-fig" markdown="span">
![Element-wise Round and clipped STE](assets/autograd-round.svg){ .light }
![Element-wise Round and clipped STE](assets/autograd-round-dark.svg){ .dark }
</div>

```python
y = Round(fp)(x)
y.sum().backward()   # x.grad is 1 inside range, 0 outside
```

## Block-scaled

\[
y_i = \operatorname{Round}_{\mathrm{elem}}\!\left(\frac{x_i}{s}\right) s.
\]

Absmax mode detaches \(s\). Learnable `scales=` keeps \(s\) in the graph. The gate is on \(x/s\), not on \(x\):

\[
\frac{\partial y_i}{\partial x_i}
=
\mathbf{1}_{x_i/s \,\in\, [e_{\min},\, e_{\max}]},
\qquad
\frac{\partial y_i}{\partial s}
=
\begin{cases}
\operatorname{Round}(x_i/s) - x_i/s
  & \text{if } x_i/s \in [e_{\min},\, e_{\max}], \\[0.35em]
\operatorname{Round}(x_i/s)
  & \text{otherwise}.
\end{cases}
\]

The second branch is \(y = e\,s\) with \(\partial e/\partial s = 0\) (clipped STE). See [Block scale](block.md) for absmax vs `scales=`.

<div class="autograd-fig" markdown="span">
![Block-scaled Round with learnable s = 1](assets/autograd-block.svg){ .light }
![Block-scaled Round with learnable s = 1](assets/autograd-block-dark.svg){ .dark }
</div>

## Other estimators

Keep the **forward** (outputs must stay on `fp.values`). Replace only **backward**. Save the pre-clamp \(x\); do not `fill_` the caller's tensor.

Quantization-aware training still starts from the straight-through estimator. MAD, ReSTE, and RDFS change how \(\partial L/\partial y\) is masked or scaled. LSQ keeps the element Jacobian and learns \(s\) — already `BlockRound(..., scales=s)`.

| Surrogate | Backward | Reference |
| --- | --- | --- |
| Clipped STE | \(\mathbf{1}_{[x_{\min},x_{\max}]}\) | Default here |
| Learned step size quantization (LSQ) | STE on round; \(\partial y/\partial s = \mathrm{Round}(x/s) - x/s\) | [Esser et al., 2020](https://arxiv.org/abs/1902.08153) — already used for \(s\) in `BlockRound` |
| Element-wise gradient scaling (EWGS) | \(g\,(1+\delta\,\mathrm{sign}(g)\,(x-y))\) on the clip gate | [Lee et al., 2021](https://arxiv.org/abs/2104.00903) |
| Magnitude-aware differentiation (MAD) | \(1\) in range; \(x_{\max}/\lvert x\rvert\) in saturation | [Sakr et al., 2022](https://arxiv.org/abs/2206.06501) |
| Rectified STE (ReSTE) | Power-function STE \(\lvert x\rvert^{1/o-1}\) | [Wu et al., 2023](https://arxiv.org/abs/2308.06689) |
| Rotated damped Fourier surrogate (RDFS) | Cosine STE of round; identity when amplitude \(A=0\) | [Chen et al., 2026](https://arxiv.org/abs/2601.19320) |

## EWGS example

EWGS scales the clipped STE by the rounding error; \(\delta=0\) is the default. The paper also adapts \(\delta\) from a Hessian trace — extra state, not shown. Save \(y\) as well as \(x\). Wrap with a learnable \(s\) to put \(\partial y/\partial s\) in the graph (the figure is \(s=1\)). Drag \(\delta\); \(0\) is clipped STE. Fixed values around \(10^{-3}\)–\(10^{-2}\) are what [Lee et al.](https://arxiv.org/abs/2104.00903) used on a unit-interval quantizer — on E2M1 they sit on top of STE, so the slider default is \(1\) to make the sawtooth visible:

<div class="ewgs-widget" data-src="../assets/ewgs-slider.json">
  <noscript>
    <div class="autograd-fig">
      <img class="light" src="../assets/autograd-ewgs.svg" alt="EWGS Round with learnable s = 1">
      <img class="dark" src="../assets/autograd-ewgs-dark.svg" alt="EWGS Round with learnable s = 1">
    </div>
  </noscript>
  <div class="ewgs-widget__stage">
    <canvas role="img" aria-label="EWGS Round with learnable s = 1. Drag delta to update the gradients."></canvas>
  </div>
  <div class="ewgs-widget__controls">
    <label for="ewgs-delta">δ</label>
    <input id="ewgs-delta" class="ewgs-widget__slider" type="range" min="0" max="1" step="0.001" value="1" aria-valuemin="0" aria-valuemax="1">
    <output class="ewgs-widget__value" for="ewgs-delta" data-ewgs-value aria-live="polite">1</output>
    <div class="ewgs-widget__chips">
      <button type="button" class="ewgs-widget__chip" data-delta="0">0 (STE)</button>
      <button type="button" class="ewgs-widget__chip" data-delta="0.001">10⁻³</button>
      <button type="button" class="ewgs-widget__chip" data-delta="0.01">0.01</button>
      <button type="button" class="ewgs-widget__chip" data-delta="0.1">0.1</button>
      <button type="button" class="ewgs-widget__chip" data-delta="1">1</button>
    </div>
    <p class="ewgs-widget__hint">Forward is unchanged. Only ∂y/∂x and ∂y/∂s depend on δ.</p>
  </div>
</div>

```python
from floating_point.round import Round, StraightThroughEstimator

DELTA = 1.0  # 0 recovers clipped STE

class EWGS(StraightThroughEstimator):
    @staticmethod
    def forward(ctx, x, dtype, min, max):
        y = StraightThroughEstimator.forward(ctx, x, dtype, min, max)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        if x.grad_fn is not None and x.grad_fn.__class__.__name__ == ctx.__class__.__name__:
            raise RuntimeError("Double quantization detected.")
        in_range = (x >= ctx.min) & (x <= ctx.max)
        scale = 1 + DELTA * grad_output.sign() * (x - y)
        grad_x = grad_output * scale * in_range.to(dtype=grad_output.dtype)
        return grad_x, None, None, None

class EWGSRound(Round):
    def forward(self, x):
        fp = self.data_type
        return EWGS.apply(x, fp, fp.minimum, fp.maximum)

s = torch.tensor(1.0, requires_grad=True)
y = EWGSRound(fp)(x / s) * s
y.sum().backward()  # x.grad: EWGS on x/s; s.grad: product rule
```

Do not wrap `Round` inside another `Function` — nested `Round` on the same tensor raises `Double quantization detected.` `BlockRound` instantiates `Round` internally, so this subclass is not used for NVFP4/MX until `BlockRound` takes a custom rounder.
