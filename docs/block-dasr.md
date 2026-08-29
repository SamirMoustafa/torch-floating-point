# Example

One pick per axis. Do not mix a hard STE backward onto this softmax. Estimators: [Autograd](autograd.md). NVFP4 packing: [Block scale](block.md).

NVIDIA’s NVFP4 ([blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)): E2M1 elements (`__nv_fp4_e2m1`, max \(\pm 6\)), one E4M3-FN scale (`__nv_fp8_e4m3`) per block of 16, \(s = \operatorname{encode}(\mathrm{amax}/6)\). Reconstruction \(y = e\,s\).

| Axis | Pick |
| --- | --- |
| Block scale | NVFP4 as above — `FloatingPoint` constructors match [Formats](formats.md); `M=6`, `block_size=16` |
| Forward | DASR on the **elements** \(x/s\) (two nearest E2M1 codes), then \(\times s\) |
| Backward | \(\frac{2}{\tau}\operatorname{Var}_\pi(c)\) on \(x/s\) |

\[
s=\operatorname{encode}_{\mathrm{E4M3}}(\mathrm{amax}/6),\qquad
\tilde e=\sum_{c\in C}\pi_\tau(c\mid x/s)\,c,\qquad
y=\tilde e\,s.
\]

On this axis \(\mathrm{amax}=10\), so \(s=\operatorname{Round}(10/6)=1.625\) (an E4M3 code). Plateaus sit on E2M1\(\times s\) and saturate at \(\pm 6s=\pm 9.75\). \(\partial y/\partial x = f'(x/s)\). \(\partial y/\partial s = \tilde e - (x/s)\,f'(x/s)\) — product rule, not the hard LSQ residual. NVIDIA absmax **detaches** \(s\); the third panel is that Jacobian if you keep \(s\) in the graph (`scales=`). Drag \(\tau\). \(\partial y/\partial x\) is cut at 12; \(\partial y/\partial s\) uses the EWGS window — spikes on wide bins go off-scale.

<div class="ste-widget" data-estimator="dasr-block" data-src="../assets/ewgs-slider.json">
  <div class="ste-widget__stage">
    <canvas role="img" aria-label="NVFP4 DASR with E4M3 scale encode(amax/6)=1.625. Drag tau to update the forward and the gradients."></canvas>
  </div>
  <div class="ste-widget__controls">
    <label for="dasr-block-tau">τ</label>
    <input id="dasr-block-tau" class="ste-widget__slider" type="range" min="0.02" max="2" step="0.01" value="0.15" aria-valuemin="0.02" aria-valuemax="2">
    <output class="ste-widget__value" for="dasr-block-tau" data-ste-value aria-live="polite">0.15</output>
    <div class="ste-widget__chips">
      <button type="button" class="ste-widget__chip" data-value="0.02">0.02 (≈Round)</button>
      <button type="button" class="ste-widget__chip" data-value="0.15">0.15</button>
      <button type="button" class="ste-widget__chip" data-value="0.5">0.5</button>
      <button type="button" class="ste-widget__chip" data-value="2">2</button>
    </div>
    <p class="ste-widget__hint">NVFP4: s = E4M3(amax/6) = 1.625 on this axis. y, ∂y/∂x, and ∂y/∂s depend on τ.</p>
  </div>
</div>

`BlockRound` still constructs stock `Round`, so wrap by hand.

NVFP4 types: E2M1 elements, E4M3 scales, block 16, \(M=6\).

```python
import math

import torch
from torch.autograd import Function

from floating_point import FloatingPoint, Round
from floating_point.block_round import encode_scale

TAU = 0.15  # small τ ≈ Round
B, M = 16, 6.0

fp4 = FloatingPoint(1, 2, 1, 1, 4, reserved_exponent=False)  # __nv_fp4_e2m1
fp8 = FloatingPoint(1, 4, 3, 7, 8, max_mantissa_at_max_exponent=6, reserved_exponent=False)  # __nv_fp8_e4m3
```

Forward: softmax over the two codes that straddle \(x\); clamp outside the range. Backward: \(\frac{2}{\tau}\operatorname{Var}_\pi(c)\); zero in saturation. Not STE — no `cpp_round`.

```python
class DASR(Function):
    @staticmethod
    def forward(ctx, x, codes):
        tau = max(TAU, 1e-4)
        idx = torch.searchsorted(codes, x, right=True).clamp(1, codes.numel() - 1)
        pair = torch.stack((codes[idx - 1], codes[idx]), dim=-1)
        w = torch.softmax(-(x.unsqueeze(-1) - pair).pow(2) / tau, dim=-1)
        y = (w * pair).sum(-1)
        inside = (x >= codes[0]) & (x <= codes[-1])
        y = torch.where(inside, y, x.clamp(codes[0], codes[-1]))
        ctx.save_for_backward(pair, w, inside)
        ctx.tau = tau
        return y

    @staticmethod
    def backward(ctx, grad_output):
        pair, w, inside = ctx.saved_tensors
        y = (w * pair).sum(-1, keepdim=True)
        var = (w * (pair - y).pow(2)).sum(-1)
        jac = (2.0 / ctx.tau) * var
        grad_x = grad_output * jac * inside.to(dtype=grad_output.dtype)
        return grad_x, None
```

Drop-in `Round`. Finite unique `fp.values` only.

```python
class DASRRound(Round):
    def forward(self, x):
        fp = self.data_type
        codes = x.new_tensor(sorted({v for v in fp.values if math.isfinite(v)}))
        return DASR.apply(x, codes)
```

Absmax scale (detached, same as `BlockRound(...)(x)`), then \(y=\mathrm{DASR}(x/s)\,s\).

```python
x = torch.randn(4, 64, requires_grad=True)
blocks = x.reshape(*x.shape[:-1], x.shape[-1] // B, B)
s = encode_scale(blocks.detach().abs().amax(dim=-1, keepdim=True) / M, fp8)
y = DASRRound(fp4)(blocks / s) * s
y = y.reshape_as(x)
y.sum().backward()  # x.grad: DASR on x/s; s detached
```

For QAT, `s = Round(fp8)(scale)` instead of `encode_scale`. Nested `Round` on the same tensor raises `Double quantization detected.`
