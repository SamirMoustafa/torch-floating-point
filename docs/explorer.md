---
hide:
  - toc
---

# Explorer

Pick a documented layout from the chips, or expand the codebook to edit the `FloatingPoint` knobs. The plots are the same language as [Autograd](autograd.md): identity \(x\) dashed, \(y\) in orange, rounding error underneath. The codebook rug is every finite code in the current \(x\) window. Gradients stay on the Autograd page; this is the **forward** map.

In block mode, chips pick a recipe. Expand the codebook to customize encode, \(M\), and geometry. Scrub \(s\), \(s_g\), and \(z\) under the plot (\(y=(e-z)\,s\,s_g\); dragging \(s\) is a `scales=` override until absmax / \(M\) / the scale codebook change). Geometry (`block_size`) does not change that 1-D transfer; the demo-block plot shows a linspace of length \(\min(k,64)\).

<div class="format-explorer" id="format-explorer">
  <noscript>
    <p>Enable JavaScript to use the explorer. Constructors are listed under <a href="../formats/">Formats</a> and <a href="../block/">Block scale</a>.</p>
  </noscript>
  <p class="format-explorer__error" data-fe-error hidden></p>
  <div class="format-explorer__layout">
    <div class="format-explorer__config">
      <div class="format-explorer__toolbar">
        <table class="format-explorer__table">
          <tbody>
            <tr>
              <th>Mode</th>
              <td class="format-explorer__mode">
                <label><input type="radio" name="fe-mode" id="fe-mode-element" checked> Element-wise <code>Round</code></label>
                <label><input type="radio" name="fe-mode" id="fe-mode-block"> Block-scaled <code>BlockRound</code></label>
              </td>
            </tr>
            <tr>
              <th><label for="fe-xrange">x window</label></th>
              <td><select id="fe-xrange"></select></td>
            </tr>
            <tr>
              <th><label for="fe-logx">log x</label></th>
              <td><label class="format-explorer__check"><input type="checkbox" id="fe-logx"> <span>x&gt;0</span></label></td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="format-explorer__chips" id="fe-elem-chips" aria-label="Element presets"></div>
      <div class="format-explorer__chips" id="fe-recipes" aria-label="Block recipes"></div>
      <details class="format-explorer__details">
        <summary class="format-explorer__details-summary"><span data-fe-summary>Codebook</span></summary>
        <section class="format-explorer__panel format-explorer__elem">
        <div class="format-explorer__panel-title">Element codebook</div>
        <table class="format-explorer__table">
          <tbody>
            <tr id="fe-elem-preset-row">
              <th><label for="fe-elem-preset">Preset</label></th>
              <td><select id="fe-elem-preset"></select></td>
            </tr>
            <tr>
              <th><label for="fe-elem-sign">sign_bits</label></th>
              <td><input id="fe-elem-sign" type="number" min="0" max="1" step="1" value="1" autocomplete="off"></td>
            </tr>
            <tr>
              <th><label for="fe-elem-exp">exponent_bits</label></th>
              <td><input id="fe-elem-exp" type="number" min="0" max="16" step="1" value="2" autocomplete="off"></td>
            </tr>
            <tr>
              <th><label for="fe-elem-mant">mantissa_bits</label></th>
              <td><input id="fe-elem-mant" type="number" min="0" max="16" step="1" value="1" autocomplete="off"></td>
            </tr>
            <tr>
              <th><label for="fe-elem-bias">bias</label></th>
              <td><input id="fe-elem-bias" type="number" step="1" value="1" autocomplete="off"></td>
            </tr>
            <tr>
              <th><label for="fe-elem-bits">bits</label></th>
              <td><input id="fe-elem-bits" type="number" value="4" readonly></td>
            </tr>
            <tr>
              <th><label for="fe-elem-maxmant">max_mantissa</label></th>
              <td><input id="fe-elem-maxmant" type="number" min="0" step="1" value="1" autocomplete="off" title="max_mantissa_at_max_exponent"></td>
            </tr>
            <tr>
              <th><label for="fe-elem-reserved">reserved_exponent</label></th>
              <td><input type="checkbox" id="fe-elem-reserved"></td>
            </tr>
          </tbody>
        </table>
      </section>
      <section class="format-explorer__panel format-explorer__block">
        <div class="format-explorer__panel-title">Block scale</div>
        <table class="format-explorer__table">
          <tbody>
            <tr>
              <th><label for="fe-scale-preset">Scale preset</label></th>
              <td><select id="fe-scale-preset"></select></td>
            </tr>
            <tr>
              <th><label for="fe-scale-sign">sign_bits</label></th>
              <td><input id="fe-scale-sign" type="number" min="0" max="1" step="1" value="1" autocomplete="off"></td>
            </tr>
            <tr>
              <th><label for="fe-scale-exp">exponent_bits</label></th>
              <td><input id="fe-scale-exp" type="number" min="0" max="16" step="1" value="4" autocomplete="off"></td>
            </tr>
            <tr>
              <th><label for="fe-scale-mant">mantissa_bits</label></th>
              <td><input id="fe-scale-mant" type="number" min="0" max="16" step="1" value="3" autocomplete="off"></td>
            </tr>
            <tr>
              <th><label for="fe-scale-bias">bias</label></th>
              <td><input id="fe-scale-bias" type="number" step="1" value="7" autocomplete="off"></td>
            </tr>
            <tr>
              <th><label for="fe-scale-bits">bits</label></th>
              <td><input id="fe-scale-bits" type="number" value="8" readonly></td>
            </tr>
            <tr>
              <th><label for="fe-scale-maxmant">max_mantissa</label></th>
              <td><input id="fe-scale-maxmant" type="number" min="0" step="1" value="6" autocomplete="off" title="max_mantissa_at_max_exponent"></td>
            </tr>
            <tr>
              <th><label for="fe-scale-reserved">reserved_exponent</label></th>
              <td><input type="checkbox" id="fe-scale-reserved"></td>
            </tr>
            <tr>
              <th><label for="fe-encode">scale_encode</label></th>
              <td><select id="fe-encode"></select></td>
            </tr>
            <tr>
              <th><label for="fe-M">M</label></th>
              <td><input id="fe-M" type="number" min="0" step="any" value="6" autocomplete="off"></td>
            </tr>
            <tr>
              <th><label for="fe-amax">block absmax</label></th>
              <td><input id="fe-amax" type="number" step="any" value="6" autocomplete="off"></td>
            </tr>
            <tr id="fe-block-size-wrap">
              <th><label for="fe-block-size">block_size</label></th>
              <td><input id="fe-block-size" type="number" min="1" step="1" value="16" autocomplete="off"></td>
            </tr>
            <tr>
              <th><label for="fe-tile">2-D tile</label></th>
              <td><input type="checkbox" id="fe-tile"></td>
            </tr>
          </tbody>
          <tbody id="fe-tile-fields" hidden>
            <tr>
              <th><label for="fe-block-h">tile h</label></th>
              <td><input id="fe-block-h" type="number" min="1" step="1" value="16" autocomplete="off"></td>
            </tr>
            <tr>
              <th><label for="fe-block-w">tile w</label></th>
              <td><input id="fe-block-w" type="number" min="1" step="1" value="16" autocomplete="off"></td>
            </tr>
          </tbody>
        </table>
      </section>
        <input id="fe-sg" type="number" hidden aria-hidden="true" value="1" autocomplete="off">
        <input id="fe-z" type="number" hidden aria-hidden="true" value="0" autocomplete="off">
      </details>
    </div>
    <div class="format-explorer__figures">
      <p class="format-explorer__stats" data-fe-stats></p>
      <div class="format-explorer__probe">
        <div class="format-explorer__knob">
          <label for="fe-probe">probe x</label>
          <input id="fe-probe" class="ste-widget__slider" type="range" min="0" max="1" step="0.0005" value="0.6" aria-valuemin="0" aria-valuemax="1">
          <input id="fe-probe-num" type="number" step="any" value="1.5" autocomplete="off" aria-label="probe x value">
        </div>
        <div class="format-explorer__block-knobs">
          <div class="format-explorer__knob">
            <label for="fe-s-range">s</label>
            <input id="fe-s-range" class="ste-widget__slider" type="range" min="0" max="1" step="0.0005" value="0.5" aria-valuemin="0" aria-valuemax="1">
            <input id="fe-s-num" type="number" step="any" value="1" autocomplete="off" aria-label="s value">
          </div>
          <div class="format-explorer__knob">
            <label for="fe-sg-range">s_g</label>
            <input id="fe-sg-range" class="ste-widget__slider" type="range" min="0" max="1" step="0.0005" value="0.5" aria-valuemin="0" aria-valuemax="1">
            <input id="fe-sg-num" type="number" step="any" value="1" autocomplete="off" aria-label="s_g value">
          </div>
          <div class="format-explorer__knob">
            <label for="fe-z-range">z</label>
            <input id="fe-z-range" class="ste-widget__slider" type="range" min="0" max="1" step="0.0005" value="0.5" aria-valuemin="0" aria-valuemax="1">
            <input id="fe-z-num" type="number" step="any" value="0" autocomplete="off" aria-label="z value">
          </div>
        </div>
      </div>
      <div class="format-explorer__plot-frame">
        <div class="format-explorer__plot" id="fe-plot" role="img" aria-label="Transfer function, codebook, and rounding error"></div>
      </div>
      <div class="format-explorer__plot-frame format-explorer__block-plot">
        <div class="format-explorer__plot" id="fe-block-plot"></div>
      </div>
    </div>
  </div>
  <p class="format-explorer__hint">Copy the constructor into Python. <code>bits</code> is capped at 16 in the browser.</p>
  <pre class="format-explorer__snippet"><code data-fe-snippet></code></pre>
</div>

## Caveats

- **FNUZ** presets match bias and finite max; negative-zero-as-NaN is not modeled ([Formats](formats.md)).
- **CFloat8** omits Inf/NaN; the chips use Tesla’s usual biases \(7\) / \(15\). Change `bias` in the codebook for the 6-bit parameter.
- **GGUF Q4_0** / **KleidiAI INT4** use an unsigned \(0\ldots 15\) codebook plus `zero_point=8` (nibble \(-8\)), IEEE FP16 scales, \(k=32\). They are not two’s-complement `Round`.
- **Tensix BFP8** is MXINT8 mag with a shared exponent over \(k=16\) (not OCP \(k=32\)).
- **UE4M3** is the E4M3-FN constructor; block `nearest` already takes \(\lvert\cdot\rvert\).
- **HiF8** is tapered — a single `FloatingPoint` cannot represent it. It is not in the preset list.
- Rounding follows the CPU kernel (ties to even) except \(E=0\) (MXINT / BFP mag / UINT4), which snaps to the nearest finite codebook value.
- Autograd estimators (STE, EWGS, ReSTE, …) are on [Autograd](autograd.md).

## Maintaining the explorer

The page runs a hand-ported kernel in `javascripts/fp-codec.js`. Python/C++ stays the oracle; identifiers are the same so you can grep either tree. After a kernel change, `python scripts/gen_fp_codec_goldens.py --write` then `python scripts/check_fp_codec.py`.

| Python | JS |
| --- | --- |
| `FloatingPoint`, `sign_bits`, `bit_pattern_to_custom_fp` | `fp-codec.js` class `FloatingPoint` (`floating_point/data_types.py`) |
| CPU round loop | `round_kernel` (`float_round_cpu_inplace` in `float_round.cpp`) |
| `Round(x)` | `round_scalar` (clamp; \(E=0\) uses `nearest_finite`) |
| `_SCALE_ENCODES` / `encode_scale` | `SCALE_ENCODES` / `encode_scale` |
| `BlockFormat` `elem_fp`, `scale_fp`, `scale_encode`, `M` | `encode_scale(stat, spec)` keys |
| Named formats / chips | `ELEM_PRESETS` / `BLOCK_RECIPES` in `format-explorer.js` |
| Goldens copy of presets | `ELEM_PRESETS` in `scripts/gen_fp_codec_goldens.py` |

**New `scale_encode`:** add the string to `_SCALE_ENCODES` and a branch in `encode_scale`; copy both into JS `SCALE_ENCODES` and `encode_scale`; add a golden row; `--write`; `check_fp_codec.py`.

**New format preset:** add the id to `ELEM_PRESETS` in the explorer (Python kwargs names) and in the generator (`chip: false` skips the element chip bar; omit 16-bit types from the generator — 2¹⁶ codes). Optional `BLOCK_RECIPES`; `--write`; check.

