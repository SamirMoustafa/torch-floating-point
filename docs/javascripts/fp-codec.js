/**
 * Browser port of floating_point.data_types.FloatingPoint, the CPU round
 * kernel (float_round.cpp), and scalar encode_scale (block_round.py).
 * No PyTorch — used by the docs format explorer only.
 * Names match the Python/C++ API so grep finds both copies.
 */
(() => {
  const MAX_BITS = 16;
  // Same strings and order as _SCALE_ENCODES in floating_point/block_round.py.
  const SCALE_ENCODES = Object.freeze([
    "nearest",
    "ue8m0_ceil",
    "ue8m0_floor",
    "ocp_floor",
    "ocp_floor_x2",
    "amax_over_M",
    "signed_peak",
  ]);

  function pow2(n) {
    return 2 ** n;
  }

  function clamp(x, lo, hi) {
    if (x < lo) {
      return lo;
    }
    if (x > hi) {
      return hi;
    }
    return x;
  }

  /** IEEE nearbyint: nearest integer, ties to even. */
  function round_ties_to_even(x) {
    const n = Math.floor(x);
    const frac = x - n;
    if (frac < 0.5) {
      return n;
    }
    if (frac > 0.5) {
      return n + 1;
    }
    return n % 2 === 0 ? n : n + 1;
  }

  function copysign1(x) {
    return x < 0 || Object.is(x, -0) ? -1 : 1;
  }

  // FloatingPoint — floating_point/data_types.py
  class FloatingPoint {
    constructor(sign_bits, exponent_bits, mantissa_bits, bias, bits, options) {
      const opts = options && typeof options === "object" ? options : {};
      if (bits !== sign_bits + exponent_bits + mantissa_bits) {
        throw new Error(`bits must equal S+E+M, got ${bits} vs ${sign_bits}+${exponent_bits}+${mantissa_bits}`);
      }
      if (sign_bits < 0 || sign_bits > 1) {
        throw new Error(`sign_bits must be 0 or 1, got ${sign_bits}`);
      }
      if (bits <= 0 || bits > MAX_BITS) {
        throw new Error(`bits must be in 1..${MAX_BITS}, got ${bits}`);
      }
      if (exponent_bits < 0 || mantissa_bits < 0) {
        throw new Error("exponent_bits and mantissa_bits must be non-negative");
      }
      this.sign_bits = sign_bits;
      this.exponent_bits = exponent_bits;
      this.mantissa_bits = mantissa_bits;
      this.bias = bias;
      this.bits = bits;
      this.reserved_exponent = Boolean(opts.reserved_exponent ?? true);
      const explicitMax = opts.max_mantissa_at_max_exponent;
      this.max_mantissa_at_max_exponent = explicitMax || pow2(mantissa_bits) - 1;
      this._values = null;
      this._finite = null;
    }

    get is_signed() {
      return this.sign_bits > 0;
    }

    get epsilon() {
      return pow2(-this.mantissa_bits);
    }

    get minimum() {
      if (this.is_signed) {
        return -this.maximum;
      }
      if (this.mantissa_bits === 0 && this.reserved_exponent) {
        return pow2(-this.bias);
      }
      return 0;
    }

    get maximum() {
      if (this.exponent_bits === 0) {
        const max_exponent = 1 - this.bias;
        const max_mantissa = pow2(this.mantissa_bits) - 1;
        return (max_mantissa / pow2(this.mantissa_bits)) * pow2(max_exponent);
      }
      const max_stored = this.reserved_exponent ? pow2(this.exponent_bits) - 2 : pow2(this.exponent_bits) - 1;
      if (this.mantissa_bits === 0) {
        return pow2(max_stored - this.bias);
      }
      const max_exponent = max_stored - this.bias;
      return (1 + this.max_mantissa_at_max_exponent / pow2(this.mantissa_bits)) * pow2(max_exponent);
    }

    bit_pattern_to_custom_fp(bit_pattern) {
      const total_bits = this.sign_bits + this.exponent_bits + this.mantissa_bits;
      const sign_mask = this.is_signed ? 1 << (total_bits - 1) : 0;
      const exponent_mask = (pow2(this.exponent_bits) - 1) << this.mantissa_bits;
      const mantissa_mask = pow2(this.mantissa_bits) - 1;
      const sign = this.is_signed ? (bit_pattern & sign_mask) >> (this.exponent_bits + this.mantissa_bits) : 0;
      const exponent = (bit_pattern & exponent_mask) >> this.mantissa_bits;
      const mantissa = bit_pattern & mantissa_mask;
      const sign_factor = sign ? -1 : 1;
      if (this.exponent_bits === 0) {
        const exponent_value = 1 - this.bias;
        const mantissa_value = mantissa / pow2(this.mantissa_bits);
        return mantissa === 0 ? sign_factor * 0 : sign_factor * mantissa_value * pow2(exponent_value);
      }
      const max_exponent = pow2(this.exponent_bits) - 1;
      if (this.mantissa_bits === 0) {
        if (this.reserved_exponent && exponent === max_exponent) {
          return NaN;
        }
        if (exponent === 0 && !this.reserved_exponent) {
          return sign_factor * 0;
        }
        return sign_factor * pow2(exponent - this.bias);
      }
      if (this.reserved_exponent && exponent === max_exponent) {
        return mantissa === 0 ? sign_factor * Infinity : NaN;
      }
      if (!this.reserved_exponent && exponent === max_exponent && mantissa > this.max_mantissa_at_max_exponent) {
        return NaN;
      }
      if (exponent === 0) {
        if (mantissa === 0) {
          return sign_factor * 0;
        }
        return sign_factor * (mantissa / pow2(this.mantissa_bits)) * pow2(1 - this.bias);
      }
      return sign_factor * (1 + mantissa / pow2(this.mantissa_bits)) * pow2(exponent - this.bias);
    }

    get values() {
      if (this._values) {
        return this._values;
      }
      const n = pow2(this.bits);
      const values = new Array(n);
      for (let i = 0; i < n; i++) {
        values[i] = this.bit_pattern_to_custom_fp(i);
      }
      values.sort((a, b) => {
        const aNan = Number.isNaN(a);
        const bNan = Number.isNaN(b);
        const aSign = aNan ? Infinity : copysign1(a);
        const bSign = bNan ? Infinity : copysign1(b);
        if (aSign !== bSign) {
          return aSign - bSign;
        }
        const aVal = aNan ? Infinity : a;
        const bVal = bNan ? Infinity : b;
        return aVal - bVal;
      });
      this._values = values;
      return values;
    }

    finite_values() {
      if (this._finite) {
        return this._finite;
      }
      const finite = [];
      const seen = new Set();
      for (const v of this.values) {
        if (!Number.isFinite(v)) {
          continue;
        }
        const key = Object.is(v, -0) ? "-0" : v;
        if (seen.has(key)) {
          continue;
        }
        seen.add(key);
        finite.push(v);
      }
      this._finite = finite;
      return finite;
    }

    codebook_stats() {
      let nFinite = 0;
      let nInf = 0;
      let nNan = 0;
      for (const v of this.values) {
        if (Number.isNaN(v)) {
          nNan += 1;
        } else if (!Number.isFinite(v)) {
          nInf += 1;
        } else {
          nFinite += 1;
        }
      }
      return {
        minimum: this.minimum,
        maximum: this.maximum,
        epsilon: this.epsilon,
        nFinite,
        nInf,
        nNan,
        nCodes: pow2(this.bits),
      };
    }

    to_constructor(name) {
      const ident = name || "fp";
      const parts = [
        `${this.sign_bits}`,
        `${this.exponent_bits}`,
        `${this.mantissa_bits}`,
        `${this.bias}`,
        `${this.bits}`,
      ];
      const defaultMax = pow2(this.mantissa_bits) - 1;
      if (this.max_mantissa_at_max_exponent !== defaultMax) {
        parts.push(`max_mantissa_at_max_exponent=${this.max_mantissa_at_max_exponent}`);
      }
      if (!this.reserved_exponent) {
        parts.push("reserved_exponent=False");
      }
      return `${ident} = FloatingPoint(${parts.join(", ")})`;
    }
  }

  // round_kernel — floating_point/float_round.cpp float_round_cpu_inplace
  function round_kernel(xVal, fp) {
    if (xVal === 0) {
      return xVal;
    }
    const exponent_bits = fp.exponent_bits;
    const mantissa_bits = fp.mantissa_bits;
    const bias = fp.bias;
    const max_stored_exp = fp.reserved_exponent ? pow2(exponent_bits) - 2 : pow2(exponent_bits) - 1;
    const max_exp = max_stored_exp - bias;
    const min_normal_exp = mantissa_bits === 0 ? -bias : 1 - bias;
    const subnormal_scale = pow2(1 - bias);
    const mantissa_upper_bound = pow2(mantissa_bits);
    const max_mant_at_max = fp.max_mantissa_at_max_exponent;
    const mantissa_scale = Math.max(mantissa_upper_bound, 1);
    const inv_mantissa_scale = 1 / mantissa_scale;
    const s = copysign1(xVal);
    const x_abs = Math.abs(xVal);
    const exponent_floor = Math.floor(Math.log2(x_abs));
    if (mantissa_bits > 0 && exponent_floor < min_normal_exp) {
      const mant_unrounded = (x_abs / subnormal_scale) * mantissa_scale;
      const mant = round_ties_to_even(mant_unrounded);
      if (mant <= 0) {
        return s * 0;
      }
      if (mant >= mantissa_upper_bound) {
        return s * subnormal_scale;
      }
      return s * (mant * inv_mantissa_scale) * subnormal_scale;
    }
    const exponent = Math.max(Math.min(exponent_floor, max_exp), min_normal_exp);
    let exp2_val = pow2(exponent);
    let scaled = x_abs / exp2_val;
    if (scaled < 1) {
      scaled = 1;
    }
    const mantissa_unrounded = (scaled - 1) * mantissa_scale;
    const mantissa = round_ties_to_even(mantissa_unrounded);
    const at_max_exp = exponent >= max_exp;
    const effective_ub = at_max_exp ? max_mant_at_max + 1 : mantissa_upper_bound;
    let final_exp2 = exp2_val;
    let final_mantissa = mantissa;
    if (mantissa >= effective_ub) {
      if (at_max_exp) {
        final_mantissa = max_mant_at_max;
      } else {
        const exponent_overflow = Math.max(Math.min(exponent + 1, max_exp), min_normal_exp);
        final_exp2 = pow2(exponent_overflow);
        final_mantissa = 0;
      }
    } else if (mantissa < 0) {
      final_mantissa = 0;
    }
    return s * (1 + final_mantissa * inv_mantissa_scale) * final_exp2;
  }

  function nearest_finite(x, codes) {
    const n = codes.length;
    if (n === 0) {
      return x;
    }
    if (x <= codes[0]) {
      return codes[0];
    }
    if (x >= codes[n - 1]) {
      return codes[n - 1];
    }
    let lo = 0;
    let hi = n - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (codes[mid] <= x) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    return x - codes[lo] <= codes[hi] - x ? codes[lo] : codes[hi];
  }

  function round_scalar(x, fp) {
    if (Number.isNaN(x)) {
      return NaN;
    }
    if (!Number.isFinite(x)) {
      return copysign1(x) > 0 ? fp.maximum : fp.minimum;
    }
    const clamped = clamp(x, fp.minimum, fp.maximum);
    if (fp.exponent_bits === 0) {
      return nearest_finite(clamped, fp.finite_values());
    }
    return round_kernel(clamped, fp);
  }

  function round_array(xs, fp) {
    const out = new Float64Array(xs.length);
    for (let i = 0; i < xs.length; i++) {
      out[i] = round_scalar(xs[i], fp);
    }
    return out;
  }

  // encode_scale — floating_point/block_round.py
  function emax_elem(fp) {
    if (fp.exponent_bits === 0) {
      const m = fp.maximum;
      if (m <= 0) {
        return 0;
      }
      return Math.floor(Math.log2(m));
    }
    const max_stored = fp.reserved_exponent ? pow2(fp.exponent_bits) - 2 : pow2(fp.exponent_bits) - 1;
    return max_stored - fp.bias;
  }

  function scale_bounds(scale_fp) {
    const min = scale_fp.minimum;
    const max = scale_fp.maximum;
    if (min > 0) {
      return [min, max];
    }
    const finite = scale_fp.finite_values();
    let lo = 1;
    for (let i = 0; i < finite.length; i++) {
      if (finite[i] > 0) {
        lo = finite[i];
        break;
      }
    }
    return [lo, max];
  }

  function log2_clamp(x, lo) {
    return Math.log2(Math.max(x, lo));
  }

  function pow2_from_unbiased_exp(exp, scale_fp) {
    const max_stored = pow2(scale_fp.exponent_bits) - (scale_fp.reserved_exponent ? 2 : 1);
    const stored = clamp(exp + scale_fp.bias, 0, max_stored);
    return pow2(stored - scale_fp.bias);
  }

  /**
   * Scalar encode_scale. ``stat`` is block absmax of x/s_global, or the signed
   * peak for signed_peak. Branch order matches block_round.encode_scale.
   */
  function encode_scale(stat, spec) {
    const scale_fp = spec.scale_fp;
    const elem_fp = spec.elem_fp;
    const encode = spec.scale_encode;
    const M = spec.M;
    const [lo, hi] = scale_bounds(scale_fp);
    if (encode === "signed_peak") {
      const raw = stat === 0 ? lo : stat / -M;
      let s = round_scalar(raw, scale_fp);
      const sign = raw === 0 ? 1 : Math.sign(raw) || 1;
      if (s === 0) {
        s = sign * lo;
      }
      return s;
    }
    if (encode === "ocp_floor") {
      return pow2_from_unbiased_exp(Math.floor(log2_clamp(stat, lo)) - emax_elem(elem_fp), scale_fp);
    }
    if (encode === "ocp_floor_x2") {
      return pow2_from_unbiased_exp(Math.floor(log2_clamp(stat, lo)) - (emax_elem(elem_fp) - 1), scale_fp);
    }
    // Torch encode_scale is float32; fround keeps amax/M on that grid.
    let raw = Math.fround(clamp(stat / M, lo, hi));
    if (stat === 0) {
      raw = lo;
    }
    if (encode === "ue8m0_ceil") {
      return pow2_from_unbiased_exp(Math.ceil(log2_clamp(raw, lo)), scale_fp);
    }
    if (encode === "ue8m0_floor") {
      return pow2_from_unbiased_exp(Math.floor(log2_clamp(raw, lo)), scale_fp);
    }
    if (encode === "amax_over_M") {
      return raw;
    }
    if (encode === "nearest") {
      const rounded = round_scalar(Math.abs(raw), scale_fp);
      return clamp(Math.abs(rounded), lo, hi);
    }
    throw new Error(`unknown scale_encode ${encode}`);
  }

  function reconstruct(x, elem_fp, s, s_global, zero_point) {
    const denom = s * s_global;
    const e = round_scalar(x / denom + zero_point, elem_fp);
    return { y: (e - zero_point) * denom, e, s, s_global, zero_point };
  }

  window.TFP = {
    MAX_BITS,
    SCALE_ENCODES,
    FloatingPoint,
    round_scalar,
    round_array,
    round_kernel,
    nearest_finite,
    encode_scale,
    reconstruct,
    emax_elem,
    scale_bounds,
    clamp,
    pow2,
  };
})();
