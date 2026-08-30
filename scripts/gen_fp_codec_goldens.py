#!/usr/bin/env python3
"""Dump decode / round / encode_scale goldens from the Python+C++ kernels.

The explorer JS port (docs/javascripts/fp-codec.js) must match this JSON.
Regenerate:

    python scripts/gen_fp_codec_goldens.py --write

CI / local check (do not write):

    python scripts/gen_fp_codec_goldens.py --check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

import floating_point
from floating_point.block_round import _SCALE_ENCODES, BlockFormat, encode_scale
from floating_point.data_types import FloatingPoint
from floating_point.round import Round

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "test" / "testdata" / "fp_codec_goldens.json"

# Explorer ELEM_PRESETS (format-explorer.js), snake_case constructors.
ELEM_PRESETS: dict[str, dict[str, Any]] = {
    "e2m1": {"sign_bits": 1, "exponent_bits": 2, "mantissa_bits": 1, "bias": 1, "reserved_exponent": False},
    "e3m2": {"sign_bits": 1, "exponent_bits": 3, "mantissa_bits": 2, "bias": 3, "reserved_exponent": False},
    "e2m3": {"sign_bits": 1, "exponent_bits": 2, "mantissa_bits": 3, "bias": 1, "reserved_exponent": False},
    "e1m2": {"sign_bits": 1, "exponent_bits": 1, "mantissa_bits": 2, "bias": 0, "reserved_exponent": False},
    "e4m3fn": {
        "sign_bits": 1,
        "exponent_bits": 4,
        "mantissa_bits": 3,
        "bias": 7,
        "reserved_exponent": False,
        "max_mantissa_at_max_exponent": 6,
    },
    "e4m3_240": {"sign_bits": 1, "exponent_bits": 4, "mantissa_bits": 3, "bias": 7, "reserved_exponent": True},
    "e5m2": {"sign_bits": 1, "exponent_bits": 5, "mantissa_bits": 2, "bias": 15, "reserved_exponent": True},
    "ue8m0": {"sign_bits": 0, "exponent_bits": 8, "mantissa_bits": 0, "bias": 127, "reserved_exponent": True},
    "e4m3fnuz": {"sign_bits": 1, "exponent_bits": 4, "mantissa_bits": 3, "bias": 8, "reserved_exponent": False},
    "e5m2fnuz": {"sign_bits": 1, "exponent_bits": 5, "mantissa_bits": 2, "bias": 16, "reserved_exponent": False},
    "cfloat8_e4m3": {"sign_bits": 1, "exponent_bits": 4, "mantissa_bits": 3, "bias": 7, "reserved_exponent": False},
    "cfloat8_e5m2": {"sign_bits": 1, "exponent_bits": 5, "mantissa_bits": 2, "bias": 15, "reserved_exponent": False},
    "mxint8": {"sign_bits": 1, "exponent_bits": 0, "mantissa_bits": 7, "bias": 0, "reserved_exponent": True},
    "uint4": {"sign_bits": 0, "exponent_bits": 0, "mantissa_bits": 4, "bias": -3, "reserved_exponent": True},
}


def dump_number(x: float) -> Any:
    if math.isnan(x):
        return {"nan": True}
    if math.isinf(x):
        return {"inf": 1 if x > 0 else -1}
    if x == 0.0 and math.copysign(1.0, x) < 0:
        return {"neg0": True}
    return x


def dump_fp(fp: Any) -> dict[str, Any]:
    return {
        "sign_bits": fp.sign_bits,
        "exponent_bits": fp.exponent_bits,
        "mantissa_bits": fp.mantissa_bits,
        "bias": fp.bias,
        "bits": fp.bits,
        "reserved_exponent": fp.reserved_exponent,
        "max_mantissa_at_max_exponent": fp.max_mantissa_at_max_exponent,
    }


def make_fp(preset: dict[str, Any]) -> FloatingPoint:
    sign_bits = preset["sign_bits"]
    exponent_bits = preset["exponent_bits"]
    mantissa_bits = preset["mantissa_bits"]
    kwargs: dict[str, Any] = {
        "sign_bits": sign_bits,
        "exponent_bits": exponent_bits,
        "mantissa_bits": mantissa_bits,
        "bias": preset["bias"],
        "bits": sign_bits + exponent_bits + mantissa_bits,
        "reserved_exponent": preset.get("reserved_exponent", True),
    }
    if "max_mantissa_at_max_exponent" in preset:
        kwargs["max_mantissa_at_max_exponent"] = preset["max_mantissa_at_max_exponent"]
    return FloatingPoint(**kwargs)


def finite_codes(fp) -> list[float]:
    seen = set()
    out: list[float] = []
    for v in fp.values:
        if not math.isfinite(v):
            continue
        key = "-0" if v == 0.0 and math.copysign(1.0, v) < 0 else v
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def nearest_finite(x: float, codes: list[float]) -> float:
    """Match docs/javascripts/fp-codec.js nearest_finite (midpoint prefers the lower code)."""
    if not codes:
        return x
    if x <= codes[0]:
        return codes[0]
    if x >= codes[-1]:
        return codes[-1]
    lo = 0
    hi = len(codes) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if codes[mid] <= x:
            lo = mid
        else:
            hi = mid
    return codes[lo] if x - codes[lo] <= codes[hi] - x else codes[hi]


def round_e0_scalar(x: float, fp) -> float:
    lo, hi = float(fp.minimum), float(fp.maximum)
    if math.isnan(x):
        return math.nan
    if not math.isfinite(x):
        return hi if math.copysign(1.0, x) > 0 else lo
    clamped = min(max(x, lo), hi)
    return nearest_finite(clamped, finite_codes(fp))


def round_cpp(xs: list[float], fp) -> list[float]:
    if floating_point.cpp_round is None:
        raise RuntimeError(
            "C++ extension missing; import torch before floating_point and install with pip install -e ."
        )

    y = Round(fp)(torch.tensor(xs, dtype=torch.float32))
    return [float(v.item()) for v in y]


def encode_one(stat: float, spec) -> float:
    s = encode_scale(torch.tensor([[stat]], dtype=torch.float32), spec)
    return float(s.reshape(-1)[0].item())


def reconstruct_one(x: float, elem_fp, s: float, s_global: float, zero_point: float):
    denom = s * s_global
    e = float(Round(elem_fp)(torch.tensor([x / denom + zero_point], dtype=torch.float32))[0].item())
    y = (e - zero_point) * denom
    return e, y


def build_goldens() -> dict[str, Any]:

    fps = {name: make_fp(preset) for name, preset in ELEM_PRESETS.items()}

    decode = {}
    for name, fp in fps.items():
        decode[name] = {
            "fp": dump_fp(fp),
            "minimum": dump_number(float(fp.minimum)),
            "maximum": dump_number(float(fp.maximum)),
            "epsilon": dump_number(float(fp.epsilon)),
            "codes": [dump_number(fp.bit_pattern_to_custom_fp(code)) for code in range(2**fp.bits)],
        }

    e2m1 = fps["e2m1"]
    e4m3fn = fps["e4m3fn"]
    ue8m0 = fps["ue8m0"]
    mxint8 = fps["mxint8"]

    round_cases = [
        {"name": "e2m1_ties", "fp": dump_fp(e2m1), "x": [0.25, 0.75, -0.25, -0.75, 0.7, 5.0, 6.0]},
        {"name": "e4m3fn_sat", "fp": dump_fp(e4m3fn), "x": [448.0, 449.0, 480.0, 1000.0, -448.0, -480.0]},
        {"name": "e4m3fn_subnormal", "fp": dump_fp(e4m3fn), "x": [0.01, 0.015625, -0.01, -0.015625]},
    ]
    for case in round_cases:
        fp = make_fp(
            {
                "sign_bits": case["fp"]["sign_bits"],
                "exponent_bits": case["fp"]["exponent_bits"],
                "mantissa_bits": case["fp"]["mantissa_bits"],
                "bias": case["fp"]["bias"],
                "reserved_exponent": case["fp"]["reserved_exponent"],
                "max_mantissa_at_max_exponent": case["fp"]["max_mantissa_at_max_exponent"],
            }
        )
        case["y"] = [dump_number(v) for v in round_cpp(case["x"], fp)]

    e0_x = [0.0, -0.0, 0.5, 1.0, 1.5, 1.984375, 2.0, -1.0, 0.0078125, 0.01171875]
    round_e0 = {
        "name": "mxint8_nearest_finite",
        "fp": dump_fp(mxint8),
        "x": [dump_number(v) for v in e0_x],
        "y": [dump_number(round_e0_scalar(v, mxint8)) for v in e0_x],
    }

    e4m3 = e4m3fn
    mxfp8_rows = [
        (256.0, "ocp_floor"),
        (256.0, "ocp_floor_x2"),
        (256.0, "ue8m0_ceil"),
        (256.0, "ue8m0_floor"),
        (256.0, "amax_over_M"),
        (448.0, "ocp_floor"),
        (448.0, "ocp_floor_x2"),
        (448.0, "ue8m0_ceil"),
        (448.0, "ue8m0_floor"),
        (448.0, "amax_over_M"),
        (500.0, "ocp_floor"),
        (500.0, "ocp_floor_x2"),
        (500.0, "ue8m0_ceil"),
        (500.0, "ue8m0_floor"),
        (500.0, "amax_over_M"),
        (672.0, "ocp_floor"),
        (672.0, "ocp_floor_x2"),
        (672.0, "ue8m0_ceil"),
        (672.0, "ue8m0_floor"),
        (672.0, "amax_over_M"),
        (1024.0, "ocp_floor"),
        (1024.0, "ocp_floor_x2"),
        (1024.0, "ue8m0_ceil"),
        (1024.0, "ue8m0_floor"),
        (1024.0, "amax_over_M"),
    ]
    encode_scale_cases: list[dict[str, Any]] = []
    for amax, encode in mxfp8_rows:
        spec = BlockFormat(e4m3, ue8m0, 32, 448.0, encode)
        encode_scale_cases.append(
            {
                "name": f"mxfp8_{encode}_{int(amax)}",
                "elem": dump_fp(e4m3),
                "scale": dump_fp(ue8m0),
                "M": 448.0,
                "scale_encode": encode,
                "stat": amax,
                "s": dump_number(encode_one(amax, spec)),
            }
        )

    nvfp4 = BlockFormat(e2m1, e4m3, 16, 6.0, "nearest")
    for amax in (0.0, 6.0, 12.0, 18.0, 24.0):
        encode_scale_cases.append(
            {
                "name": f"nvfp4_nearest_{int(amax)}",
                "elem": dump_fp(e2m1),
                "scale": dump_fp(e4m3),
                "M": 6.0,
                "scale_encode": "nearest",
                "stat": amax,
                "s": dump_number(encode_one(amax, nvfp4)),
            }
        )

    signed_spec = BlockFormat(e2m1, e4m3, 16, 6.0, "signed_peak")
    for stat in (0.0, 4.0, -4.0, 12.0):
        encode_scale_cases.append(
            {
                "name": f"signed_peak_{stat}",
                "elem": dump_fp(e2m1),
                "scale": dump_fp(e4m3),
                "M": 6.0,
                "scale_encode": "signed_peak",
                "stat": stat,
                "s": dump_number(encode_one(stat, signed_spec)),
            }
        )

    mxint8_spec = BlockFormat(mxint8, ue8m0, 32, 127.0 / 64.0, "ocp_floor")
    for amax in (0.0, 1.0, 1.984375, 4.0):
        encode_scale_cases.append(
            {
                "name": f"mxint8_ocp_{amax}",
                "elem": dump_fp(mxint8),
                "scale": dump_fp(ue8m0),
                "M": 127.0 / 64.0,
                "scale_encode": "ocp_floor",
                "stat": amax,
                "s": dump_number(encode_one(amax, mxint8_spec)),
            }
        )

    reconstruct_specs = [
        (1.5, e2m1, 1.0, 1.0, 0.0),
        (0.7, e2m1, 1.0, 1.0, 0.0),
        (3.0, e2m1, 2.0, 1.0, 0.0),
        (1.5, e2m1, 1.0, 2.0, 0.0),
        (2.0, e2m1, 1.0, 1.0, 1.0),
    ]
    reconstruct = []
    for x, elem, s, sg, z in reconstruct_specs:
        e, y = reconstruct_one(x, elem, s, sg, z)
        reconstruct.append(
            {
                "fp": dump_fp(elem),
                "x": dump_number(x),
                "s": dump_number(s),
                "s_global": dump_number(sg),
                "zero_point": dump_number(z),
                "e": dump_number(e),
                "y": dump_number(y),
            }
        )

    return {
        "decode": decode,
        "round": round_cases,
        "round_e0": [round_e0],
        "encode_scale": encode_scale_cases,
        "reconstruct": reconstruct,
        "scale_encodes": sorted(_SCALE_ENCODES),
    }


def canonical_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="overwrite the committed JSON")
    group.add_argument("--check", action="store_true", help="fail if JSON does not match the kernels")
    args = parser.parse_args()

    payload = build_goldens()
    text = canonical_dumps(payload)
    if args.write:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {GOLDEN_PATH.relative_to(ROOT)}")
        return 0

    if not GOLDEN_PATH.is_file():
        print(f"missing {GOLDEN_PATH}; run with --write", file=sys.stderr)
        return 1
    committed = GOLDEN_PATH.read_text(encoding="utf-8")
    if committed != text:
        rel = GOLDEN_PATH.relative_to(ROOT)
        print(f"{rel} is stale; run python scripts/gen_fp_codec_goldens.py --write", file=sys.stderr)
        return 1
    print(f"ok {GOLDEN_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
