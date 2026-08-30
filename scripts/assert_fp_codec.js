#!/usr/bin/env node
/**
 * Load the shipped explorer kernel and assert it against fp_codec_goldens.json.
 * Usage: node scripts/assert_fp_codec.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.resolve(__dirname, "..");
const CODEC = path.join(ROOT, "docs", "javascripts", "fp-codec.js");
const GOLDEN = path.join(ROOT, "test", "testdata", "fp_codec_goldens.json");

function loadTfp() {
  const sandbox = { window: {} };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(CODEC, "utf8"), sandbox, { filename: "fp-codec.js" });
  if (!sandbox.window.TFP) {
    throw new Error("fp-codec.js did not set window.TFP");
  }
  return sandbox.window.TFP;
}

function parseNumber(v) {
  if (v !== null && typeof v === "object") {
    if (v.nan) {
      return NaN;
    }
    if (v.inf === 1) {
      return Infinity;
    }
    if (v.inf === -1) {
      return -Infinity;
    }
    if (v.neg0) {
      return -0;
    }
  }
  return v;
}

function sameNumber(got, expectedRaw) {
  const expected = parseNumber(expectedRaw);
  if (Number.isNaN(expected)) {
    return Number.isNaN(got);
  }
  return Object.is(got, expected);
}

function fail(where, got, expected) {
  const g = Object.is(got, -0) ? "-0" : String(got);
  const e = JSON.stringify(expected);
  console.error(`${where}: got ${g}, expected ${e}`);
  process.exit(1);
}

function makeFp(TFP, spec) {
  return new TFP.FloatingPoint(spec.sign_bits, spec.exponent_bits, spec.mantissa_bits, spec.bias, spec.bits, {
    reserved_exponent: spec.reserved_exponent,
    max_mantissa_at_max_exponent: spec.max_mantissa_at_max_exponent,
  });
}

function sortedSet(xs) {
  return [...new Set(xs)].slice().sort();
}

function main() {
  const TFP = loadTfp();
  const gold = JSON.parse(fs.readFileSync(GOLDEN, "utf8"));

  if (!Array.isArray(gold.scale_encodes)) {
    fail("scale_encodes", gold.scale_encodes, "sorted(_SCALE_ENCODES) from Python");
  }
  const jsEncodes = sortedSet(TFP.SCALE_ENCODES);
  const pyEncodes = sortedSet(gold.scale_encodes);
  if (jsEncodes.length !== pyEncodes.length || jsEncodes.some((name, i) => name !== pyEncodes[i])) {
    fail(
      "TFP.SCALE_ENCODES set — add the string to SCALE_ENCODES and a branch in encode_scale (docs/javascripts/fp-codec.js)",
      jsEncodes.join(","),
      pyEncodes,
    );
  }

  for (const [name, block] of Object.entries(gold.decode)) {
    const fp = makeFp(TFP, block.fp);
    if (!sameNumber(fp.minimum, block.minimum)) {
      fail(`decode.${name}.minimum`, fp.minimum, block.minimum);
    }
    if (!sameNumber(fp.maximum, block.maximum)) {
      fail(`decode.${name}.maximum`, fp.maximum, block.maximum);
    }
    if (!sameNumber(fp.epsilon, block.epsilon)) {
      fail(`decode.${name}.epsilon`, fp.epsilon, block.epsilon);
    }
    if (block.codes.length !== 2 ** fp.bits) {
      fail(`decode.${name}.codes.length`, block.codes.length, 2 ** fp.bits);
    }
    for (let code = 0; code < block.codes.length; code++) {
      const got = fp.bit_pattern_to_custom_fp(code);
      if (!sameNumber(got, block.codes[code])) {
        fail(`decode.${name}.codes[${code}]`, got, block.codes[code]);
      }
    }
  }

  for (const case_ of gold.round) {
    const fp = makeFp(TFP, case_.fp);
    for (let i = 0; i < case_.x.length; i++) {
      const got = TFP.round_scalar(parseNumber(case_.x[i]), fp);
      if (!sameNumber(got, case_.y[i])) {
        fail(`round.${case_.name}[${i}] x=${case_.x[i]}`, got, case_.y[i]);
      }
    }
  }

  for (const case_ of gold.round_e0) {
    const fp = makeFp(TFP, case_.fp);
    for (let i = 0; i < case_.x.length; i++) {
      const got = TFP.round_scalar(parseNumber(case_.x[i]), fp);
      if (!sameNumber(got, case_.y[i])) {
        fail(`round_e0.${case_.name}[${i}]`, got, case_.y[i]);
      }
    }
  }

  for (const case_ of gold.encode_scale) {
    const elem_fp = makeFp(TFP, case_.elem);
    const scale_fp = makeFp(TFP, case_.scale);
    const got = TFP.encode_scale(parseNumber(case_.stat), {
      elem_fp,
      scale_fp,
      scale_encode: case_.scale_encode,
      M: case_.M,
    });
    if (!sameNumber(got, case_.s)) {
      fail(`encode_scale.${case_.name}`, got, case_.s);
    }
  }

  for (let i = 0; i < gold.reconstruct.length; i++) {
    const case_ = gold.reconstruct[i];
    const fp = makeFp(TFP, case_.fp);
    const got = TFP.reconstruct(
      parseNumber(case_.x),
      fp,
      parseNumber(case_.s),
      parseNumber(case_.s_global),
      parseNumber(case_.zero_point),
    );
    if (!sameNumber(got.e, case_.e)) {
      fail(`reconstruct[${i}].e`, got.e, case_.e);
    }
    if (!sameNumber(got.y, case_.y)) {
      fail(`reconstruct[${i}].y`, got.y, case_.y);
    }
  }

  console.log(`ok ${path.relative(ROOT, GOLDEN)} vs ${path.relative(ROOT, CODEC)}`);
}

main();
