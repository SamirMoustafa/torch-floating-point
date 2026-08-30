let explorerCleanup = null;

document$.subscribe(() => {
  const root = document.querySelector(".format-explorer");
  if (!root) {
    if (explorerCleanup) {
      explorerCleanup();
      explorerCleanup = null;
    }
    return;
  }
  if (root.dataset.feReady === "1") {
    return;
  }
  if (explorerCleanup) {
    explorerCleanup();
    explorerCleanup = null;
  }
  explorerCleanup = mountExplorer(root) || null;
});

const FE_LIGHT = {
  ink: "#111111",
  muted: "#6b7280",
  grid: "#e6e6e6",
  orange: "#ee4c2c",
  green: "#2a9d6e",
  blue: "#3b6fd8",
  legend: "#ffffff",
  paper: "rgba(0,0,0,0)",
};
const FE_DARK = {
  ink: "#e2e4ea",
  muted: "#8b93a3",
  grid: "#3a4050",
  orange: "#ff7a5c",
  green: "#8fb389",
  blue: "#8aa4cc",
  legend: "#1c2029",
  paper: "rgba(0,0,0,0)",
};

const ELEM_PRESETS = {
  e2m1: { label: "E2M1", sign_bits: 1, exponent_bits: 2, mantissa_bits: 1, bias: 1, reserved_exponent: false },
  e1m2: { label: "E1M2", sign_bits: 1, exponent_bits: 1, mantissa_bits: 2, bias: 0, reserved_exponent: false },
  e3m2: { label: "E3M2", sign_bits: 1, exponent_bits: 3, mantissa_bits: 2, bias: 3, reserved_exponent: false },
  e2m3: { label: "E2M3", sign_bits: 1, exponent_bits: 2, mantissa_bits: 3, bias: 1, reserved_exponent: false },
  e4m3fn: {
    label: "E4M3-FN",
    sign_bits: 1,
    exponent_bits: 4,
    mantissa_bits: 3,
    bias: 7,
    reserved_exponent: false,
    max_mantissa_at_max_exponent: 6,
  },
  e4m3_240: {
    label: "E4M3 max 240",
    sign_bits: 1,
    exponent_bits: 4,
    mantissa_bits: 3,
    bias: 7,
    reserved_exponent: true,
  },
  e5m2: { label: "E5M2", sign_bits: 1, exponent_bits: 5, mantissa_bits: 2, bias: 15, reserved_exponent: true },
  ue8m0: { label: "UE8M0", sign_bits: 0, exponent_bits: 8, mantissa_bits: 0, bias: 127, reserved_exponent: true },
  e4m3fnuz: {
    label: "E4M3 FNUZ",
    sign_bits: 1,
    exponent_bits: 4,
    mantissa_bits: 3,
    bias: 8,
    reserved_exponent: false,
  },
  e5m2fnuz: {
    label: "E5M2 FNUZ",
    sign_bits: 1,
    exponent_bits: 5,
    mantissa_bits: 2,
    bias: 16,
    reserved_exponent: false,
  },
  mxint8: { label: "MXINT8", sign_bits: 1, exponent_bits: 0, mantissa_bits: 7, bias: 0, reserved_exponent: true },
};

const BLOCK_RECIPES = {
  nvfp4: {
    label: "NVFP4",
    elem: "e2m1",
    scale: "e4m3fn",
    block_size: 16,
    M: 6,
    scale_encode: "nearest",
    s_global: 1,
    zero_point: 0,
  },
  mxfp8_nvidia: {
    label: "MXFP8 NVIDIA",
    elem: "e4m3fn",
    scale: "ue8m0",
    block_size: 32,
    M: 448,
    scale_encode: "ue8m0_ceil",
    s_global: 1,
    zero_point: 0,
  },
  mxfp8_ocp: {
    label: "MXFP8 OCP",
    elem: "e4m3fn",
    scale: "ue8m0",
    block_size: 32,
    M: 448,
    scale_encode: "ocp_floor",
    s_global: 1,
    zero_point: 0,
  },
  mxfp8_aws: {
    label: "MXFP8 AWS",
    elem: "e4m3fn",
    scale: "ue8m0",
    block_size: 32,
    M: 448,
    scale_encode: "ocp_floor_x2",
    s_global: 1,
    zero_point: 0,
  },
  mxfp4: {
    label: "MXFP4",
    elem: "e2m1",
    scale: "ue8m0",
    block_size: 32,
    M: 6,
    scale_encode: "ue8m0_ceil",
    s_global: 1,
    zero_point: 0,
  },
  mxint8: {
    label: "MXINT8",
    elem: "mxint8",
    scale: "ue8m0",
    block_size: 32,
    M: 127 / 64,
    scale_encode: "ocp_floor",
    s_global: 1,
    zero_point: 0,
  },
  hopper: {
    label: "Hopper FP8",
    elem: "e4m3fn",
    scale: "ue8m0",
    block_size: 128,
    M: 448,
    scale_encode: "amax_over_M",
    s_global: 1,
    zero_point: 0,
  },
};

const X_WINDOWS = [
  { id: "full", label: "Full range" },
  { id: "1", label: "±1" },
  { id: "2", label: "±2" },
  { id: "4", label: "±4" },
  { id: "8", label: "±8" },
  { id: "16", label: "±16" },
  { id: "32", label: "±32" },
  { id: "256", label: "±256" },
];

function colorScheme() {
  return (
    document.body.getAttribute("data-md-color-scheme") ||
    document.documentElement.getAttribute("data-md-color-scheme") ||
    "default"
  );
}

function palette() {
  return colorScheme() === "slate" ? FE_DARK : FE_LIGHT;
}

function defaultMaxMant(mantissa_bits) {
  return 2 ** mantissa_bits - 1;
}

function presetToKnobs(preset) {
  return {
    sign_bits: preset.sign_bits,
    exponent_bits: preset.exponent_bits,
    mantissa_bits: preset.mantissa_bits,
    bias: preset.bias,
    reserved_exponent: preset.reserved_exponent,
    max_mantissa_at_max_exponent:
      preset.max_mantissa_at_max_exponent == null
        ? defaultMaxMant(preset.mantissa_bits)
        : preset.max_mantissa_at_max_exponent,
  };
}

function knobsMatchPreset(knobs, preset) {
  const p = presetToKnobs(preset);
  return (
    knobs.sign_bits === p.sign_bits &&
    knobs.exponent_bits === p.exponent_bits &&
    knobs.mantissa_bits === p.mantissa_bits &&
    knobs.bias === p.bias &&
    knobs.reserved_exponent === p.reserved_exponent &&
    knobs.max_mantissa_at_max_exponent === p.max_mantissa_at_max_exponent
  );
}

function matchPresetId(knobs) {
  for (const [id, preset] of Object.entries(ELEM_PRESETS)) {
    if (knobsMatchPreset(knobs, preset)) {
      return id;
    }
  }
  return "custom";
}

function fmt(x, digits = 6) {
  if (Number.isNaN(x)) {
    return "NaN";
  }
  if (!Number.isFinite(x)) {
    return x > 0 ? "∞" : "−∞";
  }
  if (Object.is(x, -0)) {
    return "-0";
  }
  const ax = Math.abs(x);
  if (ax !== 0 && (ax >= 1e6 || ax < 1e-4)) {
    return x.toExponential(4);
  }
  return String(Number(x.toPrecision(digits)));
}

function superInt(n) {
  const digits = "⁰¹²³⁴⁵⁶⁷⁸⁹";
  const body = String(Math.abs(n))
    .split("")
    .map((d) => digits[Number(d)])
    .join("");
  return n < 0 ? `⁻${body}` : body;
}

function fmtDisp(x, digits = 6) {
  if (Object.is(x, -0)) {
    return "−0";
  }
  const raw = fmt(x, digits);
  const sci = /^(-?)(\d+(?:\.\d+)?)e([+-]\d+)$/.exec(raw);
  if (!sci) {
    return raw.replace("-", "−");
  }
  const mant = `${sci[1] ? "−" : ""}${sci[2]}`;
  return `${mant}×10${superInt(Number(sci[3]))}`;
}

function appendStat(parent, key, value) {
  const chip = document.createElement("span");
  chip.className = "format-explorer__stat";
  const kEl = document.createElement("span");
  kEl.className = "format-explorer__stat-key";
  kEl.textContent = key;
  const vEl = document.createElement("span");
  vEl.className = "format-explorer__stat-value";
  vEl.textContent = value;
  chip.appendChild(kEl);
  chip.appendChild(vEl);
  parent.appendChild(chip);
}

function fmtM(M) {
  if (Math.abs(M - 127 / 64) < 1e-12) {
    return "127 / 64";
  }
  if (Number.isInteger(M)) {
    return String(M);
  }
  return String(Number(M.toPrecision(8)));
}

function linspace(lo, hi, n) {
  const out = [];
  if (n <= 1) {
    out.push(lo);
    return out;
  }
  const step = (hi - lo) / (n - 1);
  for (let i = 0; i < n; i++) {
    out.push(lo + step * i);
  }
  return out;
}

function logspace(lo, hi, n) {
  return linspace(Math.log(lo), Math.log(hi), n).map(Math.exp);
}

function uniqueSorted(xs) {
  const copy = xs.slice().sort((a, b) => a - b);
  const out = [];
  let last = NaN;
  for (const x of copy) {
    if (!Number.isFinite(x)) {
      continue;
    }
    if (x !== last) {
      out.push(x);
      last = x;
    }
  }
  return out;
}

function sampleXs(lo, hi, fp, useLog) {
  const raw = useLog ? logspace(Math.max(lo, Number.MIN_VALUE), hi, 1600) : linspace(lo, hi, 2200);
  const extra = [];
  const finite = fp.finite_values();
  for (let i = 0; i < finite.length; i++) {
    const c = finite[i];
    if (c >= lo && c <= hi && (!useLog || c > 0)) {
      extra.push(c);
    }
    if (i + 1 < finite.length) {
      const mid = 0.5 * (finite[i] + finite[i + 1]);
      if (mid >= lo && mid <= hi && (!useLog || mid > 0)) {
        extra.push(mid);
      }
    }
  }
  extra.push(lo, hi);
  return uniqueSorted(raw.concat(extra));
}

function downsample(arr, maxN) {
  if (arr.length <= maxN) {
    return arr;
  }
  const out = [];
  const step = (arr.length - 1) / (maxN - 1);
  for (let i = 0; i < maxN; i++) {
    out.push(arr[Math.round(i * step)]);
  }
  return out;
}

function fillSelect(select, entries, withCustom) {
  select.replaceChildren();
  for (const [id, preset] of Object.entries(entries)) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = preset.label;
    select.appendChild(opt);
  }
  if (withCustom) {
    const opt = document.createElement("option");
    opt.value = "custom";
    opt.textContent = "Custom";
    select.appendChild(opt);
  }
}

function constructorLine(name, fp) {
  return fp.to_constructor(name);
}

function pythonSnippet(state, elem_fp, scale_fp) {
  const lines = [];
  if (state.block) {
    lines.push("from floating_point import BlockFormat, BlockRound, FloatingPoint");
    lines.push("");
    lines.push(constructorLine("elem_fp", elem_fp));
    lines.push(constructorLine("scale_fp", scale_fp));
    const size =
      state.tile && state.blockH > 0 && state.blockW > 0
        ? `(${state.blockH}, ${state.blockW})`
        : String(state.block_size);
    const extra = [];
    if (state.s_global !== 1) {
      extra.push(`s_global=${fmt(state.s_global, 8)}`);
    }
    if (state.zero_point !== 0) {
      extra.push(`zero_point=${fmt(state.zero_point, 8)}`);
    }
    const extraStr = extra.length ? `, ${extra.join(", ")}` : "";
    lines.push(`spec = BlockFormat(elem_fp, scale_fp, ${size}, ${fmtM(state.M)}, ${JSON.stringify(state.scale_encode)}${extraStr})`);
    if (state.sOverride != null) {
      lines.unshift("import torch");
      lines.push(`y = BlockRound(spec)(x, scales=torch.tensor([${fmt(state.sOverride, 8)}]))`);
    } else {
      lines.push("y = BlockRound(spec)(x)");
    }
  } else {
    lines.push("from floating_point import FloatingPoint, Round");
    lines.push("");
    lines.push(constructorLine("fp", elem_fp));
    lines.push("y = Round(fp)(x)");
  }
  return lines.join("\n");
}

function positiveMin(fp) {
  for (const v of fp.finite_values()) {
    if (v > 0) {
      return v;
    }
  }
  return Math.max(fp.epsilon, 1e-45);
}

function xWindow(fp, windowId, useLog) {
  const min = fp.minimum;
  const max = fp.maximum;
  const pad = 0.2 * Math.max(Math.abs(min), Math.abs(max), 1e-6);
  if (useLog) {
    const lo = positiveMin(fp);
    if (windowId === "full") {
      return [lo, max > 0 ? max * 1.2 : lo * 4];
    }
    const half = Number(windowId);
    return [lo, Math.max(half, lo * 2)];
  }
  if (windowId !== "full") {
    const half = Number(windowId);
    const lo = fp.is_signed ? -half : Math.min(0, min);
    return [lo, half];
  }
  return [min - pad, max + pad];
}

function domainToValue(t, lo, hi, useLog) {
  const u = window.TFP.clamp(t, 0, 1);
  if (useLog && lo > 0 && hi > 0) {
    return Math.exp(Math.log(lo) + u * (Math.log(hi) - Math.log(lo)));
  }
  return lo + u * (hi - lo);
}

function valueToDomain(value, lo, hi, useLog) {
  if (hi === lo) {
    return 0;
  }
  if (useLog && lo > 0 && hi > 0 && value > 0) {
    return window.TFP.clamp((Math.log(value) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)), 0, 1);
  }
  return window.TFP.clamp((value - lo) / (hi - lo), 0, 1);
}

function scaleSliderDomain(fp) {
  const lo = positiveMin(fp);
  const hi = Math.max(fp.maximum, lo * 2);
  return [lo, hi];
}

function sgSliderDomain(sg) {
  if (!Number.isFinite(sg) || sg === 0) {
    return [0.125, 8];
  }
  if (sg < 0) {
    const mag = Math.max(Math.abs(sg), 0.125);
    return [-mag * 4, -mag / 4];
  }
  return [Math.min(0.125, sg / 4), Math.max(8, sg * 4)];
}

function zSliderDomain(fp) {
  const lo = fp.minimum;
  const hi = fp.maximum;
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) {
    const pad = Math.max(Math.abs(lo) || 1, Math.abs(hi) || 1);
    return [lo - pad, hi + pad];
  }
  const pad = 0.05 * (hi - lo);
  return [lo - pad, hi + pad];
}

function setSlider(rangeEl, numEl, value, lo, hi, useLog) {
  if (!rangeEl || !Number.isFinite(value)) {
    return;
  }
  if (document.activeElement !== rangeEl) {
    rangeEl.value = String(valueToDomain(value, lo, hi, useLog));
  }
  if (numEl && document.activeElement !== numEl) {
    numEl.value = String(Number(value.toPrecision(8)));
  }
}

function autoLog(fp) {
  if (fp.is_signed) {
    return false;
  }
  const min = Math.max(fp.minimum, Number.MIN_VALUE);
  return fp.maximum > 64 || fp.maximum / min > 1e6;
}

function mapValues(xs, fn) {
  const y = new Array(xs.length);
  for (let i = 0; i < xs.length; i++) {
    y[i] = fn(xs[i]);
  }
  return y;
}

function axisStyle(pal, useLog, lo, hi, title, matches, setRange) {
  const ax = {
    gridcolor: pal.grid,
    zerolinecolor: pal.muted,
    linecolor: pal.ink,
    tickfont: { color: pal.ink, size: 11 },
    title: title ? { text: title, font: { color: pal.ink, size: 12 }, standoff: 8 } : undefined,
    type: useLog ? "log" : "linear",
    color: pal.ink,
    showline: true,
    mirror: false,
    automargin: true,
    ticklabeloverflow: "allow",
    ticks: "outside",
    ticklen: 4,
    showspikes: true,
    spikemode: "across",
    spikesnap: "cursor",
    spikedash: "dot",
    spikethickness: 1,
    spikecolor: pal.ink,
  };
  if (matches) {
    ax.matches = matches;
  } else if (setRange !== false) {
    if (useLog) {
      ax.range = [Math.log10(lo), Math.log10(hi)];
    } else {
      ax.range = [lo, hi];
    }
  }
  return ax;
}

function yAxisStyle(pal, title, domain) {
  return {
    domain,
    gridcolor: pal.grid,
    zerolinecolor: pal.muted,
    linecolor: pal.ink,
    tickfont: { color: pal.ink, size: 11 },
    title: { text: title, font: { color: pal.ink, size: 12 }, standoff: 8 },
    color: pal.ink,
    automargin: true,
    ticklabeloverflow: "allow",
    ticks: "outside",
    ticklen: 4,
  };
}

function readKnobs(root, prefix) {
  const num = (id) => Number(root.querySelector(id).value);
  const reserved_exponent = root.querySelector(`${prefix}-reserved`).checked;
  const mantissa_bits = Math.max(0, Math.trunc(num(`${prefix}-mant`)));
  let max_mantissa_at_max_exponent = Math.trunc(num(`${prefix}-maxmant`));
  const maxAllowed = defaultMaxMant(mantissa_bits);
  if (!Number.isFinite(max_mantissa_at_max_exponent) || max_mantissa_at_max_exponent < 0) {
    max_mantissa_at_max_exponent = maxAllowed;
  }
  max_mantissa_at_max_exponent = Math.min(max_mantissa_at_max_exponent, maxAllowed);
  return {
    sign_bits: Math.min(1, Math.max(0, Math.trunc(num(`${prefix}-sign`)))),
    exponent_bits: Math.max(0, Math.trunc(num(`${prefix}-exp`))),
    mantissa_bits,
    bias: Math.trunc(num(`${prefix}-bias`)),
    reserved_exponent,
    max_mantissa_at_max_exponent,
  };
}

function writeKnobs(root, prefix, knobs) {
  root.querySelector(`${prefix}-sign`).value = String(knobs.sign_bits);
  root.querySelector(`${prefix}-exp`).value = String(knobs.exponent_bits);
  root.querySelector(`${prefix}-mant`).value = String(knobs.mantissa_bits);
  root.querySelector(`${prefix}-bias`).value = String(knobs.bias);
  root.querySelector(`${prefix}-reserved`).checked = knobs.reserved_exponent;
  root.querySelector(`${prefix}-maxmant`).value = String(knobs.max_mantissa_at_max_exponent);
  root.querySelector(`${prefix}-maxmant`).max = String(defaultMaxMant(knobs.mantissa_bits));
  const bits = knobs.sign_bits + knobs.exponent_bits + knobs.mantissa_bits;
  const bitsEl = root.querySelector(`${prefix}-bits`);
  bitsEl.value = String(bits);
  bitsEl.classList.toggle("is-invalid", bits < 1 || bits > window.TFP.MAX_BITS);
}

function makeFp(knobs) {
  const bits = knobs.sign_bits + knobs.exponent_bits + knobs.mantissa_bits;
  return new window.TFP.FloatingPoint(knobs.sign_bits, knobs.exponent_bits, knobs.mantissa_bits, knobs.bias, bits, {
    reserved_exponent: knobs.reserved_exponent,
    max_mantissa_at_max_exponent: knobs.max_mantissa_at_max_exponent,
  });
}

function blockSizeK(state) {
  if (state.tile) {
    return Math.max(1, state.blockH * state.blockW);
  }
  return Math.max(1, state.block_size);
}

function frameSize(el) {
  const frame = el && el.parentElement;
  if (!frame) {
    return null;
  }
  const w = Math.round(frame.clientWidth);
  const h = Math.round(frame.clientHeight);
  if (w < 64 || h < 64) {
    return null;
  }
  return { w, h };
}

function mountExplorer(root) {
  if (root.dataset.feReady === "1") {
    return;
  }
  if (!window.TFP) {
    root.querySelector("[data-fe-error]").hidden = false;
    root.querySelector("[data-fe-error]").textContent = "Codec failed to load.";
    return;
  }
  if (typeof Plotly === "undefined") {
    root.querySelector("[data-fe-error]").hidden = false;
    root.querySelector("[data-fe-error]").textContent = "Plotly failed to load.";
    return;
  }
  root.dataset.feReady = "1";

  const elemPreset = root.querySelector("#fe-elem-preset");
  const scalePreset = root.querySelector("#fe-scale-preset");
  const encodeSelect = root.querySelector("#fe-encode");
  const xRange = root.querySelector("#fe-xrange");
  const recipes = root.querySelector("#fe-recipes");
  const plotEl = root.querySelector("#fe-plot");
  const blockPlotEl = root.querySelector("#fe-block-plot");
  const snippet = root.querySelector("[data-fe-snippet]");
  const stats = root.querySelector("[data-fe-stats]");
  const readout = root.querySelector("[data-fe-readout]");
  const probeRange = root.querySelector("#fe-probe");
  const probeNum = root.querySelector("#fe-probe-num");
  const sRange = root.querySelector("#fe-s-range");
  const sNum = root.querySelector("#fe-s-num");
  const sgRange = root.querySelector("#fe-sg-range");
  const sgNum = root.querySelector("#fe-sg-num");
  const zRange = root.querySelector("#fe-z-range");
  const zNum = root.querySelector("#fe-z-num");
  const errorBox = root.querySelector("[data-fe-error]");

  fillSelect(elemPreset, ELEM_PRESETS, true);
  fillSelect(scalePreset, ELEM_PRESETS, true);
  encodeSelect.replaceChildren();
  for (const name of window.TFP.SCALE_ENCODES) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    encodeSelect.appendChild(opt);
  }
  xRange.replaceChildren();
  for (const w of X_WINDOWS) {
    const opt = document.createElement("option");
    opt.value = w.id;
    opt.textContent = w.label;
    xRange.appendChild(opt);
  }
  recipes.replaceChildren();
  for (const [id, recipe] of Object.entries(BLOCK_RECIPES)) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "format-explorer__chip";
    btn.dataset.recipe = id;
    btn.textContent = recipe.label;
    recipes.appendChild(btn);
  }

  let syncing = false;
  let probe = 1.5;
  let sOverride = null;
  let lastLogAxis = false;
  let lastDomain = [0, 1];
  let lastSDomain = [1e-3, 1];
  let lastSLog = true;
  let lastSgDomain = [0.125, 8];
  let lastSgLog = true;
  let lastZDomain = [-1, 1];
  let lastEncodeKey = "";
  let dragging = null;
  const abort = new AbortController();
  const on = { signal: abort.signal };

  function isBlock() {
    return root.querySelector("#fe-mode-block").checked;
  }

  function readState() {
    const tile = root.querySelector("#fe-tile").checked;
    return {
      block: isBlock(),
      scale_encode: encodeSelect.value,
      M: Number(root.querySelector("#fe-M").value),
      s_global: Number(root.querySelector("#fe-sg").value),
      zero_point: Number(root.querySelector("#fe-z").value),
      amax: Number(root.querySelector("#fe-amax").value),
      block_size: Math.max(1, Math.trunc(Number(root.querySelector("#fe-block-size").value) || 16)),
      tile,
      blockH: Math.max(1, Math.trunc(Number(root.querySelector("#fe-block-h").value) || 16)),
      blockW: Math.max(1, Math.trunc(Number(root.querySelector("#fe-block-w").value) || 16)),
      xWindow: xRange.value,
      logX: root.querySelector("#fe-logx").checked,
      sOverride,
    };
  }

  function applyPreset(prefix, id) {
    if (id === "custom" || !ELEM_PRESETS[id]) {
      return;
    }
    const knobs = presetToKnobs(ELEM_PRESETS[id]);
    writeKnobs(root, prefix, knobs);
    if (prefix === "#fe-elem") {
      try {
        root.querySelector("#fe-logx").checked = autoLog(makeFp(knobs));
      } catch (_err) {
        root.querySelector("#fe-logx").checked = false;
      }
    }
  }

  function applyRecipe(id) {
    const recipe = BLOCK_RECIPES[id];
    if (!recipe) {
      return;
    }
    syncing = true;
    sOverride = null;
    lastEncodeKey = "";
    root.querySelector("#fe-mode-block").checked = true;
    elemPreset.value = recipe.elem;
    scalePreset.value = recipe.scale;
    applyPreset("#fe-elem", recipe.elem);
    applyPreset("#fe-scale", recipe.scale);
    encodeSelect.value = recipe.scale_encode;
    root.querySelector("#fe-M").value = String(recipe.M);
    root.querySelector("#fe-sg").value = String(recipe.s_global);
    root.querySelector("#fe-z").value = String(recipe.zero_point);
    root.querySelector("#fe-amax").value = String(recipe.M);
    root.querySelector("#fe-tile").checked = false;
    root.querySelector("#fe-block-size").value = String(recipe.block_size);
    syncing = false;
    syncPresetSelects();
    updateModeClass();
    draw(true);
  }

  function syncPresetSelects() {
    const elemId = matchPresetId(readKnobs(root, "#fe-elem"));
    const scaleId = matchPresetId(readKnobs(root, "#fe-scale"));
    if (elemPreset.value !== elemId) {
      elemPreset.value = elemId;
    }
    if (scalePreset.value !== scaleId) {
      scalePreset.value = scaleId;
    }
    const state = readState();
    for (const btn of recipes.querySelectorAll("[data-recipe]")) {
      const r = BLOCK_RECIPES[btn.dataset.recipe];
      const on =
        state.block &&
        r &&
        elemId === r.elem &&
        scaleId === r.scale &&
        state.scale_encode === r.scale_encode &&
        Math.abs(state.M - r.M) < 1e-12 &&
        !state.tile &&
        state.block_size === r.block_size;
      btn.classList.toggle("is-active", on);
    }
  }

  function updateModeClass() {
    root.classList.toggle("is-block", isBlock());
    root.querySelector("#fe-tile-fields").hidden = !root.querySelector("#fe-tile").checked;
    root.querySelector("#fe-block-size-wrap").hidden = root.querySelector("#fe-tile").checked;
  }

  function setProbe(value, fromSlider) {
    if (!Number.isFinite(value)) {
      return;
    }
    probe = value;
    if (!fromSlider) {
      const [lo, hi] = lastDomain;
      probeRange.value = String(valueToDomain(clampProbe(value, lo, hi), lo, hi, lastLogAxis));
    }
    probeNum.value = String(value);
  }

  function clampProbe(value, lo, hi) {
    return window.TFP.clamp(value, lo, hi);
  }

  function draw(resetRange, fromSlider) {
    errorBox.hidden = true;
    updateModeClass();
    const state = readState();
    let elem_fp;
    let scale_fp;
    try {
      elem_fp = makeFp(readKnobs(root, "#fe-elem"));
      scale_fp = makeFp(readKnobs(root, "#fe-scale"));
    } catch (err) {
      errorBox.hidden = false;
      errorBox.textContent = err.message;
      return;
    }
    const bits = elem_fp.bits;
    root.querySelector("#fe-elem-bits").value = String(bits);
    root.querySelector("#fe-scale-bits").value = String(scale_fp.bits);

    const logAxis = Boolean(state.logX);
    const [lo, hi] = xWindow(elem_fp, state.xWindow, logAxis);
    lastDomain = [lo, hi];
    lastLogAxis = logAxis;
    if (logAxis && probe <= 0) {
      probe = lo;
    }
    if (!fromSlider && (resetRange || probe < lo || probe > hi)) {
      const mid = logAxis ? Math.exp(0.5 * (Math.log(Math.max(lo, 1e-45)) + Math.log(hi))) : 0.25 * (lo + hi);
      const finite = elem_fp.finite_values().filter((v) => v >= lo && v <= hi && (!logAxis || v > 0));
      probe = finite.length ? finite[Math.min(finite.length - 1, Math.floor(finite.length * 0.7))] : mid;
    }
    if (!fromSlider) {
      setProbe(probe, false);
    } else {
      probeNum.value = String(Number(probe.toPrecision(8)));
    }

    const pal = palette();
    const xs = sampleXs(lo, hi, elem_fp, logAxis).filter((x) => !logAxis || x > 0);
    let s = 1;
    const sg = Number.isFinite(state.s_global) && state.s_global !== 0 ? state.s_global : 1;
    const z = Number.isFinite(state.zero_point) ? state.zero_point : 0;
    const M = Number.isFinite(state.M) && state.M > 0 ? state.M : 1;
    if (state.block) {
      const amax = Number.isFinite(state.amax) ? state.amax : M;
      const stat = state.scale_encode === "signed_peak" ? amax / sg : Math.abs(amax) / sg;
      const encodeKey = [
        amax,
        M,
        state.scale_encode,
        scale_fp.to_constructor("s"),
        state.block,
      ].join("|");
      if (encodeKey !== lastEncodeKey) {
        lastEncodeKey = encodeKey;
        sOverride = null;
      }
      try {
        s = window.TFP.encode_scale(stat, {
          elem_fp,
          scale_fp,
          M,
          scale_encode: state.scale_encode,
        });
      } catch (err) {
        errorBox.hidden = false;
        errorBox.textContent = err.message;
        return;
      }
      if (sOverride != null && Number.isFinite(sOverride) && sOverride !== 0) {
        s = sOverride;
      }
      state.sOverride = sOverride;
      lastSLog = autoLog(scale_fp) || scale_fp.maximum / Math.max(positiveMin(scale_fp), 1e-45) > 64;
      lastSDomain = scaleSliderDomain(scale_fp);
      lastZDomain = zSliderDomain(elem_fp);
      if (dragging !== sgRange) {
        lastSgLog = sg > 0;
        lastSgDomain = sgSliderDomain(sg);
      }
      setSlider(sRange, sNum, s, lastSDomain[0], lastSDomain[1], lastSLog);
      setSlider(sgRange, sgNum, sg, lastSgDomain[0], lastSgDomain[1], lastSgLog);
      setSlider(zRange, zNum, z, lastZDomain[0], lastZDomain[1], false);
      if (document.activeElement !== root.querySelector("#fe-sg")) {
        root.querySelector("#fe-sg").value = String(sg);
      }
      if (document.activeElement !== root.querySelector("#fe-z")) {
        root.querySelector("#fe-z").value = String(z);
      }
    }

    const yOf = (x) => {
      if (state.block) {
        return window.TFP.reconstruct(x, elem_fp, s, sg, z).y;
      }
      return window.TFP.round_scalar(x, elem_fp);
    };
    const ys = mapValues(xs, yOf);
    const err = xs.map((x, i) => x - ys[i]);
    const yProbe = yOf(probe);
    const recon = state.block ? window.TFP.reconstruct(probe, elem_fp, s, sg, z) : null;

    const elemStats = elem_fp.codebook_stats();
    const scaleStats = scale_fp.codebook_stats();
    const sat =
      state.block
        ? probe / (s * sg) + z < elem_fp.minimum || probe / (s * sg) + z > elem_fp.maximum
        : probe < elem_fp.minimum || probe > elem_fp.maximum;

    stats.replaceChildren();
    appendStat(stats, "range", `${fmtDisp(elemStats.minimum)} … ${fmtDisp(elemStats.maximum)}`);
    appendStat(stats, "ε", fmtDisp(elemStats.epsilon));
    appendStat(stats, "finite", String(elemStats.nFinite));
    if (elemStats.nInf) {
      appendStat(stats, "Inf", String(elemStats.nInf));
    }
    if (elemStats.nNan) {
      appendStat(stats, "NaN", String(elemStats.nNan));
    }
    if (state.block) {
      appendStat(stats, "s", fmtDisp(s));
      appendStat(stats, "scale", `${fmtDisp(scaleStats.minimum)} … ${fmtDisp(scaleStats.maximum)}`);
    }

    if (state.block && recon) {
      readout.textContent = `x ${fmtDisp(probe)} → y ${fmtDisp(yProbe)} · |x−y| ${fmtDisp(Math.abs(probe - yProbe))} · e ${fmtDisp(recon.e)} · s ${fmtDisp(s)} · s_g ${fmtDisp(sg)} · z ${fmtDisp(z)}${sat ? " · saturated" : ""}`;
    } else {
      readout.textContent = `x ${fmtDisp(probe)} → y ${fmtDisp(yProbe)} · |x−y| ${fmtDisp(Math.abs(probe - yProbe))}${sat ? " · saturated" : ""}`;
    }

    snippet.textContent = pythonSnippet(state, elem_fp, scale_fp);

    const yName = state.block ? "y = (e − z) s s_g" : "y = Round(x)";
    const elemCodes = downsample(
      elem_fp.finite_values().filter((v) => v >= lo && v <= hi && (!logAxis || v > 0)),
      2048,
    );
    const scaleCodes = state.block
      ? downsample(
          scale_fp.finite_values().filter((v) => v >= lo && v <= hi && (!logAxis || v > 0)),
          2048,
        )
      : [];

    const yBound = Math.max(
      Math.abs(lo),
      Math.abs(hi),
      ...ys.filter(Number.isFinite).map(Math.abs),
      Math.abs(elem_fp.maximum),
    );
    const yPad = yBound * 1.08 || 1;
    const errBound = Math.max(1e-12, ...err.filter(Number.isFinite).map(Math.abs));

    const traces = [
      {
        x: xs,
        y: xs,
        name: "x",
        xaxis: "x",
        yaxis: "y",
        type: "scatter",
        mode: "lines",
        line: { color: pal.muted, width: 1.2, dash: "dash" },
        hovertemplate: "x=%{x}<extra>x</extra>",
      },
      {
        x: xs,
        y: ys,
        name: yName,
        xaxis: "x",
        yaxis: "y",
        type: "scatter",
        mode: "lines",
        line: { color: pal.orange, width: 2.2 },
        hovertemplate: "x=%{x}<br>y=%{y}<extra></extra>",
      },
      {
        x: [probe],
        y: [yProbe],
        name: "probe",
        xaxis: "x",
        yaxis: "y",
        type: "scatter",
        mode: "markers",
        marker: { color: pal.ink, size: 10, symbol: "diamond" },
        hovertemplate: `x=${fmt(probe)}<br>y=${fmt(yProbe)}<extra>probe</extra>`,
      },
      {
        x: elemCodes,
        y: elemCodes.map(() => 1),
        name: "element codes",
        xaxis: "x2",
        yaxis: "y2",
        type: "scatter",
        mode: "markers",
        marker: { color: pal.orange, size: 7, symbol: "line-ns-open", line: { width: 1.6, color: pal.orange } },
        hovertemplate: "%{x}<extra>element</extra>",
      },
      {
        x: scaleCodes,
        y: scaleCodes.map(() => 0.45),
        name: "scale codes",
        xaxis: "x2",
        yaxis: "y2",
        type: "scatter",
        mode: "markers",
        marker: { color: pal.blue, size: 7, symbol: "line-ns-open", line: { width: 1.6, color: pal.blue } },
        hovertemplate: "%{x}<extra>scale</extra>",
        visible: Boolean(state.block && scaleCodes.length),
      },
      {
        x: [probe],
        y: [1],
        name: "probe",
        xaxis: "x2",
        yaxis: "y2",
        type: "scatter",
        mode: "markers",
        marker: { color: pal.ink, size: 10, symbol: "diamond" },
        showlegend: false,
        hoverinfo: "skip",
      },
      {
        x: xs,
        y: err,
        name: "x − y",
        xaxis: "x3",
        yaxis: "y3",
        type: "scatter",
        mode: "lines",
        line: { color: pal.green, width: 1.8 },
        hovertemplate: "x=%{x}<br>x−y=%{y}<extra></extra>",
      },
      {
        x: [probe],
        y: [probe - yProbe],
        name: "probe",
        xaxis: "x3",
        yaxis: "y3",
        type: "scatter",
        mode: "markers",
        marker: { color: pal.ink, size: 10, symbol: "diamond" },
        showlegend: false,
        hovertemplate: `x−y=${fmt(probe - yProbe)}<extra>probe</extra>`,
      },
    ];

    const layout = {
      paper_bgcolor: pal.paper,
      plot_bgcolor: pal.paper,
      font: { family: "Inter, system-ui, sans-serif", color: pal.ink, size: 12 },
      margin: { l: 62, r: 28, t: 36, b: 52, pad: 0 },
      legend: {
        bgcolor: pal.legend,
        bordercolor: pal.grid,
        borderwidth: 1,
        font: { size: 12 },
        orientation: "h",
        y: 1.08,
        x: 0,
      },
      hovermode: "x",
      hoverdistance: 40,
      spikedistance: -1,
      dragmode: false,
      autosize: true,
      uirevision: [
        elem_fp.to_constructor("e"),
        scale_fp.to_constructor("s"),
        state.block,
        state.scale_encode,
        lo,
        hi,
        logAxis,
      ].join("|"),
      xaxis: {
        ...axisStyle(pal, logAxis, lo, hi, "", null, !fromSlider),
        domain: [0, 1],
        anchor: "y",
        showticklabels: true,
      },
      xaxis2: {
        ...axisStyle(pal, logAxis, lo, hi, "", "x", !fromSlider),
        domain: [0, 1],
        anchor: "y2",
        showticklabels: true,
      },
      xaxis3: {
        ...axisStyle(pal, logAxis, lo, hi, "x", "x", !fromSlider),
        domain: [0, 1],
        anchor: "y3",
        showticklabels: true,
      },
      yaxis: {
        ...yAxisStyle(pal, "x, y", [0.74, 1]),
        type: "linear",
        range: fromSlider
          ? undefined
          : elem_fp.is_signed && !logAxis
            ? [-yPad, yPad]
            : [Math.min(0, lo) - (logAxis ? 0 : 0.05 * yPad), yPad],
      },
      yaxis2: {
        ...yAxisStyle(pal, "codes", [0.42, 0.62]),
        range: [0, 1.35],
        showticklabels: false,
        zeroline: false,
      },
      yaxis3: {
        ...yAxisStyle(pal, "x − y", [0, 0.28]),
        range: fromSlider ? undefined : [-errBound * 1.15, errBound * 1.15],
      },
    };

    const config = {
      responsive: false,
      displaylogo: false,
      displayModeBar: false,
      scrollZoom: false,
      doubleClick: false,
    };
    const plotSize = frameSize(plotEl);
    if (plotSize) {
      layout.width = plotSize.w;
      layout.height = plotSize.h;
      layout.autosize = false;
    }
    Plotly.react(plotEl, traces, layout, config);
    if (!plotEl.dataset.feBound) {
      plotEl.dataset.feBound = "1";
      plotEl.on("plotly_click", (ev) => {
        const pt = ev.points && ev.points[0];
        if (!pt || !Number.isFinite(pt.x)) {
          return;
        }
        setProbe(pt.x, false);
        draw(false, true);
      });
    }

    if (state.block) {
      const k = Math.min(64, blockSizeK(state));
      const amax = Number.isFinite(state.amax) ? Math.abs(state.amax) : M;
      const demoLo = elem_fp.is_signed ? -amax : 0;
      const idx = [];
      const xDemo = [];
      const yDemo = [];
      for (let i = 0; i < k; i++) {
        const x = k === 1 ? amax : demoLo + ((amax - demoLo) * i) / (k - 1);
        idx.push(i);
        xDemo.push(x);
        yDemo.push(yOf(x));
      }
      const blockLayout = {
        paper_bgcolor: pal.paper,
        plot_bgcolor: pal.paper,
        font: { family: "Inter, system-ui, sans-serif", color: pal.ink, size: 12 },
        autosize: false,
        dragmode: false,
        hovermode: "closest",
        margin: { l: 62, r: 28, t: 48, b: 48 },
        legend: { bgcolor: pal.legend, orientation: "h", y: 1.15, x: 0 },
        title: {
          text: `Demo block of ${k} (linspace in [${fmt(demoLo)}, ${fmt(amax)}])`,
          font: { size: 13, color: pal.ink },
        },
        xaxis: { ...axisStyle(pal, false, -0.5, k - 0.5, "index"), range: [-0.5, k - 0.5] },
        yaxis: yAxisStyle(pal, "value", [0, 1]),
      };
      const blockSize = frameSize(blockPlotEl);
      if (blockSize) {
        blockLayout.width = blockSize.w;
        blockLayout.height = blockSize.h;
      }
      Plotly.react(
        blockPlotEl,
        [
          {
            x: idx,
            y: xDemo,
            name: "xᵢ",
            type: "scatter",
            mode: "lines+markers",
            line: { color: pal.muted, width: 1.2, dash: "dash" },
            marker: { size: 6, color: pal.muted },
          },
          {
            x: idx,
            y: yDemo,
            name: "yᵢ",
            type: "scatter",
            mode: "lines+markers",
            line: { color: pal.orange, width: 2 },
            marker: { size: 7, color: pal.orange },
          },
        ],
        blockLayout,
        config,
      );
    } else if (blockPlotEl.data) {
      Plotly.purge(blockPlotEl);
    }
  }

  function onKnobChange() {
    if (syncing) {
      return;
    }
    const prefix = this.id.startsWith("fe-scale") ? "#fe-scale" : "#fe-elem";
    if (this.id.endsWith("-mant") || this.id.endsWith("-exp") || this.id.endsWith("-sign")) {
      const knobs = readKnobs(root, prefix);
      knobs.max_mantissa_at_max_exponent = Math.min(
        knobs.max_mantissa_at_max_exponent,
        defaultMaxMant(knobs.mantissa_bits),
      );
      writeKnobs(root, prefix, knobs);
    } else {
      writeKnobs(root, prefix, readKnobs(root, prefix));
    }
    syncPresetSelects();
    draw(true);
  }

  elemPreset.addEventListener(
    "change",
    () => {
      if (syncing) {
        return;
      }
      syncing = true;
      applyPreset("#fe-elem", elemPreset.value);
      syncing = false;
      draw(true);
    },
    on,
  );
  scalePreset.addEventListener(
    "change",
    () => {
      if (syncing) {
        return;
      }
      syncing = true;
      applyPreset("#fe-scale", scalePreset.value);
      syncing = false;
      draw(true);
    },
    on,
  );
  recipes.addEventListener(
    "click",
    (event) => {
      const btn = event.target.closest("[data-recipe]");
      if (btn) {
        applyRecipe(btn.dataset.recipe);
      }
    },
    on,
  );

  for (const sel of root.querySelectorAll(
    "#fe-elem-sign, #fe-elem-exp, #fe-elem-mant, #fe-elem-bias, #fe-elem-maxmant, #fe-elem-reserved, #fe-scale-sign, #fe-scale-exp, #fe-scale-mant, #fe-scale-bias, #fe-scale-maxmant, #fe-scale-reserved",
  )) {
    sel.addEventListener("change", onKnobChange, on);
    sel.addEventListener("input", onKnobChange, on);
  }
  for (const sel of root.querySelectorAll(
    "#fe-mode-element, #fe-mode-block, #fe-encode, #fe-M, #fe-sg, #fe-z, #fe-amax, #fe-block-size, #fe-block-h, #fe-block-w, #fe-tile, #fe-xrange, #fe-logx",
  )) {
    sel.addEventListener(
      "change",
      () => {
        updateModeClass();
        syncPresetSelects();
        draw(true);
      },
      on,
    );
    if (sel.type === "number") {
      sel.addEventListener(
        "input",
        () => {
          updateModeClass();
          draw(false);
        },
        on,
      );
    }
  }

  probeRange.addEventListener(
    "input",
    () => {
      const [lo, hi] = lastDomain;
      probe = domainToValue(Number(probeRange.value), lo, hi, lastLogAxis);
      probeNum.value = String(Number(probe.toPrecision(8)));
      draw(false, true);
    },
    on,
  );
  probeNum.addEventListener(
    "change",
    () => {
      setProbe(Number(probeNum.value), false);
      draw(false, true);
    },
    on,
  );

  function bindDomainSlider(rangeEl, numEl, domainOf, logOf, write) {
    if (!rangeEl || !numEl) {
      return;
    }
    rangeEl.addEventListener(
      "input",
      () => {
        dragging = rangeEl;
        const [lo, hi] = domainOf();
        const useLog = logOf();
        write(domainToValue(Number(rangeEl.value), lo, hi, useLog));
        draw(false, true);
        dragging = null;
      },
      on,
    );
    numEl.addEventListener(
      "change",
      () => {
        const v = Number(numEl.value);
        if (!Number.isFinite(v)) {
          return;
        }
        write(v);
        draw(false, true);
      },
      on,
    );
  }

  bindDomainSlider(
    sRange,
    sNum,
    () => lastSDomain,
    () => lastSLog,
    (v) => {
      sOverride = v === 0 ? null : v;
    },
  );
  bindDomainSlider(
    sgRange,
    sgNum,
    () => lastSgDomain,
    () => lastSgLog,
    (v) => {
      if (v === 0 || !Number.isFinite(v)) {
        return;
      }
      root.querySelector("#fe-sg").value = String(v);
    },
  );
  bindDomainSlider(
    zRange,
    zNum,
    () => lastZDomain,
    () => false,
    (v) => {
      if (!Number.isFinite(v)) {
        return;
      }
      root.querySelector("#fe-z").value = String(v);
    },
  );

  function resizePlot(el) {
    if (!el || !el.isConnected || !el.data) {
      return;
    }
    const size = frameSize(el);
    if (!size) {
      return;
    }
    const layout = el._fullLayout;
    if (layout && Math.abs(layout.width - size.w) < 2 && Math.abs(layout.height - size.h) < 2) {
      return;
    }
    Plotly.relayout(el, { width: size.w, height: size.h, autosize: false });
  }

  let resizeTimer = 0;
  const resize = new ResizeObserver(() => {
    if (!root.isConnected) {
      return;
    }
    if (resizeTimer) {
      cancelAnimationFrame(resizeTimer);
    }
    resizeTimer = requestAnimationFrame(() => {
      resizeTimer = 0;
      resizePlot(plotEl);
      resizePlot(blockPlotEl);
    });
  });
  const plotFrame = plotEl.parentElement;
  const blockFrame = blockPlotEl.parentElement;
  if (plotFrame) {
    resize.observe(plotFrame);
  }
  if (blockFrame) {
    resize.observe(blockFrame);
  }

  const theme = new MutationObserver(() => {
    if (root.isConnected) {
      draw(false);
    }
  });
  for (const node of [document.documentElement, document.body]) {
    theme.observe(node, { attributes: true, attributeFilter: ["data-md-color-scheme", "data-md-color-media"] });
  }
  for (const input of document.querySelectorAll("input[data-md-color-scheme]")) {
    input.addEventListener("change", () => draw(false), on);
  }
  const darkMq = window.matchMedia("(prefers-color-scheme: dark)");
  const onDark = () => {
    if (root.isConnected) {
      draw(false);
    }
  };
  if (darkMq.addEventListener) {
    darkMq.addEventListener("change", onDark, on);
  } else {
    darkMq.addListener(onDark);
  }

  function cleanup() {
    abort.abort();
    if (resizeTimer) {
      cancelAnimationFrame(resizeTimer);
      resizeTimer = 0;
    }
    resize.disconnect();
    theme.disconnect();
    if (darkMq.removeListener) {
      darkMq.removeListener(onDark);
    }
    if (plotEl.data) {
      Plotly.purge(plotEl);
    }
    if (blockPlotEl.data) {
      Plotly.purge(blockPlotEl);
    }
  }

  window.addEventListener("beforeunload", cleanup, { once: true, signal: abort.signal });

  syncing = true;
  applyPreset("#fe-elem", "e2m1");
  applyPreset("#fe-scale", "e4m3fn");
  elemPreset.value = "e2m1";
  scalePreset.value = "e4m3fn";
  encodeSelect.value = "nearest";
  root.querySelector("#fe-M").value = "6";
  root.querySelector("#fe-amax").value = "6";
  root.querySelector("#fe-sg").value = "1";
  root.querySelector("#fe-z").value = "0";
  root.querySelector("#fe-block-size").value = "16";
  xRange.value = "full";
  syncing = false;
  updateModeClass();
  syncPresetSelects();
  draw(true);
  return cleanup;
}
