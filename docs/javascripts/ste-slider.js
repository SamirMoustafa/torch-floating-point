document$.subscribe(() => {
  document.querySelectorAll(".ste-widget").forEach(mountWidget);
});

const LIGHT = {
  ink: "#111111",
  muted: "#6b7280",
  grid: "#e6e6e6",
  orange: "#ee4c2c",
  green: "#2a9d6e",
  blue: "#3b6fd8",
  legend: "#ffffff",
};
const DARK = {
  ink: "#e2e4ea",
  muted: "#8b93a3",
  grid: "#3a4050",
  orange: "#ff7a5c",
  green: "#8fb389",
  blue: "#8aa4cc",
  legend: "#1c2029",
};

const dataCache = new Map();

const ESTIMATORS = {
  ewgs: {
    panels: 3,
    height: 6.6,
    padBottom: 0.09,
    ratios: [1.45, 0.95, 1.1],
    forwardLabel: "y = ⌊x/s⌉ s",
    gxYlim: [-0.15, 2.25],
    gxStep: 0.5,
    gsYlim: [-7.2, 7.2],
    gsStep: 2,
    format: formatDelta,
    jac(x, y, xmin, xmax, delta) {
      const gate = x >= xmin && x <= xmax ? 1 : 0;
      return gate * (1 + delta * (x - y));
    },
  },
  reste: {
    panels: 2,
    height: 4.6,
    padBottom: 0.11,
    ratios: [1.45, 1.0],
    forwardLabel: "y = ⌊x⌉",
    gxYlim: [-0.2, 6.0],
    gxStep: 1,
    format: (value) => formatNumber(value, 2),
    jac(x, _y, xmin, xmax, o) {
      const gate = x >= xmin && x <= xmax ? 1 : 0;
      const ax = Math.max(Math.abs(x), 1e-4);
      return gate * (1 / o) * Math.pow(ax, 1 / o - 1);
    },
  },
  rdfs: {
    panels: 2,
    height: 4.6,
    padBottom: 0.11,
    ratios: [1.45, 1.0],
    forwardLabel: "y = ⌊x⌉",
    gxYlim: [-1.0, 30.0],
    gxStep: 5,
    format: (value) => formatNumber(value, 3),
    jac(x, y, xmin, xmax, a) {
      const gate = x >= xmin && x <= xmax ? 1 : 0;
      const amp = a * Math.SQRT2 * Math.PI;
      const c = Math.cos(Math.PI * (x + y));
      return gate * ((1 - amp * c) / (1 + amp * c));
    },
  },
  dasr: {
    panels: 2,
    height: 4.6,
    padBottom: 0.11,
    ratios: [1.45, 1.0],
    forwardLabel: "y = two-nearest",
    gxYlim: [-0.2, 12.0],
    gxStep: 2,
    format: (value) => formatNumber(value, 2),
    softForward: true,
    softMode: "dasr",
  },
  hestia: {
    panels: 2,
    height: 4.6,
    padBottom: 0.11,
    ratios: [1.45, 1.0],
    forwardLabel: "y = codebook softmax",
    gxYlim: [-0.2, 12.0],
    gxStep: 2,
    format: (value) => formatNumber(value, 2),
    softForward: true,
    softMode: "hestia",
  },
  "dasr-block": {
    panels: 3,
    height: 6.6,
    padBottom: 0.09,
    ratios: [1.45, 0.95, 1.1],
    forwardLabel: "y = DASR(x/s) s",
    gxYlim: [-0.2, 12.0],
    gxStep: 2,
    gsYlim: [-7.2, 7.2],
    gsStep: 2,
    format: (value) => formatNumber(value, 2),
    softForward: true,
    softMode: "dasr",
    scale: 1.625,
  },
};

function jsonUrl(root) {
  const attr = root.getAttribute("data-src");
  if (attr) {
    return new URL(attr, document.location.href).href;
  }
  for (const script of document.querySelectorAll("script[src]")) {
    if (script.src.includes("ste-slider.js") || script.src.includes("ewgs-slider.js")) {
      return new URL("../assets/ewgs-slider.json", script.src).href;
    }
  }
  return new URL("../assets/ewgs-slider.json", document.location.href).href;
}

function loadData(root) {
  const url = jsonUrl(root);
  if (!dataCache.has(url)) {
    dataCache.set(
      url,
      fetch(url).then((response) => {
        if (!response.ok) {
          throw new Error("ewgs-slider.json " + response.status);
        }
        return response.json();
      }),
    );
  }
  return dataCache.get(url);
}

function formatDelta(delta) {
  if (delta === 0) {
    return "0";
  }
  if (delta >= 0.01) {
    return String(Number(delta.toFixed(3)));
  }
  const exp = Math.round(Math.log10(delta));
  const digits = "⁰¹²³⁴⁵⁶⁷⁸⁹";
  const sup = String(Math.abs(exp))
    .split("")
    .map((d) => digits[Number(d)])
    .join("");
  return exp < 0 ? `10⁻${sup}` : `10${sup}`;
}

function formatNumber(value, digits) {
  return String(Number(value.toFixed(digits)));
}

function linspace(lo, hi, n) {
  const x = new Float64Array(n);
  const step = (hi - lo) / (n - 1);
  for (let i = 0; i < n; i++) {
    x[i] = lo + step * i;
  }
  return x;
}

function bisectRight(codes, value) {
  let lo = 0;
  let hi = codes.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (codes[mid] <= value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

function mixCodes(x, subset, tau) {
  let zMax = -Infinity;
  const z = new Float64Array(subset.length);
  for (let i = 0; i < subset.length; i++) {
    z[i] = -((x - subset[i]) * (x - subset[i])) / tau;
    if (z[i] > zMax) {
      zMax = z[i];
    }
  }
  let denom = 0;
  const w = new Float64Array(subset.length);
  for (let i = 0; i < subset.length; i++) {
    w[i] = Math.exp(z[i] - zMax);
    denom += w[i];
  }
  let y = 0;
  for (let i = 0; i < subset.length; i++) {
    w[i] /= denom;
    y += w[i] * subset[i];
  }
  let variance = 0;
  for (let i = 0; i < subset.length; i++) {
    const d = subset[i] - y;
    variance += w[i] * d * d;
  }
  return { y, gx: (2 / tau) * variance };
}

function mixAt(x, codes, tau, mode) {
  if (mode === "dasr") {
    const idx = bisectRight(codes, x);
    if (idx <= 0) {
      return { y: codes[0], gx: 0 };
    }
    if (idx >= codes.length) {
      return { y: codes[codes.length - 1], gx: 0 };
    }
    return mixCodes(x, [codes[idx - 1], codes[idx]], tau);
  }
  return mixCodes(x, codes, tau);
}

function forwardSoft(x, codes, tau, mode) {
  const n = x.length;
  const y = new Float64Array(n);
  const gx = new Float64Array(n);
  const t = Math.max(tau, 1e-4);
  for (let i = 0; i < n; i++) {
    const mixed = mixAt(x[i], codes, t, mode);
    y[i] = mixed.y;
    gx[i] = mixed.gx;
  }
  return { y, gx };
}

function grads(spec, data, x, param) {
  if (spec.softForward) {
    const s = spec.scale ?? 1;
    const n = x.length;
    const u = s === 1 ? x : new Float64Array(n);
    if (s !== 1) {
      for (let i = 0; i < n; i++) {
        u[i] = x[i] / s;
      }
    }
    const mixed = forwardSoft(u, data.codes, param, spec.softMode);
    if (s === 1 && spec.panels !== 3) {
      return mixed;
    }
    const y = new Float64Array(n);
    const gs = spec.panels === 3 ? new Float64Array(n) : undefined;
    for (let i = 0; i < n; i++) {
      y[i] = mixed.y[i] * s;
      if (gs) {
        gs[i] = mixed.y[i] - u[i] * mixed.gx[i];
      }
    }
    return { y, gx: mixed.gx, gs };
  }
  const n = data.n;
  const y = data.y;
  const gx = new Float64Array(n);
  const gs = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const jac = spec.jac(x[i], y[i], data.xmin, data.xmax, param);
    gx[i] = jac;
    gs[i] = y[i] - x[i] * jac;
  }
  return { y, gx, gs };
}

function colorScheme() {
  return (
    document.body.getAttribute("data-md-color-scheme") ||
    document.documentElement.getAttribute("data-md-color-scheme") ||
    "default"
  );
}

function palette() {
  return colorScheme() === "slate" ? DARK : LIGHT;
}

function mapX(rect, lo, hi, value) {
  return rect.x + ((value - lo) / (hi - lo)) * rect.w;
}

function mapY(rect, lo, hi, value) {
  return rect.y + rect.h - ((value - lo) / (hi - lo)) * rect.h;
}

function ticks(lo, hi, step) {
  const out = [];
  const start = Math.ceil(lo / step - 1e-9) * step;
  for (let v = start; v <= hi + 1e-9; v += step) {
    out.push(Number(v.toFixed(8)));
  }
  return out;
}

function strokePolyline(ctx, rect, x, y, xlo, xhi, ylo, yhi) {
  ctx.beginPath();
  for (let i = 0; i < x.length; i++) {
    const px = mapX(rect, xlo, xhi, x[i]);
    const py = mapY(rect, ylo, yhi, y[i]);
    if (i === 0) {
      ctx.moveTo(px, py);
    } else {
      ctx.lineTo(px, py);
    }
  }
  ctx.stroke();
}

function panelChrome(ctx, rect, pal, xlo, xhi, ylo, yhi, xticks, yticks, xlabel) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.clip();
  ctx.strokeStyle = pal.grid;
  ctx.lineWidth = 0.65;
  ctx.globalAlpha = 0.9;
  for (const t of xticks) {
    const px = mapX(rect, xlo, xhi, t);
    ctx.beginPath();
    ctx.moveTo(px, rect.y);
    ctx.lineTo(px, rect.y + rect.h);
    ctx.stroke();
  }
  for (const t of yticks) {
    const py = mapY(rect, ylo, yhi, t);
    ctx.beginPath();
    ctx.moveTo(rect.x, py);
    ctx.lineTo(rect.x + rect.w, py);
    ctx.stroke();
  }
  ctx.globalAlpha = 0.22;
  ctx.strokeStyle = pal.ink;
  ctx.lineWidth = 0.4;
  if (ylo < 0 && yhi > 0) {
    const z = mapY(rect, ylo, yhi, 0);
    ctx.beginPath();
    ctx.moveTo(rect.x, z);
    ctx.lineTo(rect.x + rect.w, z);
    ctx.stroke();
  }
  if (xlo < 0 && xhi > 0) {
    const z = mapX(rect, xlo, xhi, 0);
    ctx.beginPath();
    ctx.moveTo(z, rect.y);
    ctx.lineTo(z, rect.y + rect.h);
    ctx.stroke();
  }
  ctx.restore();

  ctx.strokeStyle = pal.ink;
  ctx.globalAlpha = 1;
  ctx.lineWidth = 0.9;
  ctx.beginPath();
  ctx.moveTo(rect.x, rect.y);
  ctx.lineTo(rect.x, rect.y + rect.h);
  ctx.lineTo(rect.x + rect.w, rect.y + rect.h);
  ctx.stroke();

  ctx.fillStyle = pal.ink;
  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const t of yticks) {
    ctx.fillText(String(t), rect.x - 8, mapY(rect, ylo, yhi, t));
  }
  if (xlabel) {
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const t of xticks) {
      ctx.fillText(String(t), mapX(rect, xlo, xhi, t), rect.y + rect.h + 6);
    }
  }
}

function legend(ctx, rect, pal, items) {
  ctx.font = "15px Inter, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  const padX = 10;
  const padY = 7;
  const rowH = 20;
  const swatch = 22;
  let textW = 0;
  for (const item of items) {
    textW = Math.max(textW, ctx.measureText(item.label).width);
  }
  const boxW = padX * 2 + swatch + 8 + textW;
  const boxH = padY * 2 + items.length * rowH - 4;
  const bx = rect.x + 8;
  const by = rect.y + 6;
  ctx.save();
  ctx.globalAlpha = 0.88;
  ctx.fillStyle = pal.legend;
  ctx.fillRect(bx, by, boxW, boxH);
  ctx.restore();
  items.forEach((item, i) => {
    const cy = by + padY + i * rowH + 6;
    ctx.strokeStyle = item.color;
    ctx.lineWidth = item.lw;
    ctx.setLineDash(item.dash || []);
    ctx.beginPath();
    ctx.moveTo(bx + padX, cy);
    ctx.lineTo(bx + padX + swatch, cy);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = pal.ink;
    ctx.fillText(item.label, bx + padX + swatch + 8, cy);
  });
}

function yLabel(ctx, rect, pal, text) {
  ctx.save();
  ctx.fillStyle = pal.ink;
  ctx.font = "14px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const lx = rect.x - 44;
  const ly = rect.y + rect.h / 2;
  ctx.translate(lx, ly);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

function render(canvas, spec, data, x, param) {
  const pal = palette();
  const dpr = window.devicePixelRatio || 1;
  const cssW = Math.max(320, canvas.clientWidth || canvas.parentElement.clientWidth);
  const cssH = cssW * (spec.height / 8.4);
  canvas.style.height = cssH + "px";
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const padL = 0.16 * cssW;
  const padR = 0.02 * cssW;
  const padT = 0.01 * cssH;
  const padB = spec.padBottom * cssH;
  const gap = 0.02 * cssH;
  const innerH = cssH - padT - padB - (spec.panels - 1) * gap;
  const sum = spec.ratios.reduce((a, b) => a + b, 0);
  const plotW = cssW - padL - padR;
  let y0 = padT;
  const rects = spec.ratios.map((ratio) => {
    const h = (ratio / sum) * innerH;
    const rect = { x: padL, y: y0, w: plotW, h };
    y0 += h + gap;
    return rect;
  });

  const { y, gx, gs } = grads(spec, data, x, param);
  const xlo = data.lo;
  const xhi = data.hi;
  const xticks = ticks(-10, 10, 2.5);
  const panels = [
    {
      rect: rects[0],
      ylim: [-10.8, 10.8],
      ystep: 2,
      ylabel: "x, y",
      xlabel: false,
      series: [
        { y: x, color: pal.muted, lw: 1.15, dash: [6, 4] },
        { y, color: pal.orange, lw: 2.2, dash: [] },
      ],
      legend: [
        { label: "x", color: pal.muted, lw: 1.15, dash: [6, 4] },
        { label: spec.forwardLabel, color: pal.orange, lw: 2.2 },
      ],
    },
    {
      rect: rects[1],
      ylim: spec.gxYlim,
      ystep: spec.gxStep,
      ylabel: "∂y/∂x",
      xlabel: spec.panels === 2,
      series: [{ y: gx, color: pal.green, lw: 2.2, dash: [] }],
      legend: [{ label: "∂y/∂x", color: pal.green, lw: 2.2 }],
    },
  ];
  if (spec.panels === 3) {
    panels.push({
      rect: rects[2],
      ylim: spec.gsYlim,
      ystep: spec.gsStep,
      ylabel: "∂y/∂s",
      xlabel: true,
      series: [{ y: gs, color: pal.blue, lw: 2.0, dash: [] }],
      legend: [{ label: "∂y/∂s", color: pal.blue, lw: 2.0 }],
    });
  }

  for (const panel of panels) {
    const [ylo, yhi] = panel.ylim;
    const yticks = ticks(ylo, yhi, panel.ystep);
    panelChrome(ctx, panel.rect, pal, xlo, xhi, ylo, yhi, xticks, yticks, panel.xlabel);
    ctx.save();
    ctx.beginPath();
    ctx.rect(panel.rect.x, panel.rect.y, panel.rect.w, panel.rect.h);
    ctx.clip();
    ctx.lineJoin = "miter";
    ctx.lineCap = "butt";
    for (const series of panel.series) {
      ctx.strokeStyle = series.color;
      ctx.lineWidth = series.lw;
      ctx.setLineDash(series.dash);
      strokePolyline(ctx, panel.rect, x, series.y, xlo, xhi, ylo, yhi);
    }
    ctx.setLineDash([]);
    ctx.restore();
    yLabel(ctx, panel.rect, pal, panel.ylabel);
    legend(ctx, panel.rect, pal, panel.legend);
  }

  ctx.fillStyle = pal.ink;
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText("x", padL + plotW / 2, cssH - 4);
}

function syncChips(root, value) {
  for (const chip of root.querySelectorAll("[data-value]")) {
    const chipValue = Number(chip.getAttribute("data-value"));
    chip.classList.toggle("is-active", Math.abs(chipValue - value) < 1e-9);
  }
}

function mountWidget(root) {
  if (root.dataset.steReady === "1") {
    return;
  }
  const spec = ESTIMATORS[root.getAttribute("data-estimator")];
  const canvas = root.querySelector("canvas");
  const slider = root.querySelector("input[type='range']");
  const output = root.querySelector("[data-ste-value]");
  if (!spec || !canvas || !slider || !output) {
    return;
  }
  root.dataset.steReady = "1";

  loadData(root)
    .then((data) => {
      const x = linspace(data.lo, data.hi, data.n);
      const draw = () => {
        const param = Number(slider.value);
        output.textContent = spec.format(param);
        syncChips(root, param);
        render(canvas, spec, data, x, param);
      };
      slider.addEventListener("input", draw);
      for (const chip of root.querySelectorAll("[data-value]")) {
        chip.addEventListener("click", () => {
          slider.value = chip.getAttribute("data-value");
          draw();
        });
      }
      const resize = new ResizeObserver(draw);
      resize.observe(root);
      const theme = new MutationObserver(draw);
      for (const node of [document.documentElement, document.body]) {
        theme.observe(node, {
          attributes: true,
          attributeFilter: ["data-md-color-scheme", "data-md-color-media"],
        });
      }
      for (const input of document.querySelectorAll("input[data-md-color-scheme]")) {
        input.addEventListener("change", draw);
      }
      const darkMq = window.matchMedia("(prefers-color-scheme: dark)");
      if (darkMq.addEventListener) {
        darkMq.addEventListener("change", draw);
      } else {
        darkMq.addListener(draw);
      }
      draw();
    })
    .catch((error) => {
      output.textContent = "could not load plot data";
      console.error(error);
    });
}
