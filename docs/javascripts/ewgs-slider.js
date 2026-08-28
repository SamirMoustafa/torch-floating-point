document$.subscribe(() => {
  const root = document.querySelector(".ewgs-widget");
  if (root) mountEwgs(root);
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

let dataPromise;

function jsonUrl(root) {
  const attr = root.getAttribute("data-src");
  if (attr) {
    return new URL(attr, document.location.href).href;
  }
  for (const script of document.querySelectorAll("script[src]")) {
    if (script.src.includes("ewgs-slider.js")) {
      return new URL("../assets/ewgs-slider.json", script.src).href;
    }
  }
  return new URL("../assets/ewgs-slider.json", document.location.href).href;
}

function loadData(root) {
  if (!dataPromise) {
    dataPromise = fetch(jsonUrl(root)).then((response) => {
      if (!response.ok) {
        throw new Error("ewgs-slider.json " + response.status);
      }
      return response.json();
    });
  }
  return dataPromise;
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

function linspace(lo, hi, n) {
  const x = new Float64Array(n);
  const step = (hi - lo) / (n - 1);
  for (let i = 0; i < n; i++) {
    x[i] = lo + step * i;
  }
  return x;
}

function grads(data, x, delta) {
  const n = data.n;
  const y = data.y;
  const gx = new Float64Array(n);
  const gs = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const gate = x[i] >= data.xmin && x[i] <= data.xmax ? 1 : 0;
    const jac = gate * (1 + delta * (x[i] - y[i]));
    gx[i] = jac;
    gs[i] = y[i] - x[i] * jac;
  }
  return { gx, gs };
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

function render(canvas, data, x, delta) {
  const pal = palette();
  const dpr = window.devicePixelRatio || 1;
  const cssW = Math.max(320, canvas.clientWidth || canvas.parentElement.clientWidth);
  const cssH = cssW * (6.6 / 8.4);
  canvas.style.height = cssH + "px";
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const padL = 0.16 * cssW;
  const padR = 0.02 * cssW;
  const padT = 0.01 * cssH;
  const padB = 0.09 * cssH;
  const gap = 0.02 * cssH;
  const innerH = cssH - padT - padB - 2 * gap;
  const ratios = [1.45, 0.95, 1.1];
  const sum = ratios[0] + ratios[1] + ratios[2];
  const plotW = cssW - padL - padR;
  let y0 = padT;
  const rects = ratios.map((ratio) => {
    const h = (ratio / sum) * innerH;
    const rect = { x: padL, y: y0, w: plotW, h };
    y0 += h + gap;
    return rect;
  });

  const { gx, gs } = grads(data, x, delta);
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
        { y: data.y, color: pal.orange, lw: 2.2, dash: [] },
      ],
      legend: [
        { label: "x", color: pal.muted, lw: 1.15, dash: [6, 4] },
        { label: "y = Round(x/s) s", color: pal.orange, lw: 2.2 },
      ],
    },
    {
      rect: rects[1],
      ylim: [-0.15, 2.25],
      ystep: 0.5,
      ylabel: "∂y/∂x",
      xlabel: false,
      series: [{ y: gx, color: pal.green, lw: 2.2, dash: [] }],
      legend: [{ label: "∂y/∂x", color: pal.green, lw: 2.2 }],
    },
    {
      rect: rects[2],
      ylim: [-7.2, 7.2],
      ystep: 2,
      ylabel: "∂y/∂s",
      xlabel: true,
      series: [{ y: gs, color: pal.blue, lw: 2.0, dash: [] }],
      legend: [{ label: "∂y/∂s", color: pal.blue, lw: 2.0 }],
    },
  ];

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

function syncChips(root, delta) {
  for (const chip of root.querySelectorAll("[data-delta]")) {
    const value = Number(chip.getAttribute("data-delta"));
    chip.classList.toggle("is-active", Math.abs(value - delta) < 1e-9);
  }
}

function mountEwgs(root) {
  if (root.dataset.ewgsReady === "1") {
    return;
  }
  root.dataset.ewgsReady = "1";
  const canvas = root.querySelector("canvas");
  const slider = root.querySelector("input[type='range']");
  const output = root.querySelector("[data-ewgs-value]");
  if (!canvas || !slider || !output) {
    return;
  }

  loadData(root)
    .then((data) => {
      const x = linspace(data.lo, data.hi, data.n);
      const draw = () => {
        const delta = Number(slider.value);
        output.textContent = formatDelta(delta);
        syncChips(root, delta);
        render(canvas, data, x, delta);
      };
      slider.addEventListener("input", draw);
      for (const chip of root.querySelectorAll("[data-delta]")) {
        chip.addEventListener("click", () => {
          slider.value = chip.getAttribute("data-delta");
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
