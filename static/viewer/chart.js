/* Canvas time-series charts for the episode viewer.
 *
 * Why canvas and not SVG: a 159 s episode resampled at 100 Hz is ~16k samples
 * per series, and a joints panel puts ~90 series on screen at once. That is far
 * past the point where one DOM node per point stays interactive.
 *
 * Every chart shares one time axis, one crosshair and one playhead through the
 * store in app.js, so the whole dashboard reads as a single instrument.
 */

const DPR = () => Math.min(window.devicePixelRatio || 1, 2);

export const SERIES_VARS = [
  "--series-1", "--series-2", "--series-3", "--series-4",
  "--series-5", "--series-6", "--series-7", "--series-8",
];

/** Categorical hues are assigned in fixed slot order and never cycled. Past 8
 *  entries the caller must fold to "Other" or facet — see dataviz rules. */
export function seriesColor(index) {
  return `var(${SERIES_VARS[index % SERIES_VARS.length]})`;
}

function cssValue(el, name) {
  return getComputedStyle(el).getPropertyValue(name).trim();
}

/** Nice axis ticks: 1/2/5 × 10^n covering [lo, hi]. */
function ticks(lo, hi, target = 5) {
  if (!isFinite(lo) || !isFinite(hi) || lo === hi) return [lo];
  const span = hi - lo;
  const raw = span / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) {
    out.push(Math.abs(v) < step * 1e-6 ? 0 : v);
  }
  return out;
}

function fmtNum(v, span) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const a = Math.abs(span ?? v);
  const digits = a >= 100 ? 1 : a >= 10 ? 2 : a >= 1 ? 3 : 4;
  return v.toFixed(digits);
}

export class TimeChart {
  /**
   * @param {object} opts
   *   title, unit           chart header
   *   series                [{ name, data: Float32Array, color, dash?, width? }]
   *   store                 shared view/playhead state
   *   height                plot height in CSS px (axis band added on top)
   *   yDomain               optional [lo, hi]; otherwise auto from visible range
   *   zeroLine              draw a hairline at y=0 when the domain spans it
   *   annotate              optional fn(ctx, geom) for extra marks
   *   collapsed             start folded (default true) — a joints panel opens
   *                         with ~30 charts, and unfolding is cheaper to do
   *                         than to undo
   *   onCollapse            fn(collapsed) when the user folds/unfolds
   */
  constructor(opts) {
    this.o = Object.assign({ height: 132, zeroLine: true }, opts);
    this.store = opts.store;
    this.hidden = new Set();
    this.showTable = false;
    this.collapsed = opts.collapsed !== false;
    this.el = this._build();
    this._resize = this._resize.bind(this);
    this.ro = new ResizeObserver(this._resize);
    this.ro.observe(this.wrap);
  }

  _build() {
    const card = document.createElement("div");
    card.className = "chart-card";

    const head = document.createElement("div");
    head.className = "chart-head";

    // The whole title row is the disclosure control, so the hit target is the
    // full width of the card rather than a chevron-sized corner.
    const toggle = document.createElement("button");
    toggle.className = "chart-toggle";
    toggle.setAttribute("aria-expanded", String(!this.collapsed));
    const chevron = document.createElement("span");
    chevron.className = "chevron";
    chevron.setAttribute("aria-hidden", "true");
    chevron.textContent = "▸";
    toggle.appendChild(chevron);

    const title = document.createElement("div");
    title.className = "chart-title";
    title.textContent = this.o.title;
    if (this.o.unit) {
      const u = document.createElement("span");
      u.className = "chart-unit";
      u.textContent = this.o.unit;
      title.appendChild(u);
    }
    toggle.appendChild(title);

    // Folded, the card is one scannable row, so it still has to say what is
    // inside it. The series count is the cheapest honest summary.
    const count = document.createElement("span");
    count.className = "chart-count";
    count.textContent = `${this.o.series.length} series`;
    toggle.appendChild(count);
    toggle.addEventListener("click", () => {
      this.setCollapsed(!this.collapsed);
      this.o.onCollapse?.(this.collapsed);
    });
    head.appendChild(toggle);

    const tools = document.createElement("div");
    tools.className = "chart-tools";
    const tableBtn = document.createElement("button");
    tableBtn.className = "icon-btn";
    tableBtn.textContent = "Table";
    tableBtn.setAttribute("aria-pressed", "false");
    tableBtn.title = "Show the same values as a table";
    tableBtn.addEventListener("click", () => {
      this.showTable = !this.showTable;
      tableBtn.setAttribute("aria-pressed", String(this.showTable));
      this.tableWrap.hidden = !this.showTable;
      this.renderTable();
    });
    tools.appendChild(tableBtn);
    head.appendChild(tools);
    card.appendChild(head);

    this.toggleBtn = toggle;
    this.tools = tools;

    // A legend is always present for >= 2 series; a single series is named by
    // the title instead, so no legend box is drawn for it.
    this.legend = document.createElement("div");
    this.legend.className = "legend";
    if (this.o.series.length >= 2) card.appendChild(this.legend);
    this._buildLegend();

    this.wrap = document.createElement("div");
    this.wrap.className = "chart-canvas-wrap";
    this.canvas = document.createElement("canvas");
    this.wrap.appendChild(this.canvas);
    this.tooltip = document.createElement("div");
    this.tooltip.className = "tooltip";
    this.tooltip.hidden = true;
    this.wrap.appendChild(this.tooltip);
    card.appendChild(this.wrap);

    this.tableWrap = document.createElement("div");
    this.tableWrap.className = "table-wrap";
    this.tableWrap.hidden = true;
    card.appendChild(this.tableWrap);

    this._bindPointer();
    // Assigned here rather than only at the call site, because the collapse
    // pass below reads `this.el`.
    this.el = card;
    this._applyCollapsed();
    return card;
  }

  _applyCollapsed() {
    const folded = this.collapsed;
    this.el.dataset.collapsed = String(folded);
    this.toggleBtn.setAttribute("aria-expanded", String(!folded));
    this.legend.hidden = folded;
    this.wrap.hidden = folded;
    this.tools.hidden = folded;
    this.tableWrap.hidden = folded || !this.showTable;
  }

  setCollapsed(next) {
    if (next === this.collapsed) return;
    this.collapsed = next;
    this._applyCollapsed();
    // A folded canvas has zero width, so it must be measured again on the way
    // out rather than painted at its stale size.
    if (!next) {
      this._resize();
      this.renderTable();
    }
  }

  _buildLegend() {
    this.legend.textContent = "";
    this.o.series.forEach((s, i) => {
      const item = document.createElement("span");
      item.className = "legend-item";
      item.dataset.off = String(this.hidden.has(i));
      const sw = document.createElement("span");
      sw.className = "legend-line";
      sw.style.background = s.color;
      if (s.dash) {
        sw.style.background = `repeating-linear-gradient(90deg, ${s.color} 0 3px, transparent 3px 6px)`;
      }
      item.appendChild(sw);
      item.appendChild(document.createTextNode(s.name));
      item.title = `Toggle ${s.name}`;
      item.addEventListener("click", () => {
        if (this.hidden.has(i)) this.hidden.delete(i);
        else this.hidden.add(i);
        item.dataset.off = String(this.hidden.has(i));
        this.render();
        this.renderTable();
      });
      this.legend.appendChild(item);
    });
  }

  _bindPointer() {
    const c = this.canvas;
    let dragStart = null;

    const toTime = (ev) => {
      const r = c.getBoundingClientRect();
      const g = this.geom;
      if (!g) return null;
      const x = ev.clientX - r.left;
      const frac = (x - g.padL) / Math.max(g.plotW, 1);
      const { start, end } = this.store.view;
      return start + Math.min(Math.max(frac, 0), 1) * (end - start);
    };

    c.addEventListener("pointermove", (ev) => {
      const t = toTime(ev);
      if (t === null) return;
      if (dragStart !== null) {
        this.store.setBrush([Math.min(dragStart, t), Math.max(dragStart, t)]);
      } else {
        this.store.setHover(t);
      }
    });
    c.addEventListener("pointerleave", () => this.store.setHover(null));
    c.addEventListener("pointerdown", (ev) => {
      c.setPointerCapture(ev.pointerId);
      dragStart = toTime(ev);
    });
    c.addEventListener("pointerup", (ev) => {
      const t = toTime(ev);
      const moved = dragStart !== null && Math.abs(t - dragStart) > 0.02;
      if (moved) this.store.commitBrush();
      else {
        this.store.clearBrush();
        this.store.seek(t); // a click seeks; a drag zooms
      }
      dragStart = null;
    });
    c.addEventListener("dblclick", () => this.store.resetView());
  }

  _resize() {
    if (this.collapsed) return;
    const w = this.wrap.clientWidth;
    if (!w) return;
    const axisBand = 20;
    const h = this.o.height + axisBand;
    const dpr = DPR();
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.height = `${h}px`;
    this.render();
  }

  /** Index range of the store's visible window inside the sample grid. */
  _window() {
    const { rate, samples } = this.store.timeline;
    const { start, end } = this.store.view;
    const i0 = Math.max(0, Math.floor(start * rate));
    const i1 = Math.min(samples - 1, Math.ceil(end * rate));
    return [i0, Math.max(i1, i0 + 1)];
  }

  /** Auto y-domain over the visible window across all shown series. */
  _yDomain(i0, i1) {
    if (this.o.yDomain) return this.o.yDomain;
    let lo = Infinity, hi = -Infinity;
    this.o.series.forEach((s, si) => {
      if (this.hidden.has(si)) return;
      const d = s.data;
      // Sub-sample the scan on wide windows; exact extremes still come from the
      // min/max decimation during draw, this only sets the axis.
      const step = Math.max(1, Math.floor((i1 - i0) / 4000));
      for (let i = i0; i <= i1; i += step) {
        const v = d[i];
        if (!isFinite(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    });
    if (!isFinite(lo) || !isFinite(hi)) return [-1, 1];
    if (lo === hi) { lo -= 0.5; hi += 0.5; }
    const pad = (hi - lo) * 0.12;
    return [lo - pad, hi + pad];
  }

  render() {
    // Folded charts are the common case, and playback repaints ~60x/s: drawing
    // them would spend the whole frame budget on canvases nobody is looking at.
    if (this.collapsed) return;
    const ctx = this.canvas.getContext("2d");
    const dpr = DPR();
    const W = this.canvas.width / dpr;
    const H = this.canvas.height / dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const root = document.documentElement;
    const col = {
      grid: cssValue(root, "--grid"),
      axis: cssValue(root, "--axis"),
      muted: cssValue(root, "--text-muted"),
      text: cssValue(root, "--text-primary"),
      surface: cssValue(root, "--surface-1"),
    };

    // Canvas cannot consume `var(--series-n)`, and the value changes with the
    // theme, so resolve each series colour against the live computed style on
    // every render rather than caching it at construction.
    for (const s of this.o.series) {
      s.resolved = s.color.startsWith("var(")
        ? cssValue(root, s.color.slice(4, -1))
        : s.color;
    }

    const padL = 44, padR = 10, padT = 6, padB = 20;
    const plotW = Math.max(W - padL - padR, 10);
    const plotH = Math.max(H - padT - padB, 10);
    const [i0, i1] = this._window();
    const [ylo, yhi] = this._yDomain(i0, i1);
    const { start, end } = this.store.view;

    const xOf = (t) => padL + ((t - start) / (end - start)) * plotW;
    const yOf = (v) => padT + (1 - (v - ylo) / (yhi - ylo)) * plotH;
    this.geom = { padL, padT, plotW, plotH, xOf, yOf, ylo, yhi, i0, i1 };

    // --- grid: solid hairlines, one shade off the surface (never dashed) ---
    ctx.lineWidth = 1;
    ctx.strokeStyle = col.grid;
    ctx.fillStyle = col.muted;
    ctx.font = "10px system-ui, -apple-system, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const yt = ticks(ylo, yhi, 4);
    for (const v of yt) {
      const y = Math.round(yOf(v)) + 0.5;
      if (y < padT || y > padT + plotH) continue;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
      ctx.fillText(fmtNum(v, yhi - ylo), padL - 6, y);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xt = ticks(start, end, 6);
    for (const t of xt) {
      const x = Math.round(xOf(t)) + 0.5;
      if (x < padL || x > padL + plotW) continue;
      ctx.strokeStyle = col.grid;
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + plotH);
      ctx.stroke();
      ctx.fillText(`${t.toFixed(t >= 100 ? 0 : 1)}s`, x, padT + plotH + 5);
    }

    if (this.o.zeroLine && ylo < 0 && yhi > 0) {
      const y = Math.round(yOf(0)) + 0.5;
      ctx.strokeStyle = col.axis;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
    }

    // --- series: min/max decimated so spikes survive downsampling ---
    ctx.save();
    ctx.beginPath();
    ctx.rect(padL, padT, plotW, plotH);
    ctx.clip();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    const n = i1 - i0 + 1;
    const perPx = n / plotW;
    this.o.series.forEach((s, si) => {
      if (this.hidden.has(si)) return;
      ctx.strokeStyle = s.resolved || s.color;
      ctx.lineWidth = s.width || 2;
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      const d = s.data;
      if (perPx > 2) {
        // Two points per pixel column (min and max) keeps extremes visible.
        for (let px = 0; px < plotW; px++) {
          const a = i0 + Math.floor(px * perPx);
          const b = Math.min(i1, i0 + Math.floor((px + 1) * perPx));
          let mn = Infinity, mx = -Infinity;
          for (let i = a; i <= b; i++) {
            const v = d[i];
            if (!isFinite(v)) continue;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
          }
          const x = padL + px;
          if (mn === Infinity) { ctx.moveTo(x, yOf(ylo)); continue; }
          ctx.lineTo(x, yOf(mx));
          ctx.lineTo(x, yOf(mn));
        }
      } else {
        let pen = false;
        for (let i = i0; i <= i1; i++) {
          const v = d[i];
          if (!isFinite(v)) { pen = false; continue; } // NaN => real gap
          const x = xOf(i / this.store.timeline.rate);
          const y = yOf(v);
          if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
        }
      }
      ctx.stroke();
    });
    ctx.setLineDash([]);

    if (this.o.annotate) this.o.annotate(ctx, this.geom, col);

    // --- playhead + crosshair ---
    const drawRule = (t, color, width) => {
      if (t === null || t < start || t > end) return;
      const x = Math.round(xOf(t)) + 0.5;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + plotH);
      ctx.stroke();
    };
    drawRule(this.store.playhead, cssValue(root, "--series-2"), 1.5);
    const hov = this.store.hover;
    drawRule(hov, col.axis, 1);

    // Markers on the hovered sample, ringed with the surface so overlapping
    // series stay separable without drawing borders around marks.
    if (hov !== null && hov >= start && hov <= end) {
      const idx = Math.round(hov * this.store.timeline.rate);
      this.o.series.forEach((s, si) => {
        if (this.hidden.has(si)) return;
        const v = s.data[idx];
        if (!isFinite(v)) return;
        const x = xOf(idx / this.store.timeline.rate);
        const y = yOf(v);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = s.resolved || s.color;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = col.surface;
        ctx.stroke();
      });
    }
    ctx.restore();
    this._renderTooltip();
  }

  _renderTooltip() {
    const hov = this.store.hover;
    const g = this.geom;
    if (hov === null || !g) { this.tooltip.hidden = true; return; }
    const { start, end } = this.store.view;
    if (hov < start || hov > end) { this.tooltip.hidden = true; return; }
    const idx = Math.round(hov * this.store.timeline.rate);

    const rows = this.o.series
      .map((s, si) => ({ s, si, v: s.data[idx] }))
      .filter((r) => !this.hidden.has(r.si));
    let html = `<div class="tt-time">t = ${hov.toFixed(3)} s</div>`;
    for (const r of rows) {
      html += `<div class="tt-row"><span class="tt-name">` +
        `<span class="legend-line" style="background:${r.s.resolved || r.s.color}"></span>` +
        `${r.s.name}</span><span class="tt-val">${fmtNum(r.v, g.yhi - g.ylo)}</span></div>`;
    }
    this.tooltip.innerHTML = html;
    this.tooltip.hidden = false;

    const x = g.xOf(hov);
    const w = this.tooltip.offsetWidth;
    const flip = x + w + 18 > this.wrap.clientWidth;
    this.tooltip.style.left = `${flip ? x - w - 12 : x + 12}px`;
    this.tooltip.style.top = `4px`;
  }

  /** The table-view twin: every value the chart encodes, in text. */
  renderTable() {
    if (!this.showTable || this.collapsed) return;
    const g = this.geom;
    const idx = Math.round((this.store.playhead ?? 0) * this.store.timeline.rate);
    const [i0, i1] = this._window();
    let html = "<table><thead><tr><th>Series</th><th>At playhead</th>" +
      "<th>Min</th><th>Max</th><th>Mean</th></tr></thead><tbody>";
    this.o.series.forEach((s, si) => {
      if (this.hidden.has(si)) return;
      let mn = Infinity, mx = -Infinity, sum = 0, k = 0;
      const step = Math.max(1, Math.floor((i1 - i0) / 5000));
      for (let i = i0; i <= i1; i += step) {
        const v = s.data[i];
        if (!isFinite(v)) continue;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v; k++;
      }
      const span = isFinite(mx - mn) ? mx - mn : 1;
      html += `<tr><td class="name">` +
        `<span class="legend-line" style="background:${s.resolved || s.color}"></span>${s.name}</td>` +
        `<td>${fmtNum(s.data[idx], span)}</td><td>${fmtNum(mn, span)}</td>` +
        `<td>${fmtNum(mx, span)}</td><td>${fmtNum(k ? sum / k : NaN, span)}</td></tr>`;
    });
    html += "</tbody></table>";
    this.tableWrap.innerHTML = html;
  }

  destroy() { this.ro.disconnect(); }
}

export { fmtNum, ticks };
