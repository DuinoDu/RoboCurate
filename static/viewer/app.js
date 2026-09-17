/* Episode viewer entry point: boot, transport, camera sync and panel wiring. */

import { EpisodeData } from "./data.js";
import { Store } from "./store.js";
import { TimeChart, seriesColor } from "./chart.js";
import { buildPanels } from "./panels.js";
import { renderHealth } from "./health.js";
import { buildRobotPanel } from "./robotpanel.js";
import { Rail, initCollapse } from "./sidebar.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let data, store, charts = [], scrub, camera;
let rail = null, epEntry = null;
let panelCharts = new Map(), activePanel = null;
let panelsHashWriter = null, hashSeek = null;

/** Charts on the visible tab. Playback repaints ~60×/s, so repainting the
 *  hidden tabs' canvases as well would waste most of the frame budget. */
function visibleCharts() {
  return panelCharts.get(activePanel) || [];
}

/* ---------------------------------------------------------------- theme -- */

function initTheme() {
  // `#theme=light` wins over the stored preference so a shared link carries the
  // mode it was captured in.
  const fromHash = new URLSearchParams(location.hash.slice(1)).get("theme");
  const saved = fromHash || localStorage.getItem("hm-theme") || "dark";
  applyTheme(saved);
  document.querySelectorAll("[data-theme-set]").forEach((btn) => {
    btn.addEventListener("click", () => applyTheme(btn.dataset.themeSet));
  });
}

function applyTheme(mode) {
  document.documentElement.dataset.theme = mode;
  localStorage.setItem("hm-theme", mode);
  document.querySelectorAll("[data-theme-set]").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.themeSet === mode));
  });
  // Canvases hold resolved colours, so they must repaint on a theme change.
  charts.forEach((c) => c.render());
  if (scrub) scrub.draw();
  robotView?.applyTheme();
}

/* ------------------------------------------------------------ headline -- */

function tile(label, value, unit, note) {
  return `<div class="tile">
    <div class="tile-label">${label}</div>
    <div class="tile-value">${value}${unit ? `<span class="unit">${unit}</span>` : ""}</div>
    ${note ? `<div class="tile-note">${note}</div>` : ""}
  </div>`;
}

/** What the operator said about this take.
 *
 *  The raw MCAP still carries no success/failure flag — but the recorder
 *  writes one *beside* it, in `pico_episode_summary.json`, at the moment the
 *  operator labelled the episode. When that sidecar is there the verdict is
 *  evidence and is shown as such; when it is not, the pill says the outcome
 *  is absent rather than implying a pass.
 */
function outcomePill() {
  if (!epEntry || epEntry.outcome === null || epEntry.outcome === undefined) {
    return `<span class="pill warning"><span class="pill-dot"></span>` +
      `outcome not recorded beside this episode</span>`;
  }
  if (epEntry.outcome) {
    return `<span class="pill good"><span class="pill-dot"></span>` +
      `operator marked success</span>`;
  }
  const why = epEntry.failure_reason
    ? `: ${esc(epEntry.failure_reason)}` : "";
  return `<span class="pill critical"><span class="pill-dot"></span>` +
    `operator marked failure${why}</span>`;
}

function renderHeadline() {
  const ep = data.meta.episode;
  const cam = data.meta.camera;
  // Inside a library, "which episode is this" is the useful headline — the
  // file is always called mcap_0.mcap, the session and take are what differ.
  const where = epEntry
    ? [epEntry.session, epEntry.task, epEntry.episode_id]
      .filter(Boolean).join(" / ")
    : data.meta.source.name;
  $("source-name").textContent =
    `${where} · ${(data.meta.source.size_bytes / 1e9).toFixed(2)} GB`;

  $("instruction").textContent = ep.instruction || "— no instruction recorded —";
  const meta = [];
  meta.push(ep.instruction_locked
    ? `<span class="pill good"><span class="pill-dot"></span>instruction locked</span>`
    : `<span class="pill critical"><span class="pill-dot"></span>${ep.instructions.length} different instructions</span>`);
  const mode = data.meta.bands.robot_state.at(-1);
  if (mode) {
    meta.push(`<span class="pill"><span class="pill-dot"></span>controller ${mode.value}</span>`);
  }
  meta.push(`<span class="pill"><span class="pill-dot"></span>${data.meta.topics.length} recorded topics</span>`);
  meta.push(outcomePill());
  $("instruction-meta").innerHTML = meta.join("");

  const camInfo = cam.info;
  $("tiles").innerHTML = [
    tile("Duration", ep.duration_s.toFixed(1), "s"),
    tile("Messages", ep.message_count.toLocaleString(), "", `${ep.chunk_count} chunks`),
    tile("Camera frames", cam.count.toLocaleString(), "",
      camInfo ? `${camInfo.width}×${camInfo.height}` : ""),
    tile("Body DOF", "29", "", "native lowstate order"),
    tile("Sample grid", data.meta.timeline.rate_hz.toFixed(0), "Hz",
      `${data.meta.timeline.samples.toLocaleString()} samples`),
  ].join("");

  $("clock-end").textContent = ep.duration_s.toFixed(3);
  updateCameraSub();
}

/* ---------------------------------------------------------------- bands -- */

const BAND_COLORS = new Map();
function bandColor(value) {
  if (!BAND_COLORS.has(value)) BAND_COLORS.set(value, seriesColor(BAND_COLORS.size));
  return BAND_COLORS.get(value);
}

function renderBands() {
  const host = $("bands");
  host.textContent = "";
  const duration = data.meta.episode.duration_s;
  for (const [key, label] of [["robot_state", "controller"], ["policy_mode", "mode"]]) {
    const segs = data.meta.bands[key];
    if (!segs || !segs.length) continue;
    const row = document.createElement("div");
    row.className = "band-row";
    row.innerHTML = `<span class="band-name">${label}</span>`;
    const track = document.createElement("div");
    track.className = "band-track";
    for (const s of segs) {
      const el = document.createElement("div");
      el.className = "band-seg";
      el.style.left = `${(s.start_s / duration) * 100}%`;
      el.style.width = `${Math.max(((s.end_s - s.start_s) / duration) * 100, 0.15)}%`;
      el.style.background = bandColor(s.value);
      el.title = `${s.value} · ${s.start_s.toFixed(1)}–${s.end_s.toFixed(1)} s`;
      track.appendChild(el);
    }
    row.appendChild(track);
    host.appendChild(row);
  }
}

/* ------------------------------------------------------------- scrubber -- */

class Scrubber {
  constructor(canvas) {
    this.canvas = canvas;
    this.dragging = false;
    this.overview = this._overview();
    this._bind();
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  /** A context strip: whole-body effort across the full episode, never zoomed. */
  _overview() {
    const n = data.samples;
    const out = new Float32Array(n);
    if (!data.has("state.tau")) return out;
    for (let j = 0; j < 29; j++) {
      const s = data.series("state.tau", j);
      for (let i = 0; i < n; i++) out[i] += Math.abs(s[i]) || 0;
    }
    return out;
  }

  _timeAt(ev) {
    const r = this.canvas.getBoundingClientRect();
    const frac = (ev.clientX - r.left) / Math.max(r.width, 1);
    return Math.min(Math.max(frac, 0), 1) * data.meta.episode.duration_s;
  }

  _bind() {
    const c = this.canvas;
    let downAt = null;
    c.addEventListener("pointerdown", (ev) => {
      c.setPointerCapture(ev.pointerId);
      downAt = this._timeAt(ev);
      this.dragging = true;
    });
    c.addEventListener("pointermove", (ev) => {
      const t = this._timeAt(ev);
      if (this.dragging && downAt !== null && Math.abs(t - downAt) > 0.15) {
        this.pendingRange = [Math.min(downAt, t), Math.max(downAt, t)];
        this.draw();
      }
    });
    c.addEventListener("pointerup", (ev) => {
      const t = this._timeAt(ev);
      if (this.pendingRange) {
        store.setView(this.pendingRange[0], this.pendingRange[1]);
        this.pendingRange = null;
      } else {
        store.seek(t);
      }
      this.dragging = false;
      downAt = null;
      this.draw();
    });
    c.addEventListener("dblclick", () => store.resetView());
  }

  draw() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w) return;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const cs = getComputedStyle(document.documentElement);
    const v = (k) => cs.getPropertyValue(k).trim();
    const duration = data.meta.episode.duration_s;
    const xOf = (t) => (t / duration) * w;

    // effort silhouette
    const d = this.overview;
    let mx = 0;
    for (let i = 0; i < d.length; i += 7) if (d[i] > mx) mx = d[i];
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let px = 0; px < w; px++) {
      const a = Math.floor((px / w) * d.length);
      const b = Math.floor(((px + 1) / w) * d.length);
      let peak = 0;
      for (let i = a; i < b; i++) if (d[i] > peak) peak = d[i];
      ctx.lineTo(px, h - (peak / (mx || 1)) * (h - 4));
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fillStyle = v("--grid");
    ctx.fill();

    // dim everything outside the active window
    const range = this.pendingRange || [store.view.start, store.view.end];
    ctx.fillStyle = v("--page");
    ctx.globalAlpha = 0.66;
    ctx.fillRect(0, 0, xOf(range[0]), h);
    ctx.fillRect(xOf(range[1]), 0, w - xOf(range[1]), h);
    ctx.globalAlpha = 1;

    ctx.strokeStyle = v("--axis");
    ctx.lineWidth = 1;
    for (const t of range) {
      ctx.beginPath();
      ctx.moveTo(Math.round(xOf(t)) + 0.5, 0);
      ctx.lineTo(Math.round(xOf(t)) + 0.5, h);
      ctx.stroke();
    }

    // playhead
    const x = Math.round(xOf(store.playhead)) + 0.5;
    ctx.strokeStyle = v("--series-2");
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
}

/* ---------------------------------------------------------------- camera -- */

/** Pearson correlation of two equal-length samples, or null when either side is
 *  flat — correlation is undefined there, and reporting it as 0 would look like
 *  a confident "not stereo" rather than "cannot tell". */
function pearson(a, b) {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let num = 0, va = 0, vb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    num += da * db; va += da * da; vb += db * db;
  }
  return va > 0 && vb > 0 ? num / Math.sqrt(va * vb) : null;
}

/** Wire the Left / Right / Full control, defaulting from what was detected. */
function initStereoControls() {
  const seg = $("stereo-seg");
  const detect = $("camera-detect");
  const buttons = [...seg.querySelectorAll("button")];

  const apply = (eye, remember = true) => {
    camera.setEye(eye);
    for (const b of buttons) {
      b.setAttribute("aria-pressed", String((b.dataset.eye || "") === eye));
    }
    if (remember) localStorage.setItem("hm-eye", eye);
    updateCameraSub();
  };

  for (const b of buttons) {
    b.addEventListener("click", () => apply(b.dataset.eye || ""));
  }

  camera.onDetect = (isStereo, r) => {
    detect.textContent = isStereo
      ? `side-by-side stereo detected (halves correlate ${r.toFixed(2)})`
      : `single frame (halves correlate ${r.toFixed(2)})`;
    // A stored preference is the operator's decision and outranks detection;
    // otherwise show one eye when the frame really is a pair, because the
    // second copy is redundant at this size.
    const saved = localStorage.getItem("hm-eye");
    apply(saved !== null ? saved : (isStereo ? "left" : ""), false);
  };
  apply(localStorage.getItem("hm-eye") ?? "", false);
}

/** The caption has to describe what is on screen, not what the file holds. */
function updateCameraSub() {
  const info = data.meta.camera.info;
  if (!info) { $("camera-sub").textContent = ""; return; }
  const w = camera?.eye ? Math.floor(info.width / 2) : info.width;
  const perEye = camera?.eye ? " per eye" : "";
  $("camera-sub").textContent =
    `${w}×${info.height}${perEye} · ${info.frame_id}`;
}

class Camera {
  constructor(img, overlay, stage) {
    this.img = img;
    this.overlay = overlay;
    this.stage = stage;
    this.current = -1;
    this.prefetch = new Map();
    this.eye = "";            // "" = whole frame, "left" / "right" = one eye
    this.stereo = null;       // null until the first frame has been measured
    // Without this the viewport is just black while a frame decodes or when one
    // fails, under a caption that confidently names a frame that isn't shown.
    this.img.addEventListener("load", () => {
      this.img.dataset.state = "ok";
      if (this.stereo === null) this._detectStereo();
    });
    this.img.addEventListener("error", () => {
      this.img.dataset.state = "error";
      this.overlay.textContent = `frame ${this.current + 1} failed to load`;
    });
  }

  /** Is this frame a side-by-side stereo pair?
   *
   *  Measured, not assumed: `camera_info` reports the stitched 1280x720 frame
   *  with a placeholder intrinsic matrix, so it cannot answer this. Correlating
   *  the two halves can. A stereo pair differs only by a small horizontal
   *  disparity, which leaves correlation very high; two halves of one ordinary
   *  scene are effectively unrelated.
   */
  _detectStereo() {
    const W = 64, H = 72;
    let r = null;
    try {
      const cv = document.createElement("canvas");
      cv.width = W * 2;
      cv.height = H;
      const ctx = cv.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(this.img, 0, 0, W * 2, H);
      const px = ctx.getImageData(0, 0, W * 2, H).data;
      const lum = (x, y) => {
        const i = (y * W * 2 + x) * 4;
        return 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
      };
      const a = [], b = [];
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) { a.push(lum(x, y)); b.push(lum(x + W, y)); }
      }
      r = pearson(a, b);
    } catch {
      r = null; // a tainted or undecodable canvas simply means "cannot tell"
    }
    // Undecided, so stay undecided: `stereo` is left null and the next frame to
    // load tries again. Latching a verdict from one blank or unreadable frame
    // would silently fix the wrong layout for the rest of the session.
    if (r === null) return;

    // Threshold placed from measurement, not taste. Across sampled frames of
    // this episode the two halves of the real pair correlate 0.80-0.88, while
    // the two halves of a *single* eye - the closest thing to a mono frame of
    // the same scene - reach only 0.47. The gap is wide, so 0.65 sits in the
    // empty middle: a low-texture or dark frame can drag a true pair well under
    // 0.85 without ever approaching a mono frame's score.
    this.stereo = r > 0.65;
    this.correlation = r;
    this.onDetect?.(this.stereo, r);
  }

  setEye(eye) {
    this.eye = eye;
    if (eye) this.stage.dataset.eye = eye;
    else delete this.stage.dataset.eye;
  }

  show(seconds) {
    const idx = data.frameAt(seconds);
    if (idx < 0 || idx === this.current) return;
    this.current = idx;
    this.img.dataset.state = "loading";
    this.img.src = data.frameUrl(idx);
    const t = data.meta.camera.times_s[idx];
    this.overlay.textContent =
      `frame ${idx + 1} / ${data.meta.camera.count} · t = ${t.toFixed(3)} s`;
    this._warm(idx);
  }

  /** Warm the browser cache a few frames ahead so playback does not stutter. */
  _warm(idx) {
    for (let k = 1; k <= 4; k++) {
      const j = idx + k;
      if (j >= data.meta.camera.count || this.prefetch.has(j)) continue;
      const im = new Image();
      im.src = data.frameUrl(j);
      this.prefetch.set(j, im);
    }
    if (this.prefetch.size > 90) {
      for (const key of [...this.prefetch.keys()].slice(0, 45)) {
        this.prefetch.delete(key);
      }
    }
  }
}

/* -------------------------------------------------------------- readout -- */

const READOUT_ROWS = [
  ["Base", [
    ["roll", "imu.rpy", 0, "rad"],
    ["pitch", "imu.rpy", 1, "rad"],
    ["yaw", "imu.rpy", 2, "rad"],
  ]],
  ["Reference root", [
    ["x", "ref.root_pos", 0, "m"],
    ["y", "ref.root_pos", 1, "m"],
    ["z", "ref.root_pos", 2, "m"],
  ]],
  ["Teleop link", [
    ["pico fps", "teleop.pico", 0, ""],
    ["pico dt", "teleop.pico", 1, "ms"],
    ["latency", "teleop.latency", 0, "ms"],
  ]],
  ["Hands (middle)", [
    ["left cmd", "hand.left.cmd_q", 3, ""],
    ["left meas", "hand.left.state_q", 3, ""],
    ["right cmd", "hand.right.cmd_q", 3, ""],
    ["right meas", "hand.right.state_q", 3, ""],
  ]],
];

function renderReadout() {
  const host = $("readout");
  const sample = store.indexAt(store.playhead);
  let html = "";
  for (const [title, rows] of READOUT_ROWS) {
    const live = rows.filter(([, key]) => data.has(key));
    if (!live.length) continue;
    html += `<div class="ro-group"><div class="ro-title">${title}</div>`;
    for (const [label, key, idx, unit] of live) {
      const v = data.value(key, idx, sample);
      const txt = isFinite(v) ? v.toFixed(3) : "—";
      html += `<div class="ro-row"><span class="k">${label}</span>` +
        `<span class="v">${txt}${unit ? ` ${unit}` : ""}</span></div>`;
    }
    html += "</div>";
  }

  // Largest absolute tracking error right now — the one number that says
  // whether the robot is following the policy at this instant.
  if (data.has("action.q") && data.has("state.q")) {
    let worst = 0, worstName = "—";
    const names = data.meta.joints.names;
    for (let j = 0; j < 29; j++) {
      const e = Math.abs(
        data.value("action.q", j, sample) - data.value("state.q", j, sample),
      );
      if (isFinite(e) && e > worst) { worst = e; worstName = names[j]; }
    }
    html += `<div class="ro-group"><div class="ro-title">Worst tracking error</div>
      <div class="ro-row"><span class="k">${worstName}</span>
      <span class="v">${worst.toFixed(4)} rad</span></div></div>`;
  }
  host.innerHTML = html;
}

/* ---------------------------------------------------------------- panels -- */

const OPEN_KEY = "hm-open";
let openMap = {};
let robotView = null, robotBuilt = false;

function loadOpenMap() {
  try { return JSON.parse(localStorage.getItem(OPEN_KEY)) || {}; }
  catch { return {}; }
}

/** `#panel=joints-left_arm&t=42.5&open=0,3` — a shareable pointer at one
 *  moment, in one view, with the same charts unfolded. Which is what flagging
 *  an episode for review actually needs. */
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const open = p.get("open");
  return {
    panel: p.get("panel"),
    t: Number(p.get("t")),
    open: open === null
      ? null
      : open.split(",").map(Number).filter(Number.isInteger),
  };
}

function writeHash() {
  if (!activePanel) return;
  const p = new URLSearchParams();
  p.set("panel", activePanel);
  p.set("t", store.playhead.toFixed(3));
  p.set("theme", document.documentElement.dataset.theme);
  const open = openMap[activePanel];
  if (open && open.length) p.set("open", open.join(","));
  history.replaceState(null, "", `#${p}`);
}

function renderPanels() {
  const defs = buildPanels(data);
  const tabs = $("tabs");
  const host = $("panels");
  tabs.textContent = "";
  host.textContent = "";

  const entries = [...defs];
  entries.push({ id: "health", label: "Stream health", health: true });

  openMap = loadOpenMap();
  const initial = readHash();
  if (initial.panel && initial.open) openMap[initial.panel] = initial.open;

  entries.forEach((def, i) => {
    const btn = document.createElement("button");
    btn.textContent = def.label;
    btn.setAttribute("role", "tab");
    btn.setAttribute("aria-selected", String(i === 0));
    btn.addEventListener("click", () => select(def.id));
    tabs.appendChild(btn);

    const panel = document.createElement("div");
    panel.className = "panel";
    panel.id = `panel-${def.id}`;
    panel.hidden = i !== 0;

    if (def.note) {
      const p = document.createElement("p");
      p.className = "panel-note";
      p.innerHTML = def.note;
      panel.appendChild(p);
    }

    if (def.health) {
      renderHealth(panel, data);
      panelCharts.set(def.id, []);
    } else {
      buildChartPanel(panel, def);
    }
    host.appendChild(panel);
  });

  activePanel = entries[0].id;
  panelsHashWriter = writeHash;

  function select(id, push = true) {
    if (!entries.some((d) => d.id === id)) return;
    entries.forEach((d, i) => {
      const on = d.id === id;
      $(`panel-${d.id}`).hidden = !on;
      tabs.children[i].setAttribute("aria-selected", String(on));
    });
    activePanel = id;
    // Charts in a previously hidden panel have zero width; size then repaint.
    visibleCharts().forEach((c) => { c._resize(); c.renderTable(); });
    if (push) writeHash();
  }

  if (initial.panel) select(initial.panel, false);
  if (Number.isFinite(initial.t) && location.hash.includes("t=")) {
    hashSeek = initial.t;
  }
}

/** One chart panel: a folded list of cards plus the control that unfolds them. */
function buildChartPanel(panel, def) {
  const specs = def.charts.filter(Boolean);
  const open = new Set(openMap[def.id] || []);

  const bar = document.createElement("div");
  bar.className = "panel-bar";
  const count = document.createElement("span");
  count.className = "panel-count";

  const mine = [];
  const persist = () => {
    openMap[def.id] = [...open].sort((a, b) => a - b);
    localStorage.setItem(OPEN_KEY, JSON.stringify(openMap));
    count.textContent = `${open.size} of ${specs.length} open`;
    if (activePanel === def.id) writeHash();
  };

  const setAll = (collapsed) => {
    mine.forEach((chart, idx) => {
      chart.setCollapsed(collapsed);
      if (collapsed) open.delete(idx); else open.add(idx);
    });
    persist();
  };

  const expandBtn = document.createElement("button");
  expandBtn.className = "btn";
  expandBtn.textContent = "Expand all";
  expandBtn.addEventListener("click", () => setAll(false));
  const collapseBtn = document.createElement("button");
  collapseBtn.className = "btn";
  collapseBtn.textContent = "Collapse all";
  collapseBtn.addEventListener("click", () => setAll(true));

  bar.append(expandBtn, collapseBtn, count);
  panel.appendChild(bar);

  const grid = document.createElement("div");
  grid.className = `chart-grid cols-${def.cols || 2}`;
  specs.forEach((spec, idx) => {
    const chart = new TimeChart({
      ...spec,
      store,
      collapsed: !open.has(idx),
      onCollapse: (collapsed) => {
        if (collapsed) open.delete(idx); else open.add(idx);
        persist();
      },
    });
    charts.push(chart);
    mine.push(chart);
    grid.appendChild(chart.el);
  });
  panelCharts.set(def.id, mine);
  panel.appendChild(grid);
  count.textContent = `${open.size} of ${specs.length} open`;
}

/** Fill the right-hand column with the posed robot.
 *
 * Not awaited by `boot`: the meshes are ~19 MB and the rest of the viewer is
 * usable long before they arrive, so the card appears immediately with its own
 * progress line and the page stays interactive while it loads. If the URDF is
 * missing the column collapses and the camera/readout pair reverts to the
 * side-by-side layout it had before, rather than leaving a hole.
 */
async function mountRobotPanel() {
  if (robotBuilt) return;
  robotBuilt = true;
  const host = $("robot-host");
  const result = await buildRobotPanel(host, data, store);
  if (!result.view) {
    $("split").classList.add("no-robot");
    host.textContent = "";
    console.warn(`3D view unavailable: ${result.reason}`);
    return;
  }
  robotView = result.view;
  robotView.resize();
  robotView.pose(store.indexAt(store.playhead));
}

/* ------------------------------------------------------------- transport -- */

function initTransport() {
  const playBtn = $("btn-play");
  const setPlay = (on) => {
    store.setPlaying(on);
    $("play-glyph").textContent = on ? "❚❚" : "▶";
    $("play-label").textContent = on ? "Pause" : "Play";
  };

  playBtn.addEventListener("click", () => setPlay(!store.playing));
  $("btn-start").addEventListener("click", () => store.seek(store.view.start));
  $("speed").addEventListener("change", (e) => store.setSpeed(Number(e.target.value)));
  $("btn-reset-zoom").addEventListener("click", () => store.resetView());

  const step = (dir) => {
    const idx = data.frameAt(store.playhead) + dir;
    const times = data.meta.camera.times_s;
    if (idx >= 0 && idx < times.length) store.seek(times[idx]);
  };
  $("btn-prev").addEventListener("click", () => step(-1));
  $("btn-next").addEventListener("click", () => step(1));

  document.addEventListener("keydown", (ev) => {
    if (ev.target.matches("input, select, textarea")) return;
    if (ev.code === "Space") { ev.preventDefault(); setPlay(!store.playing); }
    else if (ev.code === "ArrowRight") { ev.preventDefault(); step(ev.shiftKey ? 10 : 1); }
    else if (ev.code === "ArrowLeft") { ev.preventDefault(); step(ev.shiftKey ? -10 : -1); }
    else if (ev.code === "Home") { ev.preventDefault(); store.seek(store.view.start); }
    else if (ev.code === "Escape") { store.resetView(); }
  });

  // Playback advances against the wall clock, so a slow frame does not make the
  // episode play back slower than the requested speed.
  let last = performance.now();
  const tick = (now) => {
    const dt = (now - last) / 1000;
    last = now;
    if (store.playing) {
      let next = store.playhead + dt * store.speed;
      if (next >= store.view.end) {
        next = store.view.end;
        setPlay(false);
      }
      store.seek(next);
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/* ------------------------------------------------------------------ boot -- */

let lastHashWrite = 0;
function onStoreChange(kinds) {
  if (kinds.has("view") || kinds.has("playhead") || kinds.has("hover") ||
      kinds.has("brush")) {
    visibleCharts().forEach((c) => c.render());
  }
  if (kinds.has("playhead") || kinds.has("view")) {
    visibleCharts().forEach((c) => c.renderTable());
    camera.show(store.playhead);
    renderReadout();
    robotView?.pose(store.indexAt(store.playhead));
    $("clock-now").textContent = store.playhead.toFixed(3);
    // Throttled: playback emits ~60x/s and replaceState is not free.
    const now = performance.now();
    if (panelsHashWriter && now - lastHashWrite > 400) {
      lastHashWrite = now;
      panelsHashWriter();
    }
  }
  if (kinds.has("view")) {
    const { start, end } = store.view;
    $("btn-reset-zoom").hidden = !store.zoomed;
    $("range-note").textContent = store.zoomed
      ? `window ${start.toFixed(2)}–${end.toFixed(2)} s (${(end - start).toFixed(2)} s)`
      : "";
  }
  scrub.draw();
}

/** Show the rail before anything else is known.
 *
 * The library is the one thing that is useful even when this page cannot load
 * an episode at all — nothing selected, an episode still extracting, a cache
 * that failed to build. So it renders first and stays put, and every failure
 * below leaves the operator somewhere to go next.
 */
async function initRail() {
  rail = new Rail($("sidebar"));
  initCollapse($("shell"), $("rail-toggle"), $("sidebar"));
  try {
    await rail.refresh();
  } catch (err) {
    rail.message(`could not read the library: ${err.message}`, "bad");
  }
  return rail;
}

function bootFailed(msg) {
  $("boot-msg").textContent = msg;
  $("boot").querySelector(".spinner").hidden = true;
}

async function boot() {
  initTheme();
  await initRail();

  // A link written against a library that has since changed must say so
  // rather than quietly showing a different episode under the same URL.
  const asked = new URLSearchParams(location.search).get("ep");
  if (asked && !rail.entry(asked)) {
    rail.message(
      `The linked episode (${asked}) is not in this library. Add the ` +
      `directory that holds it, or pick another episode.`, "warn",
    );
  }

  const target = rail.target;
  if (!target) {
    bootFailed(
      "No episode selected. Choose one from the list, or add a directory "
      + "of recordings to it.",
    );
    return;
  }
  rail.setActive(target);
  epEntry = rail.entry(target);

  // An episode with no cache is a multi-minute extraction, not an error: run
  // it where it can be watched, and keep the rail usable while it runs.
  if (epEntry && epEntry.cache !== "ready") {
    const ok = await rail.prepare(target, (msg) => {
      $("boot-msg").textContent = msg;
    });
    if (!ok) {
      bootFailed($("boot-msg").textContent || "Extraction failed.");
      return;
    }
    epEntry = rail.entry(target);
    rail.setActive(target);
  }

  try {
    data = await EpisodeData.load(
      target, (msg) => { $("boot-msg").textContent = msg; },
    );
  } catch (err) {
    bootFailed(`Failed to load episode: ${err.message}`);
    return;
  }

  store = new Store({
    rate: data.meta.timeline.rate_hz,
    samples: data.meta.timeline.samples,
    duration: data.meta.episode.duration_s,
  });

  $("boot").hidden = true;
  $("app").hidden = false;
  if (epEntry) {
    document.title = `${epEntry.episode_id} · Episode Viewer`;
  }

  renderHeadline();
  renderBands();
  renderPanels();

  scrub = new Scrubber($("scrub"));
  camera = new Camera($("frame"), $("camera-overlay"), $("camera-stage"));
  initStereoControls();
  store.subscribe(onStoreChange);
  initTransport();

  // Sample 0 sits before several streams' first message (coverage < 1.0), so
  // opening there shows a readout of "—" and an empty camera. The first camera
  // frame is both inside every stream's coverage and the first thing the
  // operator actually saw. A `t=` in the URL wins over both.
  store.seek(hashSeek ?? data.meta.camera.times_s[0] ?? 0);
  requestAnimationFrame(() => visibleCharts().forEach((c) => c._resize()));

  // Last, and deliberately not awaited: ~19 MB of meshes must not stand
  // between the operator and a working transport. Seeking during the load is
  // safe — the view poses from the store's current playhead once it is ready.
  mountRobotPanel();
}

boot();
