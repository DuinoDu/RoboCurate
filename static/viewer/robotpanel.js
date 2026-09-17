/* The 3D robot card: controls, legend, and the caveat that makes it readable. */

import { RobotView, POSE_SOURCES } from "./robot3d.js";

function seg(label, options, value, onChange) {
  const wrap = document.createElement("label");
  wrap.className = "r3-control";
  wrap.append(document.createTextNode(label));
  const group = document.createElement("span");
  group.className = "seg";
  group.setAttribute("role", "group");
  for (const opt of options) {
    const btn = document.createElement("button");
    btn.textContent = opt.label;
    if (opt.title) btn.title = opt.title;
    btn.setAttribute("aria-pressed", String(opt.id === value));
    btn.addEventListener("click", () => {
      for (const sib of group.children) sib.setAttribute("aria-pressed", "false");
      btn.setAttribute("aria-pressed", "true");
      onChange(opt.id);
    });
    group.appendChild(btn);
  }
  wrap.appendChild(group);
  return wrap;
}

function check(label, value, onChange, title) {
  const wrap = document.createElement("label");
  wrap.className = "r3-control r3-check";
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = value;
  if (title) wrap.title = title;
  box.addEventListener("change", () => onChange(box.checked));
  wrap.append(box, document.createTextNode(label));
  return wrap;
}

/**
 * Build the 3D robot card into `host`.
 * Resolves to the RobotView, or null when there is no robot description —
 * the caller then collapses the column rather than showing a broken card.
 */
export async function buildRobotPanel(host, data, store) {
  const probe = await RobotView.describe();
  if (!probe.ok) return { view: null, reason: probe.reason };

  // Only sources this episode actually recorded — a picker offering a stream
  // that is not in the cache would produce a robot frozen at zero. Checked
  // before any DOM is built, so a bail-out leaves no half-made card behind.
  const sources = POSE_SOURCES.filter((s) => data.has(s.id));
  if (!sources.length) {
    return { view: null, reason: "no joint-position stream in this episode" };
  }

  const card = document.createElement("div");
  card.className = "card r3-card";

  const head = document.createElement("div");
  head.className = "card-head";
  const h2 = document.createElement("h2");
  h2.textContent = "Robot pose at playhead";
  const sub = document.createElement("span");
  sub.className = "card-sub";
  sub.textContent = probe.desc.name;
  head.append(h2, sub);
  card.appendChild(head);

  const bar = document.createElement("div");
  bar.className = "r3-bar";
  card.appendChild(bar);

  const stage = document.createElement("div");
  stage.className = "r3-stage";
  card.appendChild(stage);

  const legend = document.createElement("div");
  legend.className = "r3-legend";
  card.appendChild(legend);

  const boot = document.createElement("p");
  boot.className = "r3-boot";
  boot.textContent = "Loading robot meshes…";
  stage.appendChild(boot);
  host.appendChild(card);

  const view = new RobotView(stage, data, store, probe.desc);

  if (!sources.some((s) => s.id === view.opts.source)) {
    view.opts.source = sources[0].id;
  }
  const ghostOptions = sources.filter((s) => s.id !== view.opts.source);
  if (ghostOptions.length) view.opts.ghost = ghostOptions[0].id;

  const renderLegend = () => {
    const solid = sources.find((s) => s.id === view.opts.source);
    const ghost = sources.find((s) => s.id === view.opts.ghost);
    const items = [
      `<span class="legend-item"><span class="legend-swatch r3-sw-solid"></span>` +
        `${solid ? solid.label : "pose"} — solid</span>`,
    ];
    if (view.opts.showGhost && ghost) {
      items.push(
        `<span class="legend-item"><span class="legend-swatch r3-sw-ghost"></span>` +
          `${ghost.label} — translucent, same root</span>`,
      );
    }
    if (view.opts.colorByError) {
      items.push(
        `<span class="legend-item r3-ramp"><span class="r3-ramp-bar"></span>` +
          `tracking error 0 → ${view.errorScale.toFixed(3)} rad (p99)</span>`,
      );
    }
    legend.innerHTML = items.join("");
  };

  if (sources.length > 1) {
    bar.appendChild(
      seg(
        "Pose",
        sources.map((s) => ({ id: s.id, label: s.label, title: s.note })),
        view.opts.source,
        (id) => {
          view.opts.source = id;
          if (view.opts.ghost === id) {
            const other = sources.find((s) => s.id !== id);
            if (other) view.opts.ghost = other.id;
          }
          view.pose(store.indexAt(store.playhead));
          renderLegend();
        },
      ),
    );
    bar.appendChild(
      check(
        "Overlay",
        false,
        (on) => {
          view.opts.showGhost = on;
          view.pose(store.indexAt(store.playhead));
          renderLegend();
        },
        "Draw a second pose source in the same root frame",
      ),
    );
  }

  if (data.has("action.q") && data.has("state.q")) {
    bar.appendChild(
      check(
        "Colour by tracking error",
        false,
        (on) => {
          view.opts.colorByError = on;
          view.pose(store.indexAt(store.playhead));
          renderLegend();
        },
        "Tint each link by |policy action − measured| of its own joint",
      ),
    );
  }

  if (data.has("imu.quat")) {
    bar.appendChild(
      check(
        "Apply IMU base orientation",
        true,
        (on) => {
          view.opts.applyImu = on;
          view.pose(store.indexAt(store.playhead));
        },
        "Off compares joint pose alone, with the pelvis held level",
      ),
    );
  }

  const spacer = document.createElement("span");
  spacer.className = "spacer";
  bar.appendChild(spacer);
  bar.appendChild(
    seg(
      "View",
      [
        { id: "iso", label: "Iso" },
        { id: "front", label: "Front" },
        { id: "side", label: "Side" },
        { id: "top", label: "Top" },
      ],
      "iso",
      (id) => view.setView(id),
    ),
  );

  // The caveat is what makes this view honest, so the load-bearing sentence —
  // that the position on screen is not measured — is always on screen. The
  // rest is elaboration and folds away, because this now sits in a narrow
  // column beside the camera rather than filling a tab.
  const note = document.createElement("details");
  note.className = "panel-note r3-note";
  note.innerHTML =
    "<summary>The pelvis is <strong>pinned at the origin</strong> — the raw " +
    "episode records no odometry.</summary>" +
    "<p>Drawing base translation would therefore invent it. Standing height " +
    "is set once, at the first sample, so the feet rest on the grid. Base " +
    "<em>rotation</em> is real — it comes from the IMU. " +
    "<code>ref.root_pos</code> exists but lives in the retarget task frame, " +
    "not the measured robot's, so it stays on its own chart in Overview " +
    "rather than being overlaid here. The hands are the URDF's rubber " +
    "end effectors; BrainCo finger motion is not in this description and " +
    "does not animate — see the Hands tab for it.</p>";
  card.appendChild(note);

  await view.mount((msg) => { boot.textContent = msg; });
  boot.remove();
  view.resize();
  view.pose(store.indexAt(store.playhead));
  renderLegend();

  return { view, reason: null, renderLegend };
}
