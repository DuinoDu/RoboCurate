/* Panel definitions: which charts exist, and what each one plots.
 *
 * Charts are described declaratively so the tab machinery in app.js stays
 * generic and every panel obeys the same series/legend/colour rules.
 * Series counts are held at or below three wherever the marks overlap, which is
 * the all-pairs-safe span of the categorical palette.
 */

import { seriesColor } from "./chart.js";

const S1 = seriesColor(0); // measured / primary
const S2 = seriesColor(1); // commanded / policy action
const S3 = seriesColor(2); // reference
const S4 = seriesColor(3);
const S8 = seriesColor(7); // error / deviation

/** Measured vs policy action vs retarget reference, for one joint. */
function jointChart(data, name, index) {
  const series = [
    { name: "measured", data: data.series("state.q", index), color: S1 },
  ];
  if (data.has("action.q")) {
    series.push({
      name: "policy action",
      data: data.series("action.q", index),
      color: S2,
    });
  }
  if (data.has("ref.dof")) {
    series.push({
      name: "retarget ref",
      data: data.series("ref.dof", index),
      color: S3,
      dash: [4, 3],
      width: 1.5,
    });
  }
  return { title: name, unit: "rad", series, height: 120 };
}

function chainCharts(data, group) {
  const names = data.meta.joints.names;
  const out = [];
  for (let i = group.start; i < group.stop; i++) {
    out.push(jointChart(data, names[i], i));
  }
  return out;
}

/** Per-joint tracking error, one chart per kinematic chain (<= 7 lines). */
function trackingErrorChart(data, group) {
  const names = data.meta.joints.names;
  const series = [];
  for (let i = group.start; i < group.stop; i++) {
    series.push({
      name: names[i].replace(/^(left|right)_/, ""),
      data: data.derive("action.q", "state.q", i, (a, b) => a - b),
      color: seriesColor(i - group.start),
      width: 1.5,
    });
  }
  return {
    title: `${group.name.replace("_", " ")} tracking error`,
    unit: "rad (action − measured)",
    series,
    height: 150,
  };
}

function multiAxisChart(data, key, title, unit, height = 150) {
  const ch = data.channel(key);
  if (!ch) return null;
  return {
    title,
    unit,
    height,
    series: ch.names.map((n, i) => ({
      name: n,
      data: data.series(key, i),
      color: seriesColor(i),
    })),
  };
}

function handChart(data, side) {
  const ch = data.channel(`hand.${side}.state_q`);
  if (!ch) return null;
  // Index/middle/ring/pinky carry the grip amount; thumb and thumb_aux are
  // driven by separate scale/offset rules, so they are drawn but de-emphasised.
  return {
    title: `${side === "left" ? "Left" : "Right"} hand — measured closure`,
    unit: "0 = open, 1 = closed",
    height: 150,
    yDomain: [-0.05, 1.05],
    series: ch.names.map((n, i) => ({
      name: n,
      data: data.series(`hand.${side}.state_q`, i),
      color: seriesColor(i),
      width: i >= 2 ? 2 : 1.25,
      dash: i < 2 ? [4, 3] : null,
    })),
  };
}

function handTrackChart(data, side) {
  if (!data.has(`hand.${side}.cmd_q`)) return null;
  const idx = 3; // middle finger: representative of the grip-carrying group
  return {
    title: `${side === "left" ? "Left" : "Right"} hand — command vs measured`,
    unit: "0 = open, 1 = closed (middle finger)",
    height: 150,
    yDomain: [-0.05, 1.05],
    series: [
      { name: "commanded", data: data.series(`hand.${side}.cmd_q`, idx), color: S2 },
      { name: "measured", data: data.series(`hand.${side}.state_q`, idx), color: S1 },
    ],
  };
}

function handCurrentChart(data, side) {
  const key = `hand.${side}.state_tau`;
  const ch = data.channel(key);
  if (!ch) return null;
  return {
    title: `${side === "left" ? "Left" : "Right"} hand — motor current`,
    unit: ch.unit,
    height: 130,
    series: ch.names.map((n, i) => ({
      name: n,
      data: data.series(key, i),
      color: seriesColor(i),
      width: 1.5,
    })),
  };
}

export function buildPanels(data) {
  const groups = data.meta.joints.groups;
  const panels = [];

  // ---- Overview ----------------------------------------------------------
  const overview = [];
  const rpy = multiAxisChart(data, "imu.rpy", "Base orientation (IMU)", "rad");
  if (rpy) overview.push(rpy);
  const gyro = multiAxisChart(data, "imu.gyro", "Base angular rate", "rad/s");
  if (gyro) overview.push(gyro);
  const acc = multiAxisChart(data, "imu.accel", "Base linear acceleration", "m/s²");
  if (acc) overview.push(acc);
  if (data.has("ref.root_pos")) {
    overview.push(
      multiAxisChart(data, "ref.root_pos", "Reference root position", "m"),
    );
  }
  if (data.has("state.tau")) {
    // Whole-body effort as one number per sample keeps the story readable
    // instead of putting 29 torque lines on one axis.
    const n = data.samples;
    const total = new Float32Array(n);
    const peak = new Float32Array(n);
    for (let j = 0; j < 29; j++) {
      const s = data.series("state.tau", j);
      for (let i = 0; i < n; i++) {
        const v = Math.abs(s[i]);
        if (!isFinite(v)) continue;
        total[i] += v;
        if (v > peak[i]) peak[i] = v;
      }
    }
    overview.push({
      title: "Whole-body joint effort",
      unit: "N·m",
      height: 150,
      series: [
        { name: "sum |torque|", data: total, color: S1 },
        { name: "peak |torque|", data: peak, color: S2 },
      ],
    });
  }
  if (data.has("power.voltage")) {
    overview.push({
      title: "Bus voltage",
      unit: "V",
      height: 130,
      series: [{ name: "voltage", data: data.series("power.voltage", 0), color: S1 }],
    });
  }
  panels.push({
    id: "overview",
    label: "Overview",
    cols: 2,
    note: "Base motion, whole-body effort and power for the visible window. " +
      "Drag any chart to zoom every panel to the same slice; click to seek.",
    charts: overview.filter(Boolean),
  });

  // ---- Joints, one tab per kinematic chain -------------------------------
  for (const group of groups) {
    const label = group.name.replace("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
    const charts = chainCharts(data, group);
    if (group.stop - group.start > 1 && data.has("action.q")) {
      charts.unshift(trackingErrorChart(data, group));
    }
    panels.push({
      id: `joints-${group.name}`,
      label,
      cols: group.stop - group.start > 4 ? 3 : 2,
      note: `Native lowstate indices ${group.start}–${group.stop - 1}. ` +
        "Policy action is the absolute joint target the controller writes to " +
        "the motor after clamping, so action minus measured is true tracking error.",
      charts,
    });
  }

  // ---- Hands -------------------------------------------------------------
  const handCharts = [];
  for (const side of ["left", "right"]) {
    const a = handTrackChart(data, side);
    if (a) handCharts.push(a);
    const b = handChart(data, side);
    if (b) handCharts.push(b);
    const c = handCurrentChart(data, side);
    if (c) handCharts.push(c);
  }
  if (handCharts.length) {
    panels.push({
      id: "hands",
      label: "Hands",
      cols: 2,
      note: "BrainCo closure is normalised 0 = open to 1 = closed. Index, " +
        "middle, ring and pinky track the grip amount directly; thumb is " +
        "scaled and thumb_aux is confined to [0.8, 1.0], so those two are " +
        "drawn dashed and are excluded from the dataset's gripper value.",
      charts: handCharts,
    });
  }

  // ---- Teleop link -------------------------------------------------------
  const teleop = [];
  if (data.has("teleop.pico")) {
    teleop.push({
      title: "PICO source rate",
      unit: "fps",
      height: 150,
      series: [{ name: "pico_fps", data: data.series("teleop.pico", 0), color: S1 }],
    });
    teleop.push({
      title: "PICO frame interval",
      unit: "ms",
      height: 150,
      series: [{ name: "pico_dt", data: data.series("teleop.pico", 1), color: S2 }],
    });
  }
  if (data.has("teleop.latency")) {
    teleop.push({
      title: "Source → telemetry latency",
      unit: "ms",
      height: 150,
      series: [
        { name: "latency", data: data.series("teleop.latency", 0), color: S4 },
      ],
    });
  }
  if (data.has("ref.root_quat")) {
    teleop.push(
      multiAxisChart(data, "ref.root_quat", "Reference root rotation (wxyz)", "unit quaternion"),
    );
  }
  if (teleop.length) {
    panels.push({
      id: "teleop",
      label: "Teleop link",
      cols: 2,
      note: "Health of the PICO → retarget path. These are separate charts " +
        "rather than one dual-axis plot, because fps and milliseconds share no " +
        "scale.",
      charts: teleop.filter(Boolean),
    });
  }

  // ---- Gains and thermal -------------------------------------------------
  const diag = [];
  if (data.has("state.temp")) {
    diag.push({
      title: "Hottest joints",
      unit: "°C",
      height: 170,
      series: hottest(data, 6),
    });
  }
  if (data.has("cmd.kp")) {
    diag.push({
      title: "Applied position gain (kp)",
      unit: "—",
      height: 150,
      series: gainSummary(data, "cmd.kp"),
    });
  }
  if (data.has("cmd.kd")) {
    diag.push({
      title: "Applied damping gain (kd)",
      unit: "—",
      height: 150,
      series: gainSummary(data, "cmd.kd"),
    });
  }
  if (diag.length) {
    panels.push({
      id: "diagnostics",
      label: "Thermal & gains",
      cols: 2,
      note: "Gains come from /action/lowcmd, which the v1.4 converter never " +
        "reads — treat them as diagnostic provenance, not dataset content.",
      charts: diag,
    });
  }

  return panels;
}

/** The N joints reaching the highest temperature — 6 lines stays under the cap. */
function hottest(data, n) {
  const names = data.meta.joints.names;
  const ch = data.channel("state.temp");
  const ranked = ch.max
    .map((v, i) => ({ v: v ?? -Infinity, i }))
    .sort((a, b) => b.v - a.v)
    .slice(0, n);
  return ranked.map((r, slot) => ({
    name: names[r.i],
    data: data.series("state.temp", r.i),
    color: seriesColor(slot),
    width: 1.5,
  }));
}

/** Gains are near-constant per joint; show the envelope rather than 29 lines. */
function gainSummary(data, key) {
  const n = data.samples;
  const lo = new Float32Array(n).fill(Infinity);
  const hi = new Float32Array(n).fill(-Infinity);
  for (let j = 0; j < 29; j++) {
    const s = data.series(key, j);
    for (let i = 0; i < n; i++) {
      const v = s[i];
      if (!isFinite(v)) continue;
      if (v < lo[i]) lo[i] = v;
      if (v > hi[i]) hi[i] = v;
    }
  }
  for (let i = 0; i < n; i++) {
    if (!isFinite(lo[i])) lo[i] = NaN;
    if (!isFinite(hi[i])) hi[i] = NaN;
  }
  return [
    { name: "max across joints", data: hi, color: S2 },
    { name: "min across joints", data: lo, color: S1 },
  ];
}
