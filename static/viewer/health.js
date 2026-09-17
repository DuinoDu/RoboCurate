/* Stream health panel.
 *
 * The validation contract cares about recorder completeness, source rates and
 * worst-case gaps, so this renders those as a table rather than a chart: the
 * numbers are the answer, and a table is the accessible form for them.
 */

/** Gap budgets from contract.json's default_max_gap_ms, keyed by topic. */
const GAP_BUDGET_MS = {
  "/observations/holomotion/local_retarget_frame": 80,
  "/observations/robot_state/lowstate": 50,
  "/observations/brainco/left/state": 80,
  "/observations/brainco/right/state": 80,
  "/action/brainco/left/cmd": 80,
  "/action/brainco/right/cmd": 80,
  "/action/humanoid": 80,
  "/observations/camera/head/color/image": 50,
  "/observations/camera/head/depth/image": 50,
  "/observations/camera/head/color/camera_info": 2000,
  "/observations/camera/head/depth/camera_info": 2000,
};

function classify(topic) {
  const budget = GAP_BUDGET_MS[topic.topic];
  if (budget === undefined) return { level: "none", label: "—" };
  if (!topic.monotonic) return { level: "critical", label: "non-monotonic" };
  if (topic.max_gap_ms > budget) {
    return { level: "critical", label: `over ${budget} ms budget` };
  }
  if (topic.max_gap_ms > budget * 0.7) {
    return { level: "warning", label: `near ${budget} ms budget` };
  }
  return { level: "good", label: `within ${budget} ms` };
}

const ICON = { good: "✓", warning: "!", critical: "✕", none: "·" };

export function renderHealth(container, data) {
  const topics = [...data.meta.topics].sort((a, b) => b.count - a.count);
  const ep = data.meta.episode;

  const worst = topics
    .map((t) => ({ t, c: classify(t) }))
    .filter((x) => x.c.level === "critical" || x.c.level === "warning");

  const summary = document.createElement("div");
  summary.className = "card";
  const okCount = topics.length - worst.length;
  summary.innerHTML = `
    <div class="card-head">
      <h2>Contract check</h2>
      <span class="card-sub">gap budgets from contract.json</span>
    </div>
    <p class="panel-note">
      ${okCount} of ${topics.length} recorded streams sit inside their gap
      budget. Budgets only exist for topics the v1.4 converter consumes; the
      rest are recorded for provenance and show “—”.
      ${ep.instruction_locked
        ? "The episode instruction is locked to a single value."
        : "<strong>The instruction changed during the episode</strong>, which " +
          "violates the one-instruction-per-episode rule."}
    </p>`;
  container.appendChild(summary);

  const card = document.createElement("div");
  card.className = "card";
  const rows = topics.map((t) => {
    const c = classify(t);
    const status = c.level === "none"
      ? `<span class="mono">—</span>`
      : `<span class="status-${c.level}">${ICON[c.level]} ${c.label}</span>`;
    return `<tr>
      <td class="name">${t.topic}</td>
      <td>${t.count.toLocaleString()}</td>
      <td>${t.rate_hz.toFixed(2)}</td>
      <td>${t.mean_dt_ms.toFixed(2)}</td>
      <td>${t.p99_dt_ms.toFixed(2)}</td>
      <td>${t.max_gap_ms.toFixed(2)}</td>
      <td>${t.max_gap_at_s.toFixed(1)}</td>
      <td style="text-align:left">${status}</td>
    </tr>`;
  }).join("");

  card.innerHTML = `
    <div class="card-head"><h2>Per-topic timing</h2>
      <span class="card-sub">native rates, recorder log clock</span></div>
    <div class="table-wrap" style="max-height:none">
      <table>
        <thead><tr>
          <th>Topic</th><th>Messages</th><th>Rate (Hz)</th><th>Mean Δt (ms)</th>
          <th>p99 Δt (ms)</th><th>Max gap (ms)</th><th>At (s)</th>
          <th style="text-align:left">Budget</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  container.appendChild(card);

  const notes = topics.filter((t) => t.note);
  if (notes.length) {
    const prov = document.createElement("div");
    prov.className = "card";
    prov.innerHTML = `
      <div class="card-head"><h2>Stream provenance</h2>
        <span class="card-sub">what the converter does with each stream</span></div>
      <div class="table-wrap" style="max-height:none">
        <table><thead><tr><th>Topic</th><th style="text-align:left">Role</th></tr></thead>
        <tbody>${notes.map((t) =>
          `<tr><td class="name">${t.topic}</td>
           <td style="text-align:left">${t.note}</td></tr>`).join("")}
        </tbody></table>
      </div>`;
    container.appendChild(prov);
  }

  const schema = document.createElement("div");
  schema.className = "card";
  schema.innerHTML = `
    <div class="card-head"><h2>Recorded schemas</h2>
      <span class="card-sub">${data.meta.source.name}</span></div>
    <div class="table-wrap" style="max-height:none">
      <table><thead><tr><th>Topic</th><th style="text-align:left">Schema</th></tr></thead>
      <tbody>${topics.map((t) =>
        `<tr><td class="name">${t.topic}</td>
         <td style="text-align:left" class="mono">${t.schema}</td></tr>`).join("")}
      </tbody></table>
    </div>`;
  container.appendChild(schema);
}
