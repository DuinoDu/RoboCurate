/* The episode rail: what else is on disk, and how to get at it.
 *
 * A collection session is a *sequence* of episodes — the recorder numbers them
 * `episode_000013`, `episode_000014`, … under one timestamped session — and
 * the questions worth asking are comparative: which take succeeded, which one
 * has the gap, is this drift in every episode or only this one. A viewer that
 * can only open the file named on the command line cannot answer any of them.
 *
 * Two things make this rail more than a file list:
 *
 *   * **The operator's verdict is already on disk.** `pico_episode_summary.json`
 *     carries `task_outcome.success`, so the list can mark a failed take before
 *     you open it. The raw MCAP has no such flag; an episode copied without its
 *     sidecar reads as unknown rather than as a pass.
 *   * **Extraction costs minutes.** So the rail says which episodes are already
 *     extracted, and choosing an unextracted one starts a build with visible
 *     progress rather than a silent hang.
 *
 * Switching episodes reloads the page. The store, the charts, the camera and
 * the 3D view are all built around one episode's shape, and a reload is the
 * one refresh guaranteed to leave none of the previous one behind.
 */

const $ = (sel, root = document) => root.querySelector(sel);

const OUTCOME = {
  true: { key: "ok", glyph: "✓", label: "operator marked this a success" },
  false: { key: "bad", glyph: "✕", label: "operator marked this a failure" },
  null: { key: "unknown", glyph: "·", label: "no outcome recorded on disk" },
};

const CACHE_LABEL = {
  ready: "extracted",
  absent: "not extracted — first open takes a few minutes",
  building: "extracting…",
  error: "extraction failed",
};

const COLLAPSE_KEY = "hm-rail-collapsed";

/** `scp`/`rsync` is the operator's job, not the server's.
 *
 *  The viewer reads local paths. Typing a remote one is a reasonable mistake —
 *  the episodes do live on the robot — so answer it with the command that
 *  would put them within reach instead of a flat "no such directory". */
function remoteHint(text) {
  const m = text.match(/^([\w.-]+@[\w.-]+):(\/\S*)$/);
  if (!m) return null;
  const [, host, path] = m;
  return `That path is on ${host}. Copy it here first, then add the local ` +
    `copy:\n\nrsync -a --info=progress2 ${host}:${path} ~/recordings/`;
}

function human(bytes) {
  if (!bytes) return "";
  return bytes >= 1e9
    ? `${(bytes / 1e9).toFixed(2)} GB`
    : `${Math.round(bytes / 1e6)} MB`;
}

export class Rail {
  constructor(host) {
    this.host = host;
    this.data = { episodes: [], roots: [], default: null, cache_root: "" };
    this.active = null;
    this._build();
  }

  /* ------------------------------------------------------------- chrome -- */

  _build() {
    this.host.innerHTML = `
      <div class="rail-head">
        <h2>Episodes</h2>
        <span class="rail-count" id="rail-count"></span>
        <button class="icon-btn" id="rail-rescan" title="Re-scan every root">
          Rescan</button>
      </div>
      <form class="rail-add" id="rail-add">
        <input id="rail-path" type="text" spellcheck="false"
               placeholder="/path/to/recordings"
               aria-label="Local directory to scan for episodes">
        <button class="btn" type="submit">Add</button>
      </form>
      <p class="rail-msg" id="rail-msg" hidden></p>
      <div class="rail-list" id="rail-list"></div>
      <p class="rail-foot" id="rail-foot"></p>`;

    $("#rail-add", this.host).addEventListener("submit", (ev) => {
      ev.preventDefault();
      this.addPath($("#rail-path", this.host).value.trim());
    });
    $("#rail-rescan", this.host).addEventListener("click", () => this.rescan());
  }

  message(text, kind = "") {
    const el = $("#rail-msg", this.host);
    el.textContent = text || "";
    el.dataset.kind = kind;
    el.hidden = !text;
  }

  /* --------------------------------------------------------------- data -- */

  async refresh() {
    const res = await fetch("/api/episodes");
    if (!res.ok) throw new Error(`episodes ${res.status}`);
    this.data = await res.json();
    this.render();
    return this.data;
  }

  entry(id) {
    return this.data.episodes.find((e) => e.id === id) || null;
  }

  /** Which episode this page is for: an explicit `?ep=` wins, else the
   *  server's default (what was named on the command line, or the first
   *  episode already extracted). */
  get target() {
    const asked = new URLSearchParams(location.search).get("ep");
    if (asked && this.entry(asked)) return asked;
    return this.data.default || null;
  }

  async addPath(raw) {
    if (!raw) return;
    const hint = remoteHint(raw);
    if (hint) return this.message(hint, "warn");
    this.message("scanning…");
    let res;
    try {
      res = await fetch("/api/roots", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: raw }),
      });
    } catch (err) {
      return this.message(`could not reach the server: ${err.message}`, "bad");
    }
    const body = await res.json().catch(() => ({}));
    if (!res.ok) return this.message(body.error || `error ${res.status}`, "bad");
    this.data = body;
    this.render();
    $("#rail-path", this.host).value = "";
    this.message(
      body.added
        ? `found ${body.added} episode(s) under ${raw}`
        : `no .mcap under ${raw}`,
      body.added ? "" : "warn",
    );
  }

  /** Re-scan the known roots, so episodes recorded since startup show up. */
  async rescan() {
    const roots = [...this.data.roots];
    this.message("rescanning…");
    let total = 0;
    for (const root of roots) {
      const res = await fetch("/api/roots", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: root }),
      });
      if (res.ok) {
        this.data = await res.json();
        total += this.data.added || 0;
      }
    }
    this.render();
    this.message(`${total} episode(s) across ${roots.length} root(s)`);
  }

  /* ------------------------------------------------------------ extract -- */

  /** Make sure `id` has a cache, reporting progress until it does.
   *
   *  Returns true when the episode is ready to load. Polling `/api/episodes`
   *  rather than holding a request open means a reload during a build rejoins
   *  the same build instead of starting a second one. */
  async prepare(id, onProgress = () => {}) {
    let state = this.entry(id)?.cache;
    if (state === "ready") return true;

    const res = await fetch("/api/prepare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ep: id }),
    });
    const started = await res.json().catch(() => ({}));
    if (!res.ok) {
      onProgress(started.error || `could not start extraction (${res.status})`);
      return false;
    }
    if (started.state === "ready") return true;

    for (;;) {
      await new Promise((r) => setTimeout(r, 1200));
      let entry;
      try {
        await this.refresh();
        entry = this.entry(id);
      } catch {
        continue; // a dropped poll is not a failed build
      }
      if (!entry) return false;
      state = entry.cache;
      if (state === "ready") return true;
      if (state === "error") {
        onProgress(`Extraction failed: ${entry.cache_message}`);
        return false;
      }
      onProgress(`Extracting this episode — ${entry.cache_message}`);
    }
  }

  /* ------------------------------------------------------------- render -- */

  setActive(id) {
    this.active = id;
    for (const btn of this.host.querySelectorAll(".ep")) {
      btn.setAttribute("aria-current", String(btn.dataset.ep === id));
    }
  }

  render() {
    const list = $("#rail-list", this.host);
    const eps = this.data.episodes;
    $("#rail-count", this.host).textContent = eps.length
      ? `${eps.length}`
      : "";
    $("#rail-foot", this.host).textContent = this.data.cache_root
      ? `cache · ${this.data.cache_root}`
      : "";

    if (!eps.length) {
      list.innerHTML = `<p class="rail-empty">No episodes yet. Add a
        directory above — a session tree from the robot, or any folder of
        <code>.mcap</code> files.</p>`;
      return;
    }

    // One block per session, because that is the unit the recorder writes and
    // the unit an operator reasons about ("the 15:44 run").
    const groups = new Map();
    for (const ep of eps) {
      // Keyed as JSON, not a joined string: a task name may contain
      // whatever the operator typed, and a naive separator would fold
      // two different tasks into one block.
      const key = JSON.stringify([ep.session || "", ep.task || ""]);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(ep);
    }

    list.textContent = "";
    for (const [key, items] of groups) {
      const [session, task] = JSON.parse(key);
      const block = document.createElement("section");
      block.className = "rail-group";
      const head = document.createElement("div");
      head.className = "rail-group-head";
      head.innerHTML = session
        ? `<span class="rail-session">${session}</span>` +
          (task ? `<span class="rail-task">${task}</span>` : "")
        : `<span class="rail-session">loose files</span>`;
      block.appendChild(head);
      for (const ep of items) block.appendChild(this._item(ep));
      list.appendChild(block);
    }
    this.setActive(this.active);
  }

  _item(ep) {
    const btn = document.createElement("button");
    btn.className = "ep";
    btn.dataset.ep = ep.id;
    btn.dataset.cache = ep.cache;
    btn.setAttribute("aria-current", String(ep.id === this.active));

    const outcome = OUTCOME[String(ep.outcome)];
    const facts = [];
    if (ep.duration_s) facts.push(`${ep.duration_s.toFixed(0)} s`);
    if (ep.frames) facts.push(`${ep.frames} frames`);
    facts.push(human(ep.size_bytes));

    const detail = ep.cache === "building"
      ? ep.cache_message
      : CACHE_LABEL[ep.cache] || ep.cache;

    btn.innerHTML = `
      <span class="ep-mark" data-outcome="${outcome.key}"
            title="${outcome.label}">${outcome.glyph}</span>
      <span class="ep-body">
        <span class="ep-name">${ep.episode_id}${
          ep.part ? ` <span class="ep-part">${ep.part}</span>` : ""}</span>
        <span class="ep-facts">${facts.filter(Boolean).join(" · ")}</span>
        <span class="ep-state" data-cache="${ep.cache}">${detail}</span>
      </span>`;
    btn.title = [ep.path, ep.instruction, ep.failure_reason]
      .filter(Boolean).join("\n");
    btn.addEventListener("click", () => openEpisode(ep.id));
    return btn;
  }
}

/** Go to another episode.
 *
 *  `panel`, `theme` and `open` survive the jump — you are usually looking for
 *  the same chart in the next take — but `t=` does not: two episodes are
 *  different performances, and the same wall-clock second in each is not the
 *  same moment. The new episode opens at its own first camera frame.
 */
export function openEpisode(id) {
  const hash = new URLSearchParams(location.hash.slice(1));
  hash.delete("t");
  const query = new URLSearchParams(location.search);
  query.set("ep", id);
  location.href = `${location.pathname}?${query}${
    hash.toString() ? `#${hash}` : ""}`;
}

/** Collapse state lives across reloads: switching episodes reloads the page,
 *  and a rail that reopened every time would fight the operator. */
export function initCollapse(shell, toggle, rail) {
  const apply = (collapsed) => {
    shell.classList.toggle("rail-off", collapsed);
    rail.hidden = collapsed;
    toggle.setAttribute("aria-pressed", String(!collapsed));
    localStorage.setItem(COLLAPSE_KEY, collapsed ? "1" : "0");
  };
  apply(localStorage.getItem(COLLAPSE_KEY) === "1");
  toggle.addEventListener("click", () => {
    apply(!shell.classList.contains("rail-off"));
  });
}
