import {
  sortEvents,
  nextPending,
  applyPendingBatch,
  undoBatch,
} from "./review-state.mjs";

const labels = {
  left_leg: "左腿",
  right_leg: "右腿",
  waist: "腰部",
  left_arm: "左臂",
  right_arm: "右臂",
  left_hand: "左手",
  right_hand: "右手",
};
const modes = { motion_enabled: "动作模式", velocity_enabled: "速度模式" };

export class ReviewTools {
  constructor(ctx) {
    this.ctx = ctx;
    this.ep = ctx.current().id;
    this.data = null;
    this.loading = false;
    this.host = null;
    this.group = "";
    this.verdict = "";
    this.active = null;
    this.page = 0;
    this.alive = true;
    this.order = "time";
    this.fastMode = false;
    this.lastBatch = null;
    this.overviewOpen = false;
    this.optionsOpen = false;
    this.positionKey = "robocurate-review-position-" + this.ep;
  }
  destroy() {
    this.alive = false;
    this.host = null;
  }
  mount(host) {
    this.host = host;
    this.render();
    this.ensureLoaded();
  }
  async ensureLoaded() {
    if (!this.alive) return;
    if (this.ctx.current()?.cache !== "ready") {
      this.data = null;
      const timeline = document.querySelector("#event-timeline");
      if (timeline) timeline.innerHTML = "";
      this.render();
      return;
    }
    if (this.data || this.loading) return;
    this.loading = true;
    this.error = "";
    this.render();
    try {
      const data = await this.ctx.api("/api/diagnostics?ep=" + this.ep);
      if (!this.alive || this.ctx.current()?.id !== this.ep) return;
      this.data = data;
      this.restorePosition();
      this.render();
      this.renderTimeline();
    } catch (e) {
      this.error = e.message;
      this.render();
    } finally {
      this.loading = false;
    }
  }
  decisions() {
    return this.ctx.draft().event_decisions || {};
  }
  events() {
    return sortEvents(
      (this.data?.events || []).filter(
        (e) =>
          (!this.group || e.group === this.group) &&
          (!this.verdict ||
            (this.verdict === "pending"
              ? !this.decisions()[e.id]
              : this.decisions()[e.id] === this.verdict)),
      ),
      this.order,
    );
  }
  savePosition() {
    if (!this.data) return;
    try {
      localStorage.setItem(
        this.positionKey,
        JSON.stringify({
          signature: this.data.signature,
          active: this.active,
          group: this.group,
          verdict: this.verdict,
          order: this.order,
          fastMode: this.fastMode,
          page: this.page,
        }),
      );
    } catch {}
  }
  restorePosition() {
    let saved;
    try {
      saved = JSON.parse(localStorage.getItem(this.positionKey));
    } catch {
      return;
    }
    if (!saved) return;
    this.order = saved.order === "priority" ? "priority" : "time";
    this.fastMode = !!saved.fastMode;
    this.group = labels[saved.group] ? saved.group : "";
    this.verdict = ["pending", "accepted", "confirmed"].includes(saved.verdict)
      ? saved.verdict
      : "";
    if (saved.signature === this.data.signature) {
      const event = this.events().find((e) => e.id === saved.active);
      if (event) {
        this.active = event.id;
        this.page = Math.floor(
          this.events().findIndex((e) => e.id === event.id) / 6,
        );
        this.ctx.seek(event.peak_s);
      }
    }
  }
  clearHiddenSelection() {
    if (!this.events().some((e) => e.id === this.active)) this.active = null;
  }
  mountHandlers(host) {
    if (host) host.onclick = (e) => this.click(e);
  }
  renderTimeline() {
    const host = document.querySelector("#event-timeline");
    if (!host || !this.data || !this.alive) return;
    const { escape: esc } = this.ctx,
      d = this.data,
      duration = d.duration_s;
    const lanes = [
      [
        "上肢",
        (e) =>
          ["left_arm", "right_arm", "left_hand", "right_hand"].includes(
            e.group,
          ),
      ],
      ["腿 / 腰", (e) => ["left_leg", "right_leg", "waist"].includes(e.group)],
      ["数据 / 温度", (e) => !e.group],
    ];
    host.innerHTML = `<div class="event-timeline-title"><span>问题时间轴 <small>点击标记定位 · 提醒不等于任务失败</small></span><button class="text-btn" data-tool="open">${d.events.length} 处线索 ↗</button></div><div class="event-lane"><span>控制模式</span><div class="event-lane-track">${(d.bands.policy_mode || []).map((b) => `<button class="mode-marker ${b.value === "velocity_enabled" ? "velocity" : ""}" style="left:${(b.start_s / duration) * 100}%;width:${((b.end_s - b.start_s) / duration) * 100}%" data-time="${b.start_s}" title="${esc(modes[b.value] || b.value)} ${b.start_s.toFixed(2)}–${b.end_s.toFixed(2)}s">${b.end_s - b.start_s > 2 ? esc(modes[b.value] || b.value) : ""}</button>`).join("")}</div></div>${lanes
      .map(
        ([name, filter]) =>
          `<div class="event-lane"><span>${name}</span><div class="event-lane-track">${d.events
            .filter(filter)
            .map(
              (e) =>
                `<button class="event-marker ${e.severity} ${this.decisions()[e.id] || ""} ${e.id === this.active ? "active" : ""}" style="left:${(e.start_s / duration) * 100}%;width:${Math.max(0.4, ((e.end_s - e.start_s) / duration) * 100)}%" data-event="${e.id}" title="${esc(e.label)} · ${e.start_s.toFixed(2)}–${e.end_s.toFixed(2)}s" aria-label="定位 ${esc(e.label)} ${e.start_s.toFixed(2)} 秒"></button>`,
            )
            .join("")}</div></div>`,
      )
      .join("")}`;
    this.mountHandlers(host);
  }
  render() {
    this.ctx.selection?.(this.data?.events.find(e=>e.id===this.active));
    const host = this.host;
    if (!host?.isConnected || !this.alive) return;
    const { escape: esc } = this.ctx;
    if (!this.data) {
      const state=this.ctx.current()?.cache;
      const failed=state==='error';
      host.innerHTML = `<div class="empty">${this.error ? esc(this.error) : failed?'解析失败':state==='absent'?'等待解析':'<span class="spinner"></span> 加载诊断数据…'}<br><span class="hint">${failed?esc(this.ctx.current().cache_message||'请重新解析此记录。'):'按原始时间戳定位关节、手部、采样间隔和温度变化。'}</span>${this.error?'<br><button data-tool="retry">重试</button>':failed||state==='absent'?'<br><button data-tool="analyze">重新解析</button>':''}</div>`;
      this.mountHandlers(host);
      return;
    }
    const d = this.data,
      decisions = this.decisions(),
      list = this.events(),
      done = d.events.filter((e) => decisions[e.id]).length;
    this.page = Math.max(
      0,
      Math.min(this.page, Math.ceil(list.length / 6) - 1),
    );
    const active = d.events.find((e) => e.id === this.active),
      visible = list.slice(this.page * 6, this.page * 6 + 6);
    host.innerHTML = `<div class="review-progress"><div><b>问题复核</b><span>${done} / ${d.events.length} 已记录判断</span></div><div class="actions"><button class="small" data-tool="previous" title="快捷键 P">← 上一处</button><button class="small primary" data-tool="next" title="快捷键 N">下一处未复核 →</button></div></div><div class="event-filters"><select id="event-group" aria-label="问题部位"><option value="">全部部位</option>${Object.entries(labels).map(([value,label])=>`<option value="${value}" ${this.group===value?"selected":""}>${label}</option>`).join("")}</select><select id="event-verdict" aria-label="问题复核状态"><option value="">全部判断状态</option><option value="pending" ${this.verdict === "pending" ? "selected" : ""}>待复核</option><option value="confirmed" ${this.verdict === "confirmed" ? "selected" : ""}>需处理</option><option value="accepted" ${this.verdict === "accepted" ? "selected" : ""}>可接受</option></select><span>${list.length} 处${this.group ? " · " + labels[this.group] : ""}</span></div><details class="review-options" ${this.optionsOpen ? "open" : ""}><summary>审核选项</summary><details class="diagnostic-overview" ${this.overviewOpen ? "open" : ""}><summary>部位指标 · P99</summary><div class="body-metrics">${d.groups.map((g) => `<button data-group="${g.group}" class="body-metric ${this.group === g.group ? "selected" : ""}"><span>${g.label}</span><b>${g.p99 == null ? "—" : g.p99.toFixed(3)}</b><small>P99 · ${g.unit}</small></button>`).join("")}</div></details><div class="review-fastbar"><label><input id="fast-review" type="checkbox" ${this.fastMode ? "checked" : ""}> 判断后自动下一处</label><select id="event-order" aria-label="问题排序"><option value="time" ${this.order === "time" ? "selected" : ""}>时间顺序</option><option value="priority" ${this.order === "priority" ? "selected" : ""}>技术优先</option></select><span title="优先无效数值和断流，再按提醒级别、超阈比例与持续时间排序；不是任务失败概率。">按技术证据排序 ⓘ</span></div><div class="event-batchbar"><span>当前筛选 ${list.filter((e) => !decisions[e.id]).length} 条待复核</span><button class="small" data-tool="batch-confirmed" ${!list.some((e) => !decisions[e.id]) ? "disabled" : ""} title="将当前筛选中的未复核项标为需处理">批量需处理</button><button class="small" data-tool="batch-accepted" ${!list.some((e) => !decisions[e.id]) ? "disabled" : ""} title="将当前筛选中的未复核项标为可接受">批量可接受</button><button class="text-btn" data-tool="undo-batch" ${!this.lastBatch ? "disabled" : ""}>撤销批量</button></div></details><div class="event-workspace"><div class="event-list">${visible.map((e) => `<button class="event-list-row ${e.id === this.active ? "selected" : ""}" data-event="${e.id}"><span class="event-dot ${e.severity}"></span><span><b>${esc(e.label)}</b><small>${e.start_s.toFixed(2)}–${e.end_s.toFixed(2)} s · ${e.value == null ? "无效数值" : e.value.toFixed(3) + " " + esc(e.unit)}</small></span><em class="${decisions[e.id] || ""}">${decisions[e.id] === "confirmed" ? "需处理" : decisions[e.id] === "accepted" ? "可接受" : "待复核"}</em></button>`).join("") || '<div class="empty">此筛选下没有问题线索。</div>'}<div class="event-pagination"><button class="small" data-tool="page-prev" ${this.page === 0 ? "disabled" : ""}>上一页</button><span>${list.length ? this.page + 1 : 0} / ${Math.ceil(list.length / 6)}</span><button class="small" data-tool="page-next" ${(this.page + 1) * 6 >= list.length ? "disabled" : ""}>下一页</button></div></div><div class="event-detail ${active ? "has-active" : "is-empty"}">${active ? `<div class="event-detail-content"><div class="event-current-label">当前问题</div><div class="event-title-line"><h3>${esc(active.label)}</h3><div class="event-peak">${active.value == null ? "NaN / Inf" : active.value.toFixed(3)} <small>${esc(active.unit)}</small></div></div><p class="event-time">峰值 ${active.peak_s.toFixed(3)} s${active.threshold == null ? "" : " · 提醒线 " + active.threshold + " " + esc(active.unit)}</p><details class="event-context"><summary>详细依据与控制状态</summary><p>${esc(active.detail)}</p><p class="hint">${esc(modes[active.context?.policy_mode] || active.context?.policy_mode || "")} · ${esc(active.context?.robot_state || "")}</p></details></div><div class="event-detail-actions"><div class="actions"><button class="small primary" data-tool="play">▷ 回放这一处</button><button class="small" data-tool="signal">对应曲线 ↗</button></div><div class="event-decisions"><button class="small ${decisions[active.id] === "confirmed" ? "chosen" : ""}" data-decision="confirmed">需处理</button><button class="small ${decisions[active.id] === "accepted" ? "chosen" : ""}" data-decision="accepted">可接受</button><button class="text-btn" data-decision="">撤销判断</button></div><small>判断随审核保存。</small></div>` : '<div class="event-placeholder">选择一条线索，查看画面与判定依据。<small>N / P 切换待复核线索</small></div>'}</div></div><details class="coverage-options"><summary>片段覆盖建议</summary><div class="coverage-summary"><div><b>30 Hz 时间与数值覆盖：${(d.coverage.fraction * 100).toFixed(2)}%</b><p>${d.coverage.aligned_samples} / ${d.coverage.samples} 个可对齐时刻。${esc(d.coverage.note || d.coverage.error || "")}</p></div><button class="small" data-tool="coverage">添加覆盖完整区间</button></div></details>`;
    this.mountHandlers(host);
    host.querySelector('.diagnostic-overview').addEventListener('toggle',e=>{if(e.target.isConnected)this.overviewOpen=e.target.open;});
    host.querySelector('.review-options').addEventListener('toggle',e=>{if(e.target.isConnected)this.optionsOpen=e.target.open;});
    host.querySelector('#event-group').onchange = e=>{
      this.group=e.target.value;this.page=0;this.clearHiddenSelection();this.savePosition();this.render();
    };
    host.querySelector("#event-verdict").onchange = (e) => {
      this.verdict = e.target.value;
      this.page = 0;
      this.clearHiddenSelection();
      this.savePosition();
      this.render();
    };
    host.querySelector("#event-order").onchange = (e) => {
      this.order = e.target.value;
      this.page = 0;
      this.savePosition();
      this.render();
    };
    host.querySelector("#fast-review").onchange = (e) => {
      this.fastMode = e.target.checked;
      this.savePosition();
    };
  }
  select(id) {
    const e = this.data?.events.find((e) => e.id === id);
    if (!e) return;
    this.active = id;
    this.ctx.open();
    this.ctx.seek(e.peak_s);
    this.render();
    this.renderTimeline();
    this.savePosition();
  }
  next(direction = 1) {
    const list = this.events();
    if (!list.length) {
      this.ctx.toast("当前筛选下没有问题。");
      return;
    }
    const next = nextPending(list, this.active, this.decisions(), direction);
    if (next) {
      this.page = Math.floor(next.index / 6);
      this.select(next.event.id);
      return;
    }
    this.ctx.toast("当前筛选下的问题已全部记录判断。");
  }
  click(event) {
    const b = event.target.closest("button");
    if (!b) return;
    event.stopPropagation();
    if (b.hasAttribute("data-event")) {
      this.select(b.dataset.event);
      return;
    }
    if (b.hasAttribute("data-time")) {
      this.ctx.seek(Number(b.dataset.time));
      return;
    }
    if (b.hasAttribute("data-group")) {
      this.group = b.dataset.group;
      this.page = 0;
      this.clearHiddenSelection();
      this.savePosition();
      this.render();
      return;
    }
    if (b.hasAttribute("data-decision")) {
      const draft = this.ctx.draft();
      draft.event_decisions ||= {};
      if (b.dataset.decision)
        draft.event_decisions[this.active] = b.dataset.decision;
      else delete draft.event_decisions[this.active];
      this.ctx.dirty();
      if (this.fastMode && b.dataset.decision) {
        this.next();
      }
      this.clearHiddenSelection();
      this.savePosition();
      this.render();
      this.renderTimeline();
      return;
    }
    const active = this.data?.events.find((e) => e.id === this.active);
    switch (b.dataset.tool) {
      case 'analyze':this.ctx.analyze?.();break;
      case "batch-confirmed":
      case "batch-accepted": {
        const status =
          b.dataset.tool === "batch-accepted" ? "accepted" : "confirmed";
        const result = applyPendingBatch(
          this.decisions(),
          this.events(),
          status,
        );
        if (!result.changes.length) {
          this.ctx.toast("当前筛选下没有未复核项目。");
          break;
        }
        this.ctx.draft().event_decisions = result.next;
        this.lastBatch = result.changes;
        this.ctx.dirty();
        this.clearHiddenSelection();
        this.savePosition();
        this.render();
        this.renderTimeline();
        this.ctx.toast(
          `已批量记录 ${result.changes.length} 条判断；已有结果保持不变，可撤销本次批量。`,
        );
        break;
      }
      case "undo-batch": {
        const result = undoBatch(this.decisions(), this.lastBatch);
        this.ctx.draft().event_decisions = result.next;
        this.lastBatch = null;
        this.ctx.dirty();
        this.render();
        this.renderTimeline();
        this.savePosition();
        this.ctx.toast(
          `已撤销 ${result.restored} 条批量判断，后续单独修改的结果保留。`,
        );
        break;
      }
      case "open":
        this.ctx.open();
        break;
      case "retry":
        this.ensureLoaded();
        break;
      case "next":
        this.next();
        break;
      case "previous":
        this.next(-1);
        break;
      case "page-next":
        this.page++;
        this.savePosition();
        this.render();
        break;
      case "page-prev":
        this.page--;
        this.savePosition();
        this.render();
        break;
      case "play":
        if (active)
          this.ctx.preview(
            Math.max(0, active.start_s - 0.5),
            Math.min(this.data.duration_s, active.end_s + 0.5),
          );
        break;
      case "signal":
        if (active) this.ctx.signal(active);
        break;
      case "coverage": {
        const draft = this.ctx.draft();
        let added = 0;
        for (const c of this.data.candidates) {
          if (!draft.segments.some((s) => s.start < c.end && s.end > c.start)) {
            draft.segments.push({ ...c });
            added++;
          }
        }
        if (!added) {
          this.ctx.toast("没有可添加的独立区间；已有片段保持不变。");
          break;
        }
        draft.segments.sort((a, b) => a.start - b.start);
        this.ctx.dirty();
        this.ctx.clips();
        this.ctx.toast(
          `已添加 ${added} 个技术覆盖区间，请结合任务回放确认后保存。`,
        );
        break;
      }
    }
  }
}
