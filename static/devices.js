// 设备管理页 + 全局磁盘预警横幅。数据来自 collection-hub（经 RoboCurate /api/hub/* 代理）。
const $ = (s, el = document) => el.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const GB = 1024 ** 3;

function bytes(n) {
  if (n == null) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, v = Number(n);
  while (Math.abs(v) >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${i ? v.toFixed(v >= 100 ? 0 : 1) : v} ${u[i]}`;
}
function ago(t) {
  if (!t) return "从未";
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return `${Math.round(s)} 秒前`;
  if (s < 3600) return `${Math.round(s / 60)} 分钟前`;
  if (s < 86400) return `${(s / 3600).toFixed(1)} 小时前`;
  return `${(s / 86400).toFixed(1)} 天前`;
}
function dur(sec) {
  if (sec == null || !isFinite(sec)) return "—";
  if (sec < 60) return `${Math.round(sec)} 秒`;
  if (sec < 3600) return `${Math.round(sec / 60)} 分钟`;
  if (sec < 86400) return `${(sec / 3600).toFixed(1)} 小时`;
  return `${(sec / 86400).toFixed(1)} 天`;
}
const clock = (t) => (t ? new Date(t * 1000).toLocaleString("zh-CN", { hour12: false }) : "—");
const LEVEL = { ok: "正常", warn: "偏低", critical: "紧张", blocked: "已阻断" };
const ONLINE = { online: ["在线", "ok"], unstable: ["不稳定", "warn"], offline: ["离线", "critical"], never: ["未连接", "muted"] };
const SYNC = {
  syncing: "同步中", paused: "已暂停", synced: "已同步", offline_pending: "离线 · 有未同步数据",
  waiting_recording: "录制中 · 暂不同步", blocked_disk: "服务器空间不足 · 已阻断", settling: "刚停止 · 等待确认",
  waiting_idle: "等待空闲", queued: "排队中",
};
const EP_STATUS = { recording: "录制中", incomplete: "写入未完成", settling: "刚停止", pending: "待同步", history: "历史数据", synced: "已同步", deleted: "设备上已删除", cleaned: "已清理" };

let ctx = null;          // {api, toast}
let timer = null;
let openId = null;       // expanded device
let detail = null;
let last = null;
let bannerAt = 0;

async function hub(path, body) {
  return ctx.api("/api/hub/" + path, body);
}

export function stopDevices() {
  clearInterval(timer);
  timer = null;
}

export async function renderDevices(main, context) {
  ctx = context;
  stopDevices();
  main.innerHTML = `<div class="page-head"><div><h1>设备管理</h1><p>采集设备在线与录制状态、双端磁盘、数据同步。录制中的设备不会被同步；收工后可在这里确认“可关机”。</p></div><div class="actions"><button data-dev-action="add">添加设备</button></div></div><div id="dev-root"><div class="loading-page"><span class="spinner"></span> 正在读取设备状态…</div></div>`;
  main.onclick = onClick;
  await tick();
  timer = setInterval(tick, 3000);
}

async function tick() {
  const root = $("#dev-root");
  if (!root) return stopDevices();
  try {
    last = await hub("overview");
    if (openId) detail = await hub("devices/" + openId).catch(() => null);
    root.innerHTML = view(last);
    paintBanner(last);
  } catch (e) {
    root.innerHTML = `<section class="panel dev-unavailable"><h2>设备服务不可用</h2><p>${esc(e.message)}</p><p class="muted">collection-hub 独立于审核工作台运行；它停止时审核功能不受影响。在存储服务器上检查：<code>systemctl --user status collection-hub</code></p></section>`;
  }
}

function view(o) {
  const t = o.totals, a = o.server_disk;
  const card = (title, value, foot, level = "") => `<div class="dev-stat ${level ? "lv-" + level : ""}"><div class="stat-title">${title}</div><div class="stat-value">${value}</div><div class="stat-foot">${foot}</div></div>`;
  const cards = `<div class="stats dev-stats">
    ${card("在线设备", `${t.online}<small>/ ${t.devices}</small>`, t.unsynced_offline_bytes ? `<b class="bad">离线设备上仍有 ${bytes(t.unsynced_offline_bytes)} 未同步</b>` : "15 秒内有心跳算在线")}
    ${card("录制中", t.recording, o.global_bwlimit_mb ? `同步限速 ${o.global_bwlimit_mb} MB/s，保护遥操作 Wi-Fi` : "无设备录制 · 同步全速")}
    ${card("待同步", bytes(t.pending_bytes), `${t.pending_episodes} 条 · ${t.syncing} 台正在同步${t.anomalies ? ` · <b class="bad">${t.anomalies} 条异常</b>` : ""}`)}
    ${card("服务器数据盘", bytes(a.free), `${LEVEL[a.level]} · 约可收 ${a.days_left > 0 ? a.days_left.toFixed(1) : 0} 天${a.intake_measured ? "" : "（按默认日入库量估算）"} · 预留 ${bytes(a.reserve)}`, a.level)}
  </div>`;
  const rows = o.devices.length
    ? o.devices.map(row).join("")
    : `<tr><td colspan="7"><div class="empty"><strong>还没有设备</strong>点击“添加设备”生成令牌，然后在采集设备上运行安装命令。</div></td></tr>`;
  const events = o.events.length
    ? `<section class="panel dev-events"><div class="panel-head"><h2>最近告警</h2></div><ul>${o.events.map((e) => `<li class="lv-${e.level}"><span>${clock(e.at)}</span><b>${esc(e.device || "服务器")}</b>${esc(e.message)}</li>`).join("")}</ul></section>`
    : "";
  return `${cards}<section class="panel"><div class="panel-head"><h2>采集设备</h2><span class="muted">每 3 秒刷新 · 空闲 ${o.settings.idle_minutes} 分钟后开始同步 · 最多 ${o.settings.max_concurrent} 台并行</span></div>
    <div class="table-wrap"><table class="dev-table"><thead><tr><th>设备</th><th>状态</th><th>录制</th><th>设备磁盘</th><th>今日</th><th>同步</th><th>操作</th></tr></thead><tbody>${rows}</tbody></table></div></section>${events}`;
}

function row(d) {
  const [onl, onlLv] = ONLINE[d.online] || [d.online, ""];
  const rec = d.recording
    ? `<span class="badge failure dev-rec">● 录制中</span><small>${esc((d.status_api?.current?.task) || "")} ${d.recording_since ? dur(Date.now() / 1000 - d.recording_since) : ""} · ${bytes(d.recording_bytes)}${d.write_rate_bps ? ` · ${bytes(d.write_rate_bps)}/s` : ""}</small>`
    : d.status_api?.awaiting_outcome
      ? `<span class="badge dev-warn">等待标注</span>`
      : d.status_api?.reachable ? `<span class="muted">空闲</span>` : `<span class="muted">录制服务未运行</span>`;
  const disk = d.disk.free == null ? "—" : `<div class="dev-disk lv-${d.disk.level}"><div class="bar"><span style="width:${d.disk.total ? Math.min(100, 100 - (d.disk.free / d.disk.total) * 100) : 0}%"></span></div><small>剩 ${bytes(d.disk.free)} · 约 ${d.disk.hours_left != null ? d.disk.hours_left.toFixed(1) : "?"} 小时</small></div>`;
  let sync = `<span class="dev-sync st-${d.sync_state}">${SYNC[d.sync_state] || d.sync_state}</span>`;
  if (d.job) sync += `<div class="export-progress"><span style="width:${Math.round(d.job.progress * 100)}%"></span></div><small>${Math.round(d.job.progress * 100)}% · ${bytes(d.job.rate)}/s${d.job.bwlimit_kb ? `（限速 ${Math.round(d.job.bwlimit_kb / 1024)} MB/s）` : ""}${d.eta_seconds ? ` · 约 ${dur(d.eta_seconds)}` : ""}</small>`;
  if (d.pending_bytes && !d.job) sync += `<small>待同步 ${d.pending_episodes} 条 · ${bytes(d.pending_bytes)}</small>`;
  if (d.counts.history) sync += `<small class="muted">历史数据 ${d.counts.history.episodes} 条 · ${bytes(d.counts.history.bytes)}（未纳入）</small>`;
  const safe = d.can_shutdown
    ? `<span class="badge success dev-safe">✓ 可关机</span>`
    : d.off_shift ? `<small class="muted" title="${esc(d.shutdown_blockers.join("；"))}">暂不可关机：${esc(d.shutdown_blockers[0] || "")}</small>` : "";
  const anomalies = d.anomalies.length ? `<small class="bad">⚠ ${d.anomalies.length} 条异常</small>` : "";
  const actions = `<div class="dev-actions">
    <button class="small" data-dev-action="${d.paused ? "resume" : "pause"}" data-id="${d.id}">${d.paused ? "恢复同步" : "暂停同步"}</button>
    <button class="small" data-dev-action="sync-now" data-id="${d.id}" title="不等空闲时间，立即同步（录制中仍不会同步）">立即同步</button>
    <button class="small ${d.off_shift ? "primary" : ""}" data-dev-action="off-shift" data-id="${d.id}" title="收工：跳过空闲等待、优先全速同步">${d.off_shift ? "已收工" : "收工"}</button>
    ${d.capabilities?.shutdown ? `<button class="small ${d.shutdown_after_sync ? "primary" : ""}" data-dev-action="shutdown-after" data-id="${d.id}">${d.shutdown_after_sync ? "同步完关机 ✓" : "同步完关机"}</button>` : ""}
  </div>`;
  const main = `<tr class="dev-row ${openId === d.id ? "open" : ""}" data-dev-open="${d.id}">
    <td><b>${esc(d.name)}</b><small class="muted">${esc(d.last_ip || d.ssh_target || "")}</small></td>
    <td><span class="dev-dot lv-${onlLv}"></span>${onl}<small class="muted">${ago(d.last_seen)}</small></td>
    <td>${rec}</td><td>${disk}</td>
    <td>${d.today.episodes} 条<small class="muted">${bytes(d.today.bytes)}</small></td>
    <td>${sync}${safe}${anomalies}</td><td>${actions}</td></tr>`;
  return main + (openId === d.id ? `<tr class="dev-detail"><td colspan="7">${detailView(d)}</td></tr>` : "");
}

function detailView(d) {
  const x = detail && detail.id === d.id ? detail : null;
  if (!x) return `<div class="loading-page"><span class="spinner"></span></div>`;
  const api = x.status_api || {};
  const cur = api.current;
  const st = `<div class="dev-kv"><div><span>录制服务</span>${api.reachable ? `可访问 · recorder ${api.recorder_available ? "可用" : "不可用"}${api.last_error ? ` · <span class="bad">${esc(api.last_error)}</span>` : ""}` : "未运行"}</div>
    <div><span>当前 episode</span>${cur ? `${esc(cur.task || "")} / ${esc(cur.episode_id || "")} · ${esc(cur.state || "")}` : "—"}</div>
    <div><span>主机</span>${esc(x.hostname || "—")} · node ${esc(x.node_version || "—")} · ${esc(x.ssh_target || "本地")}</div>
    <div><span>镜像目录</span><code>${esc(x.mirror_dir)}</code></div>
    <div><span>关机前检查</span>${x.can_shutdown ? '<b class="ok">全部满足</b>' : esc(x.shutdown_blockers.join("；") || "—")}</div></div>`;
  const anomalies = x.anomalies.length
    ? `<h3>异常 episode</h3><ul class="dev-list">${x.anomalies.map((a) => `<li><code>${esc(a.rel)}</code><span class="bad">${esc(a.reason)}</span><small>${bytes(a.bytes)}</small></li>`).join("")}</ul>` : "";
  const eps = x.episodes.length
    ? `<h3>未同步（${x.episodes.length}${x.episodes.length >= 300 ? "+" : ""}）</h3><ul class="dev-list">${x.episodes.slice(0, 80).map((e) => `<li><code>${esc(e.rel)}</code><span class="dev-ep st-${e.status}">${EP_STATUS[e.status] || e.status}</span><small>${bytes(e.bytes)}${e.sync_error ? ` · <span class="bad">${esc(e.sync_error)}</span>` : ""}</small></li>`).join("")}</ul>` : `<p class="muted">设备上的数据都已同步。</p>`;
  const jobs = x.jobs.length
    ? `<h3>同步记录</h3><table class="dev-jobs"><tr><th>开始</th><th>结果</th><th>数据量</th><th>限速</th></tr>${x.jobs.map((j) => `<tr><td>${clock(j.started_at)}</td><td class="${j.result === "failed" ? "bad" : ""}" title="${esc(j.error || "")}">${esc(j.result || "进行中")}</td><td>${bytes(j.bytes_done)} / ${bytes(j.bytes_planned)}</td><td>${j.bwlimit_kb ? Math.round(j.bwlimit_kb / 1024) + " MB/s" : "全速"}</td></tr>`).join("")}</table>` : "";
  const events = x.events.length ? `<h3>事件</h3><ul class="dev-list dev-log">${x.events.map((e) => `<li class="lv-${e.level}"><small>${clock(e.at)}</small>${esc(e.message)}</li>`).join("")}</ul>` : "";
  const history = x.counts.history
    ? `<p>设备注册前录制的历史数据 ${x.counts.history.episodes} 条（${bytes(x.counts.history.bytes)}）默认不同步。<button class="small" data-dev-action="include-history" data-id="${x.id}">纳入同步</button></p>` : "";
  const cleanup = x.capabilities?.cleanup
    ? `<h3>设备本地清理</h3><p class="muted">只删除服务器上已校验一致的数据，需要二次确认；正在录制和当前 episode 不会被删除。</p><button class="small" data-dev-action="cleanup" data-id="${x.id}">清理已同步数据…</button>` : "";
  return `<div class="dev-detail-grid"><div>${st}${history}${anomalies}${cleanup}</div><div>${eps}${jobs}${events}</div></div>`;
}

async function onClick(e) {
  const btn = e.target.closest("[data-dev-action]");
  if (btn) {
    e.stopPropagation();
    const id = btn.dataset.id;
    const d = last?.devices.find((x) => x.id === id);
    try {
      switch (btn.dataset.devAction) {
        case "pause": await hub(`devices/${id}/settings`, { paused: 1 }); break;
        case "resume": await hub(`devices/${id}/settings`, { paused: 0 }); break;
        case "sync-now": await hub(`devices/${id}/sync-now`, {}); ctx.toast("已加入同步队列"); break;
        case "off-shift": await hub(`devices/${id}/settings`, { off_shift: d?.off_shift ? 0 : 1 }); break;
        case "shutdown-after":
          if (!d.shutdown_after_sync && !confirm(`${d.name}：全部数据同步并校验后自动关机？`)) return;
          await hub(`devices/${id}/settings`, { shutdown_after_sync: d.shutdown_after_sync ? 0 : 1, off_shift: 1 }); break;
        case "include-history": await hub(`devices/${id}/settings`, { include_history: 1 }); break;
        case "cleanup": return cleanupDialog(id);
        case "add": return addDialog();
      }
      await tick();
    } catch (err) {
      ctx.toast(err.message, true);
    }
    return;
  }
  const r = e.target.closest("[data-dev-open]");
  if (r) {
    openId = openId === r.dataset.devOpen ? null : r.dataset.devOpen;
    detail = null;
    await tick();
  }
}

function dialog(title, body, actions = "") {
  document.getElementById("dialog-content").innerHTML =
    `<div class="dialog-title"><h2>${title}</h2><button data-action="close-dialog" aria-label="关闭对话框">×</button></div><div class="dialog-body">${body}</div><div class="form-error" id="dialog-error" role="alert"></div><div class="dialog-actions"><button data-action="close-dialog">取消</button>${actions}</div>`;
  const dlg = document.getElementById("dialog");
  if (!dlg.open) dlg.showModal();
  return dlg;
}
const fail = (err) => { const el = document.getElementById("dialog-error"); if (el) el.textContent = err.message || String(err); };

function addDialog() {
  const dlg = dialog("添加采集设备",
    `<label class="field-label" for="dev-name">设备名称 <span class="muted">· 同时作为服务器上的目录名</span></label><input id="dev-name" placeholder="G1-01">
     <label class="field-label" for="dev-ssh">SSH 目标 <span class="muted">· hub 从这里拉取数据</span></label><input id="dev-ssh" placeholder="user@robot-host">
     <label class="field-label" for="dev-root-path">远端目录 <span class="muted">· 用 rrsync 只读限权时留空</span></label><input id="dev-root-path" placeholder="">`,
    `<button class="primary" id="dev-create">生成令牌</button>`);
  $("#dev-create", dlg).onclick = async () => {
    try {
      const res = await hub("devices", { name: $("#dev-name").value.trim(), ssh_target: $("#dev-ssh").value.trim(), remote_root: $("#dev-root-path").value.trim() });
      const base = `http://${location.hostname}:${res.node_port || 8422}`;
      const cmd = `curl -fsSL ${base}/install/install_node.sh | bash -s -- \\\n  --hub ${base} --token ${res.token}` + (res.ssh_pubkey ? ` \\\n  --hub-key '${res.ssh_pubkey}'` : "");
      dialog(`设备已注册：${esc(res.device.name)}`,
        `<p>令牌只显示这一次。在该采集设备上用采集用户执行：</p><pre class="dev-cmd">${esc(cmd)}</pre><p class="muted">会安装用户级 systemd 服务 collection-node，并把 hub 的只读拉取公钥（rrsync）写入 authorized_keys。需要远程关机或本地清理时，追加 <code>--allow-shutdown</code> / <code>--allow-cleanup</code>。</p>`);
      tick();
    } catch (err) { fail(err); }
  };
}

function cleanupDialog(id) {
  const dlg = dialog("清理设备上已同步的数据",
    `<p class="muted">只删除服务器上已逐文件校验一致的数据；正在录制和当前 episode 不会被删除。</p><label class="field-label" for="dev-days">只清理多少天以前的数据</label><input id="dev-days" type="number" min="0" step="1" value="3">`,
    `<button class="primary" id="dev-preview">预览</button>`);
  $("#dev-preview", dlg).onclick = async () => {
    try {
      const p = await hub(`devices/${id}/cleanup/preview`, { older_than_days: Number($("#dev-days").value) });
      if (!p.episodes) return fail(Error("没有符合条件、且已在服务器上校验一致的数据"));
      dialog("确认清理",
        `<p>将删除设备上 <b>${p.episodes}</b> 条 episode，共 <b>${bytes(p.bytes)}</b>。服务器上的副本已校验（大小与修改时间一致），保持不变。</p>
         <ul class="dev-list">${p.sample.map((r) => `<li><code>${esc(r)}</code></li>`).join("")}${p.episodes > p.sample.length ? `<li class="muted">…另 ${p.episodes - p.sample.length} 条</li>` : ""}</ul>
         <label class="field-label" for="dev-code">输入确认码 <code>${esc(p.confirm_token)}</code></label><input id="dev-code" autocomplete="off">`,
        `<button class="primary danger" id="dev-confirm">确认删除</button>`);
      $("#dev-confirm").onclick = async () => {
        try {
          const r = await hub(`devices/${id}/cleanup/confirm`, { request_id: p.request_id, confirm_token: $("#dev-code").value.trim() });
          document.getElementById("dialog").close();
          ctx.toast(`已下发清理指令：${r.episodes} 条，设备下次心跳时执行`);
          tick();
        } catch (err) { fail(err); }
      };
    } catch (err) { fail(err); }
  };
}

// ---------------------------------------------------------------- global banner
function paintBanner(o) {
  const el = document.getElementById("hub-banner");
  const badge = document.getElementById("nav-devices-badge");
  if (!el) return;
  if (!o) { el.hidden = true; if (badge) badge.textContent = ""; return; }
  const msgs = [];
  const a = o.server_disk;
  if (a.level !== "ok") msgs.push([a.level, `服务器数据盘剩余 ${bytes(a.free)}，约可收 ${Math.max(0, a.days_left).toFixed(1)} 天${a.blocked ? "，已低于预留值，新同步已暂停" : ""}`]);
  for (const d of o.devices) {
    if (d.online !== "never" && d.disk.level !== "ok") msgs.push([d.disk.level, `${d.name} 磁盘剩余 ${bytes(d.disk.free)}，约可录 ${d.disk.hours_left?.toFixed(1)} 小时`]);
    if (d.online === "offline" && d.unsynced_bytes > 0) msgs.push(["critical", `${d.name} 已离线，${bytes(d.unsynced_bytes)} 只在设备上`]);
  }
  if (badge) {
    const n = msgs.length + o.totals.anomalies;
    badge.textContent = n ? String(n) : o.totals.recording ? `● ${o.totals.recording}` : "";
    badge.className = "nav-count" + (n ? " dev-badge-bad" : "");
  }
  if (!msgs.length) { el.hidden = true; return; }
  const worst = msgs.some((m) => m[0] === "critical" || m[0] === "blocked") ? "critical" : "warn";
  el.className = "hub-banner lv-" + worst;
  el.innerHTML = msgs.slice(0, 3).map((m) => `<span>${esc(m[1])}</span>`).join("") + (msgs.length > 3 ? `<span>…另 ${msgs.length - 3} 条</span>` : "") + ` <a href="#devices">查看设备</a>`;
  el.hidden = false;
}

export async function updateHubBanner(context) {
  ctx = ctx || context;
  if (Date.now() - bannerAt < 15000) return;
  bannerAt = Date.now();
  try {
    const res = await fetch("/api/hub/overview");
    if (!res.ok) { paintBanner(null); return; }
    paintBanner(await res.json());
  } catch { paintBanner(null); }
}
