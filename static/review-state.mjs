// Pure review operations shared by the browser and dependency-free tests.
const finite = (n, fallback = 0) =>
  Number.isFinite(Number(n)) ? Number(n) : fallback;

export function sortEvents(events, order = "time") {
  const rank = (e) =>
    e.kind === "invalid"
      ? 0
      : e.kind === "gap"
        ? 1
        : e.severity === "warning"
          ? 2
          : 3;
  const magnitude = (e) => {
    const threshold = Math.abs(finite(e.threshold));
    const ratio = threshold > 0 ? Math.abs(finite(e.value)) / threshold : 0;
    return (
      Math.min(ratio, 10) *
      (1 + Math.log1p(Math.max(0, finite(e.end_s) - finite(e.start_s))))
    );
  };
  return [...events].sort((a, b) => {
    if (order === "priority") {
      const priority = rank(a) - rank(b) || magnitude(b) - magnitude(a);
      if (priority) return priority;
    }
    return (
      finite(a.start_s) - finite(b.start_s) ||
      finite(a.end_s) - finite(b.end_s) ||
      String(a.id).localeCompare(String(b.id))
    );
  });
}

export function nextPending(events, activeId, decisions, direction = 1) {
  if (!events.length) return null;
  const found = events.findIndex((e) => e.id === activeId);
  const start = found >= 0 ? found : direction > 0 ? -1 : 0;
  for (let step = 1; step <= events.length; step++) {
    const i = (start + direction * step + events.length * 2) % events.length;
    if (!decisions[events[i].id]) return { event: events[i], index: i };
  }
  return null;
}

export function applyPendingBatch(decisions, events, status) {
  if (!["accepted", "confirmed"].includes(status))
    throw Error("无效的审核结果");
  const next = { ...decisions },
    changes = [];
  for (const event of events) {
    if (next[event.id]) continue;
    changes.push({ id: event.id, before: undefined, after: status });
    next[event.id] = status;
  }
  return { next, changes };
}

export function undoBatch(decisions, changes) {
  const next = { ...decisions };
  let restored = 0;
  for (const change of changes || []) {
    // Later individual edits take precedence over an older bulk operation.
    if (next[change.id] !== change.after) continue;
    if (change.before === undefined) delete next[change.id];
    else next[change.id] = change.before;
    restored++;
  }
  return { next, restored };
}
