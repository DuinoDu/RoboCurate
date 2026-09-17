// Font Awesome Free 6.7.2. See static/licenses/fontawesome.txt.
const boxes = {
  play: [384, 512],
  pause: [320, 512],
  previous: [320, 512],
  next: [320, 512],
  reset: [512, 512],
  expand: [448, 512],
  compress: [448, 512],
  camera: [512, 512],
  table: [512, 512],
  export: [576, 512],
  import: [576, 512],
  sliders: [512, 512],
  help: [512, 512],
  check: [448, 512],
  close: [384, 512],
  info: [512, 512],
  warning: [512, 512],
  left: [320, 512],
  right: [320, 512],
  "arrow-left": [448, 512],
  "arrow-right": [448, 512],
  history: [512, 512],
  save: [448, 512],
  sun: [512, 512],
  moon: [384, 512],
  robot: [640, 512],
  chart: [512, 512],
  list: [512, 512],
  plus: [448, 512],
  minus: [448, 512],
  search: [512, 512],
  zoomIn: [512, 512],
  zoomOut: [512, 512],
  crop: [512, 512],
  download: [512, 512],
  link: [512, 512],
  clock: [512, 512],
  keyboard: [576, 512],
  stop: [384, 512],
};
export function icon(name) {
  const [w, h] = boxes[name] || boxes.info;
  return `<svg class="ui-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><use href="/workspace-icons.svg#${boxes[name] ? name : "info"}"/></svg>`;
}

const actionIcons = {
  import: "import",
  analyze: "sliders",
  "analyze-selected": "sliders",
  "retry-analysis": "reset",
  "retry-signals": "reset",
  "retry-frames": "reset",
  "batch-grade": "list",
  "export-selected": "export",
  "new-export": "plus",
  "confirm-export": "export",
  "confirm-import": "import",
  "confirm-batch": "check",
  "clear-selection": "close",
  previous: "left",
  next: "right",
  "mark-in": "crop",
  "mark-out": "crop",
  "add-clip": "plus",
  "cancel-clip-edit": "close",
  save: "save",
  "save-next": "save",
  "show-quality": "info",
  "discard-draft": "reset",
  history: "history",
  "save-rules": "save",
  "reset-rules": "reset",
  "robot-reset": "reset",
  "close-dialog": "close",
};
const toolIcons = {
  open: "list",
  retry: "reset",
  next: "right",
  previous: "left",
  "page-prev": "left",
  "page-next": "right",
  play: "play",
  signal: "chart",
  coverage: "crop",
  "batch-confirmed": "warning",
  "batch-accepted": "check",
  "undo-batch": "reset",
};
function decorate(root) {
  if (!(root instanceof Element)) return;
  const targets = [
    ...(root.matches("button[data-action],button[data-tool]") ? [root] : []),
    ...root.querySelectorAll("button[data-action],button[data-tool]"),
  ];
  for (const b of targets) {
    const name = actionIcons[b.dataset.action] || toolIcons[b.dataset.tool];
    if (!name || b.querySelector("svg")) continue;
    const text = b.textContent.replace(/[＋✧↗←→⤡⤢▷▶×]/g, "").trim();
    if (text === "取消") continue;
    b.innerHTML = icon(name) + (text ? "<span></span>" : "");
    if (text) b.lastElementChild.textContent = text;
    b.classList.add("with-icon");
  }
  for (const e of root.querySelectorAll(
    ".stat-icon,.banner-icon,.issue-icon",
  )) {
    if (e.querySelector("svg")) continue;
    const name = e.classList.contains("issue-icon")
      ? "warning"
      : e.classList.contains("banner-icon")
        ? "info"
        : { "▦": "table", "◷": "clock", "☑": "list", "↗": "export" }[
            e.textContent.trim()
          ] || "info";
    e.innerHTML = icon(name);
  }
}
export function installIcons() {
  decorate(document.body);
  const observer = new MutationObserver((records) => {
    for (const r of records)
      for (const node of r.addedNodes) if (node.nodeType === 1) decorate(node);
  });
  observer.observe(document.body, { childList: true, subtree: true });
}
