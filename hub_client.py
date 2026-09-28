"""RoboCurate side of collection-hub: admin-API proxy and the synced-session follower.

RoboCurate holds no device credentials. It only reads the hub's admin API
(loopback) and, when the hub reports that a session finished syncing, re-scans
that one session directory so new episodes appear in the workbench within
seconds without walking the whole mirror.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HUB_URL = ""
POLL_SEC = 10.0
_POST_ALLOWED = re.compile(
    r"^devices(/[0-9a-f]+/(settings|sync-now|rescan|token|cleanup/preview|cleanup/confirm))?$")

state = {"reachable": None, "cursor": None, "mirror_root": None, "last_error": "", "last_merge": None}


def configure(url: str) -> None:
    global HUB_URL
    HUB_URL = (url or "").rstrip("/")


def post_allowed(rest: str) -> bool:
    return bool(_POST_ALLOWED.match(rest.strip("/")))


def forward(method: str, rest: str, query: str = "", body: dict | None = None, timeout: float = 10.0):
    if not HUB_URL:
        return 503, {"error": "未配置设备服务（collection-hub）", "hub": "disabled"}
    url = f"{HUB_URL}/api/{rest}" + (f"?{query}" if query else "")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode() or "{}")
        except ValueError:
            payload = {}
        return e.code, {"error": payload.get("error") or f"设备服务返回 {e.code}"}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return 503, {"error": "设备服务不可用，审核功能不受影响", "detail": str(e), "hub": "unreachable"}


def follow_once(lib) -> dict | None:
    """Apply new hub changes to the catalog. Returns what was merged, or None if nothing to do."""
    if state["cursor"] is None:
        saved = lib.store.setting("hub_cursor", {}) or {}
        state["cursor"] = int(saved.get("cursor", 0)) if isinstance(saved, dict) else 0
        state["mirror_root"] = saved.get("mirror_root") if isinstance(saved, dict) else None
    status, data = forward("GET", "changes", f"since={state['cursor']}&limit=2000", timeout=5)
    if status != 200:
        state["reachable"] = False
        state["last_error"] = data.get("error", "")
        return None
    state["reachable"] = True
    mirror = data.get("mirror_root")
    first = False
    if mirror and state["mirror_root"] != mirror:
        state["mirror_root"] = mirror
        state["cursor"] = 0
        first = True
    if data.get("latest", 0) < state["cursor"]:
        state["cursor"] = 0  # hub database was reset
        first = True
    root = Path(mirror) if mirror else None
    result = None
    if root is not None and root.is_dir():
        if root.resolve() not in lib.roots:
            lib.add_root(root)  # first contact: one full scan of the mirror
            lib.store.set_setting("roots", [str(p) for p in lib.roots])
            first = True
        sessions = sorted({c["session_path"] for c in data.get("changes") or []})
        if sessions and not first:
            result = lib.merge_paths(root, sessions)
        elif first and not sessions:
            result = {"full": True}
        if sessions:
            state["last_merge"] = {"at": time.time(), "sessions": len(sessions), **(result or {})}
    state["cursor"] = max(int(data.get("cursor", state["cursor"])), state["cursor"])
    lib.store.set_setting("hub_cursor", {"cursor": state["cursor"], "mirror_root": state["mirror_root"]})
    return result


def start_follower(lib) -> threading.Thread:
    def loop():
        while True:
            try:
                follow_once(lib)
            except Exception as e:  # keep following; the review app must never die for this
                state["last_error"] = repr(e)
            time.sleep(POLL_SEC)

    t = threading.Thread(target=loop, name="hub-follower", daemon=True)
    t.start()
    return t
