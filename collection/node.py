#!/usr/bin/env python3
"""collection-node — runs on every recording device (e.g. a robot's onboard computer) and reports to collection-hub.

Single file, Python standard library only, so installing it is "copy one file".
It never writes into the recording tree except when the hub sends a cleanup
command *and* the node was started with ``--allow-cleanup``.

What it reports (see ``Node.heartbeat_payload``):

* recording state, derived from disk first — the PICO server's ``/status`` is
  only a hint. Lifecycle tests showed ``recorder.active`` stays true after
  a recorder crash and reads false while an orphaned recorder keeps writing
  after a server restart. The disk signal is ``mcap/__RECORDING__`` (created at
  start, removed at a clean stop) plus MCAP growth.
* free space of the recordings disk and the measured write rate;
* an incremental manifest of episodes (files, sizes, mtimes, sidecar summary),
  which the hub diffs against its mirror to decide what to pull.

Data layout it understands (same as RoboCurate's catalog)::

    <root>/[<profile>/]<session>/data/<user>/<task>/episode_NNNNNN/{mcap/, *.json}
    <root>/[<profile>/]<session>/logs/<user>/<task>/episode_NNNNNN/
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PROTOCOL = 1
VERSION = "0.1.0"

MARKER = "__RECORDING__"
SUMMARY = "pico_episode_summary.json"
META = "episode_meta.json"
MCAP_META = "metadata.yaml"
CONVERSION = "holobrain_conversion.json"
EPISODE_RE = re.compile(r"^episode_\d+$")
MAX_DEPTH = 7
# A marker whose MCAP has not grown for this long is a crashed recording, not a live one.
STALE_MARKER_SEC = 30.0
# Right after start the recorder may wait for topics before the MCAP grows.
MARKER_GRACE_SEC = 15.0
HOT_KEEP_SEC = 600.0


def now() -> float:
    return time.time()


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, "rb") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _stat(path) -> os.stat_result | None:
    try:
        return os.stat(path)
    except OSError:
        return None


def machine_id() -> str:
    for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            v = Path(p).read_text().strip()
            if v:
                return v
        except OSError:
            pass
    return "host-" + socket.gethostname()


def local_ips() -> list[str]:
    ips = []
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3).stdout
        ips = [x for x in out.split() if ":" not in x]
    except Exception:
        pass
    return ips


# --------------------------------------------------------------------- episodes

def episode_paths(root: Path, ep_dir: Path) -> dict:
    """Split an episode directory into session/user/task/episode + its log dir."""
    rel = ep_dir.relative_to(root)
    parts = rel.parts
    # [..., session, 'data', user, task, episode]
    i = len(parts) - 4
    session_rel = Path(*parts[:i]) if i > 0 else Path(".")
    user, task, ep = parts[i + 1], parts[i + 2], parts[i + 3]
    log_dir = root / session_rel / "logs" / user / task / ep
    return {
        "rel": rel.as_posix(),
        "session_rel": session_rel.as_posix(),
        "user": user,
        "task": task,
        "episode": ep,
        "log_dir": log_dir,
    }


def is_episode_dir(path: Path) -> bool:
    return EPISODE_RE.match(path.name) is not None and path.parent.parent.parent.name == "data"


def list_files(root: Path, base: Path) -> list[list]:
    """[[relpath, size, mtime_ns], ...] for every regular file under ``base`` (recursive)."""
    out = []
    if not base.is_dir():
        return out
    stack = [base]
    while stack:
        d = stack.pop()
        try:
            it = list(os.scandir(d))
        except OSError:
            continue
        for e in it:
            if e.name.startswith(".rsync-") or e.name.startswith(".~tmp~"):
                continue
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=False):
                    st = e.stat(follow_symlinks=False)
                    out.append([Path(e.path).relative_to(root).as_posix(), st.st_size, st.st_mtime_ns])
            except OSError:
                continue
    out.sort()
    return out


def summarize_sidecar(ep_dir: Path) -> dict:
    s = _read_json(ep_dir / SUMMARY)
    if s is None:
        s = _read_json(ep_dir / META) or {}
    rec = s.get("recorder") if isinstance(s.get("recorder"), dict) else {}
    outcome = s.get("task_outcome")
    if not isinstance(outcome, dict):
        md = s.get("metadata") if isinstance(s.get("metadata"), dict) else {}
        metas = md.get("metas") if isinstance(md.get("metas"), dict) else {}
        outcome = metas.get("task_outcome") if isinstance(metas.get("task_outcome"), dict) else None
    conv = _read_json(ep_dir / CONVERSION)
    return {
        "state": str(s.get("state") or ""),
        "created_at": str(s.get("created_at") or ""),
        "stopped_at": str(s.get("stopped_at") or ""),
        "deleted": bool(s.get("deleted")),
        "error": str(s.get("error_message") or "")[:300],
        "recorder_ok": rec.get("ok") if isinstance(rec.get("ok"), bool) else None,
        "outcome": outcome.get("success") if isinstance(outcome, dict) and isinstance(outcome.get("success"), bool) else None,
        "failure_reason": str((outcome or {}).get("failure_reason") or "")[:200] if isinstance(outcome, dict) else "",
        "conversion": str(conv.get("status") or "") if conv else "",
    }


def describe_episode(root: Path, ep_dir: Path) -> dict:
    info = episode_paths(root, ep_dir)
    files = list_files(root, ep_dir) + list_files(root, info.pop("log_dir"))
    mcap_dir = ep_dir / "mcap"
    marker = (mcap_dir / MARKER).exists()
    mcap_bytes = sum(f[1] for f in files if f[0].endswith(".mcap"))
    info.update(
        files=files,
        bytes=sum(f[1] for f in files),
        mcap_bytes=mcap_bytes,
        mtime_max=max((f[2] for f in files), default=0) / 1e9,
        marker=marker,
        finalized=(mcap_dir / MCAP_META).exists() and not marker,
        has_mcap=any(f[0].endswith(".mcap") for f in files),
        sidecar=summarize_sidecar(ep_dir),
    )
    return info


def scan_episodes(root: Path) -> dict[str, dict]:
    """Walk ``root`` and describe every episode directory. Cost: a few stats per episode."""
    found: dict[str, dict] = {}
    if not root.is_dir():
        return found
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        here, depth = stack.pop()
        try:
            entries = list(os.scandir(here))
        except OSError:
            continue
        for e in entries:
            if not e.is_dir(follow_symlinks=False) or e.name.startswith("."):
                continue
            p = Path(e.path)
            if e.name == "logs" and depth >= 0 and (p.parent / "data").is_dir():
                continue
            if is_episode_dir(p):
                try:
                    d = describe_episode(root, p)
                    found[d["rel"]] = d
                except (OSError, ValueError):
                    pass
            elif depth < MAX_DEPTH:
                stack.append((p, depth + 1))
    return found


def fingerprint(ep: dict) -> str:
    import hashlib
    h = hashlib.sha1()
    for f in ep["files"]:
        h.update(f"{f[0]}\0{f[1]}\0{f[2]}\n".encode())
    h.update(b"M" if ep.get("marker") else b"-")
    return h.hexdigest()


# --------------------------------------------------------------------- live watch

class RecordingWatch:
    """1 Hz watch on recently created episodes: marker + MCAP size.

    Only "hot" episodes are stat'ed every second: those created recently or with a
    marker. New episode directories are found by listing task directories of the
    two newest sessions whenever their mtime changes.
    """

    def __init__(self, root: Path):
        self.root = root
        self.hot: dict[str, dict] = {}  # rel -> {dir, seen, size, grew_at, marker_since}
        self.task_mtimes: dict[str, int] = {}
        self.write_rate = 0.0  # EWMA bytes/s while recording
        self._last_total = None
        self._last_t = None

    def _session_dirs(self) -> list[Path]:
        sessions = []
        stack = [(self.root, 0)]
        while stack:
            here, depth = stack.pop()
            try:
                entries = list(os.scandir(here))
            except OSError:
                continue
            for e in entries:
                if not e.is_dir(follow_symlinks=False) or e.name.startswith("."):
                    continue
                p = Path(e.path)
                if e.name == "data":
                    sessions.append(here)
                elif depth < 2:
                    stack.append((p, depth + 1))
        sessions = sorted(set(sessions), key=lambda p: (_stat(p / "data").st_mtime if _stat(p / "data") else 0))
        return sessions[-2:]

    def seed(self, episodes: dict[str, dict]) -> None:
        t = now()
        for rel, ep in episodes.items():
            if ep["marker"] or t - ep["mtime_max"] < HOT_KEEP_SEC:
                self._add(rel, t)

    def _add(self, rel: str, t: float) -> None:
        if rel not in self.hot:
            self.hot[rel] = {"dir": self.root / rel, "seen": t, "size": None, "grew_at": None, "marker_since": None}

    def poll(self) -> dict:
        t = now()
        for sess in self._session_dirs():
            data = sess / "data"
            try:
                users = [Path(e.path) for e in os.scandir(data) if e.is_dir()]
            except OSError:
                continue
            for u in users:
                try:
                    tasks = [Path(e.path) for e in os.scandir(u) if e.is_dir()]
                except OSError:
                    continue
                for task in tasks:
                    st = _stat(task)
                    if st is None:
                        continue
                    key = str(task)
                    if self.task_mtimes.get(key) == st.st_mtime_ns:
                        continue
                    self.task_mtimes[key] = st.st_mtime_ns
                    try:
                        for e in os.scandir(task):
                            if e.is_dir() and EPISODE_RE.match(e.name):
                                p = Path(e.path)
                                est = _stat(p)
                                if est and t - est.st_mtime < HOT_KEEP_SEC:
                                    self._add(p.relative_to(self.root).as_posix(), t)
                    except OSError:
                        pass
        recording, stale, total = [], [], 0
        for rel, h in list(self.hot.items()):
            mdir = h["dir"] / "mcap"
            if not h["dir"].exists():
                del self.hot[rel]
                continue
            marker = (mdir / MARKER).exists()
            size = 0
            try:
                for e in os.scandir(mdir):
                    if e.name.endswith(".mcap"):
                        size += e.stat().st_size
            except OSError:
                pass
            if h["size"] is not None and size > h["size"]:
                h["grew_at"] = t
            h["size"] = size
            if marker:
                if h["marker_since"] is None:
                    h["marker_since"] = t
                live = (h["grew_at"] and t - h["grew_at"] < 10) or (t - h["marker_since"] < MARKER_GRACE_SEC)
                if live:
                    recording.append(rel)
                    total += size
                elif h["grew_at"] is None or t - h["grew_at"] > STALE_MARKER_SEC:
                    stale.append(rel)
                else:
                    recording.append(rel)
                    total += size
            else:
                h["marker_since"] = None
                if (h["grew_at"] and t - h["grew_at"] < 5):
                    # growing without a marker: an orphaned writer (e.g. after a server restart)
                    recording.append(rel)
                    total += size
                elif t - h["seen"] > HOT_KEEP_SEC and (h["grew_at"] is None or t - h["grew_at"] > HOT_KEEP_SEC):
                    del self.hot[rel]
        # write-rate EWMA over recording periods only
        if recording and self._last_total is not None and self._last_t is not None and total >= self._last_total:
            dt = t - self._last_t
            if dt > 0:
                inst = (total - self._last_total) / dt
                self.write_rate = inst if self.write_rate == 0 else 0.9 * self.write_rate + 0.1 * inst
        self._last_total = total if recording else None
        self._last_t = t
        return {"active": bool(recording), "episodes": sorted(recording), "stale_markers": sorted(stale), "bytes": total}


# --------------------------------------------------------------------- node

class Node:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root = Path(os.path.expanduser(cfg["data_root"])).resolve()
        self.hub = cfg["hub"].rstrip("/")
        self.token = cfg["token"]
        self.status_url = cfg.get("status_url", "http://127.0.0.1:18080/status")
        self.container_root = cfg.get("container_root", "/recordings/raw").rstrip("/")
        self.allow_shutdown = bool(cfg.get("allow_shutdown"))
        self.allow_cleanup = bool(cfg.get("allow_cleanup"))
        self.shutdown_cmd = cfg.get("shutdown_cmd") or ["sudo", "-n", "/sbin/shutdown", "-h", "+1"]
        self.device_id = cfg.get("device_id") or machine_id()
        self.watch = RecordingWatch(self.root)
        self.episodes: dict[str, dict] = {}
        self.acked: dict[str, str] = {}  # rel -> fingerprint the hub has acknowledged
        self.acked_seq = 0
        self.need_full = True
        self.last_full_scan = 0.0
        self.recording = {"active": False, "episodes": [], "stale_markers": [], "bytes": 0}
        self.recording_since = None
        self.status_api = None
        self.last_status_poll = 0.0
        self.last_heartbeat = 0.0
        self.command_results: list[dict] = []
        self.done_commands: set[str] = set()
        self.stop = threading.Event()
        self.hub_ok = False
        self.started = now()

    # -- status api
    def poll_status_api(self) -> dict | None:
        try:
            with urllib.request.urlopen(self.status_url, timeout=2) as r:
                d = json.loads(r.read().decode()).get("data") or {}
        except Exception as e:  # service not running is normal between shifts
            return {"reachable": False, "error": type(e).__name__}
        rec = d.get("recorder") or {}
        eps = d.get("episodes") or {}
        cur = eps.get("current_episode") or {}
        return {
            "reachable": True,
            "recorder_active": rec.get("active"),
            "recorder_available": rec.get("available"),
            "last_error": str(rec.get("last_error") or "")[:300],
            "awaiting_outcome": bool(eps.get("awaiting_outcome")),
            "current": {
                "rel": self.map_container_path(cur.get("data_dir") or ""),
                "episode_id": cur.get("episode_id"),
                "state": cur.get("state"),
                "task": (cur.get("metadata") or {}).get("task_name"),
                "created_at": cur.get("created_at"),
            } if cur else None,
        }

    def map_container_path(self, p: str) -> str | None:
        if not p:
            return None
        for prefix in (self.container_root, str(self.root)):
            if p.startswith(prefix + "/"):
                return p[len(prefix) + 1:]
        return None

    # -- manifest
    def full_scan(self) -> None:
        self.episodes = scan_episodes(self.root)
        self.last_full_scan = now()
        if not self.watch.hot:
            self.watch.seed(self.episodes)

    def manifest_delta(self) -> dict | None:
        cur = {rel: fingerprint(ep) for rel, ep in self.episodes.items()}
        if self.need_full:
            return {"full": True, "seq": self.acked_seq + 1, "base": None,
                    "upserts": [self._wire(self.episodes[r]) for r in sorted(cur)], "removed": [], "_fp": cur}
        ups = [self._wire(self.episodes[r]) for r, fp in sorted(cur.items()) if self.acked.get(r) != fp]
        rem = sorted(r for r in self.acked if r not in cur)
        if not ups and not rem:
            return None
        return {"full": False, "seq": self.acked_seq + 1, "base": self.acked_seq, "upserts": ups, "removed": rem, "_fp": cur}

    @staticmethod
    def _wire(ep: dict) -> dict:
        d = dict(ep)
        d["fp"] = fingerprint(ep)
        return d

    def disk(self) -> dict:
        try:
            st = os.statvfs(self.root if self.root.exists() else self.root.parent)
            return {"path": str(self.root), "total": st.f_blocks * st.f_frsize, "free": st.f_bavail * st.f_frsize}
        except OSError:
            return {"path": str(self.root), "total": None, "free": None}

    def heartbeat_payload(self, manifest: dict | None) -> dict:
        conv_running = any(ep["sidecar"].get("conversion") == "converting" for ep in self.episodes.values())
        payload = {
            "protocol": PROTOCOL,
            "node_version": VERSION,
            "device_id": self.device_id,
            "hostname": socket.gethostname(),
            "ips": local_ips(),
            "time": now(),
            "uptime": self._uptime(),
            "node_started": self.started,
            "recording": dict(self.recording, since=self.recording_since),
            "write_rate_bps": self.watch.write_rate,
            "status_api": self.status_api,
            "disk": self.disk(),
            "conversion_running": conv_running,
            "capabilities": {"shutdown": self.allow_shutdown, "cleanup": self.allow_cleanup},
            "command_results": self.command_results,
            "episode_count": len(self.episodes),
        }
        if manifest is not None:
            payload["manifest"] = {k: v for k, v in manifest.items() if not k.startswith("_")}
        return payload

    @staticmethod
    def _uptime() -> float | None:
        try:
            return float(Path("/proc/uptime").read_text().split()[0])
        except (OSError, ValueError):
            return None

    def send(self) -> dict | None:
        manifest = self.manifest_delta()
        payload = self.heartbeat_payload(manifest)
        body = gzip.compress(json.dumps(payload, default=str).encode())
        req = urllib.request.Request(
            self.hub + "/api/node/heartbeat", data=body, method="POST",
            headers={"Content-Type": "application/json", "Content-Encoding": "gzip",
                     "Authorization": "Bearer " + self.token})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                resp = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            self.hub_ok = False
            try:
                detail = e.read().decode()[:300]
            except Exception:
                detail = ""
            log(f"hub rejected heartbeat: HTTP {e.code} {detail}")
            return None
        except Exception as e:
            self.hub_ok = False
            log(f"hub unreachable: {e}")
            return None
        self.hub_ok = True
        self.last_heartbeat = now()
        self.command_results = []
        if manifest is not None and resp.get("manifest_ack") == manifest["seq"]:
            self.acked = manifest["_fp"]
            self.acked_seq = manifest["seq"]
            self.need_full = False
        if resp.get("need_full"):
            self.need_full = True
        for cmd in resp.get("commands") or []:
            self.run_command(cmd)
        return resp

    # -- commands (P3)
    def run_command(self, cmd: dict) -> None:
        cid = str(cmd.get("id"))
        if cid in self.done_commands:
            return
        self.done_commands.add(cid)
        kind = cmd.get("type")
        try:
            if kind == "shutdown":
                result = self.cmd_shutdown(cmd)
            elif kind == "cleanup":
                result = self.cmd_cleanup(cmd)
            elif kind == "rescan":
                self.full_scan()
                result = {"ok": True}
            else:
                result = {"ok": False, "detail": f"unknown command {kind}"}
        except Exception as e:
            result = {"ok": False, "detail": repr(e)}
        self.command_results.append({"id": cid, "type": kind, **result})
        log(f"command {kind} {cid}: {result}")

    def cmd_shutdown(self, cmd: dict) -> dict:
        if not self.allow_shutdown:
            return {"ok": False, "detail": "node started without --allow-shutdown"}
        if self.recording["active"]:
            return {"ok": False, "detail": "recording in progress"}
        r = subprocess.run(self.shutdown_cmd, capture_output=True, text=True, timeout=20)
        return {"ok": r.returncode == 0, "detail": (r.stdout + r.stderr).strip()[:300]}

    def cmd_cleanup(self, cmd: dict) -> dict:
        if not self.allow_cleanup:
            return {"ok": False, "detail": "node started without --allow-cleanup"}
        current = ((self.status_api or {}).get("current") or {}).get("rel")
        results = []
        for item in cmd.get("episodes") or []:
            rel = item["rel"]
            ep_dir = (self.root / rel).resolve()
            if self.root not in ep_dir.parents or not is_episode_dir(ep_dir):
                results.append({"rel": rel, "ok": False, "detail": "path outside data root"})
                continue
            if rel == current or rel in self.recording["episodes"] or rel in self.watch.hot and (ep_dir / "mcap" / MARKER).exists():
                results.append({"rel": rel, "ok": False, "detail": "episode is current or recording"})
                continue
            now_desc = describe_episode(self.root, ep_dir) if ep_dir.exists() else None
            if now_desc is None:
                results.append({"rel": rel, "ok": True, "detail": "already absent"})
                continue
            if [list(f) for f in now_desc["files"]] != [list(f) for f in item["files"]]:
                results.append({"rel": rel, "ok": False, "detail": "files changed since hub verification"})
                continue
            log_dir = self.root / now_desc["session_rel"] / "logs" / now_desc["user"] / now_desc["task"] / now_desc["episode"]
            shutil.rmtree(ep_dir)
            if log_dir.is_dir():
                shutil.rmtree(log_dir)
            results.append({"rel": rel, "ok": True, "detail": "deleted"})
        self.full_scan()
        return {"ok": all(r["ok"] for r in results), "episodes": results}

    # -- main loop
    def tick(self) -> None:
        t = now()
        prev_active = self.recording["active"]
        self.recording = self.watch.poll()
        changed = self.recording["active"] != prev_active
        if self.recording["active"] and not prev_active:
            self.recording_since = t
        if not self.recording["active"]:
            self.recording_since = None
        api_every = 5 if (self.recording["active"] or (self.status_api or {}).get("recorder_active")) else 15
        if t - self.last_status_poll >= api_every:
            self.status_api = self.poll_status_api()
            self.last_status_poll = t
        scan_every = 60 if self.recording["active"] else 30
        if changed and not self.recording["active"]:
            scan_every = 0  # report finalized episode quickly
        if t - self.last_full_scan >= scan_every or self.need_full and self.last_full_scan == 0:
            self.full_scan()
        hb_every = 5 if self.recording["active"] else 15
        if changed or t - self.last_heartbeat >= hb_every or self.command_results:
            self.send()

    def run(self) -> None:
        log(f"collection-node {VERSION} device={self.device_id} root={self.root} hub={self.hub}")
        self.full_scan()
        while not self.stop.is_set():
            t0 = now()
            try:
                self.tick()
            except Exception as e:  # never die on a transient error
                log(f"tick error: {e!r}")
            self.stop.wait(max(0.05, 1.0 - (now() - t0)))


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def load_config(args) -> dict:
    cfg = {}
    if args.config:
        cfg.update(json.loads(Path(args.config).expanduser().read_text()))
    for k in ("hub", "token", "data_root", "status_url", "container_root", "device_id"):
        v = getattr(args, k)
        if v:
            cfg[k] = v
    if args.allow_shutdown:
        cfg["allow_shutdown"] = True
    if args.allow_cleanup:
        cfg["allow_cleanup"] = True
    missing = [k for k in ("hub", "token", "data_root") if not cfg.get(k)]
    if missing:
        raise SystemExit("missing config: " + ", ".join(missing))
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="JSON config file")
    ap.add_argument("--hub", help="collection-hub node endpoint, e.g. http://<hub-host>:8422")
    ap.add_argument("--token")
    ap.add_argument("--data-root", dest="data_root", help="host path of the recordings raw/ directory")
    ap.add_argument("--status-url", dest="status_url")
    ap.add_argument("--container-root", dest="container_root", help="path of data_root inside the recording container")
    ap.add_argument("--device-id", dest="device_id")
    ap.add_argument("--allow-shutdown", action="store_true")
    ap.add_argument("--allow-cleanup", action="store_true")
    ap.add_argument("--once", action="store_true", help="scan, send one heartbeat, print it and exit")
    args = ap.parse_args(argv)
    node = Node(load_config(args))
    if args.once:
        node.full_scan()
        node.recording = node.watch.poll()
        node.status_api = node.poll_status_api()
        resp = node.send()
        print(json.dumps({"hub_response": resp, "episodes": len(node.episodes), "recording": node.recording,
                          "status_api": node.status_api, "disk": node.disk()}, indent=2, default=str))
        return 0 if resp else 1
    import signal
    signal.signal(signal.SIGTERM, lambda *_: node.stop.set())
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
