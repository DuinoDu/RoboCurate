#!/usr/bin/env python3
"""collection-hub — device registry, sync scheduler and disk monitor for recording devices.

Runs on the storage server as its own process, separate from RoboCurate:
restarting the review app never interrupts monitoring or a transfer, and only this
process holds the SSH key that can read the robots.

Two listeners:

* node endpoint (LAN, token auth): ``POST /api/node/heartbeat``
* admin endpoint (127.0.0.1 only): overview, device settings, change cursor.
  RoboCurate proxies it under ``/api/hub/*``.

Sync pulls finished episodes with rsync over SSH (read-only ``rrsync`` on the
device) into ``<mirror_root>/<device>/<same tree as the device>``. The rules follow
lifecycle tests against the PICO episode server and recorder:

* a device that is recording is never synced; sync starts ``idle_minutes`` after
  its last recording ended (or immediately once the device is marked off-shift);
* while any device is recording, the total transfer rate is capped
  (``busy_bwlimit_mb``) to protect teleoperation Wi-Fi; otherwise full speed;
* an episode is eligible once ``mcap/metadata.yaml`` exists and the recording
  marker is gone, and it is no longer the current episode or has been quiet for
  ``settle_seconds`` (the operator can still delete the current one);
* small sidecars keep changing after stop (label, next metadata update), so each
  episode is re-synced whenever its file fingerprint changes;
* an episode deleted on the device after it was synced is moved to ``.trash``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

VERSION = "0.1.0"
GB = 1024 ** 3
MB = 1024 ** 2
ONLINE_SEC = 15 * 2 + 5     # idle heartbeats are every 15 s
UNSTABLE_SEC = 60 * 2
DEFAULTS = {
    "node_listen": "0.0.0.0:8422",
    "admin_listen": "127.0.0.1:8423",
    "db": "~/.local/share/collection-hub/hub.sqlite3",
    "mirror_root": "~/collection-hub/mirror",
    "ssh_key": "~/.config/collection-hub/id_ed25519",
    "ssh_options": ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
                    "-o", "ServerAliveCountMax=4", "-c", "aes128-gcm@openssh.com"],
    "rsync": "rsync",
    "reserve_gb": 100,
    "max_concurrent": 2,
    "busy_bwlimit_mb": 15,
    "idle_minutes": 10,
    "settle_seconds": 120,
    "batch_max_gb": 20,
    "batch_max_episodes": 40,
    "default_write_rate_gb_h": 30,
    "default_daily_intake_gb": 500,
    "device_warn_hours": 8,
    "device_critical_hours": 2,
    "server_warn_days": 3,
    "server_critical_days": 1,
    "webhook": "",
    "webhook_format": "feishu",
    "alert_cooldown_sec": 1800,
    "scheduler_interval": 2.0,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices(
  id TEXT PRIMARY KEY, name TEXT NOT NULL, token_hash TEXT NOT NULL, machine_id TEXT,
  ssh_target TEXT DEFAULT '', remote_root TEXT DEFAULT '', mirror_dir TEXT NOT NULL,
  created_at REAL, last_seen REAL, last_ip TEXT, heartbeat TEXT, manifest_seq INTEGER DEFAULT 0,
  paused INTEGER DEFAULT 0, off_shift INTEGER DEFAULT 0, shutdown_after_sync INTEGER DEFAULT 0,
  force_sync INTEGER DEFAULT 0, sync_floor REAL DEFAULT 0, include_history INTEGER DEFAULT 0,
  last_recording_end REAL, recording_since REAL, retired INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS episodes(
  device_id TEXT, rel TEXT, session_rel TEXT, user TEXT, task TEXT, episode TEXT,
  bytes INTEGER, mcap_bytes INTEGER, mtime_max REAL, files TEXT, fp TEXT,
  marker INTEGER, finalized INTEGER, has_mcap INTEGER, sidecar TEXT,
  on_device INTEGER DEFAULT 1, first_seen REAL, updated_at REAL, finalized_at REAL,
  synced_fp TEXT, synced_bytes INTEGER DEFAULT 0, synced_at REAL, sync_error TEXT, attempts INTEGER DEFAULT 0,
  deleted_on_device_at REAL, trashed INTEGER DEFAULT 0, cleaned_at REAL,
  PRIMARY KEY(device_id, rel)
);
CREATE INDEX IF NOT EXISTS episodes_dev ON episodes(device_id, on_device);
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT, started_at REAL, ended_at REAL,
  episodes TEXT, bytes_planned INTEGER, bytes_done INTEGER DEFAULT 0, bwlimit_kb INTEGER,
  result TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS changes(
  seq INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT, session_path TEXT, episode_path TEXT,
  kind TEXT, at REAL
);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, device_id TEXT, level TEXT, kind TEXT,
  message TEXT, data TEXT
);
CREATE TABLE IF NOT EXISTS commands(
  id TEXT PRIMARY KEY, device_id TEXT, type TEXT, payload TEXT, created_at REAL, sent_at REAL,
  done_at REAL, status TEXT, result TEXT
);
CREATE TABLE IF NOT EXISTS cleanup_requests(
  id TEXT PRIMARY KEY, device_id TEXT, created_at REAL, episodes TEXT, bytes INTEGER,
  confirm_token TEXT, status TEXT, command_id TEXT
);
CREATE TABLE IF NOT EXISTS intake(day TEXT PRIMARY KEY, bytes INTEGER);
"""


def now() -> float:
    return time.time()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    return s or "device"


def fmt_bytes(n: float | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


# ============================================================================ store

class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()

    def q(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def one(self, sql: str, args=()) -> sqlite3.Row | None:
        rows = self.q(sql, args)
        return rows[0] if rows else None

    def x(self, sql: str, args=()) -> sqlite3.Cursor:
        with self.lock:
            return self.db.execute(sql, args)

    def tx(self):
        return _Tx(self)


class _Tx:
    def __init__(self, store: Store):
        self.store = store

    def __enter__(self):
        self.store.lock.acquire()
        self.store.db.execute("BEGIN")
        return self.store.db

    def __exit__(self, et, ev, tb):
        try:
            self.store.db.execute("COMMIT" if et is None else "ROLLBACK")
        finally:
            self.store.lock.release()


# ============================================================================ hub

class Hub:
    def __init__(self, cfg: dict):
        self.cfg = {**DEFAULTS, **cfg}
        c = self.cfg
        self.mirror_root = Path(os.path.expanduser(c["mirror_root"])).resolve()
        self.mirror_root.mkdir(parents=True, exist_ok=True)
        marker = self.mirror_root / ".collection-hub.json"
        if not marker.exists():
            marker.write_text(json.dumps({"app": "collection-hub", "layout": "<device>/<device tree>",
                                          "created_at": now()}) + "\n")
        self.store = Store(Path(os.path.expanduser(c["db"])))
        self.jobs: dict[str, "SyncJob"] = {}   # device_id -> running job
        self.jobs_lock = threading.Lock()
        self.stop = threading.Event()
        self.alert_last: dict[tuple, float] = {}
        self.last_levels: dict[str, str] = {}
        self.last_online: dict[str, bool] = {}
        self.recording_anywhere = False
        self.reserve = float(c["reserve_gb"]) * GB

    # ------------------------------------------------------------ devices
    def add_device(self, name: str, ssh_target: str = "", remote_root: str = "", mirror_dir: str | None = None,
                   sync_floor: float | None = None) -> tuple[dict, str]:
        token = secrets.token_urlsafe(24)
        dev_id = secrets.token_hex(4)
        mdir = slug(mirror_dir or name)
        if self.store.one("SELECT 1 FROM devices WHERE mirror_dir=? AND retired=0", (mdir,)):
            raise ValueError(f"mirror directory already used by another device: {mdir}")
        self.store.x("INSERT INTO devices(id,name,token_hash,ssh_target,remote_root,mirror_dir,created_at,sync_floor)"
                     " VALUES(?,?,?,?,?,?,?,?)",
                     (dev_id, name, hash_token(token), ssh_target, remote_root, mdir, now(),
                      now() if sync_floor is None else sync_floor))
        self.event(dev_id, "info", "device_added", f"设备 {name} 已注册")
        return self.device_row(dev_id), token

    def install_info(self) -> dict:
        pub = Path(os.path.expanduser(self.cfg["ssh_key"]) + ".pub")
        try:
            key = pub.read_text().strip()
        except OSError:
            key = ""
        return {"ssh_pubkey": key, "node_port": _listen(self.cfg["node_listen"])[1]}

    def rotate_token(self, dev_id: str) -> str:
        token = secrets.token_urlsafe(24)
        self.store.x("UPDATE devices SET token_hash=?, machine_id=NULL WHERE id=?", (hash_token(token), dev_id))
        return token

    def device_row(self, dev_id: str) -> dict:
        r = self.store.one("SELECT * FROM devices WHERE id=?", (dev_id,))
        if r is None:
            raise KeyError(f"unknown device {dev_id}")
        return dict(r)

    def devices(self) -> list[dict]:
        return [dict(r) for r in self.store.q("SELECT * FROM devices WHERE retired=0 ORDER BY name")]

    def update_settings(self, dev_id: str, body: dict) -> dict:
        self.device_row(dev_id)
        allowed = {"paused": int, "off_shift": int, "shutdown_after_sync": int, "include_history": int,
                   "name": str, "ssh_target": str, "remote_root": str, "retired": int}
        sets, args = [], []
        for k, conv in allowed.items():
            if k in body:
                v = body[k]
                sets.append(f"{k}=?")
                args.append(int(bool(v)) if conv is int else str(v))
        if not sets:
            raise ValueError("no settings given")
        self.store.x(f"UPDATE devices SET {', '.join(sets)} WHERE id=?", (*args, dev_id))
        if body.get("off_shift"):
            self.event(dev_id, "info", "off_shift", "设备已标记收工，优先全速同步")
        if "paused" in body:
            self.event(dev_id, "info", "paused" if body["paused"] else "resumed",
                       "同步已暂停" if body["paused"] else "同步已恢复")
            if body["paused"]:
                self.cancel_job(dev_id, "paused")
        return self.device_row(dev_id)

    def sync_now(self, dev_id: str) -> None:
        self.device_row(dev_id)
        self.store.x("UPDATE devices SET force_sync=1, paused=0 WHERE id=?", (dev_id,))

    def authenticate(self, token: str) -> dict | None:
        if not token:
            return None
        r = self.store.one("SELECT * FROM devices WHERE token_hash=? AND retired=0", (hash_token(token),))
        return dict(r) if r else None

    # ------------------------------------------------------------ heartbeat
    def heartbeat(self, dev: dict, hb: dict, ip: str) -> dict:
        t = now()
        mid = hb.get("device_id")
        if dev["machine_id"] and mid and dev["machine_id"] != mid:
            raise PermissionError("token already bound to another machine; rotate the token to re-bind")
        rec = hb.get("recording") or {}
        api = hb.get("status_api") or {}
        prev = json.loads(dev["heartbeat"]) if dev["heartbeat"] else {}
        was_rec = bool((prev.get("recording") or {}).get("active"))
        is_rec = bool(rec.get("active"))
        sets = {"last_seen": t, "last_ip": ip, "machine_id": mid or dev["machine_id"]}
        # device clock minus hub clock; file mtimes in the manifest are in device time
        if isinstance(hb.get("time"), (int, float)):
            hb["_skew"] = float(hb["time"]) - t
        if is_rec and not was_rec:
            sets["recording_since"] = t
            self.event(dev["id"], "info", "recording_started", "开始录制：" + ", ".join(rec.get("episodes") or []))
            self.cancel_job(dev["id"], "paused_recording")
        if was_rec and not is_rec:
            sets["last_recording_end"] = t
            sets["recording_since"] = None
        if is_rec:
            sets["last_recording_end"] = t
        # disk-vs-api disagreement (P0: recorder.active is not reliable)
        api_active = api.get("recorder_active") if api.get("reachable") else None
        if api_active is not None and bool(api_active) != is_rec:
            since = prev.get("_mismatch_since") or t
            hb["_mismatch_since"] = since
            if t - since > 20:
                self.alert(dev["id"], "warn", "status_mismatch",
                           f"{dev['name']} 录制状态不一致：录制服务报告 {'录制中' if api_active else '未录制'}，磁盘{'正在' if is_rec else '没有'}写入")
        for rel in rec.get("stale_markers") or []:
            self.alert(dev["id"], "warn", "stale_marker:" + rel,
                       f"{dev['name']} 录制异常中断：{rel} 仍有录制标记但文件已停止增长")
        # command results first: a cleanup result and the manifest that no longer lists
        # those episodes arrive in the same heartbeat, and they must not go to .trash
        for res in hb.get("command_results") or []:
            self._on_command_result(dev, res)
        ack = None
        need_full = False
        man = hb.get("manifest")
        with self.store.tx() as db:
            if man is not None:
                if man.get("full"):
                    self._apply_manifest(db, dev, man, t, full=True)
                    ack = man.get("seq")
                elif man.get("base") == dev["manifest_seq"]:
                    self._apply_manifest(db, dev, man, t, full=False)
                    ack = man.get("seq")
                else:
                    need_full = True
                if ack is not None:
                    sets["manifest_seq"] = ack
            elif dev["manifest_seq"] == 0:
                need_full = True
            sets["heartbeat"] = json.dumps(hb, default=str)
            db.execute(f"UPDATE devices SET {', '.join(k + '=?' for k in sets)} WHERE id=?", (*sets.values(), dev["id"]))
            for res in hb.get("command_results") or []:
                db.execute("UPDATE commands SET status=?, done_at=?, result=? WHERE id=?",
                           ("done" if res.get("ok") else "failed", t, json.dumps(res), res.get("id")))
            cmds = db.execute("SELECT * FROM commands WHERE device_id=? AND status='pending'", (dev["id"],)).fetchall()
            out_cmds = []
            for c in cmds:
                out_cmds.append({"id": c["id"], "type": c["type"], **json.loads(c["payload"] or "{}")})
                db.execute("UPDATE commands SET status='sent', sent_at=? WHERE id=?", (t, c["id"]))
        return {"ok": True, "manifest_ack": ack, "need_full": need_full, "commands": out_cmds,
                "server_time": t, "hub_version": VERSION}

    def _apply_manifest(self, db, dev: dict, man: dict, t: float, full: bool) -> None:
        seen = set()
        for ep in man.get("upserts") or []:
            if not isinstance(ep, dict) or not valid_manifest_episode(ep):
                self.alert(dev["id"], "warn", "bad_manifest", f"{dev['name']} 上报了无效的 episode 路径，已忽略：{str(ep.get('rel') if isinstance(ep, dict) else ep)[:120]}")
                continue
            rel = ep["rel"]
            seen.add(rel)
            row = db.execute("SELECT finalized, finalized_at, on_device FROM episodes WHERE device_id=? AND rel=?",
                             (dev["id"], rel)).fetchone()
            fin_at = row["finalized_at"] if row and row["finalized_at"] else (t if ep.get("finalized") else None)
            if row and not row["finalized"] and ep.get("finalized"):
                fin_at = t
            db.execute(
                "INSERT INTO episodes(device_id,rel,session_rel,user,task,episode,bytes,mcap_bytes,mtime_max,files,fp,"
                "marker,finalized,has_mcap,sidecar,on_device,first_seen,updated_at,finalized_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?,?) "
                "ON CONFLICT(device_id,rel) DO UPDATE SET session_rel=excluded.session_rel, user=excluded.user,"
                " task=excluded.task, episode=excluded.episode, bytes=excluded.bytes, mcap_bytes=excluded.mcap_bytes,"
                " mtime_max=excluded.mtime_max, files=excluded.files, fp=excluded.fp, marker=excluded.marker,"
                " finalized=excluded.finalized, has_mcap=excluded.has_mcap, sidecar=excluded.sidecar, on_device=1,"
                " deleted_on_device_at=NULL, updated_at=excluded.updated_at, finalized_at=excluded.finalized_at",
                (dev["id"], rel, ep.get("session_rel"), ep.get("user"), ep.get("task"), ep.get("episode"),
                 ep.get("bytes", 0), ep.get("mcap_bytes", 0), ep.get("mtime_max", 0), json.dumps(ep.get("files") or []),
                 ep.get("fp"), int(bool(ep.get("marker"))), int(bool(ep.get("finalized"))), int(bool(ep.get("has_mcap"))),
                 json.dumps(ep.get("sidecar") or {}), t, t, fin_at))
        removed = list(man.get("removed") or [])
        if full:
            present = {r["rel"] for r in db.execute("SELECT rel FROM episodes WHERE device_id=? AND on_device=1", (dev["id"],))}
            removed += sorted(present - seen)
        for rel in removed:
            row = db.execute("SELECT * FROM episodes WHERE device_id=? AND rel=?", (dev["id"], rel)).fetchone()
            if row is None or not row["on_device"]:
                continue
            db.execute("UPDATE episodes SET on_device=0, deleted_on_device_at=? WHERE device_id=? AND rel=?",
                       (t, dev["id"], rel))
            if row["cleaned_at"]:
                continue  # we asked for it
            if row["synced_fp"]:
                self._trash(db, dev, row)

    def _trash(self, db, dev: dict, row) -> None:
        """Episode deleted on the device after we copied it: move our copy aside, keep it recoverable."""
        src = self.mirror_root / dev["mirror_dir"] / row["rel"]
        if not src.exists():
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dst = self.mirror_root / ".trash" / dev["mirror_dir"] / stamp / row["rel"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        session = Path(row["session_rel"] or ".")
        logs = self.mirror_root / dev["mirror_dir"] / session / "logs" / row["user"] / row["task"] / row["episode"]
        if logs.exists():
            ldst = self.mirror_root / ".trash" / dev["mirror_dir"] / stamp / session / "logs" / row["user"] / row["task"] / row["episode"]
            ldst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(logs), str(ldst))
        db.execute("UPDATE episodes SET trashed=1 WHERE device_id=? AND rel=?", (dev["id"], row["rel"]))
        db.execute("INSERT INTO changes(device_id,session_path,episode_path,kind,at) VALUES(?,?,?,?,?)",
                   (dev["id"], str(self.mirror_root / dev["mirror_dir"] / session), str(src), "trashed", now()))
        self.event(dev["id"], "warn", "trashed", f"{row['rel']} 已在设备上删除，本地副本移入回收区 {dst}")

    # ------------------------------------------------------------ derived state
    def hb(self, dev: dict) -> dict:
        return json.loads(dev["heartbeat"]) if dev["heartbeat"] else {}

    def device_now(self, dev: dict, hbd: dict | None = None) -> float:
        """Current time on the device's clock (manifest mtimes are device-clock values)."""
        hbd = hbd if hbd is not None else self.hb(dev)
        return now() + float(hbd.get("_skew") or 0.0)

    def online_state(self, dev: dict) -> str:
        if not dev["last_seen"]:
            return "never"
        age = now() - dev["last_seen"]
        return "online" if age <= ONLINE_SEC else "unstable" if age <= UNSTABLE_SEC else "offline"

    def is_recording(self, dev: dict) -> bool:
        if self.online_state(dev) == "offline":
            return False
        # Disk is the source of truth: recorder.active stays true after a recorder crash.
        return bool((self.hb(dev).get("recording") or {}).get("active"))

    def episode_status(self, dev: dict, ep: dict, hbd: dict | None = None) -> str:
        """synced | pending | settling | recording | history | incomplete | deleted | cleaned"""
        hbd = hbd if hbd is not None else self.hb(dev)
        if not ep["on_device"]:
            return "cleaned" if ep["cleaned_at"] else "deleted"
        if ep["synced_fp"] and ep["synced_fp"] == ep["fp"]:
            return "synced"
        rec = hbd.get("recording") or {}
        if ep["rel"] in (rec.get("episodes") or []):
            return "recording"
        if not dev["include_history"] and ep["first_seen"] and ep["mtime_max"] \
                and ep["mtime_max"] < dev["sync_floor"] + float(hbd.get("_skew") or 0.0) - 60 \
                and not ep["synced_fp"]:
            return "history"
        stale = ep["rel"] in (rec.get("stale_markers") or [])
        if not ep["finalized"]:
            side = json.loads(ep["sidecar"] or "{}")
            # A failed start leaves only sidecars (nothing more will be written). A crashed
            # recorder leaves an MCAP without metadata.yaml: once quiet for 10 min it is synced
            # as-is and reported as an anomaly, rather than being held on the device forever.
            only_sidecars = not ep["marker"] and not ep["has_mcap"] and side.get("state") in ("failed", "stopped")
            if not only_sidecars and not stale and self.device_now(dev, hbd) - (ep["mtime_max"] or 0) < 600:
                return "incomplete"
        cur = ((hbd.get("status_api") or {}).get("current") or {}).get("rel")
        quiet = min(self.device_now(dev, hbd) - (ep["mtime_max"] or 0), now() - (ep["finalized_at"] or 0))
        if (ep["rel"] == cur or stale) and quiet < self.cfg["settle_seconds"] and not dev["off_shift"]:
            return "settling"
        return "pending"

    def anomalies(self, dev: dict, eps: list[dict], hbd: dict) -> list[dict]:
        out = []
        rec = hbd.get("recording") or {}
        cur = ((hbd.get("status_api") or {}).get("current") or {}).get("rel")
        for ep in eps:
            if not ep["on_device"]:
                continue
            side = json.loads(ep["sidecar"] or "{}")
            reason = None
            if ep["rel"] in (rec.get("stale_markers") or []):
                reason = "录制标记残留、文件停止增长（录制进程可能崩溃）"
            elif side.get("state") == "collecting" and ep["rel"] != cur and ep["rel"] not in (rec.get("episodes") or []):
                reason = "状态停在 collecting（服务可能在录制中重启）"
            elif side.get("state") == "failed":
                reason = "录制失败：" + (side.get("error") or "未知原因")
            elif ep["finalized"] and side.get("recorder_ok") is False:
                reason = "录制服务报告失败"
            elif not ep["has_mcap"] and side.get("state") in ("stopped", "failed"):
                reason = "没有 MCAP 文件"
            if reason:
                out.append({"rel": ep["rel"], "reason": reason, "bytes": ep["bytes"], "state": side.get("state")})
        return out

    def device_summary(self, dev: dict, detail: bool = False) -> dict:
        t = now()
        hbd = self.hb(dev)
        eps = [dict(r) for r in self.store.q("SELECT * FROM episodes WHERE device_id=?", (dev["id"],))]
        counts: dict[str, list] = {}
        for ep in eps:
            st = self.episode_status(dev, ep, hbd)
            ep["status"] = st
            c = counts.setdefault(st, [0, 0])
            c[0] += 1
            c[1] += self.remaining_bytes(ep) if st in ("pending", "settling") else (ep["bytes"] or 0)
        pending = counts.get("pending", [0, 0])
        settling = counts.get("settling", [0, 0])
        disk = hbd.get("disk") or {}
        rate = hbd.get("write_rate_bps") or 0
        rate = rate if rate > 0.1 * MB else self.cfg["default_write_rate_gb_h"] * GB / 3600
        hours_left = (disk.get("free") or 0) / rate / 3600 if disk.get("free") is not None else None
        level = "ok"
        if hours_left is not None:
            level = "critical" if hours_left < self.cfg["device_critical_hours"] else "warn" if hours_left < self.cfg["device_warn_hours"] else "ok"
        today = time.strftime("%Y_%m_%d")
        today_eps = [e for e in eps if e["on_device"] or e["synced_fp"]]
        today_eps = [e for e in today_eps if today in (e["session_rel"] or "") or (e["first_seen"] and time.strftime("%Y_%m_%d", time.localtime(e["first_seen"])) == today and e["first_seen"] > dev["created_at"] + 60)]
        with self.jobs_lock:
            job = self.jobs.get(dev["id"])
        api = hbd.get("status_api") or {}
        online = self.online_state(dev)
        recording = self.is_recording(dev)
        blockers = self.shutdown_blockers(dev, hbd, counts, job)
        s = {
            "id": dev["id"], "name": dev["name"], "mirror_dir": dev["mirror_dir"], "ssh_target": dev["ssh_target"],
            "online": online, "last_seen": dev["last_seen"], "last_ip": dev["last_ip"],
            "hostname": hbd.get("hostname"), "node_version": hbd.get("node_version"),
            "recording": recording,
            "recording_episodes": (hbd.get("recording") or {}).get("episodes") or [],
            "recording_since": dev["recording_since"], "recording_bytes": (hbd.get("recording") or {}).get("bytes"),
            "write_rate_bps": hbd.get("write_rate_bps"),
            "status_api": api,
            "disk": {"free": disk.get("free"), "total": disk.get("total"), "hours_left": hours_left, "level": level},
            "today": {"episodes": len(today_eps), "bytes": sum(e["bytes"] or 0 for e in today_eps)},
            "counts": {k: {"episodes": v[0], "bytes": v[1]} for k, v in counts.items()},
            "pending_bytes": pending[1] + settling[1], "pending_episodes": pending[0] + settling[0],
            "unsynced_bytes": pending[1] + settling[1] + counts.get("recording", [0, 0])[1] + counts.get("incomplete", [0, 0])[1],
            "paused": bool(dev["paused"]), "off_shift": bool(dev["off_shift"]),
            "shutdown_after_sync": bool(dev["shutdown_after_sync"]), "include_history": bool(dev["include_history"]),
            "sync_state": self.sync_state(dev, job, pending, settling, recording, online),
            "job": job.public() if job else None,
            "can_shutdown": not blockers, "shutdown_blockers": blockers,
            "conversion_running": bool(hbd.get("conversion_running")),
            "capabilities": hbd.get("capabilities") or {},
            "idle_since": dev["last_recording_end"],
        }
        s["anomalies"] = self.anomalies(dev, eps, hbd)
        if pending[1] + settling[1] > 0:
            rate_now = job.rate if job and job.rate else None
            s["eta_seconds"] = (pending[1] + settling[1]) / rate_now if rate_now else None
        if detail:
            order = {"recording": 0, "incomplete": 1, "settling": 2, "pending": 3, "history": 4, "synced": 5, "deleted": 6, "cleaned": 7}
            eps.sort(key=lambda e: (order.get(e["status"], 9), e["rel"]), reverse=False)
            s["episodes"] = [{"rel": e["rel"], "status": e["status"], "bytes": e["bytes"], "task": e["task"],
                              "user": e["user"], "session": e["session_rel"], "synced_at": e["synced_at"],
                              "sync_error": e["sync_error"], "outcome": json.loads(e["sidecar"] or "{}").get("outcome"),
                              "state": json.loads(e["sidecar"] or "{}").get("state")}
                             for e in eps if e["status"] not in ("synced", "cleaned", "deleted")][:300]
            s["synced_recent"] = [{"rel": e["rel"], "bytes": e["bytes"], "synced_at": e["synced_at"]}
                                  for e in sorted((e for e in eps if e["status"] == "synced"),
                                                  key=lambda e: e["synced_at"] or 0, reverse=True)[:30]]
            s["jobs"] = [dict(r) for r in self.store.q(
                "SELECT id,started_at,ended_at,bytes_planned,bytes_done,bwlimit_kb,result,error FROM jobs "
                "WHERE device_id=? ORDER BY id DESC LIMIT 20", (dev["id"],))]
            s["events"] = [dict(r) for r in self.store.q(
                "SELECT at,level,kind,message FROM events WHERE device_id=? ORDER BY id DESC LIMIT 40", (dev["id"],))]
            s["commands"] = [dict(r) for r in self.store.q(
                "SELECT id,type,status,created_at,done_at,result FROM commands WHERE device_id=? ORDER BY created_at DESC LIMIT 10",
                (dev["id"],))]
            s["remote_root"] = dev["remote_root"]
        return s

    @staticmethod
    def remaining_bytes(ep: dict) -> int:
        """Bytes still to transfer: everything for a new episode, roughly the delta for a re-sync."""
        if not ep["synced_fp"]:
            return ep["bytes"] or 0
        return max(4096, (ep["bytes"] or 0) - (ep["synced_bytes"] or 0))

    def sync_state(self, dev, job, pending, settling, recording, online) -> str:
        if job:
            return "syncing"
        if dev["paused"]:
            return "paused"
        if pending[0] == 0 and settling[0] == 0:
            return "synced"
        if online == "offline":
            return "offline_pending"
        if recording:
            return "waiting_recording"
        if self.server_disk()["blocked"]:
            return "blocked_disk"
        if pending[0] == 0:
            return "settling"
        if not dev["off_shift"] and not dev["force_sync"] and dev["last_recording_end"] and \
                now() - dev["last_recording_end"] < self.cfg["idle_minutes"] * 60:
            return "waiting_idle"
        return "queued"

    def shutdown_blockers(self, dev, hbd, counts, job) -> list[str]:
        out = []
        if self.online_state(dev) == "offline":
            out.append("设备离线")
        if self.is_recording(dev):
            out.append("正在录制")
        api = hbd.get("status_api") or {}
        if api.get("awaiting_outcome"):
            out.append("有 episode 等待标注成功/失败")
        if hbd.get("conversion_running"):
            out.append("后处理转换进行中")
        for k, label in (("pending", "待同步"), ("settling", "待同步（刚停止）"), ("incomplete", "未完成写入"), ("recording", "录制中")):
            if counts.get(k, [0])[0]:
                out.append(f"{label} {counts[k][0]} 条")
        if job:
            out.append("同步进行中")
        return out

    def server_disk(self) -> dict:
        st = os.statvfs(self.mirror_root)
        free = st.f_bavail * st.f_frsize
        total = st.f_blocks * st.f_frsize
        rows = self.store.q("SELECT day, bytes FROM intake ORDER BY day DESC LIMIT 7")
        default = self.cfg["default_daily_intake_gb"] * GB
        measured = (sum(r["bytes"] for r in rows) / len(rows)) if rows else 0
        # a day or two of partial data (tests, a slow start) says little: stay conservative until 3 days
        intake = measured if len(rows) >= 3 else max(measured, default)
        intake = max(intake, 1 * GB)
        usable = free - self.reserve
        days = usable / intake
        blocked = usable <= 0
        level = "blocked" if blocked else "critical" if days < self.cfg["server_critical_days"] else \
            "warn" if days < self.cfg["server_warn_days"] else "ok"
        return {"path": str(self.mirror_root), "free": free, "total": total, "reserve": self.reserve,
                "daily_intake": intake, "intake_measured": len(rows) >= 3, "days_left": days, "level": level, "blocked": blocked}

    def overview(self) -> dict:
        devs = [self.device_summary(d) for d in self.devices()]
        server = self.server_disk()
        worst = max([server["level"]] + [d["disk"]["level"] for d in devs if d["online"] != "never"],
                    key=lambda lv: ["ok", "warn", "critical", "blocked"].index(lv))
        return {
            "version": VERSION, "time": now(), "server_disk": server, "disk_level": worst,
            "totals": {
                "devices": len(devs), "online": sum(d["online"] in ("online", "unstable") for d in devs),
                "recording": sum(d["recording"] for d in devs),
                "pending_bytes": sum(d["pending_bytes"] for d in devs),
                "pending_episodes": sum(d["pending_episodes"] for d in devs),
                "unsynced_offline_bytes": sum(d["unsynced_bytes"] for d in devs if d["online"] == "offline"),
                "syncing": sum(1 for d in devs if d["job"]),
                "anomalies": sum(len(d["anomalies"]) for d in devs),
            },
            "global_bwlimit_mb": self.cfg["busy_bwlimit_mb"] if self.recording_anywhere else None,
            "devices": devs,
            "events": [dict(r) for r in self.store.q(
                "SELECT e.at,e.level,e.kind,e.message,e.device_id,d.name AS device FROM events e LEFT JOIN devices d ON d.id=e.device_id "
                "WHERE e.level IN ('warn','critical') ORDER BY e.id DESC LIMIT 20")],
            "settings": {k: self.cfg[k] for k in ("reserve_gb", "max_concurrent", "busy_bwlimit_mb", "idle_minutes", "settle_seconds")},
        }

    # ------------------------------------------------------------ events / alerts
    def event(self, dev_id: str | None, level: str, kind: str, message: str, data: dict | None = None) -> None:
        self.store.x("INSERT INTO events(at,device_id,level,kind,message,data) VALUES(?,?,?,?,?,?)",
                     (now(), dev_id, level, kind, message, json.dumps(data or {})))
        log(f"[{level}] {message}")

    def alert(self, dev_id: str | None, level: str, kind: str, message: str) -> None:
        key = (dev_id, kind)
        t = now()
        if t - self.alert_last.get(key, 0) < self.cfg["alert_cooldown_sec"]:
            return
        self.alert_last[key] = t
        self.event(dev_id, level, kind.split(":")[0], message)
        url = self.cfg.get("webhook")
        if url:
            threading.Thread(target=self._post_webhook, args=(url, level, message), daemon=True).start()

    def _post_webhook(self, url: str, level: str, message: str) -> None:
        text = f"[collection-hub {level}] {message}"
        body = {"msg_type": "text", "content": {"text": text}} if self.cfg["webhook_format"] == "feishu" \
            else {"level": level, "text": text}
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:
            log(f"webhook failed: {e}")

    def check_alerts(self) -> None:
        server = self.server_disk()
        lv = server["level"]
        if lv != self.last_levels.get("server", "ok") and lv != "ok":
            self.alert(None, "critical" if lv in ("critical", "blocked") else "warn", "server_disk:" + lv,
                       f"存储服务器数据盘剩余 {fmt_bytes(server['free'])}，约可收 {max(server['days_left'], 0):.1f} 天"
                       + ("，已低于预留值，暂停发起新的同步" if server["blocked"] else ""))
        self.last_levels["server"] = lv
        for dev in self.devices():
            s = self.device_summary(dev)
            online = s["online"] in ("online", "unstable")
            if self.last_online.get(dev["id"]) and not online and s["unsynced_bytes"] > 0:
                self.alert(dev["id"], "critical", "offline_unsynced",
                           f"{dev['name']} 已离线，仍有 {fmt_bytes(s['unsynced_bytes'])} 只在设备上")
            self.last_online[dev["id"]] = online
            dl = s["disk"]["level"]
            if online and dl != self.last_levels.get(dev["id"], "ok") and dl != "ok":
                self.alert(dev["id"], "critical" if dl == "critical" else "warn", "device_disk:" + dl,
                           f"{dev['name']} 磁盘剩余 {fmt_bytes(s['disk']['free'])}，约可录 {s['disk']['hours_left']:.1f} 小时")
            self.last_levels[dev["id"]] = dl
            for a in s["anomalies"]:
                self.alert(dev["id"], "warn", "anomaly:" + a["rel"], f"{dev['name']} {a['rel']}：{a['reason']}")
            fails = self.store.q("SELECT result FROM jobs WHERE device_id=? ORDER BY id DESC LIMIT 3", (dev["id"],))
            if len(fails) == 3 and all(r["result"] == "failed" for r in fails):
                self.alert(dev["id"], "critical", "sync_failing", f"{dev['name']} 连续 3 次同步失败，请查看同步记录")
            # P3: shut down once everything is safe
            if dev["shutdown_after_sync"] and s["can_shutdown"]:
                if not (s["capabilities"] or {}).get("shutdown"):
                    self.alert(dev["id"], "warn", "shutdown_unsupported", f"{dev['name']} 未启用远程关机（--allow-shutdown）")
                else:
                    self.enqueue_command(dev["id"], "shutdown", {})
                    self.store.x("UPDATE devices SET shutdown_after_sync=0 WHERE id=?", (dev["id"],))
                    self.event(dev["id"], "info", "shutdown_sent", f"{dev['name']} 数据已全部同步，已下发关机指令")

    # ------------------------------------------------------------ commands (P3)
    def enqueue_command(self, dev_id: str, kind: str, payload: dict) -> str:
        cid = secrets.token_hex(6)
        self.store.x("INSERT INTO commands(id,device_id,type,payload,created_at,status) VALUES(?,?,?,?,?,'pending')",
                     (cid, dev_id, kind, json.dumps(payload), now()))
        return cid

    def _on_command_result(self, dev: dict, res: dict) -> None:
        if res.get("type") == "cleanup":
            ok_rels = [e["rel"] for e in res.get("episodes") or [] if e.get("ok")]
            t = now()
            for rel in ok_rels:
                self.store.x("UPDATE episodes SET cleaned_at=? WHERE device_id=? AND rel=?", (t, dev["id"], rel))
            self.store.x("UPDATE cleanup_requests SET status=? WHERE command_id=?",
                         ("done" if res.get("ok") else "partial", res.get("id")))
            self.event(dev["id"], "info", "cleanup_done",
                       f"{dev['name']} 本地清理完成 {len(ok_rels)} 条" + ("" if res.get("ok") else "（部分失败）"))
        elif res.get("type") == "shutdown":
            self.event(dev["id"], "info" if res.get("ok") else "warn", "shutdown_result",
                       f"{dev['name']} 关机指令{'已执行' if res.get('ok') else '失败：' + str(res.get('detail'))}")

    def cleanup_preview(self, dev_id: str, older_than_days: float) -> dict:
        dev = self.device_row(dev_id)
        hbd = self.hb(dev)
        cur = ((hbd.get("status_api") or {}).get("current") or {}).get("rel")
        cutoff = self.device_now(dev, hbd) - older_than_days * 86400
        items, total = [], 0
        for r in self.store.q("SELECT * FROM episodes WHERE device_id=? AND on_device=1 AND synced_fp=fp AND mtime_max<?",
                              (dev_id, cutoff)):
            if r["rel"] == cur or not self._verify_local(dev, r):
                continue
            items.append({"rel": r["rel"], "files": json.loads(r["files"]), "bytes": r["bytes"]})
            total += r["bytes"] or 0
        rid = secrets.token_hex(6)
        token = secrets.token_hex(4)
        self.store.x("INSERT INTO cleanup_requests(id,device_id,created_at,episodes,bytes,confirm_token,status)"
                     " VALUES(?,?,?,?,?,?,'preview')", (rid, dev_id, now(), json.dumps(items), total, token))
        return {"request_id": rid, "confirm_token": token, "episodes": len(items), "bytes": total,
                "sample": [i["rel"] for i in items[:20]], "older_than_days": older_than_days}

    def cleanup_confirm(self, dev_id: str, request_id: str, confirm_token: str) -> dict:
        r = self.store.one("SELECT * FROM cleanup_requests WHERE id=? AND device_id=?", (request_id, dev_id))
        if r is None or r["status"] != "preview":
            raise ValueError("cleanup request not found or already used")
        if not secrets.compare_digest(r["confirm_token"], str(confirm_token)):
            raise PermissionError("confirmation code does not match")
        if now() - r["created_at"] > 900:
            raise ValueError("preview expired, create a new one")
        dev = self.device_row(dev_id)
        items = [i for i in json.loads(r["episodes"]) if self._verify_local(dev, self.store.one(
            "SELECT * FROM episodes WHERE device_id=? AND rel=?", (dev_id, i["rel"])))]
        cid = self.enqueue_command(dev_id, "cleanup", {"episodes": items})
        self.store.x("UPDATE cleanup_requests SET status='sent', command_id=? WHERE id=?", (cid, request_id))
        self.event(dev_id, "warn", "cleanup_requested", f"{dev['name']} 已确认清理设备上 {len(items)} 条已校验数据")
        return {"command_id": cid, "episodes": len(items)}

    def _verify_local(self, dev: dict, row) -> bool:
        if row is None or not row["synced_fp"] or row["synced_fp"] != row["fp"]:
            return False
        base = self.mirror_root / dev["mirror_dir"]
        for rel, size, mtime_ns in json.loads(row["files"] or "[]"):
            st = _stat(base / rel)
            if st is None or st.st_size != size or int(st.st_mtime) != int(mtime_ns // 1_000_000_000):
                return False
        return True

    # ------------------------------------------------------------ scheduler
    def cancel_job(self, dev_id: str, reason: str) -> None:
        with self.jobs_lock:
            job = self.jobs.get(dev_id)
        if job:
            job.cancel(reason)

    def eligible(self, dev: dict) -> list[dict]:
        hbd = self.hb(dev)
        out = []
        for r in self.store.q("SELECT * FROM episodes WHERE device_id=? AND on_device=1 AND (synced_fp IS NULL OR synced_fp!=fp) "
                              "ORDER BY session_rel, rel", (dev["id"],)):
            ep = dict(r)
            if self.episode_status(dev, ep, hbd) == "pending":
                out.append(ep)
        return out

    def schedule_once(self) -> None:
        devs = self.devices()
        rec_any = any(self.is_recording(d) for d in devs)
        if rec_any != self.recording_anywhere:
            self.recording_anywhere = rec_any
            # bandwidth policy changed: restart running jobs with the new limit (rsync resumes with --partial)
            with self.jobs_lock:
                running = list(self.jobs.values())
            for job in running:
                job.cancel("bw_changed")
        server = self.server_disk()
        with self.jobs_lock:
            running_ids = set(self.jobs)
        slots = int(self.cfg["max_concurrent"]) - len(running_ids)
        if slots <= 0 or server["blocked"]:
            return
        cands = []
        for d in devs:
            if d["id"] in running_ids or d["paused"] or not d["ssh_target"] and not d["remote_root"]:
                continue
            if self.online_state(d) != "online" or self.is_recording(d):
                continue
            idle_ok = d["off_shift"] or d["force_sync"] or not d["last_recording_end"] or \
                now() - d["last_recording_end"] >= self.cfg["idle_minutes"] * 60
            if not idle_ok:
                continue
            eps = self.eligible(d)
            if not eps:
                if d["force_sync"]:
                    self.store.x("UPDATE devices SET force_sync=0 WHERE id=?", (d["id"],))
                continue
            free = ((self.hb(d).get("disk") or {}).get("free")) or 0
            cands.append(((0 if d["off_shift"] or d["force_sync"] else 1), free, d, eps))
        cands.sort(key=lambda c: (c[0], c[1]))
        n_new = min(slots, len(cands))
        n_total = len(running_ids) + n_new
        for _, _, d, eps in cands[:n_new]:
            batch, size = [], 0
            for ep in eps:
                b = (ep["bytes"] or 0)
                if batch and (size + b > self.cfg["batch_max_gb"] * GB or len(batch) >= self.cfg["batch_max_episodes"]):
                    break
                batch.append(ep)
                size += b
            if server["free"] - size < self.reserve:
                self.alert(None, "critical", "reserve", f"存储服务器剩余空间不足以同步 {d['name']} 的下一批数据，已暂停")
                continue
            bw = int(self.cfg["busy_bwlimit_mb"] * 1024 / max(1, n_total)) if rec_any else 0
            job = SyncJob(self, d, batch, bw)
            with self.jobs_lock:
                self.jobs[d["id"]] = job
            job.start()

    def run_scheduler(self) -> None:
        last_alert = 0.0
        while not self.stop.is_set():
            try:
                self.schedule_once()
                if now() - last_alert > 10:
                    self.check_alerts()
                    last_alert = now()
            except Exception:
                traceback.print_exc()
            self.stop.wait(self.cfg["scheduler_interval"])

    def changes(self, since: int, limit: int = 500) -> dict:
        rows = self.store.q("SELECT c.*, d.name AS device FROM changes c LEFT JOIN devices d ON d.id=c.device_id "
                            "WHERE seq>? ORDER BY seq LIMIT ?", (since, limit))
        last = self.store.one("SELECT MAX(seq) AS m FROM changes")
        return {"cursor": rows[-1]["seq"] if rows else since, "latest": (last["m"] or 0) if last else 0,
                "mirror_root": str(self.mirror_root), "changes": [dict(r) for r in rows]}


def _stat(p: Path):
    try:
        return p.stat()
    except OSError:
        return None


# ============================================================================ sync job

EPISODE_REL_RE = re.compile(r"^(?:[^/]+/)*data/[^/]+/[^/]+/episode_\d+$")


def safe_rel(path: str) -> bool:
    """A manifest path must stay inside the device mirror: relative, no '..', no empty parts."""
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path or "\0" in path:
        return False
    return all(part not in ("", ".", "..") for part in path.split("/"))


def valid_manifest_episode(ep: dict) -> bool:
    """Reject anything a buggy or hostile node could use to write or move files outside its mirror."""
    rel = ep.get("rel")
    if not safe_rel(rel) or not EPISODE_REL_RE.match(rel):
        return False
    parts = rel.split("/")
    i = len(parts) - 4
    logs = "/".join(parts[:i] + ["logs"] + parts[i + 1:]) + "/"
    for f in ep.get("files") or []:
        if not (isinstance(f, list) and len(f) == 3 and safe_rel(f[0]) and isinstance(f[1], int) and isinstance(f[2], int)):
            return False
        if not (f[0].startswith(rel + "/") or f[0].startswith(logs)):
            return False
    return True


PROGRESS_RE = re.compile(r"^\s*([\d,]+)\s+(\d+)%\s+([\d.]+)([kMG]?B)/s")


class SyncJob(threading.Thread):
    def __init__(self, hub: Hub, dev: dict, episodes: list[dict], bwlimit_kb: int):
        super().__init__(daemon=True, name=f"sync-{dev['name']}")
        self.hub, self.dev, self.episodes, self.bwlimit_kb = hub, dev, episodes, bwlimit_kb
        self.proc: subprocess.Popen | None = None
        self.cancel_reason: str | None = None
        self.bytes_planned = sum(e["bytes"] or 0 for e in episodes)
        self.bytes_done = 0
        self.rate = 0.0
        self.started_at = now()
        cur = hub.store.x("INSERT INTO jobs(device_id,started_at,episodes,bytes_planned,bwlimit_kb) VALUES(?,?,?,?,?)",
                          (dev["id"], self.started_at, json.dumps([e["rel"] for e in episodes]), self.bytes_planned, bwlimit_kb))
        self.job_id = cur.lastrowid

    def public(self) -> dict:
        return {"id": self.job_id, "episodes": len(self.episodes), "bytes_planned": self.bytes_planned,
                "bytes_done": self.bytes_done, "rate": self.rate, "bwlimit_kb": self.bwlimit_kb,
                "started_at": self.started_at,
                "progress": (self.bytes_done / self.bytes_planned) if self.bytes_planned else 1.0}

    def cancel(self, reason: str) -> None:
        self.cancel_reason = reason
        p = self.proc
        if p and p.poll() is None:
            try:
                p.terminate()
            except ProcessLookupError:
                pass

    def command(self, list_file: Path, dest: Path) -> list[str]:
        c = self.hub.cfg
        # -a without symlinks/devices: a recording tree only holds regular files, and a link
        # copied from a device must never point the mirror (or RoboCurate) elsewhere
        cmd = [c["rsync"], "-a", "--no-links", "--no-D", "--partial", "--partial-dir=.rsync-partial",
               "--info=progress2", "--no-motd",
               f"--files-from={list_file}"]
        if self.bwlimit_kb:
            cmd.append(f"--bwlimit={self.bwlimit_kb}")
        target, root = self.dev["ssh_target"], self.dev["remote_root"].rstrip("/")
        if target:
            ssh = ["ssh"]
            key = Path(os.path.expanduser(c["ssh_key"]))
            if key.exists():
                ssh += ["-i", str(key), "-o", "IdentitiesOnly=yes"]
            ssh += list(c["ssh_options"])
            cmd += ["-e", " ".join(ssh)]
            # with rrsync the remote side is already rooted at the recordings dir
            cmd.append(f"{target}:{root + '/' if root else ''}")
        else:
            cmd.append(root + "/")  # local transport (tests, or a mounted device disk)
        cmd.append(str(dest) + "/")
        return cmd

    def run(self) -> None:
        hub, dev = self.hub, self.dev
        dest = hub.mirror_root / dev["mirror_dir"]
        dest.mkdir(parents=True, exist_ok=True)
        list_file = dest / f".sync-files-{self.job_id}.txt"
        files = [f[0] for e in self.episodes for f in json.loads(e["files"] or "[]")]
        list_file.write_text("\n".join(files) + "\n")
        result, error = "failed", ""
        try:
            hub.event(dev["id"], "info", "sync_started",
                      f"{dev['name']} 开始同步 {len(self.episodes)} 条 {fmt_bytes(self.bytes_planned)}"
                      + (f"（限速 {self.bwlimit_kb // 1024} MB/s）" if self.bwlimit_kb else "（全速）"))
            cmd = self.command(list_file, dest)
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
            err_chunks = []
            t_err = threading.Thread(target=lambda: err_chunks.append(self.proc.stderr.read()), daemon=True)
            t_err.start()
            buf = b""
            t0 = now()
            while True:
                chunk = self.proc.stdout.read1(4096) if hasattr(self.proc.stdout, "read1") else self.proc.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = re.split(rb"[\r\n]", buf)
                for line in lines:
                    m = PROGRESS_RE.match(line.decode(errors="replace"))
                    if m:
                        self.bytes_done = int(m.group(1).replace(",", ""))
                        dt = now() - t0
                        self.rate = self.bytes_done / dt if dt > 0 else 0
            rc = self.proc.wait()
            t_err.join(timeout=5)
            stderr = (err_chunks[0] if err_chunks else b"").decode(errors="replace")
            if self.cancel_reason:
                result, error = self.cancel_reason, ""
            elif rc in (0, 23, 24):
                # 23/24: some files vanished or changed while copying — verify what we have
                result, error = ("ok" if rc == 0 else "partial"), stderr[-800:] if rc else ""
            else:
                result, error = "failed", f"rsync exit {rc}: {stderr[-800:]}"
            verified, changed = self.verify(dest)
            if result == "ok" and verified < len(self.episodes):
                # files newer than the manifest the job was planned from are routine (a label
                # landed mid-sync); the next heartbeat carries the new fingerprint
                result = "ok_changed" if verified + changed == len(self.episodes) else "partial"
        except Exception as e:
            result, error = "failed", repr(e)
            traceback.print_exc()
        finally:
            try:
                list_file.unlink()
            except OSError:
                pass
            hub.store.x("UPDATE jobs SET ended_at=?, bytes_done=?, result=?, error=? WHERE id=?",
                        (now(), self.bytes_done, result, error, self.job_id))
            if result in ("ok", "ok_changed", "partial"):
                hub.store.x("UPDATE devices SET force_sync=0 WHERE id=?", (dev["id"],))
            level = "info" if result in ("ok", "ok_changed", "paused_recording", "bw_changed", "paused") else "warn"
            label = {"ok": "完成", "ok_changed": "完成（部分文件同步期间有更新，稍后补传）", "partial": "部分完成", "failed": "失败", "paused_recording": "因开始录制而暂停",
                     "bw_changed": "带宽策略变化，稍后续传", "paused": "已手动暂停", "shutdown": "服务停止"}.get(result, result)
            hub.event(dev["id"], level, "sync_" + result, f"{dev['name']} 同步{label}" + (f"：{error[:200]}" if error else ""))
            with hub.jobs_lock:
                if hub.jobs.get(dev["id"]) is self:
                    del hub.jobs[dev["id"]]

    def verify(self, dest: Path) -> tuple[int, int]:
        """Mark episodes whose files all match the manifest (size + mtime seconds).

        Returns (verified, changed): ``changed`` counts episodes whose only mismatches are
        files newer than the manifest, i.e. updated on the device after the job was planned.
        """
        hub, dev = self.hub, self.dev
        ok = changed = 0
        t = now()
        day = time.strftime("%Y-%m-%d")
        for ep in self.episodes:
            files = json.loads(ep["files"] or "[]")
            bad, newer = [], 0
            for rel, size, mtime_ns in files:
                st = _stat(dest / rel)
                if st is None or st.st_size != size or int(st.st_mtime) != int(mtime_ns // 1_000_000_000):
                    bad.append(rel)
                    if st is not None and int(st.st_mtime) > int(mtime_ns // 1_000_000_000):
                        newer += 1
            if bad and newer == len(bad):
                changed += 1
            if not bad:
                ok += 1
                new_bytes = (ep["bytes"] or 0) - ((ep["synced_bytes"] or 0) if ep["synced_fp"] else 0)
                hub.store.x("UPDATE episodes SET synced_fp=?, synced_bytes=?, synced_at=?, sync_error=NULL WHERE device_id=? AND rel=?",
                            (ep["fp"], ep["bytes"], t, dev["id"], ep["rel"]))
                hub.store.x("INSERT INTO intake(day,bytes) VALUES(?,?) ON CONFLICT(day) DO UPDATE SET bytes=bytes+excluded.bytes",
                            (day, max(0, new_bytes)))
                session = Path(ep["session_rel"] or ".")
                hub.store.x("INSERT INTO changes(device_id,session_path,episode_path,kind,at) VALUES(?,?,?,?,?)",
                            (dev["id"], str(dest / session), str(dest / ep["rel"]), "synced", t))
            elif self.cancel_reason is None and newer != len(bad):
                hub.store.x("UPDATE episodes SET sync_error=?, attempts=attempts+1 WHERE device_id=? AND rel=?",
                            ("校验不一致: " + ", ".join(bad[:3]), dev["id"], ep["rel"]))
        return ok, changed


# ============================================================================ http

class _Handler(BaseHTTPRequestHandler):
    hub: Hub = None  # type: ignore
    admin = False
    server_version = "collection-hub/" + VERSION

    def log_message(self, fmt, *args):
        pass

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 64 * MB:
            raise ValueError("body too large")
        raw = self.rfile.read(n) if n else b""
        if self.headers.get("Content-Encoding") == "gzip":
            d = zlib.decompressobj(16 + zlib.MAX_WBITS)
            raw = d.decompress(raw, 256 * MB)
            if d.unconsumed_tail:
                raise ValueError("decompressed body too large")
        data = json.loads(raw or b"{}")
        if not isinstance(data, dict):
            raise ValueError("JSON object expected")
        return data

    def _dispatch(self, method: str):
        try:
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            qs = parse_qs(u.query)
            if not self.admin:
                if method == "POST" and u.path == "/api/node/heartbeat":
                    token = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
                    dev = self.hub.authenticate(token)
                    if dev is None:
                        return self._json({"ok": False, "error": "invalid token"}, 401)
                    return self._json(self.hub.heartbeat(dev, self._body(), self.client_address[0]))
                if u.path == "/api/health":
                    return self._json({"ok": True, "app": "collection-hub"})
                if method == "GET" and u.path in ("/install/node.py", "/install/install_node.sh"):
                    f = Path(__file__).resolve().parent / ("node.py" if u.path.endswith(".py") else "deploy/install_node.sh")
                    body = f.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                return self._json({"ok": False, "error": "not found"}, 404)
            return self._admin(method, parts, qs)
        except PermissionError as e:
            self._json({"ok": False, "error": str(e)}, 403)
        except KeyError as e:
            self._json({"ok": False, "error": str(e)}, 404)
        except (ValueError, TypeError) as e:
            self._json({"ok": False, "error": str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            self._json({"ok": False, "error": repr(e)}, 500)

    def _admin(self, method, parts, qs):
        hub = self.hub
        if parts[:1] != ["api"]:
            return self._json({"ok": False, "error": "not found"}, 404)
        p = parts[1:]
        if method == "GET":
            if p == ["health"]:
                return self._json({"ok": True, "app": "collection-hub", "version": VERSION, "mirror_root": str(hub.mirror_root)})
            if p == ["overview"]:
                return self._json(hub.overview())
            if p == ["devices"]:
                return self._json({"devices": [hub.device_summary(d) for d in hub.devices()]})
            if len(p) == 2 and p[0] == "devices":
                return self._json(hub.device_summary(hub.device_row(p[1]), detail=True))
            if p == ["changes"]:
                return self._json(hub.changes(int(qs.get("since", ["0"])[0]), int(qs.get("limit", ["500"])[0])))
            if p == ["events"]:
                return self._json({"events": [dict(r) for r in hub.store.q(
                    "SELECT * FROM events ORDER BY id DESC LIMIT ?", (int(qs.get("limit", ["100"])[0]),))]})
        if method == "POST":
            body = self._body()
            if p == ["devices"]:
                name = str(body.get("name") or "").strip()
                if not name:
                    raise ValueError("name is required")
                dev, token = hub.add_device(name, str(body.get("ssh_target") or ""), str(body.get("remote_root") or ""))
                return self._json({"device": hub.device_summary(dev), "token": token, **hub.install_info()})
            if len(p) >= 3 and p[0] == "devices":
                dev_id, action = p[1], "/".join(p[2:])
                if action == "settings":
                    hub.update_settings(dev_id, body)
                    return self._json(hub.device_summary(hub.device_row(dev_id)))
                if action == "sync-now":
                    hub.sync_now(dev_id)
                    return self._json({"ok": True})
                if action == "token":
                    return self._json({"token": hub.rotate_token(dev_id)})
                if action == "rescan":
                    return self._json({"command_id": hub.enqueue_command(dev_id, "rescan", {})})
                if action == "cleanup/preview":
                    return self._json(hub.cleanup_preview(dev_id, float(body.get("older_than_days", 3))))
                if action == "cleanup/confirm":
                    return self._json(hub.cleanup_confirm(dev_id, str(body.get("request_id")), str(body.get("confirm_token"))))
        return self._json({"ok": False, "error": "not found"}, 404)

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")


def _listen(spec: str) -> tuple[str, int]:
    host, _, port = spec.rpartition(":")
    return host or "0.0.0.0", int(port)


def serve(hub: Hub) -> list[ThreadingHTTPServer]:
    servers = []
    for spec, admin in ((hub.cfg["node_listen"], False), (hub.cfg["admin_listen"], True)):
        h = type("H", (_Handler,), {"hub": hub, "admin": admin})
        srv = ThreadingHTTPServer(_listen(spec), h)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True, name=("admin" if admin else "node") + "-http").start()
        servers.append(srv)
    return servers


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, file=sys.stderr, flush=True)


def load_cfg(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).expanduser().read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="collection-hub")
    ap.add_argument("--config", help="JSON config file (keys: " + ", ".join(sorted(DEFAULTS)) + ")")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("serve")
    a = sub.add_parser("add-device")
    a.add_argument("--name", required=True)
    a.add_argument("--ssh-target", default="")
    a.add_argument("--remote-root", default="")
    a.add_argument("--include-history", action="store_true", help="also sync episodes recorded before registration")
    sub.add_parser("list")
    r = sub.add_parser("rotate-token")
    r.add_argument("device_id")
    args = ap.parse_args(argv)
    hub = Hub(load_cfg(args.config))
    if args.cmd == "add-device":
        dev, token = hub.add_device(args.name, args.ssh_target, args.remote_root)
        if args.include_history:
            hub.update_settings(dev["id"], {"include_history": 1})
        print(json.dumps({"id": dev["id"], "name": dev["name"], "mirror_dir": dev["mirror_dir"], "token": token,
                          **hub.install_info()}, indent=2))
        return 0
    if args.cmd == "list":
        for d in hub.devices():
            s = hub.device_summary(d)
            print(f"{d['id']}  {d['name']:<12} {s['online']:<9} rec={int(s['recording'])} pending={fmt_bytes(s['pending_bytes'])} "
                  f"sync={s['sync_state']}")
        return 0
    if args.cmd == "rotate-token":
        print(hub.rotate_token(args.device_id))
        return 0
    servers = serve(hub)
    log(f"collection-hub {VERSION}: node {hub.cfg['node_listen']} · admin {hub.cfg['admin_listen']} · mirror {hub.mirror_root}")

    def shutdown(*_):
        hub.stop.set()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        hub.run_scheduler()
    finally:
        with hub.jobs_lock:
            jobs = list(hub.jobs.values())
        for j in jobs:
            j.cancel("shutdown")
        for j in jobs:
            j.join(timeout=10)
        for s in servers:
            s.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
