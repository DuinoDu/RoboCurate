"""collection-node / collection-hub: discovery, sync rules, commands.

Runs a real hub (HTTP + SQLite) and a real node against temporary directories,
using rsync's local transport instead of SSH.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collection import hub as hubmod  # noqa: E402
from collection import node as nodemod  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")

SESSION = "locomanip_teleop_v14/2026_09_28-10_36_15"


def make_episode(root: Path, task: str, n: int, *, recording=False, size=200_000, state=None, outcome=True,
                 session=SESSION, user="op"):
    ep = root / session / "data" / user / task / f"episode_{n:06d}"
    (ep / "mcap").mkdir(parents=True, exist_ok=True)
    (root / session / "logs" / user / task / f"episode_{n:06d}").mkdir(parents=True, exist_ok=True)
    (root / session / "logs" / user / task / f"episode_{n:06d}" / "recorder.log").write_text("log\n")
    (ep / "mcap" / "mcap_0.mcap").write_bytes(os.urandom(size))
    if recording:
        (ep / "mcap" / nodemod.MARKER).write_text("")
    else:
        (ep / "mcap" / nodemod.MCAP_META).write_text("rosbag2_bagfile_information: {}\n")
    summary = {"episode_id": ep.name, "state": state or ("collecting" if recording else "stopped"),
               "recorder": {"ok": True}, "metadata": {"task_name": task, "user_name": user},
               "task_outcome": None if recording else {"success": outcome}}
    (ep / nodemod.SUMMARY).write_text(json.dumps(summary))
    (ep / nodemod.META).write_text(json.dumps({"task_name": task}))
    return ep


def finish(ep: Path):
    (ep / "mcap" / nodemod.MARKER).unlink()
    (ep / "mcap" / nodemod.MCAP_META).write_text("done\n")
    s = json.loads((ep / nodemod.SUMMARY).read_text())
    s["state"] = "stopped"
    (ep / nodemod.SUMMARY).write_text(json.dumps(s))


@pytest.fixture
def env(tmp_path):
    device = tmp_path / "device" / "raw"
    device.mkdir(parents=True)
    cfg = {"db": str(tmp_path / "hub.sqlite3"), "mirror_root": str(tmp_path / "mirror" / "raw"),
           "node_listen": "127.0.0.1:0", "admin_listen": "127.0.0.1:0", "settle_seconds": 0, "idle_minutes": 0,
           "reserve_gb": 0, "ssh_key": str(tmp_path / "nokey"), "alert_cooldown_sec": 0}
    hub = hubmod.Hub(cfg)
    servers = hubmod.serve(hub)
    port = servers[0].server_address[1]
    dev, token = hub.add_device("G1-01", "", str(device), sync_floor=0)
    flag = tmp_path / "shutdown-called"
    node = nodemod.Node({"hub": f"http://127.0.0.1:{port}", "token": token, "data_root": str(device),
                         "status_url": "http://127.0.0.1:9/status", "device_id": "machine-A",
                         "allow_shutdown": True, "allow_cleanup": True, "shutdown_cmd": ["touch", str(flag)]})
    yield {"hub": hub, "node": node, "device": device, "dev": dev, "mirror": hub.mirror_root / "G1-01",
           "flag": flag, "port": port, "token": token, "tmp": tmp_path}
    hub.stop.set()
    for s in servers:
        s.shutdown()


def step(node):
    """One node cycle: scan, watch, heartbeat."""
    node.full_scan()
    node.recording = node.watch.poll()
    return node.send()


def run_sync(hub):
    hub.schedule_once()
    with hub.jobs_lock:
        jobs = list(hub.jobs.values())
    for j in jobs:
        j.join(timeout=60)
    return jobs


def summary(hub, dev_id):
    return hub.device_summary(hub.device_row(dev_id), detail=True)


def test_scan_classifies_episodes(tmp_path):
    make_episode(tmp_path, "pick", 1)
    make_episode(tmp_path, "pick", 2, recording=True)
    eps = nodemod.scan_episodes(tmp_path)
    assert set(eps) == {f"{SESSION}/data/op/pick/episode_000001", f"{SESSION}/data/op/pick/episode_000002"}
    done = eps[f"{SESSION}/data/op/pick/episode_000001"]
    live = eps[f"{SESSION}/data/op/pick/episode_000002"]
    assert done["finalized"] and not done["marker"] and done["sidecar"]["outcome"] is True
    assert live["marker"] and not live["finalized"]
    # log files travel with the episode
    assert any("/logs/" in f[0] for f in done["files"])
    assert done["session_rel"] == SESSION


def test_watch_detects_recording_and_stale_marker(tmp_path):
    ep = make_episode(tmp_path, "pick", 1, recording=True)
    w = nodemod.RecordingWatch(tmp_path)
    w.seed(nodemod.scan_episodes(tmp_path))
    r = w.poll()
    assert r["active"] and r["episodes"] == [f"{SESSION}/data/op/pick/episode_000001"]
    # recorder crashed: marker stays, file stops growing
    h = w.hot[r["episodes"][0]]
    h["marker_since"] = time.time() - 120
    h["grew_at"] = time.time() - 120
    r = w.poll()
    assert not r["active"] and r["stale_markers"] == [f"{SESSION}/data/op/pick/episode_000001"]
    # orphaned writer: no marker but still growing counts as recording (server restarted mid-recording)
    finish(ep)
    w.poll()
    with open(ep / "mcap" / "mcap_0.mcap", "ab") as fh:
        fh.write(b"x" * 1000)
    assert w.poll()["active"]


def test_sync_rules_end_to_end(env):
    hub, node, device, mirror = env["hub"], env["node"], env["device"], env["mirror"]
    a = make_episode(device, "pick", 1)
    b = make_episode(device, "pick", 2, recording=True)
    resp = step(node)
    assert resp and resp["manifest_ack"] == 1
    s = summary(hub, env["dev"]["id"])
    assert s["online"] == "online" and s["recording"] is True
    # never sync a device while it records
    assert run_sync(hub) == []
    assert not mirror.exists() or not any(mirror.rglob("*.mcap"))
    # recording stops -> both episodes pulled, logs included, verified
    finish(b)
    node.watch.hot.clear()
    step(node)
    jobs = run_sync(hub)
    assert jobs and jobs[0].bwlimit_kb == 0
    for ep in (a, b):
        rel = ep.relative_to(device)
        assert (mirror / rel / "mcap" / "mcap_0.mcap").read_bytes() == (ep / "mcap" / "mcap_0.mcap").read_bytes()
    assert (mirror / SESSION / "logs" / "op" / "pick" / "episode_000001" / "recorder.log").exists()
    s = summary(hub, env["dev"]["id"])
    assert s["counts"]["synced"]["episodes"] == 2 and s["sync_state"] == "synced"
    ch = hub.changes(0)
    assert {c["kind"] for c in ch["changes"]} == {"synced"} and ch["changes"][0]["session_path"] == str(mirror / SESSION)
    # a later label rewrites the small sidecar -> re-synced
    time.sleep(1.1)
    sm = json.loads((a / nodemod.SUMMARY).read_text())
    sm["task_outcome"] = {"success": False, "failure_reason": "relabel"}
    (a / nodemod.SUMMARY).write_text(json.dumps(sm))
    step(node)
    run_sync(hub)
    assert json.loads((mirror / a.relative_to(device) / nodemod.SUMMARY).read_text())["task_outcome"]["success"] is False
    # deleted on the device after sync -> our copy goes to .trash, not lost
    shutil.rmtree(a)
    step(node)
    assert not (mirror / a.relative_to(device)).exists()
    assert list((hub.mirror_root / ".trash").rglob("mcap_0.mcap"))
    assert hub.changes(0)["changes"][-1]["kind"] == "trashed"


def test_shutdown_after_sync_and_cleanup(env):
    hub, node, device = env["hub"], env["node"], env["device"]
    make_episode(device, "pick", 1)
    step(node)
    hub.update_settings(env["dev"]["id"], {"shutdown_after_sync": 1})
    hub.check_alerts()
    assert not hub.store.q("SELECT * FROM commands")  # not yet synced -> no shutdown
    run_sync(hub)
    step(node)
    s = summary(hub, env["dev"]["id"])
    assert s["can_shutdown"], s["shutdown_blockers"]
    hub.check_alerts()
    step(node)  # delivers the shutdown command
    assert env["flag"].exists()
    step(node)  # reports the result
    assert hub.store.one("SELECT status FROM commands WHERE type='shutdown'")["status"] == "done"
    # cleanup: preview -> confirm -> node deletes only verified copies
    prev = hub.cleanup_preview(env["dev"]["id"], older_than_days=0)
    assert prev["episodes"] == 1
    with pytest.raises(PermissionError):
        hub.cleanup_confirm(env["dev"]["id"], prev["request_id"], "wrong")
    hub.cleanup_confirm(env["dev"]["id"], prev["request_id"], prev["confirm_token"])
    step(node)  # executes
    assert not any(device.rglob("mcap_0.mcap"))
    step(node)  # result + manifest removal
    row = hub.store.one("SELECT * FROM episodes")
    assert row["cleaned_at"] and not row["on_device"] and not row["trashed"]
    assert list(env["mirror"].rglob("mcap_0.mcap")), "mirror copy must stay"


def test_bandwidth_limited_while_another_device_records(env):
    hub, node, device = env["hub"], env["node"], env["device"]
    make_episode(device, "pick", 1, size=50_000)
    step(node)
    # a second robot is recording
    other_root = env["tmp"] / "device2"
    other_root.mkdir()
    dev2, tok2 = hub.add_device("G1-02", "", str(other_root), sync_floor=0)
    make_episode(other_root, "place", 1, recording=True)
    node2 = nodemod.Node({"hub": f"http://127.0.0.1:{env['port']}", "token": tok2, "data_root": str(other_root),
                          "status_url": "http://127.0.0.1:9/status", "device_id": "machine-B"})
    step(node2)
    jobs = run_sync(hub)
    assert len(jobs) == 1 and jobs[0].dev["name"] == "G1-01"
    assert jobs[0].bwlimit_kb == 15 * 1024


def test_manifest_resync_and_token_binding(env):
    hub, node, device = env["hub"], env["node"], env["device"]
    make_episode(device, "pick", 1)
    step(node)
    # hub lost state (e.g. restored DB) -> asks for a full manifest
    hub.store.x("UPDATE devices SET manifest_seq=99")
    make_episode(device, "pick", 2)
    resp = step(node)
    assert resp["need_full"]
    resp = step(node)
    assert resp["manifest_ack"] and not resp["need_full"]
    assert hub.store.one("SELECT COUNT(*) AS n FROM episodes WHERE on_device=1")["n"] == 2
    # the same token used on another machine is rejected
    other = nodemod.Node({**node.cfg, "device_id": "machine-X"})
    other.full_scan()
    assert other.send() is None


def test_history_not_synced_by_default(env):
    hub, node, device = env["hub"], env["node"], env["device"]
    ep = make_episode(device, "old", 1)
    os.utime(ep / "mcap" / "mcap_0.mcap", (1_600_000_000, 1_600_000_000))
    hub.store.x("UPDATE devices SET sync_floor=?", (time.time() - 60,))
    for f in ep.rglob("*"):
        os.utime(f, (1_600_000_000, 1_600_000_000))
    for f in (device / SESSION / "logs").rglob("*"):
        os.utime(f, (1_600_000_000, 1_600_000_000))
    step(node)
    assert summary(hub, env["dev"]["id"])["counts"]["history"]["episodes"] == 1
    assert run_sync(hub) == []
    hub.update_settings(env["dev"]["id"], {"include_history": 1})
    assert run_sync(hub)


def test_admin_http_api(env):
    import urllib.request
    hub = env["hub"]
    make_episode(env["device"], "pick", 1)
    step(env["node"])
    # admin listener is the second server
    srv = [t for t in __import__("threading").enumerate() if t.name == "admin-http"]
    assert srv
    ov = hub.overview()
    assert ov["totals"]["devices"] == 1 and ov["totals"]["pending_episodes"] == 1
    assert ov["server_disk"]["level"] in ("ok", "warn", "critical")


def test_robocurate_follows_hub_changes(env, tmp_path):
    """Synced sessions appear in the RoboCurate catalog incrementally, tagged with the device."""
    studio = pytest.importorskip("studio")
    import hub_client
    hub, node, device = env["hub"], env["node"], env["device"]
    make_episode(device, "pick", 1)
    step(node)
    run_sync(hub)
    admin_port = [s for s in __import__("gc").get_objects()
                  if isinstance(s, hubmod.ThreadingHTTPServer) and s.RequestHandlerClass.admin
                  and s.RequestHandlerClass.hub is hub][0].server_address[1]
    hub_client.configure(f"http://127.0.0.1:{admin_port}")
    hub_client.state.update(cursor=None, mirror_root=None)
    lib = studio.StudioLibrary(tmp_path / "ws")
    hub_client.follow_once(lib)            # first contact: full scan of the mirror
    eps = [lib.refs[i] for i in lib.order]
    assert len(eps) == 1 and eps[0].device == "G1-01" and eps[0].task == "pick"
    make_episode(device, "pick", 2)
    step(node)
    run_sync(hub)
    res = hub_client.follow_once(lib)      # incremental: only the changed session is re-read
    assert res and res["added"] == 1 and len(lib.order) == 2
    # proxy: allowed and refused actions
    assert hub_client.post_allowed("devices/abcd1234/settings")
    assert not hub_client.post_allowed("devices/abcd1234/../../x")
    status, ov = hub_client.forward("GET", "overview")
    assert status == 200 and ov["totals"]["devices"] == 1
    hub_client.configure("http://127.0.0.1:9")
    assert hub_client.forward("GET", "overview")[0] == 503


def test_hub_rejects_paths_outside_the_mirror(env):
    """A buggy or hostile node must not make the hub copy, trash or verify files outside its mirror."""
    hub, dev = env["hub"], env["dev"]
    good = {"rel": "s/data/u/t/episode_000001", "files": [["s/data/u/t/episode_000001/mcap/mcap_0.mcap", 1, 1],
                                                          ["s/logs/u/t/episode_000001/x.log", 1, 1]]}
    assert hubmod.valid_manifest_episode(good)
    bad = [
        {"rel": "../../etc/data/u/t/episode_000001", "files": []},
        {"rel": "/abs/data/u/t/episode_000001", "files": []},
        {"rel": "s/data/u/t/not_an_episode", "files": []},
        {"rel": "s/data/u/t/episode_000001", "files": [["s/data/u/t/episode_000001/../../../../x", 1, 1]]},
        {"rel": "s/data/u/t/episode_000001", "files": [["other/place", 1, 1]]},
    ]
    for ep in bad:
        assert not hubmod.valid_manifest_episode(ep), ep
    hb = {"device_id": "machine-A", "time": time.time(), "manifest": {
        "full": True, "seq": 1, "base": None, "removed": [],
        "upserts": [dict(ep, fp="x", bytes=1, session_rel="s", user="u", task="t", episode="e") for ep in bad]}}
    hub.heartbeat(hub.device_row(dev["id"]), hb, "127.0.0.1")
    assert hub.store.one("SELECT COUNT(*) AS n FROM episodes")["n"] == 0
