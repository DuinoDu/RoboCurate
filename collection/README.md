# collection-hub / collection-node — recording-device management and sync

[中文](README.zh-CN.md)

Keeps track of the robots that record data and pulls finished episodes to the storage server that runs RoboCurate.

```
recording device ×N   collection-node  (user systemd unit: Nice=10, idle I/O, CPUQuota=5%, MemoryMax=64M)
  ├ every 1 s    marker mcap/__RECORDING__ + MCAP growth of recent episodes → immediate heartbeat on start/stop
  ├ every 5–15 s PICO episode server /status (a hint only)
  ├ every 30 s   free space + incremental manifest (files, sizes, mtimes, sidecar summary)
  └ heartbeat    POST :8422/api/node/heartbeat  (Bearer token)
storage server        collection-hub  (separate process, hub.sqlite3)
  ├ online / recording / anomaly state, disk levels on both ends, alerts (optional webhook)
  ├ rsync scheduler (SSH, read-only rrsync on the device) → <mirror_root>/<device>/<device tree>
  ├ per-file verification (size + mtime) → change cursor  GET /api/changes
  └ admin API on 127.0.0.1:8423 (RoboCurate proxies it under /api/hub/*)
storage server        RoboCurate: 设备管理 page, global disk banner; re-scans only the sessions that were synced
```

The node uses only the Python standard library, so installing it is copying one file. The hub is a separate process so that restarting the review app never interrupts monitoring or a transfer, and so that RoboCurate holds no device credentials.

## Rules

They follow lifecycle tests against the PICO episode server and the MCAP recorder (normal stop, stop then label, delete, recorder crash, server restart during recording).

| Decision | Rule |
|---|---|
| Recording | `__RECORDING__` exists and the MCAP grows (15 s grace after start), or the MCAP grows without a marker (an orphaned recorder after a server restart). `/status` `recorder.active` is shown and compared, never trusted: it stays true after a recorder crash and reads false while an orphaned recorder writes |
| Interrupted recording | marker present but the MCAP has not grown for 30 s → anomaly alert; synced as-is after 10 min |
| Ready to sync | `mcap/metadata.yaml` exists, no marker, and the episode is no longer current or has been quiet for `settle_seconds` (default 120 s — the operator can still delete the current episode). Failed starts that left only sidecars are synced too |
| Re-sync | whenever an episode's file fingerprint changes (labels and metadata updates rewrite the JSON sidecars); an unchanged MCAP is not transferred again |
| Deleted on the device | the synced copy moves to `<mirror_root>/.trash/`, it is never deleted |
| Recorded before registration | not synced unless enabled for the device |
| When | never while the device records; `idle_minutes` (default 10) after its last recording, immediately once marked off-shift or "sync now". While any device records, the total rate is capped at `busy_bwlimit_mb` (default 15 MB/s, split between jobs) to protect teleoperation Wi-Fi; otherwise full speed. A policy change restarts jobs, which resume with `--partial` |
| Capacity | no new transfer when the server's free space would drop below `reserve_gb` |
| Safe to power off | online, not recording, nothing awaiting an outcome label, no conversion running, nothing pending / settling / half-written, no sync job |
| Clock skew | measured from each heartbeat; device-clock mtimes are compared with the device's clock |

## Install

Storage server (user-level service; enable lingering so it starts at boot):

```bash
collection/deploy/install_hub.sh --mirror-root /data/recordings/raw --reserve-gb 100
cp collection/deploy/robocurate.service ~/.config/systemd/user/   # optional: RoboCurate as a service
```

In RoboCurate open **设备管理 → 添加设备**. It returns a one-time token and the command to run on the device, of the form:

```bash
curl -fsSL http://<hub-host>:8422/install/install_node.sh | bash -s -- \
  --hub http://<hub-host>:8422 --token <token> --hub-key 'ssh-ed25519 AAAA… collection-hub@<hub-host>'
```

It installs the `collection-node` user service and adds the hub's key to `authorized_keys` restricted to `rrsync -ro <data root>`. From the command line: `python3 collection/hub.py --config ~/.config/collection-hub/config.json add-device --name G1-01 --ssh-target <user>@<robot-host>`.

Optional node capabilities (off by default):

* `--allow-shutdown` — "power off after sync". Needs `<user> ALL=(root) NOPASSWD: /sbin/shutdown` in sudoers.
* `--allow-cleanup` — "delete synced data on the device". The hub builds a preview from copies it has re-verified file by file and requires the confirmation code; the node deletes only episodes whose files are unchanged since and that are neither current nor recording.

## Configuration (`~/.config/collection-hub/config.json`)

`mirror_root`, `reserve_gb`, `max_concurrent`, `busy_bwlimit_mb`, `idle_minutes`, `settle_seconds`, `batch_max_gb`, `batch_max_episodes`, `default_write_rate_gb_h`, `default_daily_intake_gb` (used until three days of intake are measured), `device_warn_hours` / `device_critical_hours`, `server_warn_days` / `server_critical_days`, `webhook` with `webhook_format` `feishu` | `generic`, `node_listen`, `admin_listen`.

## Security

The node endpoint (`:8422`) is meant for a trusted collection LAN: heartbeats need a per-device token (stored hashed), the install files are public. The admin API listens on loopback only; RoboCurate forwards it with its own same-origin rules, so anyone who can open RoboCurate can also control sync — see [SECURITY.md](../SECURITY.md). Manifest paths from nodes are validated, and rsync does not copy symlinks or device files.

## Operations

```bash
systemctl --user status collection-hub;  journalctl --user -u collection-hub -f    # server
systemctl --user status collection-node; journalctl --user -u collection-node -f   # device
python3 collection/hub.py --config ~/.config/collection-hub/config.json list
python3 ~/.local/share/collection-node/node.py --config ~/.config/collection-node/config.json --once
```

Tests: `python -m pytest tests/test_collection.py` runs a real hub, node and rsync (local transport) against temporary directories.
