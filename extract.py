#!/usr/bin/env python3
"""Extract a v1.4 collection MCAP episode into a web-ready visualization cache.

Raw episodes are large (hundreds of MB) and every message must be decoded by a
pure-Python ROS 2 decoder, so decoding on each page load is not viable. This
module does one indexed pass over the file and writes a compact cache:

    meta.json    episode metadata, per-topic QC statistics, channel
                 descriptors, discrete state bands, and the camera frame
                 timetable
    series.f32   little-endian float32, channel block by channel block; within
                 a block a single series is contiguous (``data[s * n + i]``)

JPEG frames are deliberately *not* copied into the cache. The MCAP chunk index
makes a windowed read of one frame take ~20 ms, so `server.py` serves images
straight from the raw episode and the cache stays a few MB instead of ~500.

The raw episode is only ever opened for reading; nothing here mutates it.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from g1_joints import (
    BODY_DOF,
    BODY_JOINT_NAMES,
    GROUP_OF_INDEX,
    HAND_DOF,
    HAND_JOINT_NAMES,
    JOINT_GROUPS,
    REF_ROOT_POS,
    REF_ROOT_QUAT,
)
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

CACHE_VERSION = 4

# Canonical renamed topics produced by the v1.4 recorder config.
T_LOWSTATE = "/observations/robot_state/lowstate"
T_LOWCMD = "/action/lowcmd"
T_ROBOT_STATE = "/observations/robot_state/state"
T_POLICY_MODE = "/observations/policy/mode"
T_ACTION = "/action/humanoid"
T_KPS = "/action/humanoid/kps"
T_KDS = "/action/humanoid/kds"
T_REFERENCE = "/observations/holomotion/local_retarget_frame"
T_INSTRUCTION = "/task/instruction"
T_CAM_IMAGE = "/observations/camera/head/color/image"
T_CAM_INFO = "/observations/camera/head/color/camera_info"

HAND_STATE = {
    "left": "/observations/brainco/left/state",
    "right": "/observations/brainco/right/state",
}
HAND_CMD = {
    "left": "/action/brainco/left/cmd",
    "right": "/action/brainco/right/cmd",
}

# Provenance shown next to each stream so a reader does not mistake a
# diagnostic-only topic for the dataset's action source. Resolved from
# convert_mcap_to_holobrain.py's consumed-topic set.
TOPIC_NOTES = {
    T_LOWSTATE: (
        "Dataset source: joints (motor q) + virtual_root (IMU roll/pitch)."
    ),
    T_ACTION: (
        "Dataset action source. Absolute joint targets in rad, native order."
    ),
    T_REFERENCE: (
        "Dataset source for base_pose + root action tail. Its dof_pos[29] "
        "is NOT written to the dataset."
    ),
    T_LOWCMD: (
        "Diagnostic only — the v1.4 converter never reads /action/lowcmd."
    ),
    T_CAM_IMAGE: (
        "Dataset source: cover-resized and centre-cropped to 640x360."
    ),
    T_CAM_INFO: "Dataset source: intrinsics, rescaled with the image.",
    T_INSTRUCTION: "Episode instruction; must be locked to a single value.",
    T_ROBOT_STATE: "Controller state machine.",
    T_POLICY_MODE: "Policy mode transitions.",
    T_KPS: "Static policy gains.",
    T_KDS: "Static policy gains.",
}
for _side in ("left", "right"):
    TOPIC_NOTES[HAND_STATE[_side]] = (
        "Dataset source: median of index/middle/ring/pinky -> "
        f"{_side} gripper."
    )
    TOPIC_NOTES[HAND_CMD[_side]] = (
        f"Dataset source: commanded {_side} gripper closure."
    )

# Topics whose payload we never need to decode during the telemetry pass.
SKIP_DECODE = {T_CAM_IMAGE}


@dataclass
class Track:
    """Accumulates one multi-series signal sampled at its own native rate."""

    key: str
    label: str
    unit: str
    names: list[str]
    group: str
    hold: bool = False  # commands hold the previous sample; states interpolate
    times: list[int] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)

    def add(self, t_ns: int, row) -> None:
        self.times.append(t_ns)
        self.values.append(np.asarray(row, dtype=np.float32))

    @property
    def width(self) -> int:
        return len(self.names)


def _percentile(sorted_vals: np.ndarray, q: float) -> float:
    if sorted_vals.size == 0:
        return 0.0
    return float(np.percentile(sorted_vals, q))


def _timing_stats(times_ns: np.ndarray, duration_s: float) -> dict:
    """Per-topic timing QC: rate, jitter, worst gap, monotonicity."""
    count = int(times_ns.size)
    stats = {
        "count": count,
        "rate_hz": round(count / duration_s, 3) if duration_s > 0 else 0.0,
        "monotonic": True,
        "mean_dt_ms": 0.0,
        "p99_dt_ms": 0.0,
        "max_gap_ms": 0.0,
        "max_gap_at_s": 0.0,
    }
    if count < 2:
        return stats
    diffs = np.diff(times_ns.astype(np.int64))
    stats["monotonic"] = bool(np.all(diffs >= 0))
    dt_ms = diffs.astype(np.float64) / 1e6
    worst = int(np.argmax(dt_ms))
    stats["mean_dt_ms"] = round(float(dt_ms.mean()), 4)
    stats["p99_dt_ms"] = round(_percentile(dt_ms, 99.0), 4)
    stats["max_gap_ms"] = round(float(dt_ms[worst]), 4)
    stats["max_gap_at_s"] = round(
        float((times_ns[worst] - times_ns[0]) / 1e9), 4
    )
    return stats


def _resample(track: Track, grid_ns: np.ndarray) -> np.ndarray:
    """Put a native-rate track on the uniform grid.

    States and references are linearly interpolated; commands are held from the
    previous sample, mirroring `contract.json`'s resampling policy. Samples
    outside the track's own coverage become NaN so the frontend can draw a gap
    instead of a fabricated flat line.
    """
    n = grid_ns.size
    out = np.full((track.width, n), np.nan, dtype=np.float32)
    if not track.times:
        return out
    src_t = np.asarray(track.times, dtype=np.int64)
    src_v = np.vstack(track.values).astype(np.float32)
    order = np.argsort(src_t, kind="stable")
    src_t, src_v = src_t[order], src_v[order]

    inside = (grid_ns >= src_t[0]) & (grid_ns <= src_t[-1])
    if not np.any(inside):
        return out
    g = grid_ns[inside]
    if track.hold:
        idx = np.searchsorted(src_t, g, side="right") - 1
        np.clip(idx, 0, src_t.size - 1, out=idx)
        out[:, inside] = src_v[idx].T
    else:
        tf = src_t.astype(np.float64)
        gf = g.astype(np.float64)
        for s in range(track.width):
            out[s, inside] = np.interp(gf, tf, src_v[:, s]).astype(np.float32)
    return out


def _bands(times_ns: list[int], labels: list[str], t0: int) -> list[dict]:
    """Run-length encode a string topic into contiguous spans."""
    out: list[dict] = []
    for t, label in zip(times_ns, labels, strict=True):
        rel = (t - t0) / 1e9
        if out and out[-1]["value"] == label:
            continue
        if out:
            out[-1]["end_s"] = round(rel, 4)
        out.append({"value": label, "start_s": round(rel, 4), "end_s": None})
    return out


def _make_tracks() -> dict[str, Track]:
    body = list(BODY_JOINT_NAMES)
    hands = list(HAND_JOINT_NAMES)
    t: dict[str, Track] = {}

    def add(key, label, unit, names, group, hold=False):
        t[key] = Track(key, label, unit, list(names), group, hold)

    add("state.q", "Measured position", "rad", body, "body")
    add("state.dq", "Measured velocity", "rad/s", body, "body")
    add("state.tau", "Estimated torque", "N·m", body, "body")
    add("state.temp", "Motor temperature", "°C", body, "body")
    add(
        "cmd.q",
        "Low-level target (diagnostic)",
        "rad",
        body,
        "body",
        hold=True,
    )
    add("cmd.tau", "Feed-forward torque", "N·m", body, "body", hold=True)
    add("cmd.kp", "Position gain", "-", body, "body", hold=True)
    add("cmd.kd", "Damping gain", "-", body, "body", hold=True)
    add("action.q", "Policy action", "rad", body, "body", hold=True)
    add("ref.dof", "Retarget reference", "rad", body, "body")

    add("imu.rpy", "IMU orientation", "rad", ["roll", "pitch", "yaw"], "imu")
    add("imu.gyro", "Angular rate", "rad/s", ["x", "y", "z"], "imu")
    add("imu.accel", "Linear acceleration", "m/s²", ["x", "y", "z"], "imu")
    add("imu.quat", "IMU quaternion", "-", list(REF_ROOT_QUAT), "imu")

    add(
        "ref.root_pos",
        "Reference root position",
        "m",
        list(REF_ROOT_POS),
        "reference",
    )
    add(
        "ref.root_quat",
        "Reference root rotation",
        "-",
        list(REF_ROOT_QUAT),
        "reference",
    )
    add("teleop.pico", "PICO link health", "-", ["fps", "dt_ms"], "teleop")
    add(
        "teleop.latency",
        "Reference latency",
        "ms",
        ["source_to_telemetry"],
        "teleop",
    )

    add("power.voltage", "Bus voltage", "V", ["voltage"], "power")

    for side in ("left", "right"):
        # BrainCo q is a normalised 0=open..1=closed grip fraction, not
        # radians, and `tau_est` is raw motor current telemetry (amps/1000)
        # rather than a
        # calibrated torque. Labelling them as rad/N·m would invite false
        # comparison against the body joints.
        add(
            f"hand.{side}.state_q",
            f"{side.title()} hand closure",
            "0=open 1=closed",
            hands,
            "hand",
        )
        add(
            f"hand.{side}.state_tau",
            f"{side.title()} hand motor current",
            "A (uncalibrated)",
            hands,
            "hand",
        )
        add(
            f"hand.{side}.cmd_q",
            f"{side.title()} hand command",
            "0=open 1=closed",
            hands,
            "hand",
            hold=True,
        )
    return t


def build_cache(
    mcap_path: Path,
    out_dir: Path,
    rate_hz: float = 100.0,
    progress=lambda msg: None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_stat = mcap_path.stat()
    tracks = _make_tracks()

    with mcap_path.open("rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()
        if summary is None or summary.statistics is None:
            raise ValueError(
                f"{mcap_path} has no summary section; the recorder "
                "transaction "
                "likely did not close. Treat it as diagnostic-only."
            )
        stats = summary.statistics
        t0, t1 = stats.message_start_time, stats.message_end_time
        duration_s = (t1 - t0) / 1e9
        if duration_s <= 0:
            raise ValueError("Episode has non-positive duration.")

        by_topic = {}
        for cid, ch in summary.channels.items():
            schema = summary.schemas.get(ch.schema_id)
            by_topic[ch.topic] = {
                "topic": ch.topic,
                "schema": schema.name if schema else "unknown",
                "count": int(stats.channel_message_counts.get(cid, 0)),
            }

        raw_times: dict[str, list[int]] = {t: [] for t in by_topic}
        instruction_msgs: list[tuple[int, str]] = []
        mode_t: list[int] = []
        mode_v: list[str] = []
        rstate_t: list[int] = []
        rstate_v: list[str] = []
        gains = {"kp": None, "kd": None}
        camera_info = None

        image_topics = [t for t, info in by_topic.items()
                        if info["schema"] == "sensor_msgs/msg/CompressedImage"]
        decode_topics = [t for t in by_topic if t not in image_topics]
        total = sum(by_topic[t]["count"] for t in decode_topics)
        progress(
            f"decoding {total} messages across {len(decode_topics)} topics"
        )

        seen = 0
        last_report = time.time()
        for _schema, channel, message, raw in reader.iter_decoded_messages(
            topics=decode_topics
        ):
            topic = channel.topic
            seen += 1
            if time.time() - last_report > 3.0:
                progress(
                    f"  {seen}/{total} messages "
                    f"({100 * seen / max(total, 1):.0f}%)"
                )
                last_report = time.time()
            # Every track is keyed off the recorder log clock. The recorder's
            # stamp_type policy already decides what that clock means per
            # topic,
            # and it is the only stamp present on *every* message, so using it
            # uniformly is what keeps heterogeneous streams mutually aligned.
            _dispatch(
                topic,
                raw,
                int(message.log_time),
                tracks,
                raw_times,
                instruction_msgs,
                mode_t,
                mode_v,
                rstate_t,
                rstate_v,
                gains,
            )
            if topic == T_CAM_INFO and camera_info is None:
                camera_info = {
                    "width": int(raw.width),
                    "height": int(raw.height),
                    "distortion_model": str(raw.distortion_model),
                    "k": [float(x) for x in raw.k],
                    "d": [float(x) for x in raw.d],
                    "frame_id": str(raw.header.frame_id),
                }

        progress("indexing camera frames")
        cameras = {t: [] for t in image_topics}
        for _, channel, message in reader.iter_messages(topics=image_topics):
            cameras[channel.topic].append(int(message.log_time))
        for times in cameras.values():
            times.sort()
        cam_times = cameras.get(T_CAM_IMAGE, [])

    # The decoded pass keys everything off log_time, recorded per message.
    progress("resampling onto uniform grid")
    n = max(int(math.floor(duration_s * rate_hz)) + 1, 2)
    grid_ns = t0 + (np.arange(n, dtype=np.int64) * int(1e9 / rate_hz))
    grid_ns = np.clip(grid_ns, t0, t1)

    channels: dict[str, dict] = {}
    blocks: list[np.ndarray] = []
    offset = 0
    for key, track in tracks.items():
        if not track.times:
            continue
        block = _resample(track, grid_ns)
        blocks.append(block.astype("<f4").reshape(-1))
        finite = np.isfinite(block)
        channels[key] = {
            "key": key,
            "label": track.label,
            "unit": track.unit,
            "group": track.group,
            "names": track.names,
            "series": track.width,
            "offset": offset,
            "samples": n,
            "hold": track.hold,
            "coverage": round(float(finite.any(axis=0).mean()), 4),
            "min": [
                None
                if not finite[s].any()
                else round(float(np.nanmin(block[s])), 6)
                for s in range(track.width)
            ],
            "max": [
                None
                if not finite[s].any()
                else round(float(np.nanmax(block[s])), 6)
                for s in range(track.width)
            ],
        }
        offset += track.width * n

    series_path = out_dir / "series.f32"
    with series_path.open("wb") as fh:
        for block in blocks:
            fh.write(block.tobytes())

    topics_meta = []
    for topic, info in sorted(by_topic.items()):
        times = np.asarray(
            cameras[topic] if topic in cameras else raw_times.get(topic, []),
            dtype=np.int64,
        )
        entry = dict(info)
        entry.update(_timing_stats(times, duration_s))
        entry["count"] = info["count"]
        entry["note"] = TOPIC_NOTES.get(topic, "")
        topics_meta.append(entry)

    instruction = instruction_msgs[-1][1] if instruction_msgs else None
    instructions = sorted({text for _, text in instruction_msgs})

    meta = {
        "cache_version": CACHE_VERSION,
        "source": {
            "path": str(mcap_path.resolve()),
            "name": mcap_path.name,
            "size_bytes": mcap_path.stat().st_size,
            "sha1_head": _head_digest(mcap_path),
        },
        "episode": {
            "start_ns": int(t0),
            "end_ns": int(t1),
            "duration_s": round(duration_s, 4),
            "message_count": int(stats.message_count),
            "chunk_count": int(stats.chunk_count),
            "topic_count": len(by_topic),
            "instruction": instruction,
            "instructions": instructions,
            "instruction_locked": len(instructions) <= 1,
        },
        "timeline": {
            "rate_hz": rate_hz,
            "samples": n,
            "t0_ns": int(t0),
            "step_ns": int(1e9 / rate_hz),
        },
        "topics": topics_meta,
        "channels": channels,
        "joints": {
            "names": list(BODY_JOINT_NAMES),
            "groups": [
                {"name": g, "start": s, "stop": e} for g, s, e in JOINT_GROUPS
            ],
            "group_of_index": list(GROUP_OF_INDEX),
            "hand_names": list(HAND_JOINT_NAMES),
        },
        "gains": gains,
        "camera": {
            "topic": T_CAM_IMAGE,
            "count": len(cam_times),
            "info": camera_info,
            "times_s": [round((t - t0) / 1e9, 5) for t in cam_times],
        },
        "bands": {
            "policy_mode": _bands(mode_t, mode_v, t0),
            "robot_state": _bands(rstate_t, rstate_v, t0),
        },
        "series_bytes": offset * 4,
    }
    for band_list in meta["bands"].values():
        if band_list:
            band_list[-1]["end_s"] = round(duration_s, 4)

    # Native timestamps stay int64; training exports never resample from the
    # float32 display cache. All camera streams retain their own timestamps.
    native = {}
    for key, track in tracks.items():
        if track.times:
            native[key + ".t"] = np.asarray(track.times, dtype=np.int64)
            native[key + ".v"] = np.vstack(track.values).astype(np.float32)
    meta["cameras"] = {}
    for topic, times in cameras.items():
        name = topic.split("/camera/")[-1].split("/")[0]
        native["camera." + name + ".t"] = np.asarray(times, dtype=np.int64)
        meta["cameras"][name] = {"topic": topic, "count": len(times),
                                 "times_s": [(t - t0) / 1e9 for t in times]}
    for topic, times in raw_times.items():
        native["topic:" + topic] = np.asarray(times, dtype=np.int64)
    with (out_dir / "native.tmp").open("wb") as fh:
        np.savez_compressed(fh, **native)
    (out_dir / "native.tmp").replace(out_dir / "native.npz")
    final_stat = mcap_path.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise ValueError("Source MCAP changed during extraction; please retry after recording has stopped.")
    meta["source"]["mtime_ns"] = source_stat.st_mtime_ns
    tmp = out_dir / "meta.tmp"
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    tmp.replace(out_dir / "meta.json")
    progress(f"wrote {series_path} ({offset * 4 / 1e6:.1f} MB)")
    return meta


def _dispatch(
    topic,
    raw,
    ts,
    tracks,
    raw_times,
    instruction_msgs,
    mode_t,
    mode_v,
    rstate_t,
    rstate_v,
    gains,
):
    """Route one decoded message into its track(s). `ts` is the log clock."""
    if topic == T_LOWSTATE:
        motors = raw.motor_state[:BODY_DOF]
        tracks["state.q"].add(ts, [m.q for m in motors])
        tracks["state.dq"].add(ts, [m.dq for m in motors])
        tracks["state.tau"].add(ts, [m.tau_est for m in motors])
        tracks["state.temp"].add(ts, [max(m.temperature) for m in motors])
        imu = raw.imu_state
        tracks["imu.rpy"].add(ts, list(imu.rpy))
        tracks["imu.gyro"].add(ts, list(imu.gyroscope))
        tracks["imu.accel"].add(ts, list(imu.accelerometer))
        tracks["imu.quat"].add(ts, list(imu.quaternion))
        tracks["power.voltage"].add(ts, [raw.motor_state[0].vol])
        raw_times[topic].append(ts)
        return

    if topic == T_LOWCMD:
        cmds = raw.motor_cmd[:BODY_DOF]
        tracks["cmd.q"].add(ts, [c.q for c in cmds])
        tracks["cmd.tau"].add(ts, [c.tau for c in cmds])
        tracks["cmd.kp"].add(ts, [c.kp for c in cmds])
        tracks["cmd.kd"].add(ts, [c.kd for c in cmds])
        raw_times[topic].append(ts)
        return

    if topic == T_ACTION:
        data = list(raw.data)[:BODY_DOF]
        if len(data) == BODY_DOF:
            tracks["action.q"].add(ts, data)
        raw_times[topic].append(ts)
        return

    if topic == T_REFERENCE:
        qpos = list(raw.reference_qpos)
        if len(qpos) >= 36:
            tracks["ref.root_pos"].add(ts, qpos[0:3])
            tracks["ref.root_quat"].add(ts, qpos[3:7])
            tracks["ref.dof"].add(ts, qpos[7:36])
        tracks["teleop.pico"].add(ts, [raw.pico_fps, raw.pico_dt * 1000.0])
        latency_ms = (
            raw.telemetry_timestamp_realtime - raw.source_timestamp_realtime
        ) * 1000.0
        tracks["teleop.latency"].add(ts, [latency_ms])
        raw_times[topic].append(ts)
        return

    if topic in (T_KPS, T_KDS):
        key = "kp" if topic == T_KPS else "kd"
        gains[key] = [round(float(v), 4) for v in list(raw.data)]
        raw_times[topic].append(ts)
        return

    if topic == T_INSTRUCTION:
        instruction_msgs.append((ts, str(raw.data)))
        raw_times[topic].append(ts)
        return

    if topic == T_POLICY_MODE:
        mode_t.append(ts)
        mode_v.append(str(raw.data))
        raw_times[topic].append(ts)
        return

    if topic == T_ROBOT_STATE:
        rstate_t.append(ts)
        rstate_v.append(str(raw.data))
        raw_times[topic].append(ts)
        return

    for side, name in HAND_STATE.items():
        if topic == name:
            states = raw.states[:HAND_DOF]
            tracks[f"hand.{side}.state_q"].add(ts, [m.q for m in states])
            tracks[f"hand.{side}.state_tau"].add(
                ts, [m.tau_est for m in states]
            )
            raw_times[topic].append(ts)
            return
    for side, name in HAND_CMD.items():
        if topic == name:
            cmds = raw.cmds[:HAND_DOF]
            tracks[f"hand.{side}.cmd_q"].add(ts, [c.q for c in cmds])
            raw_times[topic].append(ts)
            return

    raw_times.setdefault(topic, []).append(ts)


def _head_digest(path: Path, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        h.update(fh.read(nbytes))
    return h.hexdigest()[:16]


def cache_dir_for(mcap_path: Path, root: Path) -> Path:
    stem = mcap_path.stem
    digest = hashlib.sha1(str(mcap_path.resolve()).encode()).hexdigest()[:10]
    return root / f"{stem}-{digest}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mcap", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rate", type=float, default=100.0)
    args = ap.parse_args()

    out = args.out or cache_dir_for(args.mcap, Path(".mcap_viz_cache"))
    build_cache(
        args.mcap, out, args.rate, progress=lambda m: print(m, flush=True)
    )
    print(f"cache ready: {out}")


if __name__ == "__main__":
    sys.exit(main())
