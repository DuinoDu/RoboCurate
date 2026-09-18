"""Read-only audit of the supplied handoff and current local input metadata.

Run with Python 3.10+: python3 research/warp_rm_20260916/audit_inputs.py
Only writes audit artifacts beside this script. Does not train or score models.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import zipfile


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
ZIP = ROOT / "HoloCurate-团队交接-20260915-210906.zip"
APP = ROOT / "robot-data-studio"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    changed, missing = [], []
    with zipfile.ZipFile(ZIP) as archive:
        catalog = json.loads(archive.read("HoloCurate/handoff/data-catalog.json"))
        for member in archive.infolist():
            if member.is_dir():
                continue
            rel = Path(member.filename).relative_to("HoloCurate")
            if rel.parts[0] in ("handoff", "workspace") or rel.name == "PACKAGE_MANIFEST.json":
                continue
            local = APP / rel
            if not local.is_file():
                missing.append(str(rel))
            elif hashlib.sha256(local.read_bytes()).digest() != hashlib.sha256(archive.read(member)).digest():
                changed.append(str(rel))
    formal = [e for e in catalog["episodes"] if Path(e["root"]).name == "2026_09_10-17_40_44"]
    rows, raw_mismatch, sidecar_mismatch = [], [], []
    for ep in formal:
        rating = ep["quality"]["rating"]
        raw = Path(ep["path"])
        present = raw.is_file()
        if not present or raw.stat().st_size != ep["size_bytes"]:
            raw_mismatch.append(ep["episode_id"])
        sidecar = Path(ep["sidecar"])
        body = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        if body.get("task_outcome", {}).get("success") != ep["outcome"]:
            sidecar_mismatch.append(ep["episode_id"])
        rows.append({
            "id": ep["id"], "episode_id": ep["episode_id"], "duration_s": ep["duration_s"],
            "camera_counts": {k: v["count"] for k, v in ep["cameras"].items()},
            "rating": {k: rating[k] for k in ("version", "score", "state", "grade_gate")},
            "outcome": ep["outcome"], "raw_present": present,
        })
    db = APP / "workspace/reviews.sqlite3"
    with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as connection:
        saved = connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        history = connection.execute("SELECT COUNT(*) FROM audit").fetchone()[0]
    checksum = sha256(ZIP)
    expected = Path(str(ZIP) + ".sha256").read_text().split()[0]
    summary = {
        "scope": "Handoff catalog statistics, source comparison and read-only raw-file/sidecar checks; no model inference",
        "handoff_zip": ZIP.name, "handoff_sha256": checksum, "expected_sha256": expected,
        "handoff_sha256_matches": checksum == expected,
        "source_files_different_from_handoff": changed, "source_files_missing": missing,
        "catalog_counts": catalog["counts"], "formal_count": len(formal),
        "total_duration_s": sum(e["duration_s"] for e in formal),
        "duration_range_s": [min(e["duration_s"] for e in formal), max(e["duration_s"] for e in formal)],
        "durations_less_than_46_5_s": sum(e["duration_s"] < 46.5 for e in formal),
        "tasks": dict(Counter(e["task"] for e in formal)),
        "instructions": dict(Counter(e["instruction"] for e in formal)),
        "outcomes": dict(Counter(str(e["outcome"]) for e in formal)),
        "operator_notes": dict(Counter(e["operator_note"] for e in formal)),
        "head_camera_modes": dict(Counter(e.get("head_camera_mode") for e in formal)),
        "technical_score_range": [min(e["rating"]["score"] for e in rows), max(e["rating"]["score"] for e in rows)],
        "camera_frame_counts": {key: sum(e["cameras"][key]["count"] for e in formal)
                                for key in ("head", "left_wrist", "right_wrist")},
        "raw_files_present": sum(e["raw_present"] for e in rows),
        "raw_missing_or_size_mismatch": raw_mismatch,
        "sidecar_outcome_mismatch": sidecar_mismatch,
        "current_saved_reviews": saved, "current_review_history": history,
        "execution_environment_nvidia_smi": shutil.which("nvidia-smi"),
        "execution_environment_nvidia_devices": [str(p) for p in Path("/dev").glob("nvidia*")],
        "episodes": rows,
    }
    (OUT / "local_data_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (OUT / "episodes.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode_id", "duration_s", "technical_score", "grade", "operator_outcome"])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(episode_id=row["episode_id"], duration_s=row["duration_s"],
                                 technical_score=row["rating"]["score"], grade=row["rating"]["grade_gate"],
                                 operator_outcome=row["outcome"]))
    sources = []
    for path in sorted((OUT / "sources").rglob("*")):
        if path.is_file():
            sources.append(dict(path=str(path.relative_to(OUT)), bytes=path.stat().st_size, sha256=sha256(path)))
    (OUT / "source_file_checksums.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("episodes", "instructions")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
