"""Build a source/model/evidence handoff; runtime state is admitted by whitelist only."""
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import stat
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research/warp_rm_20260916"
APP = ROOT / "robot-data-studio"
HEAD_SHA = "a17751a7a566da2a60f976452c2d45a2ceb37142ab8c9e3752ee3904ef4658f4"
BACKBONE_SHA = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
SKIP = {".venv", "workspace", ".git", ".agents", ".codex", ".cache", ".pytest_cache",
        "__pycache__", "node_modules"}
TEXT_SUFFIXES = {".py", ".js", ".mjs", ".json", ".md", ".txt", ".sh", ".html", ".css", ".xml"}
SECRET_PATTERNS = [re.compile(rb"hf_[A-Za-z0-9]{25,}"),
                   re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{30,}"),
                   re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def selected_files(folder):
    for current, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith(".")
                         and not (Path(current)/d).is_symlink())
        for name in sorted(files):
            p = Path(current)/name
            if not p.is_symlink() and not name.endswith((".pyc", ".pyo")):
                if not name.startswith(".") or name == ".gitignore":
                    yield p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    built = datetime.now().astimezone()
    target = args.output or ROOT / ("HoloCurate-WARP方法融合-"+built.strftime("%Y%m%d-%H%M%S")+".zip")
    if target.exists():
        raise FileExistsError(target)
    payload = {}
    disk_inputs = {}

    def add(path, name=None):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symbolic input: {path}")
        name = name or path.relative_to(ROOT).as_posix()
        if name in payload or name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"Invalid or duplicate archive name: {name}")
        data = path.read_bytes()
        if path.suffix in TEXT_SUFFIXES and any(p.search(data) for p in SECRET_PATTERNS):
            raise ValueError(f"Credential-like content detected; inspect locally: {name}")
        payload[name] = data
        disk_inputs[path] = digest(data)

    project_files = list(selected_files(APP))
    for path in project_files:
        add(path)

    with zipfile.ZipFile(RESEARCH/"implementation_baseline.zip") as baseline:
        old = {n: digest(baseline.read(n)) for n in baseline.namelist() if not n.endswith("/")}
    current = {p.relative_to(APP).as_posix(): disk_inputs[p] for p in project_files}
    missing = sorted(set(old)-set(current))
    if missing:
        raise ValueError(f"Baseline source files missing: {missing}")
    preserved = ["scripts/build_preview_meshes.py", "static/solid-canvas.js", "static/viewer/g1-preview.json"]
    assert all(old[n] == current[n] for n in preserved), "Pre-existing source edits were changed"
    audit = dict(baseline="implementation_baseline.zip", baseline_sha256=digest((RESEARCH/"implementation_baseline.zip").read_bytes()),
                 baseline_file_count=len(old), current_source_file_count=len(current), missing=missing,
                 changed_existing=[dict(path=n, before_sha256=old[n], after_sha256=current[n])
                                   for n in sorted(old) if old[n] != current[n]],
                 added=sorted(set(current)-set(old)),
                 preexisting_edits_preserved=preserved,
                 scope="Current source snapshot, including existing UI/3D updates; changes are not all attributed to WARP work")
    (RESEARCH/"handoff-source-audit.json").write_bytes(encoded(audit))

    model_dir = APP/"workspace/warp/models"
    for rel in ["open-dinov2-pilot-a17751a7a566.pt", "dinov2-small-provenance.json",
                "dinov2-small/config.json", "dinov2-small/model.safetensors",
                "dinov2-small/preprocessor_config.json", "dinov2-small/README.md"]:
        add(model_dir/rel)
    profiles = json.loads((model_dir/"profiles.json").read_text())
    chosen = [p for p in profiles["models"] if p["id"] == "open-dinov2-pilot"]
    assert len(chosen) == 1 and chosen[0]["checkpoint_sha256"] == HEAD_SHA
    payload["robot-data-studio/workspace/warp/models/profiles.json"] = encoded(dict(version=1, models=chosen))
    assert digest(payload["robot-data-studio/workspace/warp/models/"+chosen[0]["checkpoint"]]) == HEAD_SHA
    assert digest(payload["robot-data-studio/workspace/warp/models/dinov2-small/model.safetensors"]) == BACKBONE_SHA

    evidence_dir = APP/"workspace/evidence"
    for rel in ["trajectory-ui-result.json", "warp/pytest.xml", "warp/node-tests.json", "warp/ui-report.json",
                "warp/progress-dark-test-fixture.png", "warp/progress-light-test-fixture.png",
                "warp/progress-export-test-fixture.png", "warp/live/report.json", "warp/live/progress-real-video.png"]:
        add(evidence_dir/rel)
    suites = ET.parse(evidence_dir/"warp/pytest.xml").getroot().findall("testsuite")
    python_tests = {key: sum(int(s.attrib[key]) for s in suites) for key in ("tests", "errors", "failures", "skipped")}
    assert python_tests == dict(tests=92, errors=0, failures=0, skipped=0)
    checks = {}
    for name, rel, expected in [("progress_ui", "warp/ui-report.json", 16),
                                ("trajectory_ui", "trajectory-ui-result.json", 38),
                                ("real_video_ui", "warp/live/report.json", 10)]:
        report = json.loads((evidence_dir/rel).read_text())
        assert report["passed"] and len(report["checks"]) == expected
        checks[name] = dict(passed=True, checks=expected)
    node = json.loads((evidence_dir/"warp/node-tests.json").read_text())
    assert node["passed"] and len(node["files"]) == 7

    for rel in ["项目改进与验证报告.md", "复现评估与改进方案.md", "sources.json", "sources.initial_20260916.json",
                "source_file_checksums.json", "public_download_audit.json", "public_selection.json",
                "audit_inputs.py", "local_data_audit.json", "local_head_samples.json", "local_head_samples.jpg",
                "episodes.csv", "fetch_public_traces.py", "replay_public_traces.py", "build_handoff.py",
                "handoff-source-audit.json"]:
        add(RESEARCH/rel)
    for rel in ["dataset_card.md", "dataset_info.json", "dataset_metadata.json", "model_card.md", "model_metadata.json",
                "dinov2_small_metadata.json", "warp_rm_commit.json", "public_episodes.parquet", "public_object_counts.json",
                "public_video_tree.json", "score_bottles.py", "abc-rabc-LICENSE", "n128_trace_card.md",
                "n128_trace_manifest.json", "n128_trace_metadata.json", "n512_trace_card.md", "n512_trace_metadata.json"]:
        add(RESEARCH/"sources"/rel)
    for path in selected_files(RESEARCH/"trace_replay"):
        add(path)
    for suite in ("n128", "n512"):
        add(RESEARCH/"traces"/suite/"local_download_audit.json")
    for path in selected_files(RESEARCH/"open_pilot"):
        if path.name != "latest.pt":
            add(path)
    assert digest(payload["research/warp_rm_20260916/open_pilot/train/best.pt"]) == HEAD_SHA
    add(RESEARCH/"交接说明.md", "交接说明.md")

    manifest = dict(schema_version=1, built_at=built.isoformat(),
                    scope="Runnable public DINOv2 WARP-method adaptation; no original DINOv3 or robot-policy reproduction claim",
                    validation=dict(python=python_tests, node_test_files=7, **checks),
                    model=dict(id=chosen[0]["id"], checkpoint_sha256=HEAD_SHA, backbone_sha256=BACKBONE_SHA,
                               validation_status="transfer_unvalidated"),
                    original_workspace_files_unchanged=True,
                    files=[dict(path=n, bytes=len(data), sha256=digest(data)) for n, data in sorted(payload.items())])
    # This field refers to immutable input snapshots during archive creation, not source edits made earlier.
    manifest["input_snapshot_stable_during_packaging"] = manifest.pop("original_workspace_files_unchanged")
    payload["ARTIFACT_MANIFEST.json"] = encoded(manifest)
    prefix = target.stem + "/"
    partial = target.with_suffix(".zip.partial")
    try:
        with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in sorted(payload.items()):
                entry = zipfile.ZipInfo(prefix+name, built.timetuple()[:6])
                entry.compress_type = zipfile.ZIP_DEFLATED
                mode = 0o755 if name.endswith(".sh") else 0o644
                entry.external_attr = (stat.S_IFREG | mode) << 16
                archive.writestr(entry, data)
        with zipfile.ZipFile(partial) as archive:
            assert archive.testzip() is None, "ZIP CRC check failed"
            names = archive.namelist()
            assert len(names) == len(set(names)) == len(payload)
            for item in manifest["files"]:
                data = archive.read(prefix+item["path"])
                assert len(data) == item["bytes"] and digest(data) == item["sha256"], item["path"]
        for path, sha in disk_inputs.items():
            if digest(path.read_bytes()) != sha:
                raise RuntimeError(f"Source changed during packaging: {path}")
        partial.rename(target)
    except Exception:
        # Preserve any partial archive for inspection; never replace an existing deliverable.
        raise
    archive_sha = digest(target.read_bytes())
    target.with_suffix(target.suffix+".sha256").write_text(archive_sha+"  "+target.name+"\n")
    verification = dict(passed=True, archive=target.name, sha256=archive_sha, bytes=target.stat().st_size,
                        archive_entries=len(payload), manifest_verified_files=len(manifest["files"]),
                        zip_crc_valid=True, duplicate_entries=False, input_snapshot_stable=True,
                        baseline_source_files_preserved=True, registered_models=[chosen[0]["id"]],
                        credentials_and_runtime_state="Excluded by explicit runtime whitelist; token/private-key text scan passed",
                        validation=manifest["validation"])
    target.with_suffix(target.suffix+".verification.json").write_bytes(encoded(verification))
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
