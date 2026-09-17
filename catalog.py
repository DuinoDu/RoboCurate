#!/usr/bin/env python3
"""Find episodes on disk and read what their sidecars claim about them.

The v1.4 PICO recorder does not write a flat pile of `.mcap` files. It writes a
tree, and the tree carries most of what you need to *choose* an episode before
paying to open one:

    <collection root>/                     e.g. locomanip_teleop_v14
      .episode_sequences.json              {"<task>": <last episode number>}
      <session>/                           2026_08_31-15_44_05
        data/<user>/<task>/episode_0000NN/
          episode_meta.json                task, instruction, runtime settings
          pico_episode_summary.json        the above + outcome, state, sizes
          mcap/
            mcap_0.mcap                    the episode itself
            metadata.yaml                  rosbag2 sidecar
        logs/<user>/<task>/episode_0000NN/

So discovery is a walk for `*.mcap`, and everything else is read from the
sidecars beside it. Nothing here opens an MCAP: a catalog of fifty episodes has
to build in milliseconds, and an episode is ~600 MB.

`pico_episode_summary.json` carries `task_outcome.success`, which the raw MCAP
does not. That is worth surfacing: the operator's verdict is the single most
useful thing to know before spending minutes extracting an episode. When the
sidecar is absent — a loose `.mcap` copied off the robot on its own — the
outcome is reported as unknown rather than guessed.

Paths that are not laid out this way still work. A directory of loose `.mcap`
files is a perfectly good catalog with empty session/task/outcome columns.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SUMMARY_NAME = "pico_episode_summary.json"
META_NAME = "episode_meta.json"

# A session directory is named for when the session started.
SESSION_RE = re.compile(r"^\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}$")
EPISODE_RE = re.compile(r"^episode_\d+$", re.IGNORECASE)

# Derived data, environments and version control never contain episodes, and
# `.mcap_viz_cache` in particular would otherwise be walked on every scan.
SKIP_DIRS = {
    ".mcap_viz_cache",
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "logs",
}

MAX_DEPTH = 8
MAX_EPISODES = 2000


def episode_key(path: Path) -> str:
    """Stable id for an episode file.

    Derived from the absolute path, exactly like the extraction cache key, so
    an id in a URL keeps meaning across restarts without a database.
    """
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]


@dataclass
class EpisodeRef:
    """One `.mcap` plus whatever its neighbours say about it."""

    path: Path
    root: Path
    id: str = ""
    name: str = ""
    size_bytes: int = 0
    mtime: float = 0.0
    episode_id: str = ""
    session: str = ""
    user: str = ""
    task: str = ""
    part: str = ""
    instruction: str = ""
    outcome: bool | None = None
    failure_reason: str = ""
    operator_note: str = ""
    created_at: str = ""
    stopped_at: str = ""
    state: str = ""
    deleted: bool = False
    hand_backend: str = ""
    sidecar: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "path": str(self.path),
            "root": str(self.root),
            "name": self.name,
            "size_bytes": self.size_bytes,
            "mtime": self.mtime,
            "episode_id": self.episode_id,
            "session": self.session,
            "user": self.user,
            "task": self.task,
            "part": self.part,
            "instruction": self.instruction,
            "outcome": self.outcome,
            "failure_reason": self.failure_reason,
            "operator_note": self.operator_note,
            "created_at": self.created_at,
            "stopped_at": self.stopped_at,
            "state": self.state,
            "deleted": self.deleted,
            "hand_backend": self.hand_backend,
            "sidecar": self.sidecar,
            "warnings": self.warnings,
        }


def _read_json(path: Path) -> dict | None:
    try:
        with path.open("rb") as fh:
            loaded = json.load(fh)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _find_sidecar(mcap_path: Path) -> tuple[dict | None, Path | None]:
    """Walk up from the episode file looking for its metadata.

    `mcap/mcap_0.mcap` puts the sidecars one level up, but a hand-copied
    episode may sit at any of a couple of depths, so try a few. The summary is
    preferred over the bare meta because it is a superset — it adds the
    outcome, the recorder's own report, and the byte counts.
    """
    here = mcap_path.parent
    for _ in range(3):
        for name in (SUMMARY_NAME, META_NAME):
            found = _read_json(here / name)
            if found is not None:
                return found, here / name
        if here.parent == here:
            break
        here = here.parent
    return None, None


def _ancestor_labels(mcap_path: Path) -> tuple[str, str, str]:
    """(session, user, task) read off the directory names, when they are there.

    `<session>/data/<user>/<task>/episode_NNNNNN/mcap/x.mcap` — anchor on the
    `data` component rather than counting levels up, so an episode that was
    copied with a different amount of surrounding tree still resolves.
    """
    parts = list(mcap_path.parts)
    session = user = task = ""
    for i, part in enumerate(parts):
        if SESSION_RE.match(part):
            session = part
        if part == "data" and i + 2 < len(parts) - 1:
            user, task = parts[i + 1], parts[i + 2]
    return session, user, task


def _episode_dir_name(mcap_path: Path) -> str:
    for parent in list(mcap_path.parents)[:3]:
        if EPISODE_RE.match(parent.name):
            return parent.name
    return ""


def describe(mcap_path: Path, root: Path) -> EpisodeRef:
    """Everything known about one episode without opening it."""
    ref = EpisodeRef(path=mcap_path, root=root)
    ref.id = episode_key(mcap_path)
    ref.name = mcap_path.name
    try:
        stat = mcap_path.stat()
        ref.size_bytes = stat.st_size
        ref.mtime = stat.st_mtime
    except OSError:
        ref.warnings.append("cannot stat the file")

    ref.session, ref.user, ref.task = _ancestor_labels(mcap_path)
    ref.episode_id = _episode_dir_name(mcap_path) or mcap_path.stem
    # Only meaningful when a recording split across several files; a lone
    # `mcap_0.mcap` is the whole episode and does not need a part label.
    siblings = sorted(p.name for p in mcap_path.parent.glob("*.mcap"))
    if len(siblings) > 1:
        ref.part = mcap_path.stem

    side, side_path = _find_sidecar(mcap_path)
    if side is None:
        return ref
    ref.sidecar = str(side_path)

    # `pico_episode_summary.json` nests the meta; `episode_meta.json` is that
    # nested object on its own. Accept either shape.
    nested = side.get("metadata")
    meta = nested if isinstance(nested, dict) else side
    metas = meta.get("metas") if isinstance(meta.get("metas"), dict) else {}

    ref.instruction = str(meta.get("instruction") or "")
    ref.user = str(meta.get("user_name") or ref.user)
    ref.task = str(meta.get("task_name") or ref.task)
    ref.hand_backend = str(metas.get("hand_backend") or "")
    ref.created_at = str(side.get("created_at") or "")
    ref.stopped_at = str(side.get("stopped_at") or "")
    ref.state = str(side.get("state") or "")
    ref.deleted = bool(side.get("deleted"))

    # The outcome may sit at the top level or inside metas, and the two are
    # written together. Prefer the top-level copy; fall back to the nested one.
    outcome = side.get("task_outcome")
    if not isinstance(outcome, dict):
        outcome = metas.get("task_outcome")
    if isinstance(outcome, dict) and "success" in outcome:
        ref.outcome = bool(outcome["success"])
        ref.failure_reason = str(outcome.get("failure_reason") or "")
        ref.operator_note = str(outcome.get("operator_note") or "")

    if ref.state and ref.state != "stopped":
        # A recorder that never reached `stopped` may have been killed
        # mid-write; the file is still readable but is not a clean episode.
        ref.warnings.append(f"recorder state is {ref.state!r}, not 'stopped'")
    if side.get("error_message"):
        ref.warnings.append(str(side["error_message"]))
    return ref


def scan(root: Path, limit: int = MAX_EPISODES) -> list[EpisodeRef]:
    """Every `.mcap` under `root`, described.

    A single file is a catalog of one, so pointing the viewer at one episode
    and pointing it at a season of them are the same operation.
    """
    root = root.expanduser()
    if root.is_file():
        return [describe(root.resolve(), root.resolve().parent)]
    if not root.is_dir():
        raise FileNotFoundError(f"no such file or directory: {root}")

    root = root.resolve()
    found: list[EpisodeRef] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack and len(found) < limit:
        here, depth = stack.pop()
        try:
            entries = sorted(here.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_symlink() and entry.is_dir():
                continue  # a symlinked tree can loop; the walk must terminate
            if entry.is_dir():
                if depth >= MAX_DEPTH or entry.name in SKIP_DIRS:
                    continue
                if entry.name.startswith(".") and entry.name != ".":
                    continue
                stack.append((entry, depth + 1))
            elif entry.suffix.lower() == ".mcap":
                found.append(describe(entry, root))
    return sort_episodes(found)


def sort_episodes(refs: list[EpisodeRef]) -> list[EpisodeRef]:
    """Newest session first, then episode order within it.

    Sessions are named for their start time and sort lexicographically by it,
    so `reverse` on that component alone puts the most recent work on top
    while episodes inside a session stay in the order they were recorded.
    """
    def key(ref: EpisodeRef):
        return (ref.session or "", ref.task or "", ref.episode_id, ref.name)

    by_session: dict[str, list[EpisodeRef]] = {}
    for ref in refs:
        by_session.setdefault(ref.session or "", []).append(ref)
    out: list[EpisodeRef] = []
    for session in sorted(by_session, reverse=True):
        out.extend(sorted(by_session[session], key=key))
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="list episodes under a path")
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    refs = scan(args.root)
    for ref in refs:
        mark = {True: "ok", False: "FAILED", None: "?"}[ref.outcome]
        size = ref.size_bytes / 1e9
        print(
            f"{ref.id}  {ref.session or '-':<21}  {ref.task or '-':<12}  "
            f"{ref.episode_id:<16}  {size:6.2f} GB  {mark}"
        )
        for warning in ref.warnings:
            print(f"    ! {warning}")
    print(f"\n{len(refs)} episode(s) under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
