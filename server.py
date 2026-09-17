#!/usr/bin/env python3
"""Web server for the HoloMotion v1.4 MCAP visualization tool.

Stdlib-only HTTP layer (no framework dependency) over two data sources:

* the extraction cache (`meta.json`, `series.f32`) built by `extract.py`
* the raw episode, opened read-only, for on-demand JPEG frames

The server holds a *library*: any number of episodes discovered under the paths
it was given, of which at most a few are resident at a time. Extraction takes
minutes, so it never happens inside a request — `POST /api/prepare` starts a
background build and the caller watches it.

Endpoints
    GET  /                      dashboard
    GET  /api/episodes          the catalog: every known episode + cache state
    POST /api/roots             {"path": "..."} — scan another local directory
    POST /api/prepare           {"ep": "<id>"} — build this episode's cache
    GET  /api/meta?ep=<id>      episode metadata + channel descriptors (JSON)
    GET  /api/series?ep=<id>    the whole float32 block (binary)
    GET  /api/frame?ep=&i=<n>   one camera frame as image/jpeg
    GET  /api/robot             URDF kinematic tree for the 3D view (JSON)
    GET  /api/mesh?f=<name>     one visual mesh referenced by that tree
    GET  /api/health            library/state probe

`ep` is optional everywhere and falls back to the default episode, so a URL
written before the library existed still resolves.

The raw MCAP is never written to. Frames are read through the chunk index, so a
single frame costs a windowed read rather than a full scan.

This binds to 127.0.0.1 and, by design, reads local paths the operator names.
Do not expose it on a public interface.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import socketserver
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from collections import OrderedDict
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import catalog
from extract import T_CAM_IMAGE, build_cache, cache_dir_for
from mcap.reader import make_reader
from urdf import find_default_urdf, parse_urdf

# Each resident episode costs its series block (~22 MB) plus a frame index.
# Three is enough to flip between the episodes under comparison without
# re-indexing, and small enough that the ceiling is obvious.
MAX_RESIDENT = 3

STATIC_DIR = Path(__file__).parent / "static"

MESH_TYPES = {
    ".stl": "model/stl",
    ".dae": "model/vnd.collada+xml",
    ".obj": "text/plain",
}


class Robot:
    """The URDF description behind the 3D panel, or nothing.

    Absent robot description is a normal state, not an error: the viewer is
    useful without the 3D view, and a standalone checkout may carry no robot
    at all. `/api/robot` then 404s and the frontend collapses that column.
    """

    def __init__(self, urdf_path: Path):
        self.path = urdf_path
        self.desc = parse_urdf(urdf_path)
        self.mesh_dir = Path(self.desc["mesh_dir"])
        # An allowlist of exact filenames, so the mesh route never has to
        # reason about `..` or symlinks in caller-supplied text.
        self.allowed = set(self.desc["mesh_files"])
        self.body = json.dumps(self.desc).encode()
        self.etag = f'"{hashlib.sha1(self.body).hexdigest()[:16]}"'

    def mesh(self, name: str) -> tuple[bytes, str] | None:
        if name not in self.allowed:
            return None
        path = self.mesh_dir / name
        if not path.is_file():
            return None
        ctype = MESH_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return path.read_bytes(), ctype


class Episode:
    """Holds the cache in memory and serves frames from the raw MCAP.

    One reader per thread: `mcap`'s reader wraps a single file object with its
    own stream position, so sharing one across worker threads would interleave
    seeks and return corrupt frames.
    """

    def __init__(self, mcap_path: Path, cache_dir: Path):
        self.mcap_path = mcap_path
        self.cache_dir = cache_dir
        self.meta_bytes = (cache_dir / "meta.json").read_bytes()
        self.meta = json.loads(self.meta_bytes)
        self.series_path = cache_dir / "series.f32"
        self.series_bytes = self.series_path.read_bytes()
        # Content-derived, so a rebuilt cache invalidates the browser copy.
        self.etag_meta = f'"{hashlib.sha1(self.meta_bytes).hexdigest()[:16]}"'
        self.etag_series = (
            f'"{hashlib.sha1(self.series_bytes).hexdigest()[:16]}"'
        )
        self._local = threading.local()
        self.frame_times_ns = self._index_frames()

    def _index_frames(self) -> list[int]:
        """Exact nanosecond log time of every camera frame.

        Deliberately re-derived from the episode rather than reconstructed from
        the cache: `meta.json` stores frame times as rounded seconds for the
        frontend, and a 10 us rounding error is enough to miss the single-
        nanosecond window used to fetch a frame. Scanning the camera channel
        without decoding costs about a second.
        """
        native_path = self.cache_dir / "native.npz"
        if native_path.exists():
            import numpy as np
            with np.load(native_path, allow_pickle=False) as native:
                if "camera.head.t" in native:
                    return native["camera.head.t"].tolist()
        times: list[int] = []
        with self.mcap_path.open("rb") as fh:
            reader = make_reader(fh)
            for _schema, _channel, message in reader.iter_messages(
                topics=[T_CAM_IMAGE]
            ):
                times.append(message.log_time)
        times.sort()
        return times

    def _reader(self):
        reader = getattr(self._local, "reader", None)
        if reader is None:
            fh = self.mcap_path.open("rb")
            reader = make_reader(fh)
            self._local.fh = fh
            self._local.reader = reader
        return reader

    def frame(self, index: int) -> bytes | None:
        if not 0 <= index < len(self.frame_times_ns):
            return None
        ts = self.frame_times_ns[index]
        reader = self._reader()
        for _schema, _channel, message in reader.iter_messages(
            topics=[T_CAM_IMAGE], start_time=ts, end_time=ts + 1
        ):
            return _jpeg_payload(message.data)
        return None


def _jpeg_payload(raw: bytes) -> bytes | None:
    """Pull the JPEG out of an unparsed CompressedImage CDR message.

    Running the full ROS 2 decoder just to reach `.data` costs far more than
    reading the field directly. `data` is the last field of CompressedImage and
    every CDR sequence is length-prefixed, so the four bytes immediately before
    the JPEG's start-of-image marker are that length: reading it back gives the
    exact payload and simultaneously validates the guess.

    A bare marker scan is not equivalent — CDR pads the tail of the message for
    alignment, so scanning to the end returns a few extra bytes.
    """
    start = raw.find(b"\xff\xd8\xff")
    if start < 4:
        return None
    length = int.from_bytes(raw[start - 4 : start], "little")
    if 0 < length <= len(raw) - start:
        return raw[start : start + length]
    return raw[start:]  # fall back to the tail if the prefix is implausible


class NotReady(Exception):
    """The episode exists but its cache does not (yet)."""

    def __init__(self, state: dict):
        super().__init__(state.get("message", "not ready"))
        self.state = state


class Library:
    """Every episode the server knows about, and the few it has open.

    Two costs shape this class. Extracting an episode takes **minutes**, so it
    runs on a background thread and the catalog reports its progress; opening
    an already-extracted one takes a second or two, so it happens on first use
    rather than at startup. Neither is allowed to block the catalog itself,
    which the sidebar polls.
    """

    def __init__(self, cache_root: Path, rate: float):
        self.cache_root = cache_root
        self.rate = rate
        self.roots: list[Path] = []
        self.refs: dict[str, catalog.EpisodeRef] = {}
        self.order: list[str] = []
        self.resident: OrderedDict[str, Episode] = OrderedDict()
        self.jobs: dict[str, dict] = {}
        self.default_id: str | None = None
        self._lock = threading.Lock()      # guards the catalog structures
        self._open_lock = threading.Lock()  # serialises episode opens
        self._summary_memo: dict[str, tuple[float, dict]] = {}

    # ---------------------------------------------------------- discovery --

    def add_root(self, path: Path) -> list[catalog.EpisodeRef]:
        """Scan a local path and merge what it holds into the catalog.

        Re-scanning a known root is how the catalog picks up episodes recorded
        since startup, so this is deliberately idempotent rather than an error.
        """
        found = catalog.scan(path)
        resolved = path.expanduser().resolve()
        with self._lock:
            if resolved not in self.roots:
                self.roots.append(resolved)
            for ref in found:
                if ref.id not in self.refs:
                    self.order.append(ref.id)
                self.refs[ref.id] = ref
            everything = [self.refs[i] for i in self.order]
            self.order = [r.id for r in catalog.sort_episodes(everything)]
        return found

    def ref(self, ep_id: str) -> catalog.EpisodeRef:
        with self._lock:
            found = self.refs.get(ep_id)
        if found is None:
            raise KeyError(ep_id)
        return found

    def cache_dir(self, ref: catalog.EpisodeRef) -> Path:
        return cache_dir_for(ref.path, self.cache_root)

    # -------------------------------------------------------------- state --

    def state(self, ep_id: str) -> dict:
        """`ready`, `absent`, `building` or `error` for one episode.

        A build writes `meta.json` last, so there is no half-ready state to
        observe: the cache directory either answers for a finished extraction
        or does not answer at all.
        """
        job = self.jobs.get(ep_id)
        if job and job["state"] in ("building", "error"):
            return dict(job)
        try:
            ref = self.ref(ep_id)
        except KeyError:
            return {"state": "unknown", "message": "no such episode"}
        if (self.cache_dir(ref) / "meta.json").is_file():
            return {"state": "ready", "message": ""}
        return {"state": "absent", "message": "not extracted yet"}

    def _summary(self, cache_dir: Path) -> dict:
        """The handful of catalog-worthy numbers out of a built `meta.json`.

        Memoised on the file's mtime: the sidebar polls, `meta.json` is a few
        hundred kilobytes, and re-parsing every episode's copy on every poll
        would cost more than everything else the catalog does.
        """
        meta_file = cache_dir / "meta.json"
        try:
            mtime = meta_file.stat().st_mtime
        except OSError:
            return {}
        key = str(meta_file)
        hit = self._summary_memo.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            meta = json.loads(meta_file.read_bytes())
        except (OSError, ValueError):
            return {}
        summary = {
            "duration_s": meta["episode"]["duration_s"],
            "message_count": meta["episode"]["message_count"],
            "frames": meta["camera"]["count"],
            "topics": len(meta["topics"]),
            # The cache carries the last instruction actually recorded, which
            # outranks the sidecar's copy of what was *planned*.
            "recorded_instruction": meta["episode"].get("instruction") or "",
        }
        self._summary_memo[key] = (mtime, summary)
        return summary

    def to_json(self) -> dict:
        with self._lock:
            ids = list(self.order)
        episodes = []
        for ep_id in ids:
            ref = self.refs[ep_id]
            item = ref.to_json()
            item.update(self._summary(self.cache_dir(ref)))
            state = self.state(ep_id)
            item["cache"] = state["state"]
            item["cache_message"] = state.get("message", "")
            item["resident"] = ep_id in self.resident
            episodes.append(item)
        return {
            "roots": [str(r) for r in self.roots],
            "cache_root": str(self.cache_root),
            "default": self.default_id,
            "episodes": episodes,
        }

    # ------------------------------------------------------------- extract --

    def prepare(self, ep_id: str, rebuild: bool = False) -> dict:
        """Ensure this episode has a cache, building it in the background."""
        ref = self.ref(ep_id)
        with self._lock:
            job = self.jobs.get(ep_id)
            if job and job["state"] == "building":
                return dict(job)
            if not rebuild and (self.cache_dir(ref) / "meta.json").is_file():
                return {"state": "ready", "message": ""}
            job = {
                "state": "building",
                "message": "starting extraction…",
                "started": time.time(),
            }
            self.jobs[ep_id] = job
        thread = threading.Thread(
            target=self._build, args=(ref, job), daemon=True
        )
        thread.start()
        return dict(job)

    def _build(self, ref: catalog.EpisodeRef, job: dict) -> None:
        cache_dir = self.cache_dir(ref)
        try:
            build_cache(
                ref.path,
                cache_dir,
                self.rate,
                progress=lambda msg: job.update(message=msg),
            )
        except Exception as err:  # a bad episode must not take the server down
            traceback.print_exc()
            job.update(state="error", message=f"{type(err).__name__}: {err}")
            return
        job.update(state="ready", message="", finished=time.time())

    def build_now(self, ep_id: str, rebuild: bool = False) -> None:
        """Extract on the calling thread, reporting to the console.

        Used only for an episode named explicitly on the command line, where
        the operator asked for that episode and can see the progress. Anything
        chosen later, from the sidebar, goes through `prepare` instead.
        """
        ref = self.ref(ep_id)
        cache_dir = self.cache_dir(ref)
        if not rebuild and (cache_dir / "meta.json").is_file():
            print(f"reusing cache {cache_dir}")
            return
        print(f"building cache for {ref.path.name} -> {cache_dir}")
        build_cache(
            ref.path, cache_dir, self.rate,
            progress=lambda m: print(m, flush=True),
        )

    # --------------------------------------------------------------- open --

    def episode(self, ep_id: str) -> Episode:
        """The resident `Episode`, opening it if the cache is there.

        Raises `NotReady` when it is not, which the routes turn into a 409 the
        frontend can act on — a missing cache is a state to report, not a
        failure to log.
        """
        hit = self.resident.get(ep_id)
        if hit is not None:
            self.resident.move_to_end(ep_id)
            return hit
        # Resolve the reference before the cache state: an id that names no
        # episode is a 404, and must not be reported as one that merely has
        # not been extracted yet.
        ref = self.ref(ep_id)
        state = self.state(ep_id)
        if state["state"] != "ready":
            raise NotReady({**state, "ep": ep_id})
        # Serialised, so two requests racing on the same cold episode index
        # the camera channel once between them rather than once each.
        with self._open_lock:
            hit = self.resident.get(ep_id)
            if hit is not None:
                return hit
            opened = Episode(ref.path, self.cache_dir(ref))
            self.resident[ep_id] = opened
            while len(self.resident) > MAX_RESIDENT:
                self.resident.popitem(last=False)
        return opened


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, library: Library, robot: Robot | None, **kwargs):
        self.library = library
        self.robot = robot
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt, *args):  # quieter console
        if "--verbose" in sys.argv:
            super().log_message(fmt, *args)

    # -------------------------------------------------------------- helpers --

    def _json(self, payload, status: int = 200, **kwargs):
        return self._send(
            json.dumps(payload).encode(), "application/json",
            status=status, **kwargs,
        )

    def _body_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            loaded = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _episode(self, query: str) -> Episode | None:
        """Resolve `?ep=` to a resident episode, or answer for its absence.

        Returns None having already sent a response, so every data route can
        say `ep = self._episode(...)` / `if ep is None: return`.
        """
        ep_id = (parse_qs(query).get("ep") or [""])[0]
        ep_id = ep_id or self.library.default_id
        if not ep_id:
            self._json(
                {"error": "no episode selected",
                 "detail": "the library is empty; add a path first"},
                status=404,
            )
            return None
        try:
            return self.library.episode(ep_id)
        except NotReady as err:
            # 409, not 404: the episode is real and this is a state the caller
            # can resolve by asking for it to be built.
            self._json(err.state, status=409)
        except KeyError:
            self._json({"error": "no such episode", "ep": ep_id}, status=404)
        except (OSError, ValueError) as err:
            self._json({"error": str(err), "ep": ep_id}, status=500)
        return None

    # ---------------------------------------------------------------- POST --

    def do_POST(self):
        route = urlparse(self.path).path
        body = self._body_json()
        if route == "/api/roots":
            raw = str(body.get("path") or "").strip()
            if not raw:
                return self._json({"error": "path is required"}, status=400)
            try:
                found = self.library.add_root(Path(raw))
            except (FileNotFoundError, NotADirectoryError, OSError) as err:
                return self._json({"error": str(err)}, status=404)
            payload = self.library.to_json()
            payload["added"] = len(found)
            return self._json(payload)
        if route == "/api/prepare":
            ep_id = str(body.get("ep") or "") or self.library.default_id
            if not ep_id:
                return self._json({"error": "ep is required"}, status=400)
            try:
                state = self.library.prepare(
                    ep_id, rebuild=bool(body.get("rebuild"))
                )
            except KeyError:
                return self._json(
                    {"error": "no such episode", "ep": ep_id}, status=404
                )
            return self._json({**state, "ep": ep_id})
        return self.send_error(404, "no such endpoint")

    # ----------------------------------------------------------------- GET --

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/api/episodes":
            # Deliberately uncached: this is the one thing that changes while
            # the page is open, and the sidebar polls it during a build.
            return self._json(self.library.to_json(), cache="no-store")
        if route == "/api/meta":
            episode = self._episode(parsed.query)
            if episode is None:
                return None
            return self._send(
                episode.meta_bytes,
                "application/json",
                cache="public, max-age=3600",
                etag=episode.etag_meta,
            )
        if route == "/api/series":
            episode = self._episode(parsed.query)
            if episode is None:
                return None
            return self._send(
                episode.series_bytes,
                "application/octet-stream",
                cache="public, max-age=3600",
                etag=episode.etag_series,
            )
        if route == "/api/health":
            resident = list(self.library.resident)
            return self._json(
                {
                    "ok": True,
                    "episodes": len(self.library.order),
                    "roots": [str(r) for r in self.library.roots],
                    "default": self.library.default_id,
                    "resident": resident,
                    "robot": (
                        None if self.robot is None else self.robot.path.name
                    ),
                }
            )
        if route == "/api/robot":
            if self.robot is None:
                # A structured 404: the frontend logs this reason and
                # collapses the column instead of failing silently.
                return self._send(
                    json.dumps(
                        {
                            "error": "no robot description",
                            "detail": "checked-in G1 URDF not found; pass "
                            "--urdf to point at one",
                        }
                    ).encode(),
                    "application/json",
                    status=404,
                )
            return self._send(
                self.robot.body,
                "application/json",
                cache="public, max-age=3600",
                etag=self.robot.etag,
            )
        if route == "/api/mesh":
            if self.robot is None:
                return self.send_error(404, "no robot description")
            name = parse_qs(parsed.query).get("f", [""])[0]
            found = self.robot.mesh(name)
            if found is None:
                return self.send_error(404, "mesh not in this robot")
            payload, ctype = found
            return self._send(
                payload, ctype, cache="public, max-age=86400"
            )
        if route == "/api/frame":
            qs = parse_qs(parsed.query)
            try:
                index = int(qs.get("i", ["0"])[0])
            except ValueError:
                return self.send_error(400, "i must be an integer")
            episode = self._episode(parsed.query)
            if episode is None:
                return None
            payload = episode.frame(index)
            if payload is None:
                return self.send_error(404, "frame not found")
            return self._send(
                payload, "image/jpeg", cache="public, max-age=3600"
            )
        return super().do_GET()

    def _send(
        self,
        body: bytes,
        content_type: str,
        cache: str | None = None,
        etag: str | None = None,
        status: int = 200,
    ):
        # The cache is derived from an immutable episode, so a conditional
        # request can skip resending it. Without this every refresh
        # re-downloads
        # the whole ~22 MB series block.
        if etag and self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if etag:
            self.send_header("ETag", etag)
        if cache:
            self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def build_library(args) -> Library:
    """Turn the command line into a library, and pick what opens first.

    An episode named **explicitly** is extracted here, on this thread, with
    progress on the console — that is what the operator asked for and waiting
    for it is the honest response. A *directory* is only scanned: it may hold
    fifty episodes and extracting them all at startup would be minutes each,
    for episodes nobody asked to see. Those build on demand, from the sidebar.
    """
    library = Library(args.cache_root, args.rate)
    explicit: list[str] = []
    for path in args.paths:
        if not path.exists():
            print(f"no such file or directory: {path}", file=sys.stderr)
            continue
        found = library.add_root(path)
        if path.is_file():
            explicit.extend(ref.id for ref in found)
        print(f"  {len(found):3d} episode(s) under {path}")

    for ep_id in explicit:
        library.build_now(ep_id, rebuild=args.rebuild)

    # First choice: what was named. Otherwise the first episode already
    # extracted, so a plain directory argument opens something instantly
    # instead of committing the visitor to a multi-minute build they did not
    # choose. With nothing extracted, the sidebar asks.
    library.default_id = next(
        (ep for ep in explicit if library.state(ep)["state"] == "ready"),
        next(
            (ep for ep in library.order
             if library.state(ep)["state"] == "ready"),
            None,
        ),
    )
    return library


def load_robot(args) -> Robot | None:
    """Resolve the 3D view's robot description, tolerating its absence.

    A malformed or missing URDF costs the 3D view, not the whole viewer, so
    failures here are reported and swallowed. An explicitly passed `--urdf`
    or `$MCAP_VIZ_URDF` that cannot be read is still worth a loud warning —
    discovery is allowed to come up empty, an explicit request is not.
    """
    if args.no_3d:
        return None
    try:
        path = args.urdf or find_default_urdf(
            Path(__file__).resolve().parent
        )
    except ValueError as err:
        print(f"{err} — 3D view disabled", file=sys.stderr)
        return None
    if path is None:
        print("no robot description found — 3D view disabled")
        return None
    try:
        robot = Robot(path)
    except (OSError, ValueError, ET.ParseError) as err:
        print(f"cannot use {path}: {err}", file=sys.stderr)
        return None
    for warning in robot.desc["warnings"]:
        print(f"  urdf: {warning}", file=sys.stderr)
    return robot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "paths",
        type=Path,
        nargs="*",
        help="raw v1.4 episodes (.mcap) and/or directories holding them; "
        "more can be added from the sidebar while the server runs",
    )
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--rate", type=float, default=100.0, help="resample rate (Hz)"
    )
    ap.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="cache location (default: .mcap_viz_cache/ beside this tool)",
    )
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="robot description for the 3D view (default: the vendored G1 "
        "29-DOF URDF under robot/; $MCAP_VIZ_URDF also overrides)",
    )
    ap.add_argument(
        "--no-3d",
        action="store_true",
        help="skip the 3D view even when a URDF is available",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # One cache root for the whole library, not one beside each episode. The
    # recordings this reads are evidence and may sit on a read-only or shared
    # mount; writing derived data into that tree is exactly what the v1.4
    # workflow forbids. The key still includes each episode's absolute path,
    # so distinct episodes never collide here.
    args.cache_root = (
        args.cache_root or Path(__file__).resolve().parent / ".mcap_viz_cache"
    )

    library = build_library(args)
    robot = load_robot(args)

    handler = partial(Handler, library=library, robot=robot)
    with ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        ready = sum(
            1 for e in library.order if library.state(e)["state"] == "ready"
        )
        print()
        print(
            f"  library   {len(library.order)} episode(s), {ready} extracted"
        )
        if library.default_id:
            ref = library.ref(library.default_id)
            print(f"  opening   {ref.episode_id} · {ref.path}")
        else:
            print("  opening   nothing yet — choose an episode in the sidebar")
        print(f"  cache     {library.cache_root}")
        if robot is not None:
            movable = sum(
                1
                for j in robot.desc["joints"]
                if j["type"] in ("revolute", "continuous", "prismatic")
            )
            print(
                f"  robot     {robot.path.name} "
                f"({movable} movable joints, {len(robot.allowed)} meshes)"
            )
        else:
            print("  robot     none — 3D view disabled")
        print(f"  serving   http://{args.host}:{args.port}/")
        print()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
