/* Episode data access.
 *
 * `/api/series` returns one contiguous float32 block. Each channel occupies
 * `series * samples` floats at its recorded offset, and within a channel one
 * series is contiguous, so a per-series view is a zero-copy subarray rather
 * than a slice.
 */

export class EpisodeData {
  constructor(meta, buffer, ep = null) {
    this.meta = meta;
    this.ep = ep;
    this.all = new Float32Array(buffer);
    this.samples = meta.timeline.samples;
    this._cache = new Map();
  }

  /** Query string identifying which episode a request is about.
   *
   *  Omitted when the server's default is wanted, so a URL written before the
   *  library existed still resolves — and, more usefully, so the frame URLs
   *  are distinct per episode and the browser's HTTP cache cannot serve one
   *  episode's imagery for another's playhead. */
  static _q(ep) {
    return ep ? `ep=${encodeURIComponent(ep)}` : "";
  }

  frameUrl(index) {
    const ep = EpisodeData._q(this.ep);
    return `/api/frame?${ep ? `${ep}&` : ""}i=${index}`;
  }

  static async load(ep = null, onProgress = () => {}) {
    const q = EpisodeData._q(ep);
    const url = (route) => (q ? `${route}?${q}` : route);

    onProgress("Fetching episode metadata…");
    const meta = await fetch(url("/api/meta")).then((r) => {
      if (!r.ok) throw new Error(`meta ${r.status}`);
      return r.json();
    });

    onProgress("Downloading telemetry…");
    const res = await fetch(url("/api/series"));
    if (!res.ok) throw new Error(`series ${res.status}`);
    const total = Number(res.headers.get("Content-Length") || 0);

    // Stream so a 20 MB payload reports real progress instead of hanging.
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (total) {
        onProgress(
          `Downloading telemetry… ${Math.round((received / total) * 100)}%`,
        );
      }
    }
    const buf = new Uint8Array(received);
    let at = 0;
    for (const c of chunks) { buf.set(c, at); at += c.length; }
    return new EpisodeData(meta, buf.buffer, ep);
  }

  channel(key) { return this.meta.channels[key]; }
  has(key) { return Boolean(this.meta.channels[key]); }

  /** One series of one channel as a Float32Array view. */
  series(key, index) {
    const id = `${key}#${index}`;
    const hit = this._cache.get(id);
    if (hit) return hit;
    const ch = this.meta.channels[key];
    if (!ch) return null;
    const start = ch.offset + index * ch.samples;
    const view = this.all.subarray(start, start + ch.samples);
    this._cache.set(id, view);
    return view;
  }

  /** Series index by name within a channel (joint name, axis name, …). */
  indexOf(key, name) {
    const ch = this.meta.channels[key];
    return ch ? ch.names.indexOf(name) : -1;
  }

  byName(key, name) {
    const i = this.indexOf(key, name);
    return i < 0 ? null : this.series(key, i);
  }

  /** Derive a new array elementwise from two channels' series. */
  derive(keyA, keyB, index, fn) {
    const a = this.series(keyA, index);
    const b = this.series(keyB, index);
    if (!a || !b) return null;
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = fn(a[i], b[i]);
    return out;
  }

  value(key, index, sample) {
    const s = this.series(key, index);
    return s ? s[sample] : NaN;
  }

  topic(name) {
    return this.meta.topics.find((t) => t.topic === name) || null;
  }

  /** Nearest camera frame index for a time in seconds (binary search). */
  frameAt(seconds) {
    const times = this.meta.camera.times_s;
    if (!times.length) return -1;
    let lo = 0, hi = times.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid] < seconds) lo = mid + 1; else hi = mid;
    }
    if (lo > 0 && Math.abs(times[lo - 1] - seconds) <= Math.abs(times[lo] - seconds)) {
      return lo - 1;
    }
    return lo;
  }
}
