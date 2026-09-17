/* Shared view state for the episode viewer.
 *
 * One store owns the time window, playhead, hover and brush so that every
 * chart, the scrubber and the camera all read from the same slice. Charts never
 * hold their own filter — that is the single filter row rule.
 */

export class Store {
  constructor(timeline) {
    this.timeline = timeline;            // { rate, samples, duration }
    this.view = { start: 0, end: timeline.duration };
    this.playhead = 0;
    this.hover = null;
    this.brush = null;
    this.playing = false;
    this.speed = 1;
    this._subs = new Set();
    this._frameQueued = false;
  }

  subscribe(fn) { this._subs.add(fn); return () => this._subs.delete(fn); }

  /** Coalesce bursts of pointer events into one repaint per animation frame. */
  emit(kind) {
    this._pending = this._pending || new Set();
    this._pending.add(kind);
    if (this._frameQueued) return;
    this._frameQueued = true;
    requestAnimationFrame(() => {
      const kinds = this._pending;
      this._pending = new Set();
      this._frameQueued = false;
      for (const fn of this._subs) fn(kinds);
    });
  }

  clamp(t) {
    // A pointer event before the first layout pass can produce NaN; letting it
    // reach the playhead would blank every chart until the next seek.
    if (!Number.isFinite(t)) return this.playhead;
    return Math.min(Math.max(t, 0), this.timeline.duration);
  }

  seek(t) {
    this.playhead = this.clamp(t);
    this.emit("playhead");
  }

  setHover(t) {
    const next = t === null ? null : this.clamp(t);
    if (next === this.hover) return;
    this.hover = next;
    this.emit("hover");
  }

  setBrush(range) { this.brush = range; this.emit("brush"); }
  clearBrush() {
    if (!this.brush) return;
    this.brush = null;
    this.emit("brush");
  }

  /** Turn the in-progress drag into the new visible window. */
  commitBrush() {
    if (!this.brush) return;
    const [a, b] = this.brush;
    this.brush = null;
    if (b - a < 0.05) { this.emit("brush"); return; }
    this.view = { start: this.clamp(a), end: this.clamp(b) };
    this.playhead = this.clamp(this.playhead);
    this.emit("view");
  }

  setView(start, end) {
    this.view = { start: this.clamp(start), end: this.clamp(end) };
    this.emit("view");
  }

  resetView() {
    this.view = { start: 0, end: this.timeline.duration };
    this.emit("view");
  }

  get zoomed() {
    return this.view.start > 0.001 ||
      this.view.end < this.timeline.duration - 0.001;
  }

  setPlaying(on) { this.playing = on; this.emit("transport"); }
  setSpeed(v) { this.speed = v; this.emit("transport"); }

  /** Nearest sample index for a time, clamped to the grid. */
  indexAt(t) {
    return Math.min(
      this.timeline.samples - 1,
      Math.max(0, Math.round(t * this.timeline.rate)),
    );
  }
}
