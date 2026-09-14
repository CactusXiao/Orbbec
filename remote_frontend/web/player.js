// Backend-rendered video, bounded lookahead and exact presented-frame tracking.
export function chunkWindow(chunks, time, fps, total) {
  const low = Math.max(0, time - 15),
    high = Math.min(total / fps, time + 60);
  return chunks
    .filter((c) => (c.start + c.count) / fps > low && c.start / fps < high)
    .sort((a, b) => {
      const rank = (c) =>
        c.start / fps <= time && (c.start + c.count) / fps > time
          ? -1
          : c.start / fps >= time
            ? c.start / fps - time
            : 10000 + time - c.start / fps;
      return rank(a) - rank(b);
    });
}
// Fetch in parallel, but let the player append in timeline order. Keep only
// three compressed chunks outside SourceBuffer, including in-flight requests.
export class ChunkDownloads {
  constructor(url, fetcher = (...args) => globalThis.fetch(...args)) {
    this.url = url;
    this.fetcher = fetcher;
    this.entries = new Map();
  }
  plan(chunks) {
    const next = chunks.slice(0, 3);
    const wanted = new Set(next.map((c) => c.index));
    for (const [id, entry] of this.entries)
      if (!wanted.has(id)) {
        entry.controller.abort();
        this.entries.delete(id);
      }
    for (const chunk of next) {
      if (this.entries.has(chunk.index)) continue;
      const controller = new AbortController(),
        start = performance.now();
      const promise = (async () => {
        try {
          const response = await this.fetcher(this.url(chunk.index), {
            signal: controller.signal,
          });
          if (!response.ok) throw new Error("视频片段暂不可用");
          const bytes = await response.arrayBuffer();
          return {
            bytes,
            mbps: bytes.byteLength * 8 / Math.max(1, performance.now() - start) / 1000,
          };
        } catch (error) {
          // A later chunk may fail before the player reaches it.
          return { error };
        }
      })();
      this.entries.set(chunk.index, { controller, promise });
    }
  }
  close() {
    for (const entry of this.entries.values()) entry.controller.abort();
    this.entries.clear();
  }
}
export class EpisodePlayer {
  constructor(video, manifest, onStatus) {
    Object.assign(this, {
      video,
      id: manifest.id,
      fps: manifest.fps,
      total: manifest.frames.length,
      onStatus,
      target: 0,
      loaded: new Set(),
      closed: false,
      running: false,
      wantsPlay: false,
      playbackSerial: 0,
      playPending: null,
      seekSerial: 0,
      presented: -1,
    });
    this.abort = new AbortController();
    this.downloads = new ChunkDownloads(
      (index) => `/api/sessions/${this.id}/chunks/${index}.mp4`,
    );
    const mime = `video/mp4; codecs="${manifest.media.codec || "avc1.640028"}"`;
    this.incremental =
      !!manifest.media.chunks && window.MediaSource?.isTypeSupported(mime);
    if (this.incremental) {
      this.source = new MediaSource();
      this.url = URL.createObjectURL(this.source);
      video.src = this.url;
      this.open = new Promise((resolve, reject) =>
        this.source.addEventListener(
          "sourceopen",
          () => {
            if (this.closed) return resolve();
            try {
              this.buffer = this.source.addSourceBuffer(mime);
              this.source.duration = this.total / this.fps;
              resolve();
            } catch (e) {
              reject(e);
            }
          },
          { once: true },
        ),
      );
      // A paused workspace still fills its lookahead; a moving one evicts old chunks.
    }
    this.tick = setInterval(() => {
      this.pump();
      this.resume();
      this.reportPlayback();
    }, 250);
    this.waiting = () => {
      if (this.wantsPlay && !this.closed) {
        if (this.measurement && !this.seeking) this.measurement.waits++;
        video.pause();
        this.buffering = true;
        this.onStatus({});
      }
    };
    video.addEventListener("waiting", this.waiting);
    if (video.requestVideoFrameCallback) {
      const presented = (_, meta) => {
        if (this.measurement && this.wantsPlay && !video.paused && !this.seeking) {
          const now = performance.now();
          const m = this.measurement;
          if (m.lastFrameAt != null) m.maxFrameGapMs = Math.max(m.maxFrameGapMs, now - m.lastFrameAt);
          m.lastFrameAt = now;
          m.frames++;
        }
        this.presented = Math.min(
          this.total - 1,
          Math.floor(meta.mediaTime * this.fps + 0.001),
        );
        if (!this.closed)
          this.callback = video.requestVideoFrameCallback(presented);
      };
      this.callback = video.requestVideoFrameCallback(presented);
    }
    this.update(manifest.media);
  }
  update(state) {
    this.state = state;
    if (state.complete === false && Number.isFinite(state.prepared)) {
      const now = performance.now();
      if (!this.preparationSamples || state.prepared < this.preparationSamples.at(-1).frames)
        this.preparationSamples = [];
      const samples = this.preparationSamples;
      if (!samples.length || state.prepared !== samples.at(-1).frames)
        samples.push({ time: now, frames: state.prepared });
      // Retain a recent window so a slower concurrent task changes the estimate.
      while (samples.length > 2 && samples[1].time < now - 20000) samples.shift();
    }
    if (!this.incremental && !this.fullSource && state.complete !== false) {
      this.video.src = `/api/sessions/${this.id}/preview.mp4`;
      this.video.preload = "auto";
      this.fullSource = true;
    }
    this.pump();
  }
  ahead(time = this.video.currentTime) {
    for (let i = 0; i < this.video.buffered.length; i++)
      if (
        this.video.buffered.start(i) <= time + 0.02 &&
        this.video.buffered.end(i) > time
      )
        return this.video.buffered.end(i) - time;
    return 0;
  }
  async change(fn) {
    const buffer = this.buffer;
    await new Promise((resolve, reject) => {
      const cleanup = () => {
        buffer.removeEventListener("updateend", done);
        buffer.removeEventListener("error", failed);
        this.abort.signal.removeEventListener("abort", failed);
      };
      const done = () => {
        cleanup();
        resolve();
      };
      const failed = () => {
        cleanup();
        reject(new Error("视频片段解码中断"));
      };
      buffer.addEventListener("updateend", done, { once: true });
      buffer.addEventListener("error", failed, { once: true });
      this.abort.signal.addEventListener("abort", failed, { once: true });
      try {
        fn();
      } catch (e) {
        cleanup();
        reject(e);
      }
    });
  }
  async pump() {
    if (!this.incremental || this.running || this.closed) return;
    this.running = true;
    try {
      await this.open;
      while (!this.closed) {
        const time = this.seeking
          ? this.target / this.fps
          : this.video.currentTime;
        const plan = chunkWindow(
          this.state.chunks || [],
          time,
          this.fps,
          this.total,
        );
        const wanted = new Set(plan.map((c) => c.index));
        for (const c of this.state.chunks || [])
          if (this.loaded.has(c.index) && !wanted.has(c.index)) {
            await this.change(() =>
              this.buffer.remove(
                c.start / this.fps,
                (c.start + c.count) / this.fps,
              ),
            );
            this.loaded.delete(c.index);
          }
        // Some browsers evict media under memory pressure independently of us.
        for (const c of plan)
          if (this.loaded.has(c.index)) {
            const start = c.start / this.fps + 0.01,
              end = (c.start + c.count) / this.fps;
            if (this.ahead(start) < end - start - 0.04)
              this.loaded.delete(c.index);
          }
        const missing = plan.filter((c) => !this.loaded.has(c.index));
        this.downloads.plan(missing);
        const c = missing[0];
        if (!c) break;
        const serial = this.seekSerial;
        const { bytes, mbps, error } =
          await this.downloads.entries.get(c.index).promise;
        if (this.closed) break;
        if (serial !== this.seekSerial) continue;
        if (error) {
          this.downloads.entries.delete(c.index);
          throw error;
        }
        this.buffer.timestampOffset = c.start / this.fps;
        await this.change(() => this.buffer.appendBuffer(bytes));
        this.loaded.add(c.index);
        this.downloads.entries.delete(c.index);
        this.onStatus({ mbps });
        this.resume();
      }
      // End at the known final timestamp so native ended/playback-complete fires.
      // A later remove/append reopens an ended MediaSource for backward seeks.
      const last = this.state.chunks?.at(-1);
      if (
        !this.closed &&
        this.state.complete &&
        last &&
        this.loaded.has(last.index) &&
        this.source.readyState === "open"
      )
        this.source.endOfStream();
    } catch (e) {
      if (!this.closed)
        this.onStatus({
          error:
            e.name === "QuotaExceededError"
              ? "浏览器视频缓存不足，请关闭其他占用内存的页面后重试"
              : e.message + "；自动重试中",
        });
    } finally {
      this.running = false;
    }
  }
  async seek(position) {
    const serial = ++this.seekSerial;
    this.target = Math.max(0, Math.min(this.total - 1, position));
    this.seeking = true;
    if (this.incremental)
      this.downloads.plan(
        chunkWindow(this.state.chunks || [], this.target / this.fps, this.fps, this.total)
          .filter((c) => !this.loaded.has(c.index)),
      );
    this.pump();
    const time = (this.target + 0.25) / this.fps,
      deadline = performance.now() + 120000;
    try {
      while (
        !this.closed &&
        serial === this.seekSerial &&
        (!this.video.readyState || this.ahead(time) <= 0)
      ) {
        if (performance.now() > deadline)
          throw new Error("目标帧仍在准备，请稍后重试");
        await new Promise((r) => setTimeout(r, 40));
      }
      if (this.closed || serial !== this.seekSerial) return false;
      if (
        Math.abs(this.video.currentTime - time) > 0.0001 ||
        this.presented !== this.target
      )
        this.video.currentTime = time;
      while (!this.closed && serial === this.seekSerial) {
        const exact = this.video.requestVideoFrameCallback
          ? this.presented === this.target
          : !this.video.seeking &&
            Math.floor(this.video.currentTime * this.fps) === this.target;
        if (!this.video.seeking && this.video.readyState >= 2 && exact)
          return true;
        if (performance.now() > deadline)
          throw new Error("目标视频帧尚未显示，请重试");
        await new Promise((r) => setTimeout(r, 16));
      }
      return false;
    } finally {
      if (serial === this.seekSerial) this.seeking = false;
    }
  }
  async play(position) {
    if (this.closed) return false;
    this.measurement = {start:performance.now(),position,frames:0,waits:0,maxFrameGapMs:0};
    const serial = ++this.playbackSerial;
    this.wantsPlay = true;
    this.buffering = true;
    this.onStatus({});
    try {
      const ready = await this.seek(position);
      if (serial !== this.playbackSerial || !this.wantsPlay || this.closed)
        return false;
      if (ready) this.resume();
      return ready;
    } catch (e) {
      if (serial !== this.playbackSerial || this.closed) return false;
      this.pause();
      throw e;
    }
  }
  resume() {
    if (
      !this.wantsPlay ||
      this.closed ||
      this.seeking ||
      !this.video.paused ||
      this.playPending
    )
      return;
    const needed = Math.min(6, this.total / this.fps - this.video.currentTime);
    if (this.ahead() + 0.04 < needed || this.video.readyState < 3) return;
    if (!this.preparationReady()) return;
    const serial = this.playbackSerial;
    this.buffering = false;
    const pending = (this.playPending = { serial });
    Promise.resolve(this.video.play())
      .then(() => {
        // User pause/exit wins over a late media play completion.
        if (!this.wantsPlay || this.closed) this.video.pause();
      })
      .catch((e) => {
        // An aborted old request must not cancel a newer play request.
        if (serial !== this.playbackSerial || !this.wantsPlay || this.closed)
          return;
        this.wantsPlay = false;
        this.onStatus({ error: "请再次点击播放：" + e.message });
      })
      .finally(() => {
        if (this.playPending === pending) this.playPending = null;
      });
  }
  preparationReady(now = performance.now()) {
    if (this.state?.complete !== false) return true;
    const prepared = this.state.prepared || 0;
    if (prepared >= this.total) return true;
    const samples = this.preparationSamples || [];
    if (samples.length < 2) return false;
    const first = samples[0], last = samples.at(-1);
    const elapsed = (now - first.time) / 1000;
    if (elapsed < 6) return false;
    // Start when the remaining preparation fits inside playback, with 25%
    // throughput headroom and a six-second reserve. Pauses remain cancellable.
    const rate = (last.frames - first.frames) / this.fps / elapsed * 0.75;
    const remaining = this.total / this.fps - this.video.currentTime;
    return rate > 0 && (this.total - prepared) / this.fps / rate + 6 <= remaining;
  }
  pause() {
    if (this.measurement && this.measurement.end == null) this.measurement.end = performance.now();
    ++this.playbackSerial;
    this.wantsPlay = false;
    this.buffering = false;
    this.playPending = null;
    this.video.pause();
    this.reportPlayback();
    this.onStatus({});
  }
  reportPlayback() {
    const m = this.measurement;
    if (!m || !this.video.dataset) return;
    const quality = this.video.getVideoPlaybackQuality?.();
    // DOM-only diagnostics: no extra UI and no task/video data transmitted.
    this.video.dataset.playback = JSON.stringify({
      wallSeconds:((m.end ?? performance.now()) - m.start) / 1000,
      mediaSeconds:Math.max(0, this.video.currentTime - m.position / this.fps),
      framesPresented:m.frames, waits:m.waits, maxFrameGapMs:m.maxFrameGapMs,
      droppedFrames:quality?.droppedVideoFrames ?? this.video.webkitDroppedFrameCount ?? null,
      decodedFrames:quality?.totalVideoFrames ?? this.video.webkitDecodedFrameCount ?? null,
      ended:this.video.ended, paused:this.video.paused,
    });
  }
  close() {
    this.closed = true;
    this.seekSerial++;
    this.pause();
    this.abort.abort();
    this.downloads.close();
    clearInterval(this.tick);
    this.video.removeEventListener("waiting", this.waiting);
    if (this.callback) this.video.cancelVideoFrameCallback(this.callback);
    this.video.removeAttribute("src");
    this.video.load();
    if (this.url) URL.revokeObjectURL(this.url);
  }
}
