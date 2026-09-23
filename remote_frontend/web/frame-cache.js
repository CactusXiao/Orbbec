import { mediaURL } from "./media-routing.js";
// Cache expendable images separately from durable annotation drafts.
export function labelLookahead(frames, cameras, position, selected, overview) {
  const plan = [],
    seen = new Set();
  const add = (p, camera) => {
    if (p < 0 || p >= frames.length) return;
    const key = `${camera}:${frames[p]}`;
    if (!seen.has(key)) {
      seen.add(key);
      plan.push({ camera, frame: frames[p] });
    }
  };
  // Ego participates in the same camera contract as the fixed RGB views.
  const primary = overview ? cameras : [selected];
  for (let d = 1; d <= (overview ? 4 : 20); d++)
    for (const c of primary) add(position + d, c);
  for (let d = 1; d <= 2; d++) for (const c of primary) add(position - d, c);
  for (let d = 0; d <= 2; d++) for (const c of cameras) add(position + d, c);
  return plan;
}
export class FrameCache {
  constructor(database) {
    this.database = database;
    this.urls = new Map();
    this.active = 0;
    this.waiters = [];
    this.writes = Promise.resolve();
    this.limit = 512 * 1024 * 1024;
  }
  async slot(signal, background) {
    if (signal?.aborted) throw new DOMException("Cancelled", "AbortError");
    if (this.active >= 4)
      await new Promise((resolve) =>
        background ? this.waiters.push(resolve) : this.waiters.unshift(resolve),
      );
    else this.active++;
    if (signal?.aborted) {
      this.release();
      throw new DOMException("Cancelled", "AbortError");
    }
  }
  release() {
    const next = this.waiters.shift();
    if (next) next();
    else this.active--;
  }
  async cached(key) {
    const db = await this.database;
    return new Promise((resolve, reject) => {
      const t = db.transaction(["images", "image_meta"], "readwrite"),
        r = t.objectStore("images").get(key);
      r.onsuccess = () => {
        if (r.result)
          t.objectStore("image_meta").put(
            { size: r.result.size, used: Date.now() },
            key,
          );
      };
      t.oncomplete = () => resolve(r.result);
      t.onabort = () => reject(t.error);
    });
  }
  async store(key, blob) {
    this.writes = this.writes
      .catch(() => {})
      .then(async () => {
        const db = await this.database;
        await new Promise((resolve, reject) => {
          const t = db.transaction(["images", "image_meta"], "readwrite"),
            meta = t.objectStore("image_meta"),
            images = t.objectStore("images"),
            rows = [];
          const cursor = meta.openCursor();
          cursor.onsuccess = () => {
            const c = cursor.result;
            if (c) {
              if (c.key !== key) rows.push({ key: c.key, ...c.value });
              c.continue();
              return;
            }
            let size = rows.reduce((n, r) => n + r.size, blob.size);
            rows.sort((a, b) => a.used - b.used);
            for (const row of rows) {
              if (size <= this.limit) break;
              images.delete(row.key);
              meta.delete(row.key);
              size -= row.size;
            }
            if (blob.size <= this.limit) {
              images.put(blob, key);
              meta.put({ size: blob.size, used: Date.now() }, key);
            }
          };
          t.oncomplete = resolve;
          t.onabort = () => reject(t.error);
        });
      });
    return this.writes;
  }
  async blob(
    session,
    camera,
    frame,
    layer = "frames",
    signal,
    background = false,
  ) {
    const key = `${session}:${layer}:${camera}:${frame}`;
    const cached = await this.cached(key).catch(() => null);
    if (cached) return cached;
    await this.slot(signal, background);
    try {
      const r = await fetch(
        mediaURL(session, `${layer}/${camera}/${frame}`),
        { signal },
      );
      if (!r.ok) throw new Error("画面尚未准备完成");
      const blob = await r.blob();
      // Quota failures must never block viewing or delete a draft.
      await this.store(key, blob).catch(() => {
        this.limit = Math.max(32 * 1024 * 1024, Math.floor(this.limit / 2));
      });
      return blob;
    } finally {
      this.release();
    }
  }
  async url(session, camera, frame, layer = "frames") {
    const key = `${session}:${layer}:${camera}:${frame}`;
    if (this.urls.has(key)) {
      const url = this.urls.get(key);
      this.urls.delete(key);
      this.urls.set(key, url);
      return url;
    }
    const blob = await this.blob(session, camera, frame, layer),
      url = URL.createObjectURL(blob);
    if (this.urls.has(key)) URL.revokeObjectURL(this.urls.get(key));
    this.urls.set(key, url);
    while (this.urls.size > 64) {
      const [old, value] = this.urls.entries().next().value;
      URL.revokeObjectURL(value);
      this.urls.delete(old);
    }
    return url;
  }
  prefetch(session, plan) {
    this.cancel();
    const controller = (this.prefetchAbort = new AbortController()),
      queue = [...plan];
    const worker = async () => {
      while (queue.length && !controller.signal.aborted) {
        const { camera, frame } = queue.shift();
        try {
          await this.blob(
            session,
            camera,
            frame,
            "frames",
            controller.signal,
            true,
          );
        } catch {}
      }
    };
    worker();
    worker();
  }
  cancel() {
    this.prefetchAbort?.abort();
  }
  clearURLs() {
    this.cancel();
    for (const url of this.urls.values()) URL.revokeObjectURL(url);
    this.urls.clear();
  }
}
