// Port of label/canvas_view.py. All coordinates and pointer state stay local.
const clone = (x) => structuredClone(x);
const schematic = [
  [0, 0.95],
  [-0.35, 0.25],
  [-0.44, -0.2],
  [-0.48, -0.6],
  [0, 0.2],
  [-0.02, -0.3],
  [-0.02, -0.75],
  [0.65, 0.35],
  [0.78, 0],
  [0.86, -0.3],
  [0.35, 0.25],
  [0.42, -0.2],
  [0.45, -0.6],
  [-0.55, 0.35],
  [-0.82, 0.05],
  [-1, -0.22],
  [-1.12, -0.48],
  [-0.5, -0.95],
  [-0.02, -1.15],
  [0.48, -0.95],
  [0.92, -0.58],
];
const styles = [
  [0, 0],
  [1, 0],
  [1, 1],
  [1, 2],
  [2, 0],
  [2, 1],
  [2, 2],
  [4, 0],
  [4, 1],
  [4, 2],
  [3, 0],
  [3, 1],
  [3, 2],
  [0, 0],
  [0, 1],
  [0, 2],
  [0, 3],
  [1, 3],
  [2, 3],
  [3, 3],
  [4, 3],
];
const bases = [
  ["#0078ff", "#28b4ff", "#50dcff", "#78ffdc", "#b4ffb4"],
  ["#ff4600", "#ff7800", "#ffb400", "#ffdc3c", "#ffff78"],
];
function color(h, j) {
  if (!j) return h ? "#d2d2ff" : "#f2f2f2";
  const [f, s] = styles[j],
    hex = bases[h][f];
  return (
    "#" +
    [1, 3, 5]
      .map((i) =>
        Math.round(
          parseInt(hex.slice(i, i + 2), 16) * (1 - 0.1 * s) + 255 * 0.1 * s,
        )
          .toString(16)
          .padStart(2, "0"),
      )
      .join("")
  );
}
export class LabelCanvas {
  constructor(
    canvas,
    {
      changed = () => {},
      track = () => {},
      cancelled = () => {},
      schematics = true,
    } = {},
  ) {
    this.canvas = canvas;
    this.onChanged = changed;
    this.onTrack = track;
    this.onCancelled = cancelled;
    this.schematics = schematics;
    this.history = [];
    this.sample = null;
    this.image = null;
    this.overlay = null;
    this.readOnly = true;
    this.edges = [];
    this.tracked = [];
    this.counts = [];
    this.view = { scale: 1, x: 0, y: 0 };
    this.locate = null;
    this.press = null;
    this.selection = null;
    this.pan = null;
    this.key = "";
    this.imageGeneration = 0;
    this.resizeObserver = new ResizeObserver(() => {
      const width = this.width, height = this.height;
      this.resize();
      if (!width || !height || width <= 1 || height <= 1) this.fit();
      else {
        this.view.x += (this.width - width) / 2;
        this.view.y += (this.height - height) / 2;
      }
      this.draw();
    });
    this.resizeObserver.observe(canvas);
    canvas.tabIndex = 0;
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("pointerdown", (e) =>
      canvas.setPointerCapture(e.pointerId),
    );
    // MouseEvent.detail carries the browser's double-click count; PointerEvent
    // detail is zero in WebKit. Preserve the first click for native tracking.
    canvas.addEventListener("mousedown", (e) => this.down(e));
    canvas.addEventListener("pointermove", (e) => this.move(e));
    canvas.addEventListener("pointerup", (e) => this.up(e));
    canvas.addEventListener("pointercancel", () => this.finish());
    canvas.addEventListener("dblclick", (e) => this.double(e));
    canvas.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        this.locate = null;
        this.draw();
      }
    });
    canvas.addEventListener(
      "wheel",
      (e) => {
        if (!this.image) return;
        e.preventDefault();
        const [x, y] = this.point(e),
          [ix, iy] = this.toImage(x, y);
        this.view.scale = Math.max(
          0.05,
          Math.min(10, this.view.scale * (e.deltaY < 0 ? 1.1 : 1 / 1.1)),
        );
        this.view.x = x - ix * this.view.scale;
        this.view.y = y - iy * this.view.scale;
        this.draw();
      },
      { passive: false },
    );
  }
  resize() {
    const r = this.canvas.getBoundingClientRect(),
      d = devicePixelRatio || 1;
    this.width = Math.max(1, r.width);
    this.height = Math.max(1, r.height);
    this.canvas.width = Math.round(this.width * d);
    this.canvas.height = Math.round(this.height * d);
    this.ctx = this.canvas.getContext("2d");
    this.ctx.setTransform(d, 0, 0, d, 0, 0);
  }
  fit() {
    if (!this.image) return;
    const w = this.imageWidth,
      h = this.imageHeight;
    const s = Math.max(
      0.05,
      Math.min(10, Math.min(this.width / w, this.height / h)),
    );
    this.view = {
      scale: s,
      x: (this.width - w * s) / 2,
      y: (this.height - h * s) / 2,
    };
  }
  async setImage(url, key, viewKey = key) {
    const generation = ++this.imageGeneration;
    const image = new Image();
    image.src = url;
    await image.decode();
    if (generation !== this.imageGeneration) return;
    const preserveView = this.image && this.viewKey === viewKey &&
      this.imageWidth === image.naturalWidth && this.imageHeight === image.naturalHeight;
    this.viewKey = viewKey;
    this.image = image;
    this.imageWidth = image.naturalWidth;
    this.imageHeight = image.naturalHeight;
    this.crop = null;
    this.renderCrop = null;
    this.key = key;
    this.history = [];
    this.locate = null;
    this.resize();
    if (!preserveView) this.fit();
    this.draw();
  }
  videoLayers(video, crop, rendered, opacity) {
    this.overlayGeneration = (this.overlayGeneration || 0) + 1;
    this.overlay = null;
    this.renderCrop = rendered;
    this.overlayOpacity = opacity;
    this.videoFrame(video, crop);
  }
  videoFrame(video, crop) {
    this.image = video;
    this.crop = crop;
    if (this.imageWidth !== crop[2] || this.imageHeight !== crop[3]) {
      this.imageWidth = crop[2];
      this.imageHeight = crop[3];
      this.fit();
    }
    this.draw();
  }
  setState(
    sample,
    {
      edges = [],
      tracked = [],
      counts = [],
      readOnly = true,
      annotation = true,
    } = {},
  ) {
    this.sample = sample;
    this.edges = edges;
    this.tracked = tracked;
    this.counts = counts;
    this.readOnly = readOnly;
    this.annotation = annotation;
    this.draw();
  }
  async setOverlay(url, opacity = 1) {
    const generation = (this.overlayGeneration =
      (this.overlayGeneration || 0) + 1);
    const imageGeneration = this.imageGeneration;
    this.overlayOpacity = opacity;
    if (!url) {
      this.overlay = null;
      this.draw();
      return;
    }
    const image = new Image();
    image.src = url;
    await image.decode();
    if (
      generation !== this.overlayGeneration ||
      imageGeneration !== this.imageGeneration
    )
      return;
    this.overlay = image;
    this.draw();
  }
  point(e) {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }
  toImage(x, y) {
    return [
      (x - this.view.x) / this.view.scale,
      (y - this.view.y) / this.view.scale,
    ];
  }
  toCanvas(x, y) {
    return [
      this.view.x + x * this.view.scale,
      this.view.y + y * this.view.scale,
    ];
  }
  schematicPoints(count = false) {
    if (!this.schematics || !this.annotation) return [];
    const scale = count ? 30 : 120,
      centers = count
        ? [
            [this.width - 147, this.height - 54],
            [this.width - 55, this.height - 54],
          ]
        : [
            [175.5, this.height - 171],
            [540.5, this.height - 171],
          ];
    return centers.flatMap(([cx, cy], h) =>
      schematic.map(([x, y], j) => ({
        h,
        j,
        x: cx + (h === 0 ? -x : x) * scale,
        y: cy + y * scale,
        count,
        r: count ? 9 : 27.6,
      })),
    );
  }
  hitSchematic(x, y) {
    for (const count of [false, true]) {
      let hit = null,
        best = Infinity;
      for (const p of this.schematicPoints(count)) {
        const d = (x - p.x) ** 2 + (y - p.y) ** 2;
        if (d <= p.r * p.r && d <= best) {
          hit = p;
          best = d;
        }
      }
      if (hit) return hit;
    }
    return null;
  }
  nearest(x, y, max = 12) {
    let hit = null,
      best = max * max;
    if (!this.sample) return null;
    this.sample.points.forEach((hand, h) =>
      hand.forEach((p, j) => {
        if (p[0] === -1 && p[1] === -1) return;
        const [cx, cy] = this.toCanvas(...p),
          d = (cx - x) ** 2 + (cy - y) ** 2;
        if (d <= best) {
          hit = { h, j };
          best = d;
        }
      }),
    );
    return hit;
  }
  push() {
    this.history.push(
      clone({ points: this.sample.points, visible: this.sample.visible }),
    );
  }
  changed() {
    const cancelled = this.tracked.filter(
      ([h, j]) => !this.sample.visible[h][j],
    );
    if (cancelled.length) this.onCancelled(cancelled);
    this.onChanged();
    this.draw();
  }
  undo() {
    if (this.readOnly || !this.history.length) return;
    Object.assign(this.sample, this.history.pop());
    this.locate = null;
    this.changed();
  }
  ignore() {
    if (this.readOnly) return;
    this.push();
    this.sample.visible.forEach((h) => h.fill(false));
    this.locate = null;
    this.changed();
  }
  down(e) {
    this.canvas.focus();
    const [x, y] = this.point(e);
    if (e.button === 2) {
      e.preventDefault();
      const hit = this.hitSchematic(x, y);
      if (this.locate) {
        const [ix, iy] = this.toImage(x, y);
        if (
          !hit &&
          ix >= 0 &&
          iy >= 0 &&
          ix <= this.imageWidth &&
          iy <= this.imageHeight
        ) {
          this.push();
          this.place(this.locate, ix, iy);
          this.changed();
        }
        this.locate = null;
        this.draw();
        return;
      }
      const target = !this.readOnly && (hit || this.nearest(x, y, 14));
      if (target) {
        this.finish();
        this.locate = target;
        this.draw();
        return;
      }
      this.pan = { x, y };
      return;
    }
    if (e.button !== 0 || this.readOnly || this.locate || !this.sample) return;
    clearTimeout(this.timer);
    if (e.detail >= 2) return;
    const hit = this.hitSchematic(x, y);
    if (hit) {
      this.firstClick = {
        ...hit,
        visible: this.sample.visible[hit.h][hit.j],
        tracked: this.tracked.some(([h, j]) => h === hit.h && j === hit.j),
      };
      this.push();
      this.sample.visible[hit.h][hit.j] = !this.sample.visible[hit.h][hit.j];
      this.changed();
      return;
    }
    this.firstClick = null;
    this.press = {
      x,
      y,
      current: [x, y],
      hit: this.nearest(x, y),
      dragging: false,
    };
    this.timer = setTimeout(() => {
      if (this.press && !this.press.dragging) {
        this.selection = [this.press.x, this.press.y, ...this.press.current];
        this.draw();
      }
    }, 380);
  }
  place(hit, x, y) {
    this.sample.points[hit.h][hit.j] = [
      Math.max(0, Math.min(this.imageWidth - 1, x)),
      Math.max(0, Math.min(this.imageHeight - 1, y)),
    ];
    this.sample.visible[hit.h][hit.j] = true;
  }
  move(e) {
    const [x, y] = this.point(e);
    if (this.pan) {
      this.view.x += x - this.pan.x;
      this.view.y += y - this.pan.y;
      this.pan = { x, y };
      this.draw();
      return;
    }
    const p = this.press;
    if (!p || this.readOnly || this.locate) return;
    p.current = [x, y];
    if (this.selection) {
      this.selection = [p.x, p.y, x, y];
      this.draw();
      return;
    }
    if (p.hit && !p.dragging && (x - p.x) ** 2 + (y - p.y) ** 2 > 25) {
      clearTimeout(this.timer);
      p.dragging = true;
      this.push();
    }
    if (p.dragging) {
      this.place(p.hit, ...this.toImage(x, y));
      this.draw();
    }
  }
  up(e) {
    if (this.selection && !this.readOnly) {
      const [x0, y0] = this.selection,
        [x1, y1] = this.point(e),
        hits = [];
      this.sample.points.forEach((hand, h) =>
        hand.forEach((p, j) => {
          if (p[0] === -1 && p[1] === -1) return;
          const [x, y] = this.toCanvas(...p);
          if (
            x >= Math.min(x0, x1) &&
            x <= Math.max(x0, x1) &&
            y >= Math.min(y0, y1) &&
            y <= Math.max(y0, y1)
          )
            hits.push([h, j]);
        }),
      );
      if (hits.length) {
        this.push();
        for (const [h, j] of hits)
          this.sample.visible[h][j] = !this.sample.visible[h][j];
        this.changed();
      }
    }
    this.finish();
  }
  finish() {
    clearTimeout(this.timer);
    if (this.press?.dragging) this.changed();
    this.press = null;
    this.selection = null;
    this.pan = null;
    this.draw();
  }
  double(e) {
    if (this.readOnly || this.locate) return;
    clearTimeout(this.timer);
    this.press = null;
    const [x, y] = this.point(e),
      hit = this.hitSchematic(x, y);
    if (hit) {
      if (!hit.count) {
        const first = this.firstClick;
        if (first && first.h === hit.h && first.j === hit.j) {
          this.sample.visible[hit.h][hit.j] = first.visible;
          this.history.pop();
        }
        if (!first?.tracked) this.onTrack(hit.h, hit.j);
        this.firstClick = null;
        this.changed();
      }
      return;
    }
    const point = this.nearest(x, y, 14);
    if (point) {
      this.push();
      this.sample.visible[point.h][point.j] =
        !this.sample.visible[point.h][point.j];
      this.changed();
    }
  }
  draw() {
    const c = this.ctx;
    if (!c) return;
    c.clearRect(0, 0, this.width, this.height);
    c.fillStyle = "#202d3e";
    c.fillRect(0, 0, this.width, this.height);
    if (this.image && !(this.renderCrop && (this.overlayOpacity ?? 1) >= 1)) {
      const v = this.view;
      try {
        if (this.crop)
          c.drawImage(
            this.image,
            ...this.crop,
            v.x,
            v.y,
            this.imageWidth * v.scale,
            this.imageHeight * v.scale,
          );
        else
          c.drawImage(
            this.image,
            v.x,
            v.y,
            this.imageWidth * v.scale,
            this.imageHeight * v.scale,
          );
      } catch {}
    }
    if (this.renderCrop && this.image && (this.overlayOpacity ?? 1) > 0) {
      const v = this.view;
      c.save();
      c.globalAlpha = this.overlayOpacity ?? 1;
      try {
        c.drawImage(
          this.image,
          ...this.renderCrop,
          v.x,
          v.y,
          this.imageWidth * v.scale,
          this.imageHeight * v.scale,
        );
      } catch {}
      c.restore();
    }
    if (this.overlay) {
      const v = this.view;
      c.save();
      c.globalAlpha = this.overlayOpacity ?? 1;
      c.drawImage(
        this.overlay,
        v.x,
        v.y,
        this.imageWidth * v.scale,
        this.imageHeight * v.scale,
      );
    }
    c.globalAlpha = 1;
    if (this.overlay) c.restore();
    if (!this.sample || !this.annotation) return;
    const line = (a, b, stroke, width = 1) => {
      c.beginPath();
      c.moveTo(...a);
      c.lineTo(...b);
      c.strokeStyle = stroke;
      c.lineWidth = width;
      c.stroke();
    };
    const circle = (x, y, r, fill, stroke, width = 1) => {
      c.beginPath();
      c.arc(x, y, r, 0, Math.PI * 2);
      if (fill) {
        c.fillStyle = fill;
        c.fill();
      }
      c.strokeStyle = stroke;
      c.lineWidth = width;
      c.stroke();
    };
    const ring = (h, j, x, y, r = 7) => {
      if (this.tracked.some((p) => p[0] === h && p[1] === j)) {
        circle(x, y, r + 1, null, "#111", 4);
        circle(x, y, r, null, "#ffea00", 2);
      }
    };
    for (let h = 0; h < 2; h++) {
      for (const [a, b] of this.edges) {
        if (!this.sample.visible[h][a] || !this.sample.visible[h][b]) continue;
        const p = this.sample.points[h][a],
          q = this.sample.points[h][b];
        if (p.every((v) => v === -1) || q.every((v) => v === -1)) continue;
        const pa = this.toCanvas(...p),
          pb = this.toCanvas(...q),
          g = c.createLinearGradient(...pa, ...pb);
        g.addColorStop(0, color(h, a));
        g.addColorStop(1, color(h, b));
        line(pa, pb, g);
      }
      this.sample.points[h].forEach((p, j) => {
        if (p.every((v) => v === -1)) return;
        const [x, y] = this.toCanvas(...p),
          focus = this.locate?.h === h && this.locate?.j === j;
        circle(
          x,
          y,
          focus ? 5 : 2,
          this.sample.visible[h][j] || focus ? color(h, j) : null,
          focus ? "white" : color(h, j),
          focus ? 2 : 1,
        );
        ring(h, j, x, y);
      });
    }
    for (const count of [false, true]) {
      const points = this.schematicPoints(count);
      for (let h = 0; h < 2; h++)
        for (const [a, b] of this.edges) {
          const p = points.find((p) => p.h === h && p.j === a),
            q = points.find((p) => p.h === h && p.j === b);
          if (p && q)
            line(
              [p.x, p.y],
              [q.x, q.y],
              count ? "#6a6a6a" : "#8a8a8a",
              count ? 1 : 2,
            );
        }
      for (const p of points) {
        const n = this.counts[p.h]?.[p.j] || 0,
          co = count
            ? n <= 0
              ? null
              : n === 1
                ? "#ff3b30"
                : n === 2
                  ? "#ffd60a"
                  : "#30d158"
            : p.j === 0
              ? p.h
                ? "#ff7800"
                : "#0078ff"
              : color(p.h, p.j);
        const focus = this.locate?.h === p.h && this.locate?.j === p.j;
        circle(
          p.x,
          p.y,
          count ? 4 : 15,
          count ? co : this.sample.visible[p.h][p.j] ? co : null,
          focus ? "white" : count ? "#5a5a5a" : co,
          focus ? 2 : 1,
        );
        if (!count) ring(p.h, p.j, p.x, p.y, 19);
      }
    }
    if (this.locate) {
      c.fillStyle = "white";
      c.font = "14px sans-serif";
      c.textAlign = "center";
      c.fillText(
        `定位模式：${this.locate.h ? "右手" : "左手"}关节 ${this.locate.j}，右键图像完成定位，右键图外退出（Esc 取消）`,
        this.width / 2,
        24,
      );
    }
    if (this.selection) {
      const [x0, y0, x1, y1] = this.selection;
      c.setLineDash([4, 2]);
      c.strokeStyle = "#f6d44a";
      c.strokeRect(x0, y0, x1 - x0, y1 - y0);
      c.setLineDash([]);
    }
  }
  close() {
    this.resizeObserver.disconnect();
    this.finish();
  }
}
