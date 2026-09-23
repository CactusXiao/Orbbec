// Browser ports of src/qc/app.py and src/qc/state_store.py state transitions.
// Transport and rendering are deliberately outside these functions.
export function normalizeRanges(ranges, gap = 5) {
  const sorted = ranges
    .map(([a, b]) => [Math.min(a, b), Math.max(a, b)])
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const out = [];
  for (const [a, b] of sorted) {
    const prev = out.at(-1);
    if (prev && (a <= prev[1] + 1 || a - prev[1] - 1 < Math.max(0, gap)))
      prev[1] = Math.max(prev[1], b);
    else out.push([a, b]);
  }
  return out;
}
export function normalizeSegments(segments, gap = 5) {
  const cameras = new Map();
  for (const s of segments) {
    const camera = s.primary_camera || "";
    if (!cameras.has(camera)) cameras.set(camera, []);
    cameras.get(camera).push([s.start_frame, s.end_frame]);
  }
  return [...cameras].flatMap(([camera, ranges]) =>
    normalizeRanges(ranges, gap).map(([start_frame, end_frame]) => ({
      start_frame, end_frame, ...(camera ? {primary_camera: camera} : {}),
    })),
  ).sort((a, b) => a.start_frame - b.start_frame || a.end_frame - b.end_frame ||
    (a.primary_camera || "").localeCompare(b.primary_camera || ""));
}

// Preserve QC boundaries, including adjacent/overlapping error intervals.
export function labelSegments(manifest) {
  const segments = (manifest.qc_segments || []).map((s, i) => ({
    ...s, key: String(s.segment_id || `${s.start_frame}:${s.end_frame}:${s.primary_camera || ""}:${i}`),
    positions: manifest.frames.flatMap((f, p) => f >= s.start_frame && f <= s.end_frame ? [p] : []),
  })).filter(s => s.positions.length);
  const covered = new Set(segments.flatMap(s => s.positions));
  let last = null;
  manifest.frames.forEach((f, p) => {
    if (covered.has(p)) { last = null; return; }
    if (!last || f !== last.end_frame + 1) {
      last = {key: `frames:${f}`, start_frame: f, end_frame: f, positions: []};
      segments.push(last);
    }
    last.end_frame = f;
    last.positions.push(p);
  });
  return segments.sort((a,b) => a.start_frame - b.start_frame || a.end_frame - b.end_frame);
}
export function activeLabelSegment(draft, position) {
  const segments = labelSegments(draft.manifest);
  return segments.find(s => s.key === draft.activeSegment && s.positions.includes(position)) ||
    segments.find(s => s.positions.includes(position));
}
export function labelStep(draft, position, delta) {
  const segment = activeLabelSegment(draft, position);
  if (!segment) return position;
  return segment.positions[Math.max(0, Math.min(segment.positions.length - 1,
    segment.positions.indexOf(position) + delta))];
}

export function enterLabelSegment(draft, frame, segment = null) {
  const visited = new Set(draft.visitedSegments || []);
  let camera = null;
  for (const s of segment ? [segment] : draft.manifest.qc_segments || []) {
    if (frame < s.start_frame || frame > s.end_frame) continue;
    const key = String(s.segment_id || `${s.start_frame}:${s.end_frame}:${s.primary_camera || ""}`);
    if (visited.has(key)) continue;
    visited.add(key);
    if (camera === null && (draft.manifest.cameras.includes(s.primary_camera) || s.primary_camera === "ego"))
      camera = s.primary_camera;
  }
  draft.visitedSegments = [...visited];
  return camera;
}

export class QcWorkflow {
  constructor(frames, result, position = 0) {
    this.frames = frames;
    this.result = result;
    this.position = position;
    this.mode = "playback";
    this.playing = false;
    this.anchor = position;
    this.start = null;
    this.end = null;
    this.primaryCamera = null;
  }
  step(delta) {
    if (this.playing) return this.position;
    const lower = this.mode === "bad_range" ? Math.max(0, this.anchor - 9) : 0;
    this.position = Math.max(
      lower,
      Math.min(this.frames.length - 1, this.position + delta),
    );
    return this.position;
  }
  seek(position) {
    if (this.mode !== "playback") return false;
    this.position = Math.max(0, Math.min(this.frames.length - 1, position));
    return true;
  }
  play() {
    if (this.mode !== "playback") return false;
    if (
      this.result.playback_complete &&
      this.position === this.frames.length - 1
    )
      this.position = 0;
    this.playing = true;
    return true;
  }
  displayed(position) {
    if (!this.playing) return;
    this.position = position;
    if (position === this.frames.length - 1) {
      this.result.playback_complete = true;
      this.playing = false;
    }
  }
  enterBadRange() {
    if (this.playing) return false;
    this.mode = "bad_range";
    this.anchor = this.position;
    this.start = null;
    this.end = null;
    this.primaryCamera = null;
    return true;
  }
  boundary(side) {
    this[side] = this.frames[this.position];
  }
  confirmBadRange(kind, gap = 5) {
    if (this.start === null || this.end === null)
      throw Error("请先设置坏帧起点和终点。");
    if (this.start > this.end) throw Error("坏帧起点不能大于终点。");
    if (!this.primaryCamera) throw Error("请双击主要错误视角的图像，红色边框高亮后再确认区间。");
    const field = kind === "egopose" ? "ego_ranges" : "bad_ranges";
    const segmentField = kind === "egopose" ? "ego_segments" : "bad_segments";
    const segments = this.result[segmentField] || (this.result[field] || []).map(
      ([start_frame, end_frame]) => ({start_frame, end_frame}));
    this.result[segmentField] = normalizeSegments([...segments, {
      start_frame: this.start, end_frame: this.end, primary_camera: this.primaryCamera,
    }], gap);
    this.result[field] = normalizeRanges(
      this.result[segmentField].map(s => [s.start_frame, s.end_frame]),
      gap,
    );
    this.position = this.frames.indexOf(this.end);
    this.mode = "playback";
    this.primaryCamera = null;
    this.start = null;
    this.end = null;
  }
  cancelBadRange() {
    this.position = this.anchor;
    this.mode = "playback";
    this.primaryCamera = null;
    this.start = null;
    this.end = null;
  }
}
export const sources = ["mano", "mano_visible", "correct"];
export const sourceLabels = {
  mano: "原始视角",
  mano_visible: "原始视角（含可见性）",
  correct: "修改后视角",
};
