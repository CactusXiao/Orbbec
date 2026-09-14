import { QcWorkflow, sources, sourceLabels, enterLabelSegment } from "/workflow.js";
import { LabelCanvas } from "/label-canvas.js";
import { installDesktopLayout } from "/desktop-layout.js";
import { NativeQueue } from "/queue.js";
import { accountUI } from "/auth.js";
import { EpisodePlayer } from "/player.js";
import { FrameCache, labelLookahead } from "/frame-cache.js";
let highQuality = false;
const $ = (id) => document.getElementById(id),
  clone = (x) => structuredClone(x);
let frameReady = false;
let nativeCanvas, nativeQueue, identity, qcFlow;
const nativeTiles = new Map();
let player = null,
  source = "correct",
  overview = true,
  overlay = null,
  jobItems = [],
  mediaPoll,
  healthTimer;
let progressSignature = "",
  throughput = null;
const zoomStates = new WeakMap();
let db,
  draft = null,
  role =
    new URLSearchParams(location.search).get("role") ||
    localStorage.getItem("orbbec-workflow-role") ||
    "label",
  position = 0,
  camera = "",
  busy = false,
  durable = true,
  undo = [],
  generation = 0;
let releaseDraftLock = null;
let saveQueue = Promise.resolve(),
  pollTimer,
  heartbeatTimer,
  drag = null,
  owner = "";
const request = indexedDB.open("orbbec-outsourcing-v1", 2);
request.onupgradeneeded = () => {
  const d = request.result;
  if (!d.objectStoreNames.contains("drafts"))
    d.createObjectStore("drafts", { keyPath: "id" });
  // Old image-only cache had no size index; discard disposable images on upgrade.
  if (d.objectStoreNames.contains("images")) d.deleteObjectStore("images");
  d.createObjectStore("images");
  d.createObjectStore("image_meta");
};
const database = new Promise((resolve, reject) => {
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
});
const frameCache = new FrameCache(database);
async function read(store, key) {
  const d = await database;
  return new Promise((resolve, reject) => {
    const t = d.transaction(store);
    const r =
      key === undefined
        ? t.objectStore(store).getAll()
        : t.objectStore(store).get(key);
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}
async function put(store, value, key) {
  const d = await database;
  return new Promise((resolve, reject) => {
    const t = d.transaction(store, "readwrite");
    const o = t.objectStore(store);
    key === undefined ? o.put(value) : o.put(value, key);
    t.oncomplete = () => resolve();
    t.onabort = () => reject(t.error);
    t.onerror = () => reject(t.error);
  });
}
let noticeTimer = null;
function notice(text) {
  clearTimeout(noticeTimer);
  noticeTimer = null;
  $("notice").textContent = text;
  if (text) noticeTimer = setTimeout(() => notice(""), 1000);
}
// Clear the old reminder before the next action can produce a new one.
for (const event of ["pointerdown", "keydown", "wheel"])
  document.addEventListener(event, () => notice(""), {
    capture: true,
    passive: true,
  });
async function api(path, body) {
  const r = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers:
      body === undefined
        ? {}
        : { "Content-Type": "application/json", "X-Orbbec-Request": "1" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) {
    const e = new Error(data.error || `请求失败 ${r.status}`);
    e.status = r.status;
    if (r.status === 401 && path !== "/api/login") auth.expired(e.message);
    throw e;
  }
  return data;
}
function locked() {
  return (
    busy || !owner || !draft || !!draft.pending || !!draft.receipt || !durable
  );
}
function save() {
  if (!draft) return Promise.resolve();
  draft.updatedAt = new Date().toISOString();
  const snapshot = clone(draft);
  $("saveState").textContent = "正在保存到本机…";
  saveQueue = saveQueue
    .catch(() => {})
    .then(() => put("drafts", snapshot))
    .then(() => {
      durable = true;
      $("saveState").textContent = draft?.receipt
        ? "已提交 · 回执已保存"
        : "草稿已保存到本机";
    })
    .catch((e) => {
      durable = false;
      $("saveState").textContent = "本机保存失败";
      notice("本机存储失败，已暂停编辑。请导出草稿备份：" + e.message);
      throw e;
    });
  return saveQueue;
}
function changed() {
  draft.pending = null;
  save().catch(() => {});
  updateProgress();
}
function snapshot() {
  undo.push(clone(draft.result));
  if (undo.length > 40) undo.shift();
}
async function jobs() {
  if (!owner) return;
  try {
    jobItems = await api("/api/jobs/" + role);
    const local = (await read("drafts")).filter(
      (d) => d.manifest.operator === owner,
    );
    nativeQueue.show(role, jobItems, local, identity);
    $("labelRole").classList.toggle("selected", role === "label");
    $("qcRole").classList.toggle("selected", role === "qc");
  } catch (e) {
    notice(e.message);
  }
}
async function leaseAndOpen(item, leasedRole) {
  if (busy) return;
  busy = true;
  try {
    const m = await api("/api/lease", {
      role: leasedRole,
      job_id: item.job_id,
    });
    const d = {
      id: m.id,
      manifest: m,
      result:
        leasedRole === "label"
          ? { samples: {}, confirmed: [] }
          : {
              bad_ranges: [],
              ego_ranges: [],
              bad_episode: false,
              reviewed: [],
              playback_complete: false,
            },
      createdAt: new Date().toISOString(),
    };
    await put("drafts", d);
    await openDraft(d);
  } catch (e) {
    notice(e.message);
  } finally {
    busy = false;
    updateProgress();
  }
}
async function openDraft(d) {
  if (!owner || d.manifest.operator !== owner) return;
  notice("");
  if (!d.receipt) {
    try {
      const manifest = await api(`/api/sessions/${d.id}/resume`, {});
      if (
        JSON.stringify(manifest.frames) !== JSON.stringify(d.manifest.frames) ||
        JSON.stringify(manifest.cameras) !== JSON.stringify(d.manifest.cameras)
      )
        throw Error("任务范围变化，不能覆盖本机草稿");
      d.manifest = { ...d.manifest, ...manifest };
      d.released = false;
    } catch (e) {
      notice(e.message);
      if (e.status === 409 && !d.pending) {
        d.archived = true;
        await put("drafts", d);
        await jobs();
      }
      if (e.status || d.released) return;
    }
  }
  if (draft?.id !== d.id || !releaseDraftLock) {
    if (releaseDraftLock) {
      releaseDraftLock();
      releaseDraftLock = null;
    }
    if (!navigator.locks) {
      notice("此浏览器不支持安全的草稿编辑锁，请使用近期版本的浏览器。");
      return;
    }
    const acquired = await new Promise((resolve) => {
      navigator.locks
        .request("orbbec-draft-" + d.id, { ifAvailable: true }, (lock) => {
          if (!lock) {
            resolve(false);
            return;
          }
          return new Promise((release) => {
            releaseDraftLock = release;
            resolve(true);
          });
        })
        .catch(() => resolve(false));
    });
    if (!acquired) {
      notice("该草稿已在另一个标签页中打开，请在原标签页继续编辑。");
      return;
    }
  }
  if (player) {
    player.close();
    player = null;
  }
  clearInterval(mediaPoll);
  draft = d;
  if (d.manifest.role === "label") {
    d.confirmedSamples ||= Object.fromEntries(
      Object.entries(d.result.samples)
        .filter(([k]) => d.result.confirmed.includes(+k.split(":")[0]))
        .map(([k, v]) => [k, clone(v)]),
    );
  }
  source = "correct";
  overview = false;
  qcFlow =
    d.manifest.role === "qc"
      ? new QcWorkflow(d.manifest.frames, d.result, d.position || 0)
      : null;
  overlay = null;
  $("source").value = source;
  position =
    d.manifest.role === "label"
      ? Math.max(
          0,
          d.manifest.frames.findIndex((f) => !d.result.confirmed.includes(f)),
        )
      : d.position || 0;
  camera = d.camera || d.manifest.cameras[0];
  if (camera === "ego") overview = true;
  undo = [];
  generation++;
  $("picker").hidden = true;
  $("workspace").hidden = false;
  $("title").textContent =
    `${d.manifest.role.toUpperCase()} · ${d.manifest.episode_id}`;
  $("receipt").hidden = !d.receipt;
  $("content").hidden = true;
  $("preparing").hidden = !!d.receipt;
  $("prepareBack").hidden = !!d.receipt;
  $("retryMedia").hidden = true;
  $("submit").disabled = !!d.receipt;
  $("submit").textContent = d.pending ? "重试提交" : "确认提交";
  if (d.receipt) {
    showReceipt();
    return;
  }
  await refreshMedia();
  await heartbeat();
  clearInterval(heartbeatTimer);
  heartbeatTimer = setInterval(heartbeat, 30000);
}
async function heartbeat() {
  if (!owner || !draft || draft.receipt) return;
  try {
    await api(`/api/sessions/${draft.id}/heartbeat`, {});
    $("connection").textContent = "已连接";
  } catch (e) {
    $("connection").textContent =
      e.status === 409 ? "任务冲突 · 草稿保留" : "离线 · 草稿保留";
    if (e.status === 409) notice(e.message);
  }
}
async function refreshMedia() {
  clearTimeout(pollTimer);
  if (!owner || !draft || draft.receipt) return;
  const id = draft.id;
  try {
    const remote = await api("/api/sessions/" + id);
    if (draft?.id !== id) return;
    if (remote.revision !== draft.manifest.revision)
      throw new Error("数据版本已变化，保留草稿，请重新核对");
    draft.manifest.media = remote.media;
    draft.manifest.task_name = remote.task_name;
    draft.manifest.episode_index = remote.episode_index;
    draft.manifest.lease_until = remote.lease_until;
    $("retryMedia").hidden = !remote.media.error;
    if (remote.media.error) {
      const message = "后端准备失败：" + remote.media.error;
      if (draft.manifest.role === "qc") {
        await leaveWorkspace();
        notice(message);
        return;
      }
      throw new Error(message);
    }
    if (!remote.media.ready) {
      $("preparing").textContent =
        "后端正在解码并准备首批画面，完成后即可开始工作。";
      renderPreparation(remote.media);
      const progress = Object.entries(remote.media.progress || {});
      if (progress.length)
        $("preparing").textContent +=
          " " +
          progress
            .map(
              ([cam, p]) =>
                `${cam}：解码 ${p.decoded || 0}/${p.total}，画面 ${p.rendered || 0}/${p.total}`,
            )
            .join(" · ");
      pollTimer = setTimeout(refreshMedia, 2000);
      return;
    }
    if (
      draft.manifest.role === "label" &&
      !Object.keys(draft.result.samples).length
    ) {
      const samples = await api(`/api/sessions/${id}/samples.json`);
      if (draft?.id !== id) return;
      draft.result.samples = samples;
      draft.initialSamples = clone(samples);
      await save();
    }
    if (draft.manifest.role === "label" && !draft.initialSamples) {
      draft.initialSamples = await api(`/api/sessions/${id}/samples.json`);
      await save();
    }
    if (draft.manifest.role === "label" && !draft.sources) {
      const sources = await api(`/api/sessions/${id}/sources.json`).catch(
        () => null,
      );
      if (draft?.id !== id) return;
      if (sources) {
        draft.sources = sources;
        await save();
      }
    }
    await showContent();
  } catch (e) {
    if (draft?.id !== id || !owner || e.status === 401) return;
    if (
      (draft.manifest.role === "label" &&
        Object.keys(draft.result.samples).length) ||
      draft.manifest.media?.ready
    ) {
      await showContent();
      notice(
        draft.pending
          ? "连接不可用；此前提交内容已保留，恢复连接后可重试。"
          : "连接不可用，可以继续编辑已缓存画面；新画面和视频需要恢复连接。",
      );
    } else {
      $("preparing").textContent = e.message;
      $("retryMedia").hidden = false;
      pollTimer = setTimeout(refreshMedia, 5000);
    }
  }
}
function renderPreparation(media) {
  $("prepareBack").textContent =
    draft.manifest.role === "qc" ? "返回 Episode 列表" : "返回任务";
  const host = $("decodeStatus");
  host.hidden = false;
  host.replaceChildren();
  const title = document.createElement("h1");
  title.textContent =
    draft.manifest.role === "qc" ? "正在准备质检数据" : "正在解码 RGB 帧...";
  host.append(title);
  const p = document.createElement("p");
  p.textContent = `Episode ID：${draft.manifest.episode_index}    ${draft.manifest.role === "qc" ? "正在准备五路 RGB 与 Pico Ego MANO 投影视图" : "正在准备标注帧"}`;
  host.append(p);
  const table = document.createElement("table");
  table.className = "nativeTable";
  const row = table.createTHead().insertRow();
  for (const text of ["Camera", "状态", "帧数", "错误"]) {
    const th = document.createElement("th");
    th.textContent = text;
    row.append(th);
  }
  const body = table.createTBody(),
    labels = {
      pending: "等待",
      decoding: "解码中",
      done: "完成",
      failed: "失败",
      mesh_pending: "等待渲染 mesh",
      mesh_rendering: "渲染 mesh",
      mesh_gpu: "核显渲染 mesh",
      mesh_software_fallback: "软件渲染 mesh",
      mesh_done: "mesh 完成",
    };
  const cameras =
    draft.manifest.role === "qc"
      ? ["00", "02", "03", "05", "06"]
          .filter((c) => draft.manifest.cameras.includes(c))
          .concat("ego")
      : draft.manifest.cameras;
  for (const cam of cameras) {
    const status = media.progress?.[cam] || {},
      r = body.insertRow();
    for (const text of [
      cam,
      labels[status.status] || status.status || "等待",
      `${status.rendered || status.decoded || 0} / ${status.total || draft.manifest.frames.length}`,
      status.error || "",
    ])
      r.insertCell().textContent = text;
  }
  host.append(table);
}
async function imageURL(cam, frame, sessionId = draft.id, layer = "frames") {
  return frameCache.url(sessionId, cam, frame, layer);
}
function paintQC() {
  if (!draft || !qcFlow || highQuality) return;
  const v = $("video"),
    alpha = (draft.manoOpacity ?? 100) / 100;
  for (const part of draft.manifest.media?.layout?.tiles || []) {
    const entry = nativeTiles.get("nativeQcGrid:" + part.camera);
    if (entry) {
      entry.status.textContent = "";
      entry.canvas.videoLayers(v, part.raw, part.rendered, alpha);
    }
  }
}
async function showContent() {
  if (!owner || !draft) return;
  $("preparing").hidden = true;
  $("decodeStatus").hidden = true;
  $("prepareBack").hidden = true;
  $("content").hidden = false;
  const label = draft.manifest.role === "label";
  $("taskProgress").hidden = !label;
  $("toggleProgress").hidden = !label;
  $("content").classList.toggle("qcWorkspace", !label);
  $("labelToolbar").hidden = !label;
  $("qcDisplayToolbar").hidden = label;
  $("qcControlBar").hidden = label;
  $("nativePlaybackStatus").hidden = label || qcFlow?.mode === "bad_range";
  $("manoOpacity").value = draft.manoOpacity ?? 100;
  $("manoOpacityValue").textContent = `${draft.manoOpacity ?? 100}%`;
  $("qcPlaybackToolbar").hidden = label || qcFlow?.mode === "bad_range";
  $("qcBadToolbar").hidden = label || qcFlow?.mode !== "bad_range";
  $("editorLayout").hidden = overview;
  $("overviewGrid").hidden = !overview;
  $("labelPanel").hidden = !label;
  $("qcPanel").hidden = label;
  $("timeline").max = draft.manifest.frames.length - 1;
  $("timeline").value = position;
  if (!label) {
    if (!player || player.id !== draft.id) {
      player = new EpisodePlayer($("video"), draft.manifest, (status) => {
        if (status.error) notice(status.error);
        if (status.mbps) throughput = status.mbps;
        updateHealth();
        updateProgress();
      });
      clearInterval(mediaPoll);
      mediaPoll = setInterval(pollMedia, 2000);
    } else player.update(draft.manifest.media);
  }
  await renderFrame();
  updateProgress();
}
const ns = "http://www.w3.org/2000/svg";
function svgNode(tag, attrs) {
  const n = document.createElementNS(ns, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}
function selectedSample(cam = camera) {
  const key = `${draft.manifest.frames[position]}:${cam}`;
  const base = draft.result.samples[key];
  return source === "correct"
    ? base
    : { ...base, ...draft.sources?.[key]?.[source] };
}
function annotation(svg, sample, cam, small = false) {
  for (const n of [...svg.children]) if (n.tagName !== "image") n.remove();
  if (!sample || overlay) return;
  const colors = ["#37c7ff", "#ff8a3d"];
  for (let h = 0; h < 2; h++) {
    for (const [a, b] of draft.manifest.skeleton_edges) {
      if (!sample.visible[h][a] || !sample.visible[h][b]) continue;
      svg.append(
        svgNode("line", {
          x1: sample.points[h][a][0],
          y1: sample.points[h][a][1],
          x2: sample.points[h][b][0],
          y2: sample.points[h][b][1],
          stroke: colors[h],
          "stroke-width": 2,
          "vector-effect": "non-scaling-stroke",
        }),
      );
    }
    for (let j = 0; j < 21; j++) {
      const [x, y] = sample.points[h][j];
      if (x < 0 || y < 0) continue;
      const selected =
        !small && h === +$("hand").value && j === +$("joint").value;
      const tracked = (draft.tracked?.[cam] || []).some(
        (p) => p[0] === h && p[1] === j,
      );
      svg.append(
        svgNode("circle", {
          cx: x,
          cy: y,
          r: selected ? 10 : 6,
          fill: sample.visible[h][j] ? colors[h] : "none",
          stroke: tracked ? "#f6ec62" : selected ? "white" : colors[h],
          "stroke-width": tracked ? 4 : 2,
          "data-h": h,
          "data-j": j,
        }),
      );
    }
  }
}
function draw() {
  if (!draft || draft.manifest.role !== "label") return;
  const sample = selectedSample();
  if (!sample) return;
  const counts = [0, 1].map((h) =>
    Array.from({ length: 21 }, (_, j) =>
      draft.manifest.cameras.reduce(
        (n, cam) => n + Number(!!selectedSample(cam)?.visible[h][j]),
        0,
      ),
    ),
  );
  nativeCanvas.setState(sample, {
    edges: draft.manifest.skeleton_edges,
    tracked: draft.tracked?.[camera] || [],
    counts,
    readOnly: locked() || source !== "correct" || !!overlay || overview,
    annotation: !overlay,
  });
  $("cycleSource").textContent = "视图：" + sourceLabels[source];
  $("skeleton").textContent =
    overlay?.action === "skeleton" ? "Hide Skeleton" : "Show Skeleton";
  $("mesh").textContent =
    overlay?.action === "mesh" ? "Hide MANO" : "Show MANO";
  $("overview").title = "0：总览；1–7：单机位。滚轮缩放，右键平移。";
  $("viewNotice").textContent = overview ? "总览 · 只读" : "";
  const error =
    draft.sources?.[`${draft.manifest.frames[position]}:${camera}`]?.errors;
  if (
    error &&
    (source === "correct" ? Object.keys(error).length : error[source])
  )
    $("viewNotice").textContent += " · 原始参考缺失，已有修改仍保留";
  updateProgress();
}
async function tile(host, cam, url, sample, label) {
  const key = host.id + ":" + cam;
  let entry = nativeTiles.get(key);
  if (!entry) {
    const box = document.createElement("div");
    box.className = "nativeCameraCell";
    box.title = cam === "ego" ? "Ego" : `机位 ${cam}`;
    const title = document.createElement("div");
    title.className = "cameraTitle";
    title.textContent =
      cam === "ego"
        ? label
          ? "机位 ego · 只读"
          : "Pico Ego · MANO 外参投影"
        : label
          ? `机位 ${cam} · 只读`
          : `Camera ${cam}`;
    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-label", `Camera ${cam}`);
    canvas.addEventListener("dblclick", () => {
      if (label || locked() || !frameReady || qcFlow?.mode !== "bad_range") return;
      qcFlow.primaryCamera = cam;
      updateProgress();
    });
    const status = document.createElement("span");
    status.className = "tileStatus";
    box.append(title, canvas, status);
    host.append(box);
    entry = {
      canvas: new LabelCanvas(canvas, { schematics: false }),
      box,
      status,
    };
    nativeTiles.set(key, entry);
  }
  entry.box.hidden = false;
  entry.status.textContent = "";
  if (url) {
    await entry.canvas.setImage(url, `${draft.id}:${position}:${cam}`);
    entry.canvas.setState(sample, {
      edges: draft.manifest.skeleton_edges,
      tracked: draft.tracked?.[cam] || [],
      readOnly: true,
      annotation: label && cam !== "ego" && !overlay,
    });
  } else {
    entry.status.textContent =
      cam === "ego" ? "此帧无同步 ego 画面" : "目标帧渲染中…";
  }
  return entry;
}
async function renderFrame() {
  if (!owner || !draft) return;
  if (draft.manifest.role === "label") {
    const before = (draft.visitedSegments || []).length;
    const primary = enterLabelSegment(draft, draft.manifest.frames[position]);
    if (primary) {
      camera = primary;
      overview = primary === "ego";
      draft.camera = camera;
      draft.overview = overview;
      overlay = null;
      $("editorLayout").hidden = overview;
      $("overviewGrid").hidden = !overview;
    }
    if (before !== draft.visitedSegments.length) save().catch(() => {});
    $("overviewGrid").classList.toggle("primaryEgoView", overview && camera === "ego");
  }
  const renderStarted = performance.now();
  frameReady = false;
  frameCache.cancel();
  updateProgress();
  const gen = ++generation,
    sid = draft.id,
    frame = draft.manifest.frames[position];
  $("timeline").value = position;
  try {
    if (draft.manifest.role === "label") {
      if (!overview) {
        const url = await imageURL(camera, frame);
        if (gen !== generation) return;
        await nativeCanvas.setImage(url, `${sid}:${frame}:${camera}:${source}`);
        await nativeCanvas.setOverlay(
          overlay?.frame === frame
            ? `/api/sessions/${sid}/operations/${overlay.id}/${camera}.png`
            : null,
        );
      } else {
        const cameras = camera === "ego" ? ["ego"] : ["00", "02", "03", "05", "06"]
          .filter((c) => draft.manifest.cameras.includes(c))
          .concat("ego");
        for (const [key, entry] of nativeTiles)
          if (key.startsWith("overviewGrid:")) entry.box.hidden = !cameras.includes(key.split(":")[1]);
        for (const c of cameras) {
          let url = null;
          try {
            url = await imageURL(c, frame);
          } catch (e) {
            if (c !== "ego") throw e;
          }
          if (gen !== generation) return;
          const entry = await tile(
            $("overviewGrid"),
            c,
            url,
            c === "ego" ? null : selectedSample(c),
            true,
          );
          await entry.canvas.setOverlay(
            c !== "ego" && overlay?.frame === frame
              ? `/api/sessions/${sid}/operations/${overlay.id}/${c}.png`
              : null,
          );
        }
      }
      draw();
    } else {
      for (const cam of draft.manifest.media?.cameras ||
        draft.manifest.cameras) {
        if (highQuality) {
          const [raw, rendered] = await Promise.all([
            imageURL(cam, frame, sid, "raw_frames"),
            imageURL(cam, frame, sid),
          ]);
          if (gen !== generation) return;
          const entry = await tile($("nativeQcGrid"), cam, raw, null, false);
          await entry.canvas.setOverlay(
            rendered,
            (draft.manoOpacity ?? 100) / 100,
          );
        } else await tile($("nativeQcGrid"), cam, null, null, false);
      }
      if (!highQuality) {
        if (!player || !(await player.seek(position)) || gen !== generation)
          return;
        paintQC();
      }
    }
    if (gen === generation) {
      frameReady = true;
      // Local DOM diagnostics only; includes image retrieval, decode and draw.
      $("timeline").dataset.render = JSON.stringify({
        role: draft.manifest.role,
        milliseconds: Math.round((performance.now() - renderStarted) * 10) / 10,
        overview,
      });
      if (draft.manifest.role === "label")
        frameCache.prefetch(
          sid,
          labelLookahead(
            draft.manifest.frames,
            draft.manifest.cameras,
            position,
            camera,
            overview,
          ),
        );
      updateProgress();
    }
  } catch (e) {
    if (gen === generation) notice(e.message + "；本地结果未丢失。");
  }
}
function updateProgress() {
  if (!draft) return;
  const label = draft.manifest.role === "label",
    r = draft.result,
    frames = draft.manifest.frames,
    frame = frames[position];
  const name = `${draft.manifest.task_name} / Episode ${draft.manifest.episode_index}`;
  $("title").textContent = label
    ? `帧 ${frame} · 已确认 ${r.confirmed.length}/${frames.length}`
    : `帧 ${frame} / ${frames.at(-1)}${r.bad_ranges.length ? ` · 手部 ${r.bad_ranges.length} 段` : ""}${r.ego_ranges.length ? ` · Ego ${r.ego_ranges.length} 段` : ""}`;
  $("title").title = label
    ? `${name} · 机位 ${camera} · ${sourceLabels[source]}`
    : `${name} · 手部：${r.bad_ranges.map((p) => p.join("–")).join(", ") || "无"} · Ego：${r.ego_ranges.map((p) => p.join("–")).join(", ") || "无"}`;
  $("frameStatus").hidden = !label;
  $("viewNotice").hidden = !label;
  $("confirmFrame").textContent = "确认并继续";
  $("confirmFrame").disabled =
    locked() || !frameReady || overlay?.action === "skeleton";
  $("submit").textContent = draft.pending
    ? "重试提交"
    : `提交任务（${r.confirmed?.length || 0}/${frames.length}）`;
  const complete = label
    ? r.confirmed.length === frames.length
    : !!r.playback_complete;
  $("submit").disabled =
    busy || !!draft.receipt || !owner || (!draft.pending && !complete);
  $("qcSubmit").disabled =
    busy || !!draft.receipt || !owner || (!draft.pending && !complete);
  $("qcSubmit").textContent = draft.pending ? "重试提交" : "提交";
  $("undo").disabled =
    locked() ||
    source !== "correct" ||
    !!overlay ||
    overview ||
    !nativeCanvas?.history.length;
  $("ignore").disabled =
    locked() || source !== "correct" || !!overlay || overview;
  if (nativeCanvas)
    nativeCanvas.readOnly =
      locked() || source !== "correct" || !!overlay || overview;
  for (const id of [
    "skeleton",
    "mesh",
    "cycleSource",
    "prevCamera",
    "nextCamera",
    "overview",
    "prev",
    "next",
  ])
    $(id).disabled = busy || !owner;
  if (label) {
    $("timeline").disabled = busy || !owner;
    $("timelineMarks").classList.remove("disabled");
  }
  if (!label && qcFlow) {
    const bad = qcFlow.mode === "bad_range",
      playing = !$("video").paused || !!player?.wantsPlay;
    $("qcPlaybackToolbar").hidden = bad;
    $("qcBadToolbar").hidden = !bad;
    $("nativePlaybackStatus").hidden = bad;
    $("timeline").disabled = bad || busy;
    $("timelineMarks").classList.toggle("disabled", bad);
    for (const id of ["qcPrev", "qcNext", "prevTen", "nextTen"])
      $(id).disabled = playing || busy;
    $("enterRange").disabled = playing || busy || !frameReady;
    for (const id of ["setStart", "setEnd", "badRange", "egoRange"])
      $(id).disabled = locked() || !frameReady;
    // Keep the pause button actionable during seek/buffering and avoid replacing
    // its text node every video frame while the user is pressing it.
    const canPause = playing || !!player?.wantsPlay;
    const playDisabled = !canPause && (busy || !frameReady);
    if ($("play").disabled !== playDisabled) $("play").disabled = playDisabled;
    const playText = canPause ? "暂停" : "播放";
    if ($("play").textContent !== playText) $("play").textContent = playText;
    $("episodeException").disabled = busy;
    $("nativeBadStatus").textContent =
      `当前帧：${frame}    Start=${qcFlow.start ?? "-"}    End=${qcFlow.end ?? "-"}    主要错误视角：${qcFlow.primaryCamera || "请双击图像选择"}`;
    for (const [key, entry] of nativeTiles)
      if (key.startsWith("nativeQcGrid:")) entry.box.classList.toggle("primaryErrorCamera",
        bad && key === "nativeQcGrid:" + qcFlow.primaryCamera);
    $("nativePlaybackStatus").textContent = r.playback_complete
      ? "已完成一次播放，可随时提交"
      : !frameReady
        ? `目标帧 ${frame} 渲染中 · 可继续拖动进度条`
        : playing
          ? "播放中 · 目标 30 FPS"
          : "已暂停 · 完成一次播放后可提交";
  }
  renderProgress();
}
function step(delta) {
  if (!owner || !draft || busy) return;
  finishDrag();
  const keepMesh = overlay?.action === "mesh";
  if (qcFlow) {
    qcFlow.playing = !$("video").paused || !!player?.wantsPlay;
    position = qcFlow.step(delta);
  } else {
    position = Math.max(
      0,
      Math.min(draft.manifest.frames.length - 1, position + delta),
    );
    overlay = null;
  }
  draft.position = position;
  save().catch(() => {});
  renderFrame().then(async () => {
    if (keepMesh && draft && !qcFlow) {
      const result = await calculate("mesh");
      if (result) {
        overlay = result;
        renderFrame();
      }
    }
  });
}
function finishDrag() {
  nativeCanvas?.finish();
}
$("undo").onclick = () => nativeCanvas.undo();
$("ignore").onclick = () => nativeCanvas.ignore();
$("confirmFrame").onclick = async () => {
  if (locked() || $("confirmFrame").disabled || qcFlow) return;
  finishDrag();
  const frame = draft.manifest.frames[position];
  for (const cam of draft.manifest.cameras)
    draft.confirmedSamples[`${frame}:${cam}`] = clone(
      draft.result.samples[`${frame}:${cam}`],
    );
  draft.result.confirmed = [
    ...new Set([...draft.result.confirmed, frame]),
  ].sort((a, b) => a - b);
  await save();
  updateProgress();
  const hadTracks = Object.values(draft.tracked || {}).some((p) => p.length);
  if (hadTracks && position < draft.manifest.frames.length - 1) {
    source = "correct";
    await trackNext();
  } else step(1);
};
$("prev").onclick = () => step(-1);
$("next").onclick = () => step(1);
let timelinePlaying = false;
$("timeline").onpointerdown = () => {
  timelinePlaying = !!qcFlow && (!$("video").paused || !!player?.wantsPlay);
  if (timelinePlaying) player?.pause();
};
$("timeline").oninput = () => {
  if (qcFlow) {
    if (!qcFlow.seek(+$("timeline").value)) return;
    position = qcFlow.position;
    renderFrame();
  } else
    $("progress").textContent =
      `帧 ${draft.manifest.frames[+$("timeline").value]} · ${+$("timeline").value + 1}/${draft.manifest.frames.length}`;
};
$("timeline").onchange = () => {
  if (qcFlow) {
    draft.position = position;
    save().catch(() => {});
    if (timelinePlaying) $("play").click();
    timelinePlaying = false;
  } else step(+$("timeline").value - position);
};
window.addEventListener("keydown", (e) => {
  if (
    !owner ||
    !draft ||
    qcFlow ||
    $("workspace").hidden ||
    e.ctrlKey ||
    e.metaKey ||
    e.altKey ||
    ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName) ||
    document.querySelector("dialog[open]")
  )
    return;
  if (e.key === "0") {
    e.preventDefault();
    $("overview").click();
  } else if (/^[1-7]$/.test(e.key)) {
    e.preventDefault();
    chooseCamera(draft.manifest.cameras[+e.key - 1]);
  }
});
function renderRanges() {
  updateProgress();
}
function addRange(kind) {
  if (locked() || !frameReady || !qcFlow) return;
  try {
    qcFlow.confirmBadRange(kind);
    position = qcFlow.position;
    draft.position = position;
    changed();
    renderFrame();
    updateProgress();
  } catch (e) {
    notice(e.message);
  }
}
$("badRange").onclick = () => addRange("hand_pose");
$("egoRange").onclick = () => addRange("egopose");
let reviewTimer, playbackUiAt = 0;
$("video").onpause = () => {
  if (qcFlow) qcFlow.playing = false;
  if (draft && owner && qcFlow) {
    save().catch(() => {});
    paintQC();
    updateProgress();
  }
};
$("video").onplay = () => {
  if (qcFlow) {
    qcFlow.playing = true;
    updateProgress();
  }
};
function videoFrame(_, meta) {
  const v = $("video");
  if (owner && draft && qcFlow && !v.paused && !v.seeking && !player?.seeking) {
    const idx = Math.min(
      draft.manifest.frames.length - 1,
      Math.floor(meta.mediaTime * draft.manifest.fps + 0.001),
    );
    qcFlow.displayed(idx);
    position = idx;
    draft.position = idx;
    paintQC();
    frameReady = true;
    $("timeline").value = idx;
    if (performance.now() - playbackUiAt >= 100) {
      playbackUiAt = performance.now();
      updateProgress();
    }
    if (!reviewTimer)
      reviewTimer = setTimeout(() => {
        reviewTimer = null;
        save().catch(() => {});
      }, 1000);
  }
  v.requestVideoFrameCallback(videoFrame);
}
if ("requestVideoFrameCallback" in $("video"))
  $("video").requestVideoFrameCallback(videoFrame);
$("video").onended = () => {
  if (qcFlow) {
    player?.pause();
    qcFlow.result.playback_complete = true;
    qcFlow.playing = false;
    position = draft.manifest.frames.length - 1;
    qcFlow.position = position;
    draft.position = position;
    save().catch(() => {});
    paintQC();
    updateProgress();
  }
};
$("submit").onclick = () => {
  if (!owner || !draft || draft.receipt || busy) return;
  finishDrag();
  const complete = qcFlow
    ? draft.result.playback_complete
    : draft.result.confirmed.length === draft.manifest.frames.length;
  if (!draft.pending && !complete && !draft.result.bad_episode) {
    notice(
      qcFlow
        ? "当前 Episode 尚未完成一次播放，暂不能提交。"
        : "请先确认所有帧，再提交整个任务。",
    );
    return;
  }
  player?.pause();
  $("doSubmit").click();
};
$("cancelSubmit").onclick = () => $("submitDialog").close();
$("doSubmit").onclick = async () => {
  $("submitDialog").close();
  if (!durable) {
    notice("本机存储不可用，请先导出草稿。");
    return;
  }
  busy = true;
  $("submit").disabled = true;
  updateProgress();
  try {
    await saveQueue;
    if (!draft.pending) {
      draft.pending = {
        submission_id: crypto.randomUUID(),
        revision: draft.manifest.revision,
        result: clone(draft.result),
      };
      if (!qcFlow) draft.pending.result.samples = clone(draft.confirmedSamples);
      await save();
    }
    const receipt = await api(
      `/api/sessions/${draft.id}/submit`,
      draft.pending,
    );
    if (
      !receipt.accepted ||
      receipt.submission_id !== draft.pending.submission_id
    )
      throw new Error("回执不匹配，保留草稿");
    draft.receipt = receipt;
    await save();
    notice(qcFlow ? "后端已确认 QC 结果。" : "后端已确认整个标注任务的结果。");
    await leaveWorkspace(false);
  } catch (e) {
    notice(e.message + "；草稿和提交编号已保留，可重试。");
    $("submit").textContent = "重试提交";
  } finally {
    busy = false;
    $("submit").disabled = !!draft?.receipt;
    updateProgress();
  }
};
function showReceipt() {
  $("connection").textContent = "已提交 · 回执已保存";
  clearInterval(heartbeatTimer);
  $("content").hidden = true;
  $("preparing").hidden = true;
  $("receipt").hidden = false;
  $("receipt").textContent =
    `已提交成功。任务：${draft.receipt.job_id}；回执：${draft.receipt.submission_id}；时间：${draft.receipt.accepted_at}。草稿保留在本机供核对。`;
}
$("export").onclick = () => {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(draft, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = `orbbec-draft-${draft.id}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
async function leaveWorkspace(release = true) {
  if (!draft) return;
  finishDrag();
  player?.pause();
  await save();
  if (release && !draft.receipt) {
    if (draft.pending) {
      notice("提交结果尚未确认，请先重试提交。");
      return;
    }
    await api(`/api/sessions/${draft.id}/release`, {});
    draft.released = true;
    draft.manifest.lease_until = "";
    await save();
  }
  frameCache.clearURLs();
  highQuality = false;
  const wasQC = !!qcFlow;
  player?.close();
  player = null;
  clearInterval(mediaPoll);
  clearInterval(heartbeatTimer);
  clearTimeout(pollTimer);
  if (releaseDraftLock) {
    releaseDraftLock();
    releaseDraftLock = null;
  }
  for (const entry of nativeTiles.values()) {
    entry.canvas.close();
    entry.box.remove();
  }
  nativeTiles.clear();
  draft = null;
  qcFlow = null;
  generation++;
  $("workspace").hidden = true;
  $("picker").hidden = false;
  nativeQueue.qcEpisodes = wasQC;
  await jobs();
}
$("home").onclick = async () => {
  if (busy) return;
  if (
    !qcFlow &&
    !(await confirmAction("返回任务", "返回将放弃未确认的修改，继续？"))
  )
    return;
  try {
    if (!qcFlow && draft.initialSamples) {
      draft.result.samples = clone(draft.initialSamples);
      Object.assign(draft.result.samples, clone(draft.confirmedSamples));
    }
    await leaveWorkspace();
  } catch (e) {
    notice(e.message);
  }
};
$("labelRole").onclick = () => {
  role = "label";
  localStorage.setItem("orbbec-workflow-role", role);
  $("taskFilter").value = "";
  jobs();
};
$("qcRole").onclick = () => {
  role = "qc";
  localStorage.setItem("orbbec-workflow-role", role);
  $("taskFilter").value = "";
  jobs();
};
$("refresh").onclick = jobs;
$("retryMedia").onclick = async () => {
  try {
    await heartbeat();
    await api(`/api/sessions/${draft.id}/retry-media`, {});
    $("retryMedia").hidden = true;
    await refreshMedia();
  } catch (e) {
    notice(e.message);
  }
};
window.addEventListener("online", () => {
  heartbeat();
});
window.addEventListener("offline", () => {
  $("connection").textContent = "离线 · 本机草稿继续保存";
});
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden" && draft) save().catch(() => {});
});
function renderJobs() {
  jobs();
}
function chooseCamera(cam) {
  if (!cam || !draft) return;
  finishDrag();
  camera = cam;
  overview = false;
  draft.camera = cam;
  draft.overview = false;
  save().catch(() => {});
  showContent();
}
$("overview").onclick = () => {
  finishDrag();
  if (camera === "ego") camera = draft.manifest.cameras[0];
  overview = true;
  draft.overview = overview;
  save().catch(() => {});
  showContent();
};
$("prevCamera").onclick = () =>
  chooseCamera(
    draft.manifest.cameras[
      (draft.manifest.cameras.indexOf(camera) -
        1 +
        draft.manifest.cameras.length) %
        draft.manifest.cameras.length
    ],
  );
$("nextCamera").onclick = () =>
  chooseCamera(
    draft.manifest.cameras[
      (draft.manifest.cameras.indexOf(camera) + 1) %
        draft.manifest.cameras.length
    ],
  );
$("source").onchange = () => {
  finishDrag();
  if ($("source").value !== "correct" && !draft.sources) {
    notice("原始结果尚未准备好，请恢复连接后重新打开该草稿。");
    $("source").value = "correct";
    return;
  }
  source = $("source").value;
  overlay = null;
  renderFrame();
};
function resetView(svg, width, height) {
  const state = {
    x: 0,
    y: 0,
    width,
    height,
    baseWidth: width,
    baseHeight: height,
  };
  zoomStates.set(svg, state);
  applyView(svg);
}
function applyView(svg) {
  const z = zoomStates.get(svg);
  if (z) svg.setAttribute("viewBox", `${z.x} ${z.y} ${z.width} ${z.height}`);
}
function bindZoom(svg) {
  let pan = null;
  svg.addEventListener(
    "wheel",
    (e) => {
      const z = zoomStates.get(svg);
      if (!z) return;
      e.preventDefault();
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      const scale = Math.min(
        (z.baseWidth * 1.25) / z.width,
        Math.max(z.baseWidth / 16 / z.width, Math.exp(e.deltaY * 0.0015)),
      );
      z.x = p.x - (p.x - z.x) * scale;
      z.y = p.y - (p.y - z.y) * scale;
      z.width *= scale;
      z.height *= scale;
      applyView(svg);
    },
    { passive: false },
  );
  svg.addEventListener("contextmenu", (e) => e.preventDefault());
  svg.addEventListener("pointerdown", (e) => {
    if (e.button !== 2 && e.button !== 1) return;
    const z = zoomStates.get(svg);
    if (!z) return;
    e.preventDefault();
    svg.setPointerCapture(e.pointerId);
    pan = {
      x: e.clientX,
      y: e.clientY,
      view: { ...z },
      scale: svg.getScreenCTM().a,
    };
  });
  svg.addEventListener("pointermove", (e) => {
    if (!pan) return;
    const z = zoomStates.get(svg);
    z.x = pan.view.x - (e.clientX - pan.x) / pan.scale;
    z.y = pan.view.y - (e.clientY - pan.y) / pan.scale;
    applyView(svg);
  });
  svg.addEventListener("pointerup", () => (pan = null));
  svg.addEventListener("pointercancel", () => (pan = null));
}
bindZoom($("editor"));
bindZoom($("inspection"));
$("resetZoom").onclick = () => {
  for (const svg of [
    $("editor"),
    ...$("overviewGrid").querySelectorAll("svg"),
  ]) {
    const z = zoomStates.get(svg);
    if (z) resetView(svg, z.baseWidth, z.baseHeight);
  }
};
async function inspectImage(url, cam, frame) {
  const img = new Image();
  img.src = url;
  await img.decode();
  $("imageTitle").textContent = `机位 ${cam} · 帧 ${frame}`;
  $("inspection").replaceChildren(
    svgNode("image", {
      href: url,
      width: img.naturalWidth,
      height: img.naturalHeight,
    }),
  );
  resetView($("inspection"), img.naturalWidth, img.naturalHeight);
  $("imageDialog").showModal();
}
$("closeImage").onclick = () => $("imageDialog").close();
$("inspection").ondblclick = () => {
  const z = zoomStates.get($("inspection"));
  if (z) resetView($("inspection"), z.baseWidth, z.baseHeight);
};
function renderJointMap() {
  const sample = selectedSample();
  if (!sample) return;
  $("jointMap").replaceChildren();
  const h = +$("hand").value;
  for (let j = 0; j < 21; j++) {
    const b = document.createElement("button");
    const count = draft.manifest.cameras.reduce(
      (n, cam) =>
        n +
        Number(
          !!draft.result.samples[`${draft.manifest.frames[position]}:${cam}`]
            ?.visible[h][j],
        ),
      0,
    );
    b.textContent = `${j} · ${count}视角`;
    b.title = `${draft.manifest.joint_names[j]}：${sample.visible[h][j] ? "可见" : "不可见"}`;
    b.className = `${sample.visible[h][j] ? "visibleJoint" : "hiddenJoint"} ${j === +$("joint").value ? "selected" : ""} ${count < 2 ? "insufficient" : ""}`;
    b.onclick = () => {
      $("joint").value = j;
      draw();
    };
    $("jointMap").append(b);
  }
  const chosen = draft.tracked?.[camera] || [];
  const tracked = chosen.some((p) => p[0] === h && p[1] === +$("joint").value);
  $("trackJoint").textContent = tracked
    ? "取消跟踪该关节 · T"
    : "跟踪所选关节 · T";
  $("trackingInfo").textContent =
    `${chosen.length} 个关节已选为跟踪点；确认并继续时由后端跟踪到下一帧。黄色描边表示跟踪点。`;
}
function unconfirm() {}
$("hideHand").onclick = () => {
  if (locked() || source !== "correct" || overlay) return;
  snapshot();
  selectedSample().visible[+$("hand").value].fill(false);
  unconfirm();
  changed();
  draw();
};
$("trackJoint").onclick = () => {
  if (locked() || source !== "correct" || overlay) return;
  const h = +$("hand").value,
    j = +$("joint").value;
  draft.tracked ||= {};
  draft.tracked[camera] ||= [];
  const pairs = draft.tracked[camera],
    index = pairs.findIndex((p) => p[0] === h && p[1] === j);
  if (index >= 0) pairs.splice(index, 1);
  else {
    const s = selectedSample();
    if (!s.visible[h][j] || Math.min(...s.points[h][j]) < 0) {
      notice("先将该关节设为可见并放置到画面中。");
      return;
    }
    pairs.push([h, j]);
  }
  save().catch(() => {});
  draw();
};
async function calculate(action, extras = {}) {
  if (locked()) return null;
  finishDrag();
  busy = true;
  updateProgress();
  const sid = draft.id,
    frame = draft.manifest.frames[position];
  const samples = Object.fromEntries(
    draft.manifest.cameras.map((cam) => [
      cam,
      clone(
        action === "skeleton"
          ? selectedSample(cam)
          : draft.result.samples[`${frame}:${cam}`],
      ),
    ]),
  );
  notice(
    action === "track"
      ? "后端正在跟踪关节点…"
      : action === "mesh"
        ? "后端正在渲染 MANO…"
        : "后端正在重建三维骨架…",
  );
  try {
    const request = {
      action,
      frame,
      ...(action === "mesh" ? {} : { samples }),
      ...extras,
    };
    const op = await api(`/api/sessions/${sid}/compute`, request);
    let result;
    while (draft?.id === sid) {
      await new Promise((r) => setTimeout(r, 700));
      result = await api(`/api/sessions/${sid}/operations/${op.id}`);
      if (result.error) throw new Error(result.error);
      if (result.ready) break;
    }
    if (draft?.id !== sid) return null;
    notice("计算完成，结果尚未提交。");
    return { ...result, frame, action };
  } catch (e) {
    notice("计算未完成：" + e.message + "；本机草稿保留。");
    return null;
  } finally {
    busy = false;
    updateProgress();
  }
}
async function trackNext() {
  if (locked() || position >= draft.manifest.frames.length - 1) return;
  const target = draft.manifest.frames[position + 1];
  const selected = clone(draft.tracked || {});
  for (const cam of Object.keys(selected))
    selected[cam] = selected[cam].filter(
      ([h, j]) =>
        draft.result.samples[`${draft.manifest.frames[position]}:${cam}`]
          .visible[h][j],
    );
  const result = await calculate("track", { target, selected });
  if (!result) {
    step(1);
    return;
  }
  snapshot();
  for (const [cam, sample] of Object.entries(result.samples)) {
    const targetSample = draft.result.samples[`${target}:${cam}`];
    for (const [h, j] of sample.selected) {
      targetSample.points[h][j] = sample.points[h][j];
      targetSample.visible[h][j] = sample.visible[h][j];
      if (!sample.visible[h][j])
        draft.tracked[cam] = draft.tracked[cam].filter(
          (p) => p[0] !== h || p[1] !== j,
        );
    }
  }
  if (Object.values(result.samples).some((sample) => sample.selected.length))
    draft.result.confirmed = draft.result.confirmed.filter((f) => f !== target);
  changed();
  position = draft.manifest.frames.indexOf(target);
  await step(0);
  if (result.errors?.length)
    notice("部分机位跟踪未完成：" + result.errors.join("；"));
}
$("trackNext").onclick = trackNext;
for (const action of ["skeleton", "mesh"])
  $(action).onclick = async () => {
    if (overlay?.action === action) {
      overlay = null;
      await renderFrame();
      return;
    }
    const result = await calculate(action);
    if (result && result.frame === draft.manifest.frames[position]) {
      overlay = result;
      await renderFrame();
    }
  };
const progressCanvas = document.createElement("canvas");
progressCanvas.setAttribute(
  "aria-label",
  "帧进度，绿色已确认，红色问题区间，蓝色当前帧",
);
$("timelineMarks").append(progressCanvas);
progressCanvas.onclick = (e) => {
  if (!draft || busy || qcFlow?.mode === "bad_range") return;
  const r = progressCanvas.getBoundingClientRect(),
    target = Math.min(
      draft.manifest.frames.length - 1,
      Math.floor(
        ((e.clientX - r.left) / r.width) * draft.manifest.frames.length,
      ),
    );
  if (qcFlow) {
    const playing = !$("video").paused || !!player?.wantsPlay;
    player?.pause();
    qcFlow.seek(target);
    position = target;
    draft.position = target;
    save().catch(() => {});
    renderFrame().then(() => {
      if (playing) $("play").click();
    });
  } else step(target - position);
};
function renderProgress() {
  if (!draft || !draft.result) return;
  const label = draft.manifest.role === "label",
    done = new Set(label ? draft.result.confirmed : draft.result.reviewed);
  const frame = draft.manifest.frames[position];
  $("frameStatus").textContent =
    `当前帧 ${frame} · ${done.has(frame) ? "已确认" : "待确认"}`;
  $("frameStatus").className = done.has(frame) ? "done" : "todo";
  const signature = `${draft.id}:${position}:${[...done].join(",")}:${JSON.stringify(draft.result.bad_ranges)}:${JSON.stringify(draft.result.ego_ranges)}:${qcFlow?.mode}:${qcFlow?.start}:${qcFlow?.end}`;
  if (signature === progressSignature) return;
  progressSignature = signature;
  const width = progressCanvas.parentElement.clientWidth || 900;
  if (progressCanvas.width !== width) progressCanvas.width = width;
  if (progressCanvas.height !== 36) progressCanvas.height = 36;
  const ctx = progressCanvas.getContext("2d"),
    w = width / draft.manifest.frames.length;
  ctx.clearRect(0, 0, width, 36);
  if (label) {
    draft.manifest.frames.forEach((f, i) => {
      ctx.fillStyle = done.has(f) ? "#46d36b" : "#ff5c5c";
      ctx.fillRect(i * w, 13, Math.max(1, w - 1), 10);
    });
  } else {
    ctx.fillStyle = "#3b4d64";
    ctx.fillRect(0, 13, width, 10);
    ctx.fillStyle = "#2563eb";
    ctx.fillRect(0, 13, position * w, 10);
    for (const [field, color, y, h] of [
      ["bad_ranges", "#e5484d", 13, 10],
      ["ego_ranges", "#ff202b", 26, 5],
    ]) {
      ctx.fillStyle = color;
      for (const [a, b] of draft.result[field]) {
        const start = draft.manifest.frames.indexOf(a),
          end = draft.manifest.frames.indexOf(b);
        ctx.fillRect(start * w, y, Math.max(2, (end - start) * w), h);
      }
    }
    if (qcFlow?.mode === "bad_range" && qcFlow.start !== null) {
      const start = draft.manifest.frames.indexOf(qcFlow.start),
        end =
          qcFlow.end === null
            ? position
            : draft.manifest.frames.indexOf(qcFlow.end);
      ctx.fillStyle = "#f59e0b";
      ctx.fillRect(
        Math.min(start, end) * w,
        13,
        Math.max(2, Math.abs(end - start) * w),
        10,
      );
    }
  }
  ctx.fillStyle = "white";
  ctx.fillRect(position * w, 8, 2, 26);
  $("progress").textContent = label
    ? `帧 ${frame} · ${position + 1}/${draft.manifest.frames.length} · ${done.has(frame) ? "已确认" : "未确认"} · 已确认 ${done.size}/${draft.manifest.frames.length}`
    : "";
  if (label) {
    $("frameList").replaceChildren();
    const table = document.createElement("table");
    table.className = "nativeTable";
    const head = table.createTHead().insertRow();
    for (const t of ["任务", "完成", "总数"]) {
      const th = document.createElement("th");
      th.textContent = t;
      head.append(th);
    }
    const row = table.createTBody().insertRow();
    for (const t of [
      `${draft.manifest.task_name} / ${draft.manifest.episode_index}`,
      done.size,
      draft.manifest.frames.length,
    ])
      row.insertCell().textContent = t;
    row.className = "selected";
    row.onclick = () => {
      position = Math.max(
        0,
        draft.manifest.frames.findIndex((f) => !done.has(f)),
      );
      camera = draft.manifest.cameras[0];
      draft.tracked = {};
      step(0);
    };
    $("frameList").append(table);
  }
}
$("goFrame").onclick = () => {
  const i = draft.manifest.frames.indexOf(+$("frameJump").value);
  if (i < 0) {
    notice("该帧不在本次任务中，请按左侧帧列表选择。");
    return;
  }
  step(i - position);
};
$("frameJump").onkeydown = (e) => {
  if (e.key === "Enter") $("goFrame").click();
};
$("prevTen").onclick = () => step(-10);
$("nextTen").onclick = () => step(10);
$("play").onclick = async () => {
  const v = $("video"),
    activePlayer = player;
  if (!qcFlow || !activePlayer) return;
  // Pausing is always available, including while media is preparing.
  if (!v.paused || activePlayer.wantsPlay) {
    activePlayer.pause();
    qcFlow.playing = false;
    if (!frameReady) renderFrame();
    else paintQC();
    updateProgress();
    return;
  }
  if (busy || !frameReady) return;
  qcFlow.position = position;
  if (!qcFlow.play()) return;
  position = qcFlow.position;
  highQuality = false;
  $("highQuality").textContent = "高清检查";
  frameReady = false;
  // Register play intent synchronously, before any asynchronous positioning.
  // The button can now cancel this request even if the network is stalled.
  try {
    const ready = await activePlayer.play(position);
    if (activePlayer !== player || !qcFlow || !owner) return;
    if (ready && $("video").paused) {
      paintQC();
      frameReady = true;
    }
  } catch (e) {
    notice(e.message);
  } finally {
    if (activePlayer === player) updateProgress();
  }
};
function selectBoundary(side) {
  if (locked() || !frameReady || !qcFlow || qcFlow.mode !== "bad_range") return;
  qcFlow.boundary(side);
  updateProgress();
}
$("enterRange").onclick = () => {
  if (locked() || !frameReady || !qcFlow) return;
  qcFlow.playing = !$("video").paused || !!player?.wantsPlay;
  qcFlow.position = position;
  if (qcFlow.enterBadRange()) updateProgress();
};
$("setStart").onclick = () => selectBoundary("start");
$("setEnd").onclick = () => selectBoundary("end");
let mediaPolling = false;
async function pollMedia() {
  if (!owner || !draft || draft.manifest.role !== "qc" || draft.receipt) return;
  if (draft.manifest.media?.complete) {
    clearInterval(mediaPoll);
    return;
  }
  if (mediaPolling) return;
  mediaPolling = true;
  const sid = draft.id;
  try {
    const current = await api(`/api/sessions/${sid}`);
    if (draft?.id !== sid) return;
    draft.manifest.media = current.media;
    player?.update(current.media);
    if (current.media.error) notice("后端准备画面失败：" + current.media.error);
    if (
      !frameReady &&
      !player?.seeking &&
      $("video").paused &&
      position < (current.media.prepared || 0)
    )
      renderFrame();
    updateHealth();
  } catch {
    updateHealth();
  } finally {
    mediaPolling = false;
  }
}
function updateHealth() {
  if (draft?.manifest.role !== "qc") return;
  const v = $("video"),
    q = v.getVideoPlaybackQuality?.();
  let buffered = 0;
  for (let n = 0; n < v.buffered.length; n++)
    if (
      v.currentTime >= v.buffered.start(n) &&
      v.currentTime <= v.buffered.end(n)
    )
      buffered = v.buffered.end(n) - v.currentTime;
  $("qcBufferStatus").textContent = player?.buffering ? "缓冲中…" : "";
  $("qcDisplayToolbar").title =
    `已缓冲 ${buffered.toFixed(1)} 秒${player?.incremental ? " · 前方预取 60 秒" : " · 浏览器自动缓冲"}`;
  const prep = draft.manifest.media;
  $("playbackHealth").textContent =
    `${!v.paused && v.readyState < 3 ? "缓冲中 · " : ""}缓冲 ${buffered.toFixed(1)} 秒${prep?.prepared ? ` · 已准备 ${prep.prepared}/${draft.manifest.frames.length} 帧` : ""}${q ? ` · 丢帧 ${q.droppedVideoFrames}/${q.totalVideoFrames}` : ""}${throughput ? ` · 片段读取 ${throughput.toFixed(1)} Mbps` : ""}`;
}
healthTimer = setInterval(updateHealth, 1000);
$("importDraft").onclick = () => $("draftFile").click();
$("draftFile").onchange = async () => {
  try {
    const file = $("draftFile").files[0];
    if (!file) return;
    if (file.size > 32 * 1024 * 1024) throw new Error("草稿超过 32 MB");
    const d = JSON.parse(await file.text());
    if (!d?.id || !d.manifest || d.manifest.operator !== owner || !d.result)
      throw new Error("草稿格式或操作员不匹配");
    const existing = await read("drafts", d.id);
    if (existing) {
      notice("本机已存在该草稿，已保留现有内容，请使用恢复草稿。");
      return;
    }
    const m = await api(`/api/sessions/${encodeURIComponent(d.id)}`);
    if (
      m.revision !== d.manifest.revision ||
      JSON.stringify(m.frames) !== JSON.stringify(d.manifest.frames) ||
      JSON.stringify(m.cameras) !== JSON.stringify(d.manifest.cameras)
    )
      throw new Error("任务版本不匹配");
    const done = m.role === "label" ? d.result.confirmed : d.result.reviewed;
    if (!Array.isArray(done) || !done.every((f) => m.frames.includes(f)))
      throw new Error("确认帧无效");
    if (m.role === "label")
      for (const f of m.frames)
        for (const cam of m.cameras) {
          const sample = d.result.samples?.[`${f}:${cam}`];
          if (
            !sample ||
            sample.points?.length !== 2 ||
            sample.visible?.length !== 2 ||
            !sample.points.every(
              (h) =>
                h.length === 21 &&
                h.every((p) => p.length === 2 && p.every(Number.isFinite)),
            ) ||
            !sample.visible.every(
              (h) =>
                h.length === 21 &&
                h.every((v) => [true, false, 0, 1].includes(v)),
            ) ||
            ![sample.width, sample.height].every(
              (v) => Number.isInteger(v) && v > 0 && v <= 8192,
            )
          )
            throw new Error("标注数据无效");
        }
    d.manifest = m;
    await put("drafts", d);
    await jobs();
    notice("草稿已导入本机，尚未提交。");
  } catch (e) {
    notice("导入失败：" + e.message);
  } finally {
    $("draftFile").value = "";
  }
};

installDesktopLayout();
$("manoOpacity").oninput = () => {
  if (!draft || !qcFlow) return;
  draft.manoOpacity = +$("manoOpacity").value;
  $("manoOpacityValue").textContent = `${draft.manoOpacity}%`;
  if (highQuality)
    for (const [key, entry] of nativeTiles) {
      if (key.startsWith("nativeQcGrid:")) {
        entry.canvas.overlayOpacity = draft.manoOpacity / 100;
        entry.canvas.draw();
      }
    }
  else paintQC();
  save().catch(() => {});
};
$("highQuality").onclick = () => {
  if (!draft || !qcFlow || busy) return;
  player?.pause();
  highQuality = !highQuality;
  $("highQuality").textContent = highQuality ? "返回视频画面" : "高清检查";
  renderFrame();
};

nativeCanvas = new LabelCanvas($("nativeLabelCanvas"), {
  changed: () => {
    if (draft) {
      changed();
      draw();
    }
  },
  track: (h, j) => {
    if (locked() || source !== "correct") return;
    draft.tracked ||= {};
    const pairs = (draft.tracked[camera] ||= []),
      i = pairs.findIndex((p) => p[0] === h && p[1] === j),
      sample = selectedSample();
    if (i >= 0) pairs.splice(i, 1);
    else if (
      sample.visible[h][j] &&
      sample.points[h][j].every(Number.isFinite) &&
      Math.min(...sample.points[h][j]) >= 0
    )
      pairs.push([h, j]);
    else notice("请先将这个关节在当前视角下标为可见，并设置有效位置。");
    save().catch(() => {});
    draw();
  },
  cancelled: (pairs) => {
    if (draft?.tracked?.[camera])
      draft.tracked[camera] = draft.tracked[camera].filter(
        (p) => !pairs.some((q) => q[0] === p[0] && q[1] === p[1]),
      );
    nativeCanvas.tracked = draft?.tracked?.[camera] || [];
  },
});
nativeQueue = new NativeQueue($("nativeQueue"), {
  open: leaseAndOpen,
  restore: openDraft,
  refresh: jobs,
});
$("cycleSource").onclick = () => {
  finishDrag();
  source = sources[(sources.indexOf(source) + 1) % sources.length];
  overlay = null;
  renderFrame();
};
for (const [id, delta] of [
  ["qcPrev", -1],
  ["qcNext", 1],
  ["badPrevTen", -10],
  ["badPrev", -1],
  ["badNext", 1],
  ["badNextTen", 10],
])
  $(id).onclick = () => step(delta);
$("cancelBad").onclick = () => {
  qcFlow.cancelBadRange();
  position = qcFlow.position;
  draft.position = position;
  changed();
  renderFrame();
  updateProgress();
};
$("prepareBack").onclick = () => $("home").click();
$("qcHome").onclick = () => {
  $("home").click();
};
$("qcExit").onclick = async () => {
  try {
    await leaveWorkspace();
    notice("任务已释放，可以关闭此标签页。");
  } catch (e) {
    notice(e.message);
  }
};
$("qcSubmit").onclick = () => $("submit").click();
async function confirmAction(title, message) {
  const dialog = document.createElement("dialog"),
    h = document.createElement("h2"),
    p = document.createElement("p"),
    yes = document.createElement("button"),
    no = document.createElement("button");
  h.textContent = title;
  p.textContent = message;
  yes.textContent = "确定";
  no.textContent = "取消";
  dialog.append(h, p, no, yes);
  document.body.append(dialog);
  return new Promise((resolve) => {
    const finish = (v) => {
      dialog.close();
      dialog.remove();
      resolve(v);
    };
    yes.onclick = () => finish(true);
    no.onclick = () => finish(false);
    dialog.oncancel = (e) => {
      e.preventDefault();
      finish(false);
    };
    dialog.showModal();
  });
}
$("episodeException").onclick = async () => {
  if (locked()) return;
  player?.pause();
  if (
    await confirmAction(
      "Episode 异常",
      "确认将整个 Episode 标记为异常？该结果不会进入人工返修段。",
    )
  ) {
    draft.result.bad_episode = true;
    await save();
    $("submit").click();
  }
};
const auth = accountUI({
  api,
  notice,
  async suspend() {
    owner = "";
    frameCache.clearURLs();
    highQuality = false;
    generation++;
    finishDrag();
    player?.pause();
    player?.close();
    player = null;
    clearInterval(heartbeatTimer);
    clearInterval(mediaPoll);
    clearTimeout(pollTimer);
    $("workspace").hidden = true;
    $("picker").hidden = true;
    if (releaseDraftLock) {
      releaseDraftLock();
      releaseDraftLock = null;
    }
    await save();
  },
  async resume(account) {
    identity = account;
    owner = identity.operator;
    if (draft?.manifest.operator !== owner) draft = null;
    $("workspaceLabel").textContent =
      "外包工作台" +
      (identity.workspace_label ? " · " + identity.workspace_label : "");
    $("labelRole").hidden = !identity.roles.includes("label");
    $("qcRole").hidden = !identity.roles.includes("qc");
    if (!identity.roles.includes(role)) role = identity.roles[0];
    $("picker").hidden = false;
    notice("");
    await jobs();
    if (draft) await openDraft(draft);
  },
});
if ("serviceWorker" in navigator)
  navigator.serviceWorker.register("/sw.js").catch(() => {});
try {
  db = await database;
  if (navigator.storage?.persist) navigator.storage.persist().catch(() => {});
  await auth.start();
} catch (e) {
  notice("工作台未就绪：" + e.message);
}
