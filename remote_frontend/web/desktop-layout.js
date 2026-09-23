export function installDesktopLayout() {
  const $ = (id) => document.getElementById(id),
    make = (tag, id, cls) => {
      const e = document.createElement(tag);
      if (id) e.id = id;
      if (cls) e.className = cls;
      return e;
    };
  const hidden = make("div", "legacyControls");
  hidden.hidden = true;
  document.body.append(hidden);
  $("refresh").hidden = true;
  $("video").setAttribute("aria-hidden", "true");
  const queue = make("div", "nativeQueue");
  $("picker").append(queue);
  for (const e of [...$("picker").children])
    if (e !== queue && e.tagName !== "NAV") hidden.append(e);
  const back = make("button", "prepareBack");
  back.textContent = "返回任务";
  back.hidden = true;
  $("workspace").prepend(back);
  const decode = make("section", "decodeStatus");
  decode.hidden = true;
  $("workspace").prepend(decode);
  const info = make("div", "nativeInfo");
  info.append($("title"), $("frameStatus"), $("viewNotice"));
  const progressToggle = make("button", "toggleProgress");
  progressToggle.textContent = "标注进度";
  progressToggle.setAttribute("aria-expanded", "false");
  progressToggle.setAttribute("aria-controls", "taskProgress");
  progressToggle.onclick = () => {
    const open = $("content").classList.toggle("showProgress");
    progressToggle.setAttribute("aria-expanded", String(open));
  };
  info.append(progressToggle);
  $("workArea").prepend(info);
  const side = $("taskProgress");
  side.querySelector("h3").textContent = "待标注区间";
  $("frameList").replaceChildren();
  const editorHost = $("editorLayout");
  for (const e of [...editorHost.children]) hidden.append(e);
  const canvas = make("canvas", "nativeLabelCanvas");
  canvas.setAttribute("aria-label", "关节标注画布");
  editorHost.append(canvas);
  for (const e of [...$("labelPanel").children])
    if (!["editorLayout", "overviewGrid"].includes(e.id)) hidden.append(e);
  const labelBar = make("div", "labelToolbar", "nativeToolbar");
  const cycle = make("button", "cycleSource");
  cycle.textContent = "视图：修改后视角";
  for (const e of [
    $("prev"),
    $("next"),
    $("overview"),
    $("prevCamera"),
    $("nextCamera"),
    $("undo"),
    $("ignore"),
    cycle,
    $("skeleton"),
    $("mesh"),
    $("confirmFrame"),
    $("submit"),
    $("home"),
  ])
    labelBar.append(e);
  const playback = make("div", "qcPlaybackToolbar", "nativeToolbar"),
    bad = make("div", "qcBadToolbar", "nativeToolbar");
  const add = (id, text, target) => {
    const e = make("button", id);
    e.textContent = text;
    target.append(e);
    return e;
  };
  playback.append($("prevTen"));
  add("qcPrev", "上一帧", playback);
  playback.append($("play"));
  add("qcNext", "下一帧", playback);
  playback.append($("nextTen"), $("enterRange"));
  add("episodeException", "Episode 异常", playback).className = "danger";
  const playbackStatus = make("span", "nativePlaybackStatus");
  info.append(playbackStatus);
  add("qcSubmit", "提交", playback).className = "primary";
  add("qcExit", "退出程序", playback);
  add("qcHome", "返回 Episode 列表", playback);
  add("badPrevTen", "上十帧", bad);
  add("badPrev", "上一帧", bad);
  bad.append($("setStart"));
  bad.append(make("span", "nativeBadStatus"));
  bad.append($("setEnd"));
  add("badNext", "下一帧", bad);
  add("badNextTen", "下十帧", bad);
  bad.append($("egoRange"), $("badRange"));
  add("cancelBad", "撤销", bad);
  bad.hidden = true;
  const timeline = make("div", "nativeTimeline");
  timeline.append($("progress"), $("timelineMarks"), $("timeline"));
  const display = make("div", "qcDisplayToolbar", "nativeToolbar");
  const opacityLabel = make("label");
  opacityLabel.textContent = "MANO 不透明度 ";
  const opacity = make("input", "manoOpacity");
  opacity.type = "range";
  opacity.min = "0";
  opacity.max = "100";
  opacity.value = "100";
  opacity.setAttribute("aria-label", "MANO 不透明度");
  opacityLabel.append(opacity, make("span", "manoOpacityValue"));
  display.append(opacityLabel);
  add("highQuality", "高清检查", display);
  display.append(make("span", "qcBufferStatus"));
  const qcControls = make("div", "qcControlBar", "nativeToolbar");
  qcControls.append(playback, bad, display);
  $("workArea").append(timeline, labelBar, qcControls);
  const qcGrid = make("div", "nativeQcGrid", "nativeCameraGrid");
  $("qcPanel").append(qcGrid);
  for (const e of [...$("qcPanel").children])
    if (e !== qcGrid) hidden.append(e);
  // Retain one hidden hardware-decoded stream; the six cells display its tiles.
  $("video").removeAttribute("controls");
  $("video").muted = true;
  $("video").className = "videoTransport";
  document.body.append($("video"));
  for (const e of [...$("workArea").children])
    if (e.tagName === "FOOTER") hidden.append(e);
  const top = $("workspace").querySelector(":scope > .toolbar");
  if (top) hidden.append(top);
  $("overview").textContent = "六视角总览（0）";
  $("prev").textContent = "上一帧";
  $("next").textContent = "下一帧";
  $("prevCamera").textContent = "上一机位";
  $("nextCamera").textContent = "下一机位";
  $("setStart").textContent = "设为坏帧起点";
  $("setEnd").textContent = "设为坏帧终点";
  $("egoRange").textContent = "确认为 EgoPose 外参不准";
  $("badRange").textContent = "确认为手部 Pose 不准";
}
