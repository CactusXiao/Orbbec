// Task/Episode navigation follows Label HomePage and QC selection pages.
export class NativeQueue {
  constructor(root, { open, restore, refresh }) {
    this.root = root;
    this.open = open;
    this.restore = restore;
    this.refresh = refresh;
    this.countdown = [];
    setInterval(() => {
      this.countdown = this.countdown.filter(([cell]) => cell.isConnected);
      for (const [cell, get] of this.countdown) cell.textContent = get();
    }, 1000);
    this.task = null;
    this.episode = null;
    this.qcEpisodes = false;
  }
  table(headings, rows, onSelect, onOpen) {
    const table = document.createElement("table");
    table.className = "nativeTable";
    table.tabIndex = 0;
    const head = table.createTHead().insertRow();
    for (const text of headings) {
      const th = document.createElement("th");
      th.textContent = text;
      head.append(th);
    }
    const body = table.createTBody();
    let selected = null;
    for (const item of rows) {
      const row = body.insertRow();
      for (const text of item.values) {
        const cell = row.insertCell();
        cell.textContent = typeof text === "function" ? text() : (text ?? "");
        if (typeof text === "function") this.countdown.push([cell, text]);
      }
      row.onclick = () => {
        for (const r of body.rows) r.classList.remove("selected");
        row.classList.add("selected");
        selected = item;
        onSelect?.(item);
      };
      row.ondblclick = () => {
        row.onclick();
        onOpen?.(item);
      };
      if (!selected) {
        selected = item;
        row.classList.add("selected");
      }
    }
    table.onkeydown = (e) => {
      if (e.key === "Enter" && selected) {
        e.preventDefault();
        onOpen?.(selected);
      }
    };
    return table;
  }
  button(text, fn) {
    const b = document.createElement("button");
    b.textContent = text;
    b.onclick = fn;
    return b;
  }
  show(role, items, drafts, identity) {
    if (this.role !== role) {
      this.task = null;
      this.qcEpisodes = false;
    }
    this.role = role;
    this.items = items;
    this.drafts = drafts.filter(
      (d) => d.manifest.role === role && !d.receipt && !d.archived,
    );
    this.identity = identity;
    this.render();
  }
  groups() {
    return [
      ...new Set([
        ...this.items.map((i) => i.task_name),
        ...this.drafts.map((d) => d.manifest.task_name),
      ]),
    ]
      .filter(Boolean)
      .sort()
      .map((name) => {
        const local = this.drafts.filter((d) => d.manifest.task_name === name),
          localJobs = new Set(local.map((d) => d.manifest.job_id));
        return {
          name,
          local,
          items: this.items.filter(
            (i) => i.task_name === name && !localJobs.has(i.job_id),
          ),
        };
      });
  }
  render() {
    this.root.replaceChildren();
    const qc = this.role === "qc",
      groups = this.groups();
    if (
      !groups.some((g) => g.name === this.task) &&
      !(qc && this.qcEpisodes && this.task)
    ) {
      this.task = groups[0]?.name || null;
      this.qcEpisodes = false;
    }
    const title = document.createElement("h1");
    title.textContent = qc
      ? this.qcEpisodes
        ? `Episode 选择 - ${this.task}`
        : "人工质检"
      : "关节标注";
    const top = document.createElement("div");
    top.className = "toolbar";
    top.append(title);
    if (qc && this.qcEpisodes)
      top.append(
        this.button("返回 Task", () => {
          this.qcEpisodes = false;
          this.render();
        }),
      );
    if (qc) top.append(this.button("刷新", this.refresh));
    this.root.append(top);
    const status = document.createElement("p");
    status.textContent = qc
      ? this.qcEpisodes
        ? "双击 Episode 开始质检；进行中的任务可继续上次进度。"
        : `后端：${location.origin}    Worker：${this.identity.username}`
      : "选择任务与 Episode，检查并修正多视角手部关节。";
    this.root.append(status);
    if (!qc) {
      const info = document.createElement("div");
      info.className = "nativeConnection";
      for (const [label, value] of [
        ["服务地址", location.origin],
        ["操作员", this.identity.username],
      ]) {
        const field = document.createElement("label");
        field.textContent = label;
        const input = document.createElement("input");
        input.value = value;
        input.readOnly = true;
        field.append(input);
        info.append(field);
      }
      this.root.append(info);
    }
    const grid = document.createElement("div");
    grid.className = qc ? "nativeQueues single" : "nativeQueues";
    this.root.append(grid);
    if (!qc || !this.qcEpisodes) {
      const card = document.createElement("section");
      card.className = "queueCard";
      if (!qc) {
        const h = document.createElement("h2");
        h.textContent = "01  选择任务";
        card.append(h);
      }
      card.append(
        this.table(
          qc
            ? ["Task", "待质检 Episode", "本机进行中", "最快过期"]
            : ["任务", "片段", "批次", "受试者", "帧数"],
          groups.map((g) => ({
            group: g,
            values: qc
              ? [
                  `${g.local.length ? "● " : ""}${g.name}`,
                  g.items.length,
                  g.local.length,
                  () => this.expiry(g.local),
                ]
              : [
                  g.name,
                  [...g.items, ...g.local.map((d) => d.manifest)].reduce(
                    (n, i) => n + (i.segments || 1),
                    0,
                  ),
                  g.items.length + g.local.length,
                  [
                    ...new Set(
                      [...g.items, ...g.local.map((d) => d.manifest)].map(
                        (i) => i.subject_id,
                      ),
                    ),
                  ].join(", "),
                  [...g.items, ...g.local.map((d) => d.manifest)].reduce(
                    (n, i) =>
                      n +
                      (Array.isArray(i.frames)
                        ? i.frames.length
                        : i.frames || 0),
                    0,
                  ),
                ],
          })),
          (item) => {
            this.task = item.group.name;
            if (!qc) this.renderEpisodes(grid);
          },
          (item) => {
            if (qc) {
              this.task = item.group.name;
              this.qcEpisodes = true;
              this.render();
            }
          },
        ),
      );
      grid.append(card);
    }
    if (!qc || this.qcEpisodes) this.renderEpisodes(grid);
    if (!qc) {
      const footer = document.createElement("div");
      footer.className = "toolbar";
      footer.append(
        this.button("刷新任务", this.refresh),
        this.button("开始标注所选 Episode", () => this.enter(this.episode)),
      );
      this.root.append(footer);
    }
    if (!groups.length) {
      const empty = document.createElement("p");
      empty.textContent = qc ? "暂无待质检任务。" : "暂无待标注任务。";
      this.root.append(empty);
    }
  }
  renderEpisodes(grid) {
    grid.querySelector(".episodeCard")?.remove();
    const qc = this.role === "qc",
      g = this.groups().find((g) => g.name === this.task);
    const entries = [
      ...(g?.local || []).map((d) => ({ draft: d, manifest: d.manifest })),
      ...(g?.items || []).map((item) => ({ item, manifest: item })),
    ];
    this.episode = entries[0] || null;
    const card = document.createElement("section");
    card.className = "queueCard episodeCard";
    if (!qc) {
      const h = document.createElement("h2");
      h.textContent = "02  选择 Episode";
      card.append(h);
    }
    card.append(
      this.table(
        qc
          ? ["状态", "Episode ID", "Subject", "帧数", "租期剩余"]
          : ["Episode", "受试者", "片段", "帧数", "起始帧"],
        entries.map((entry) => {
          const m = entry.manifest,
            frames = Array.isArray(m.frames) ? m.frames.length : m.frames || 0;
          return {
            ...entry,
            values: qc
              ? [
                  entry.draft
                    ? entry.draft.released ||
                      Date.parse(entry.draft.manifest.lease_until || "") <=
                        Date.now()
                      ? "已释放 · 有本地进度"
                      : "进行中"
                    : "可领取",
                  m.episode_index,
                  m.subject_id || "",
                  frames,
                  entry.draft ? () => this.expiry([entry.draft]) : "",
                ]
              : [
                  m.episode_index,
                  m.subject_id || "",
                  m.segments || 1,
                  frames,
                  m.first_start_frame ?? m.frames?.[0] ?? 0,
                ],
          };
        }),
        (item) => (this.episode = item),
        (item) => this.enter(item),
      ),
    );
    grid.append(card);
  }
  enter(entry) {
    if (!entry) return;
    entry.draft ? this.restore(entry.draft) : this.open(entry.item, this.role);
  }
  expiry(drafts) {
    const active = drafts
      .filter((d) => !d.released)
      .map((d) => Date.parse(d.manifest.lease_until || ""))
      .filter((time) => Number.isFinite(time) && time > Date.now());
    if (!active.length) return drafts.length ? "已释放" : "";
    const n = Math.max(
      0,
      Math.floor((Math.min(...active) - Date.now()) / 1000),
    );
    return [Math.floor(n / 3600), Math.floor(n / 60) % 60, n % 60]
      .map((n) => String(n).padStart(2, "0"))
      .join(":");
  }
}
