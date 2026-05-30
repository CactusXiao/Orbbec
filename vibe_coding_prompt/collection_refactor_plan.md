# Collection 模块代码修改优化计划

## 角色设置

你是一名熟悉 C++17、OpenCV、OrbbecSDK 的系统级工程师，负责对 `src/sync/collection.cpp` 进行重构优化。你需要在不改变项目整体架构、不引入新的第三方依赖（仅使用项目已有的 cJSON、OpenCV、OrbbecSDK、C++17标准库）的前提下，完成以下所有修改。**代码仅在远程 Linux 主机上编译运行，不要尝试在本地编译测试。**

修改过程中严格遵守以下编码原则：
- 保持现有代码风格（花括号换行、snake_case 命名、`const auto &` 引用优先）
- 不添加无意义注释，不在注释中解释你做了什么改动
- 异常处理沿用现有的 `try/catch(...)` + 不抛出模式
- 线程安全通过已有的 `mtx_`、原子变量维护，新增状态变量同样遵守此原则
- 所有路径操作使用 `fs::path`（已 `using namespace std::filesystem`）

---

## 项目代码结构速查

- **主文件**：`src/sync/collection.cpp`（3775 行）
- **核心类**：`MultiDeviceMemoryRecorder`（第 1113 行开始）
- **UI 入口函数**：`run_collection()`（第 3282 行开始）
- **配置 UI 结构体**：`CollectionConfigUi`（第 337 行）
- **采集 UI 结构体**：`CollectionCaptureUi`（第 3239 行）
- **保存逻辑**：`saveToDisk()`（第 1608 行）
- **相机参数写入**：`writeParamsJson()`（第 3074 行），`writeExtrinsicsJson()`（第 3164 行）
- **可视化网格绘制**：`drawRgbGrid()`（第 3246 行）

---

## 修改项一：配置界面曝光时间默认值修改

### 修改目标
将 Config 页面曝光时间（exposure_ms）输入框的初始值由 `0.8` 改为 `0.1`。

### 相关代码位置
- `CollectionConfigUi` 结构体（第 337 行）：
  ```cpp
  std::string exposureMs = "0.8";  // ← 改为 "0.1"
  ```
- `exposureMsFloat()` 方法（第 372 行）：
  ```cpp
  float exposureMsFloat() const {
      return static_cast<float>(parseDoubleBound(exposureMs, 0.8, 0.05, 100.0));
  }
  // fallback 值 0.8 也改为 0.1
  ```

### 新代码逻辑
仅修改两处字面量，无逻辑变动。

### 注意事项
- `parseDoubleBound` 的 `fallback` 参数（0.8）同步改为 0.1，以保证用户清空输入框时默认值一致。

---

## 修改项二：键盘快捷键（Ctrl+1/2/3/4）

### 修改目标
在采集 UI 页面（Capture Page），将键盘 `Ctrl+1/2/3/4` 分别映射为 Start / Stop / Save / Reset 按钮的等效操作。

### 相关代码位置
`run_collection()` 中主循环（第 3316 行），`cv::waitKey(1)` 返回键值 `key`，后续处理 Config / Capture 页面的按键事件。

### 新代码逻辑

在 Capture 页面渲染完按钮、计算 `doStart/doStop/doSave/doReset` 之后，添加如下快捷键检测逻辑：

```
// 在 Capture 页面按钮计算之后追加：
if(page == CollectionPage::Capture && key > 0) {
    // Linux/GTK OpenCV HighGUI 中 Ctrl 修饰符检测：
    // (key & 0xFF)   = 基础键值（ASCII）
    // (key & 0x40000) != 0 表示 Ctrl 按下（GTK 后端）
    const bool ctrlHeld = (key & 0x40000) != 0;
    const int baseKey = key & 0xFF;
    if(ctrlHeld) {
        if(baseKey == '1') { doStart = true; }
        else if(baseKey == '2') { doStop  = true; }
        else if(baseKey == '3') { doSave  = true; }
        else if(baseKey == '4') { doReset = true; }
    }
}
```

快捷键生成的 `doStart/doStop/doSave/doReset` 与鼠标点击按钮产生的完全相同，后续统一走状态机处理逻辑（见修改项四）。

### 注意事项
- **平台差异**：`cv::waitKey()` 的修饰符位在不同 OpenCV 后端（GTK/Qt/xcb）表现不同。
  - GTK 后端（最常见）：`(key & 0x40000) != 0` 表示 Ctrl。
  - 若该检测不生效，备选方案：把 `cv::waitKey(1)` 改为 `cv::waitKeyEx(1)` 并检查 `(key >> 16) & 0x04`。
  - 建议在代码中加入 `#ifdef`-free 的运行时 fallback：若 Ctrl 检测始终无效，至少保证鼠标点击功能正常。
- 快捷键只在 Capture 页面生效，Config 页面不触发。
- 快捷键触发的操作同样受状态机合法性检查（非法操作无效化）。

---

## 修改项三：任务管理系统（task.json + 进度 CSV）

### 修改目标
- 移除 Capture UI 左上角的 `data_id` 手动输入框。
- 从 `save_path/task.json` 自动读取任务列表（名称、中文描述、重复次数）。
- 在 `save_path/subject_id/progress.csv` 维护采集进度，支持中断恢复。
- 进入 Capture 页面时自动定位至上次中断的任务及 episode。

### 数据结构定义（添加到匿名 namespace 中）

```cpp
struct TaskInfo {
    std::string name;
    std::string description_cn;
    std::string description_en;
    int         repeat_times = 1;
};

struct TaskProgress {
    std::string task_name;
    int         completed = 0;   // 已完成次数
    int         total     = 1;   // 总重复次数（来自 task.json）
};
```

### 新增函数

**1. `loadTaskJson(path) → vector<TaskInfo>`**

解析 `save_path/task.json`，按 JSON 对象键顺序返回任务列表。

```
loadTaskJson(path):
    读取文件 → cJSON_Parse
    遍历 root->child（每个 child 对应一个任务）:
        info.name          = child->string  （JSON key）
        info.repeat_times  = 从 "repeat_times" 字段读取，默认 1
        info.description_cn = 从 "task_discription_cn" 字段读取
        info.description_en = 从 "task_discription_en" 字段读取
    返回 vector<TaskInfo>
    解析失败 → 返回空 vector，打印 stderr 警告
```

**2. `loadProgressCsv(path) → vector<TaskProgress>`**

读取 `save_path/subject_id/progress.csv`（格式：每行 `task_name,n,total`）。

```
loadProgressCsv(path):
    若文件不存在 → 返回空 vector
    逐行读取，解析 "task_name,n,total" 格式
    返回 vector<TaskProgress>（按文件行顺序）
```

**3. `saveProgressCsv(path, progress)`**

将 `vector<TaskProgress>` 写回 CSV 文件（覆盖写入）。

**4. `buildProgressFromTasks(tasks, existingProgress) → vector<TaskProgress>`**

将 task.json 中的任务与已有进度合并：
- 对每个 task，若 existingProgress 中存在同名记录，复用 completed 值；否则 completed=0。
- total 始终以 task.json 中 repeat_times 为准（防止 task.json 更新后 CSV 数据不一致）。

**5. `getCurrentTaskIndex(progress) → int`**

返回第一个 `completed < total` 的任务索引，若全部完成返回 -1（意味着任务全部结束）。

### `CollectionCaptureUi` 结构体修改

移除 `dataId`、`dataIdError` 字段，新增：

```cpp
struct CollectionCaptureUi {
    // 移除: std::string dataId; bool dataIdError;
    std::string activeField;
    std::string msg;

    // 新增：任务管理状态
    std::vector<TaskInfo>     tasks;
    std::vector<TaskProgress> progress;
    int                       currentTaskIdx  = -1;  // 当前任务索引
    int                       currentEpisode  = 0;   // 当前 episode（1-based, 即将采集的第几次）
    bool                      taskLoadError   = false;
    std::string               taskErrorMsg;
};
```

### 进入 Capture 页面时的初始化逻辑

在 Config 页面点击 "Enter Capture" 按钮后（第 3450 行附近），添加任务加载逻辑：

```
点击 "Enter Capture" 时：
    tasks = loadTaskJson(saveRoot / "task.json")
    若 tasks 为空：
        cfgUi.error = "task.json not found or invalid at: " + saveRoot
        不切换页面，显示错误
        return
    existingProgress = loadProgressCsv(saveRoot / subjectId / "progress.csv")
    capUi.progress = buildProgressFromTasks(tasks, existingProgress)
    saveProgressCsv(saveRoot / subjectId / "progress.csv", capUi.progress)
    capUi.tasks = tasks
    capUi.currentTaskIdx = getCurrentTaskIndex(capUi.progress)
    若 currentTaskIdx == -1:
        capUi.msg = "All tasks completed!"
        currentEpisode = 0
    否则:
        capUi.currentEpisode = capUi.progress[currentTaskIdx].completed + 1
    切换到 Capture 页面
```

### 保存完成后的进度更新逻辑

当异步保存成功完成时（见修改项六，save 线程回调）：

```
保存成功后：
    progress[currentTaskIdx].completed += 1
    saveProgressCsv(saveRoot / subjectId / "progress.csv", progress)
    若 progress[currentTaskIdx].completed >= progress[currentTaskIdx].total:
        currentTaskIdx = getCurrentTaskIndex(progress)  // 移到下一个
    若 currentTaskIdx == -1:
        msg = "All tasks completed!"
        currentEpisode = 0
    否则:
        currentEpisode = progress[currentTaskIdx].completed + 1
    状态机切换到 IDLE
```

### 注意事项
- **task.json 注释兼容性**：task.json 示例文件中包含 `//` 注释（非标准 JSON），需在 `loadTaskJson` 中对文件内容做预处理，逐行去除 `//` 注释后再传入 `cJSON_Parse`。
- **CSV 格式**：建议格式 `task_name,completed,total`（三列），中间不含空格，首行为表头 `task_name,completed,total`。
- **CSV 原子写入**：先写临时文件，再 `fs::rename` 替换，防止因程序崩溃导致 CSV 损坏。
- **空任务列表防御**：若 task.json 存在但格式错误（所有字段缺失），在 UI 中显示错误而非崩溃。
- **中断恢复**：progress.csv 中的 completed 代表已成功保存到磁盘的次数；未保存的采集不计入（避免数据缺失但进度误增）。
- **currentEpisode 从 1 开始**：episode_1, episode_2, ... 与 completed 的关系：`currentEpisode = completed + 1`（下一次将是第 currentEpisode 次）。

---

## 修改项四：采集 UI 布局重构 + 状态机

### 修改目标
重构 Capture 页面布局：相机实时预览固定在左上角且尺寸缩小为原来的 1/2；右侧区域大字显示任务信息；预览窗口下方大字显示当前采集状态（绿/黄/红/白）；引入状态机防止非法操作。

### 状态机定义

```cpp
enum class CaptureState {
    IDLE,      // 白色 - 待采集（含保存完毕后的等待状态）
    RECORDING, // 绿色 - 采集中
    STOPPED,   // 黄色 - 已停止，待保存
    SAVING,    // 红色 - 异步保存中
};
```

在 `run_collection()` 中添加：
```cpp
CaptureState captureState = CaptureState::IDLE;
```

**状态转移表：**

| 当前状态 | 操作 | 目标状态 | 前置条件 |
|---------|------|---------|---------|
| IDLE | Start (Ctrl+1) | RECORDING | `tasks` 不为空 && `currentTaskIdx != -1` |
| RECORDING | Stop (Ctrl+2) | STOPPED | 无（始终有效） |
| RECORDING | 超时自动停止 | STOPPED | `recorder.autoStopIfTimeout()` |
| STOPPED | Save (Ctrl+3) | SAVING | 无（始终有效） |
| STOPPED | Reset (Ctrl+4) | IDLE | 无（始终有效） |
| SAVING | 保存完成（异步） | IDLE | 后台线程信号 |
| 任意状态 | Reset (Ctrl+4) | IDLE | 强制（停止录制 + 丢弃数据） |

**非法操作无效化规则：**
- IDLE 状态下：Stop / Save 按钮视觉禁用（灰色），点击或快捷键无效果
- RECORDING 状态下：Start / Save 按钮视觉禁用，Reset 仍然有效（先调用 `stopRecording()` 再 `reset()`）
- SAVING 状态下：所有按钮视觉禁用，等待保存完成
- 任务全部完成（`currentTaskIdx == -1`）时：Start 按钮禁用

按钮状态控制实现：
```cpp
// 统一在计算 doStart/doStop/doSave/doReset 之后，应用状态机过滤
const bool allowStart  = (captureState == CaptureState::IDLE) && (capUi.currentTaskIdx != -1);
const bool allowStop   = (captureState == CaptureState::RECORDING);
const bool allowSave   = (captureState == CaptureState::STOPPED);
const bool allowReset  = (captureState != CaptureState::SAVING);

if(!allowStart)  { doStart = false; }
if(!allowStop)   { doStop  = false; }
if(!allowSave)   { doSave  = false; }
if(!allowReset)  { doReset = false; }
```

### 新 UI 布局

窗口整体尺寸保持 `cv::namedWindow` 设定的大小（1800×1000 或用户调整后的尺寸 winW×winH）。

**区域划分：**

```
┌─────────────────────────────────────────────────────────────┐
│  [相机预览区 - 左上角, winW/2 × winH/2]  │  [任务信息区 - 右侧]     │
│  固定大小，若画面超出则缩放适应              │  大字显示任务名和中文描述  │
├─────────────────────────────────────────────────────────────┤
│  [状态显示区 - 预览下方, winW/2 宽, 约 120px 高]  │  [右侧按钮区]          │
│  醒目大字显示当前状态                                │  Start/Stop/Save/Reset │
└─────────────────────────────────────────────────────────────┘
│  [底部信息栏 - 状态/错误消息, Log]                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

**具体区域坐标（以 winW=1800, winH=1000 为例，实际应动态计算）：**

```cpp
const int previewW = winW / 2;
const int previewH = winH / 2;
cv::Rect viewRect(0, 0, previewW, previewH);                    // 左上角相机预览

const int taskPanelX = previewW + 20;
const int taskPanelW = winW - taskPanelX - 20;
cv::Rect taskPanel(taskPanelX, 0, taskPanelW, winH - 160);      // 右侧任务信息

cv::Rect statusRect(0, previewH, previewW, 120);                  // 状态显示（预览下方）

cv::Rect btnArea(taskPanelX, winH - 280, taskPanelW, 200);       // 右侧按钮区

cv::Rect logRect(0, previewH + 130, previewW, winH - previewH - 160); // 左下角 log
```

**相机预览绘制修改（`drawRgbGrid` 调用）：**

在 `drawRgbGrid()` 内部，`inner` 尺寸计算需适配缩小后的 cell 大小。原函数逻辑不变，只是传入的 `r`（`viewRect`）尺寸缩小了，grid 内图像会自动按 `cv::resize(..., inner.size(), ...)` 缩放。

**任务信息区绘制：**

```cpp
// 右侧任务信息区
cv::rectangle(ui, taskPanel, cv::Scalar(25, 25, 25), cv::FILLED);
if(capUi.currentTaskIdx >= 0 && capUi.currentTaskIdx < (int)capUi.tasks.size()) {
    const auto &task = capUi.tasks[capUi.currentTaskIdx];
    const auto &prog = capUi.progress[capUi.currentTaskIdx];

    // 任务名（大字）
    cv::putText(ui, task.name,
                cv::Point(taskPanel.x + 20, taskPanel.y + 60),
                cv::FONT_HERSHEY_DUPLEX, 1.4, cv::Scalar(255, 220, 50), 2, cv::LINE_AA);

    // Episode 信息
    std::string episodeStr = "Episode " + std::to_string(capUi.currentEpisode)
                           + " / " + std::to_string(prog.total);
    cv::putText(ui, episodeStr,
                cv::Point(taskPanel.x + 20, taskPanel.y + 100),
                cv::FONT_HERSHEY_DUPLEX, 0.9, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);

    // 中文描述（自动换行）
    // 注意：OpenCV FONT_HERSHEY 系列不支持中文 Unicode，
    // 中文描述需通过 FreeType 渲染，或退而使用英文描述。
    // 若无 FreeType 支持，改为显示 task.description_en。
    // 具体处理见"注意事项"。
    auto lines = wrapTextToWidth(task.description_en, taskPanel.width - 40,
                                 cv::FONT_HERSHEY_DUPLEX, 0.75, 1);
    int descY = taskPanel.y + 150;
    for(const auto &line: lines) {
        cv::putText(ui, line, cv::Point(taskPanel.x + 20, descY),
                    cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
        descY += 28;
    }
}
```

**状态显示区绘制（大字，彩色）：**

```cpp
// 状态显示区（位于相机预览正下方）
struct StateDisplay { const char *text; cv::Scalar color; cv::Scalar bgColor; };
StateDisplay sd;
switch(captureState) {
case CaptureState::IDLE:
    sd = {"READY", cv::Scalar(255,255,255), cv::Scalar(40,40,40)};
    break;
case CaptureState::RECORDING:
    sd = {"RECORDING", cv::Scalar(50,220,50), cv::Scalar(20,60,20)};
    break;
case CaptureState::STOPPED:
    sd = {"STOPPED - SAVE?", cv::Scalar(50,200,255), cv::Scalar(20,40,80)};
    break;
case CaptureState::SAVING:
    sd = {"SAVING...", cv::Scalar(50,50,220), cv::Scalar(20,20,60)};
    break;
}
cv::rectangle(ui, statusRect, sd.bgColor, cv::FILLED);
cv::rectangle(ui, statusRect, sd.color, 2);
int baseline = 0;
auto textSz = cv::getTextSize(sd.text, cv::FONT_HERSHEY_DUPLEX, 2.0, 3, &baseline);
cv::Point textOrg(statusRect.x + (statusRect.width - textSz.width)/2,
                  statusRect.y + (statusRect.height + textSz.height)/2);
cv::putText(ui, sd.text, textOrg, cv::FONT_HERSHEY_DUPLEX, 2.0, sd.color, 3, cv::LINE_AA);
```

**按钮绘制（带禁用视觉反馈）：**

新增 `uiButtonDisablable()` 辅助函数，允许指定按钮是否可交互：

```cpp
static bool uiButtonEx(cv::Mat &img, const cv::Rect &r, const std::string &label,
                        FrameMouse &ms, bool enabled) {
    const bool hover = enabled && r.contains(cv::Point(ms.x, ms.y));
    cv::Scalar bg = !enabled ? cv::Scalar(25,25,25)
                  : (hover  ? cv::Scalar(60,60,60) : cv::Scalar(40,40,40));
    cv::Scalar fg = enabled ? cv::Scalar(255,255,255) : cv::Scalar(80,80,80);
    cv::Scalar border = enabled ? cv::Scalar(120,120,120) : cv::Scalar(50,50,50);
    cv::rectangle(img, r, bg, cv::FILLED);
    cv::rectangle(img, r, border, 1);
    cv::putText(img, label, cv::Point(r.x+14, r.y+r.height/2+7),
                cv::FONT_HERSHEY_DUPLEX, 0.7, fg, 1, cv::LINE_AA);
    if(enabled && ms.clicked && r.contains(cv::Point(ms.clickX, ms.clickY))) {
        ms.clicked = false;
        return true;
    }
    return false;
}
```

按钮布局（右侧区域，竖向排列）：

```cpp
const int btnX = taskPanel.x + 20;
const int btnW = taskPanel.width - 40;
cv::Rect bStart(btnX, winH-280, btnW, 50);
cv::Rect bStop (btnX, winH-220, btnW, 50);
cv::Rect bSave (btnX, winH-160, btnW, 50);
cv::Rect bReset(btnX, winH-100, btnW, 50);

bool doStart = uiButtonEx(ui, bStart, "Start  [Ctrl+1]", fm, allowStart);
bool doStop  = uiButtonEx(ui, bStop,  "Stop   [Ctrl+2]", fm, allowStop);
bool doSave  = uiButtonEx(ui, bSave,  "Save   [Ctrl+3]", fm, allowSave);
bool doReset = uiButtonEx(ui, bReset, "Reset  [Ctrl+4]", fm, allowReset);
```

移除原来底部的 `bMenu/bConfig/bStart/bStop/bSave/bReset` 六个按钮布局。保留 `Back to Config` 和 `Back to Menu` 按钮于 Log 区域旁（适当位置），但仅在非 RECORDING/SAVING 状态下可用。

### 注意事项
- **中文显示问题**：OpenCV 的 `FONT_HERSHEY_*` 字体不支持中文 Unicode 字符。若系统未集成 `opencv_freetype` 模块，需要改用英文描述（`description_en`）。如确需显示中文，可通过 FreeType 扩展（`#include <opencv2/freetype.hpp>`）实现，但需确认远端主机是否有该模块。建议优先用英文，把 `task_discription_cn` 存入 Log 以文本形式输出或写入文件。
- **窗口尺寸自适应**：所有区域坐标需基于运行时 `winW`/`winH` 动态计算，不能硬编码。
- **视频流画面缩放**：`drawRgbGrid` 内部已使用 `cv::resize` 适配 cell 大小，缩小 viewRect 后图像会自动缩小，无需额外改动 drawRgbGrid 函数体。

---

## 修改项五：保存路径修改（新增 episode 层级）

### 修改目标
保存路径由 `saveRoot/subjectId/dataId` 变更为 `saveRoot/subjectId/task_name/episode_N`。

### 相关代码位置
`saveToDisk()` 函数，其中 `ensureDirs` lambda（第 1778 行）和调用 `saveToDisk` 的位置（第 3709 行附近）。

### 修改方案

**`saveToDisk()` 签名变更：**

```cpp
// 旧：
bool saveToDisk(const fs::path &saveRoot, const std::string &subjectId, const std::string &dataId)

// 新：
bool saveToDisk(const fs::path &saveRoot, const std::string &subjectId,
                const std::string &taskName, int episodeN)
```

**`ensureDirs` lambda 内部修改：**

```cpp
dest = saveRoot / subjectId / taskName / ("episode_" + std::to_string(episodeN));
// 其余目录创建逻辑不变
```

**调用处修改：**

在 `doSave` 处理逻辑中（原来的第 3704 行），将参数从 `(root, subject, dataId)` 改为：

```cpp
const std::string taskName = capUi.tasks[capUi.currentTaskIdx].name;
const int episodeN = capUi.currentEpisode;
const bool ok = recorder.saveToDiskAsync(root, subject, taskName, episodeN,
                                          [&]() { /* 保存完成回调，见修改项六 */ });
```

### 注意事项
- `episode_N` 中 N 是 1-based 的 currentEpisode（第 1 次采集保存到 `episode_1`）。
- 若同名目录已存在（程序重启后意外重复），`fs::create_directories` 不会报错，但会覆盖已有帧文件。建议在 `ensureDirs` 前检查目标路径是否存在，若存在则追加后缀（如 `episode_1_dup`）或打印警告，当前实现以简单覆盖为主。

---

## 修改项六：保存异步化（解决 UI 卡顿问题）

### 修改目标
点击 Save 后立即返回 UI，后台异步完成磁盘写入；用户无需等待即可查看状态（SAVING 状态），保存完成后自动切换回 IDLE 并更新进度。

### 问题分析

原 `saveToDisk()` 同步执行，包含：
1. 从 `buffers_` 移出数据到 `local`（快，毫秒级，需持锁）
2. 时间戳对齐计算（中等，取决于帧数）
3. 磁盘写入（慢，几秒到几十秒）

步骤 1 完成后，`buffers_` 即为空，新录制即可开始——但原代码将三步合并为同步调用，UI 被阻塞在步骤 3。

### 解决方案：异步 saveToDisk

**在 `MultiDeviceMemoryRecorder` 类中新增成员变量：**

```cpp
std::thread          saveThread_;
std::atomic<bool>    saveInProgress_{ false };
std::atomic<bool>    saveSucceeded_{ false };
std::function<void(bool)> saveCompleteCb_;  // 保存完成回调 (bool = 成功?)
std::mutex           saveCbMtx_;
```

**新增 `saveToDiskAsync()` 公有方法：**

```cpp
bool saveToDiskAsync(const fs::path &saveRoot, const std::string &subjectId,
                     const std::string &taskName, int episodeN,
                     std::function<void(bool)> onComplete) {
    // 1. 前置检查（主线程，快速）
    if(recording_.load() || !hasData_.load() || saveInProgress_.load()) {
        return false;
    }

    // 2. 从 buffers_ 移出数据（持锁，快速）
    // 这部分与原 saveToDisk 开头相同，将 local/typesSaving/refType 准备好
    // ...

    // 3. 若数据为空则返回失败
    if(local.empty()) { return false; }

    saveInProgress_.store(true);
    saveSucceeded_.store(false);
    {
        std::lock_guard<std::mutex> g(saveCbMtx_);
        saveCompleteCb_ = std::move(onComplete);
    }

    // 4. 启动后台线程执行实际 I/O
    if(saveThread_.joinable()) { saveThread_.join(); }
    saveThread_ = std::thread([this, saveRoot, subjectId, taskName, episodeN,
                                local=std::move(local), typesSaving, refType]() mutable {
        const bool ok = this->doSaveToDisk(saveRoot, subjectId, taskName, episodeN,
                                            local, typesSaving, refType);
        saveSucceeded_.store(ok);
        saveInProgress_.store(false);
        std::function<void(bool)> cb;
        {
            std::lock_guard<std::mutex> g(saveCbMtx_);
            cb = std::move(saveCompleteCb_);
        }
        if(cb) { cb(ok); }
    });

    return true;  // 返回"已启动"，不代表成功
}
```

**将原 `saveToDisk()` 的 I/O 逻辑提取为 `doSaveToDisk()` 私有方法：**

`doSaveToDisk()` 接收已移出的 `local` 数据，执行原来的对齐、深度处理（新）、磁盘写入逻辑，无需再持 `mtx_` 锁，线程安全。

**UI 层回调处理：**

在 `run_collection()` 的 `doSave` 处理中：

```cpp
if(doSave) {
    const std::string taskName = capUi.tasks[capUi.currentTaskIdx].name;
    const int episodeN = capUi.currentEpisode;
    const bool started = recorder.saveToDiskAsync(root, subject, taskName, episodeN,
        [&](bool ok) {
            // 此回调在 saveThread_ 中执行，不可直接修改 UI 变量
            // 通过原子变量传递结果给主线程
            pendingSaveResult.store(ok ? 1 : -1);
        });
    if(started) {
        captureState = CaptureState::SAVING;
        pushUiLog("Save started in background: " + taskName + " episode_" + std::to_string(episodeN));
    } else {
        pushUiLog("Save failed to start");
        capUi.msg = "Save failed: not ready";
    }
}
```

在主循环中检查 `pendingSaveResult`（`std::atomic<int>`，0=未完成，1=成功，-1=失败）：

```cpp
// 在每帧渲染开始前（主循环顶部）检查：
if(captureState == CaptureState::SAVING) {
    const int result = pendingSaveResult.load();
    if(result != 0) {
        pendingSaveResult.store(0);
        if(result == 1) {
            // 保存成功：更新进度
            capUi.progress[capUi.currentTaskIdx].completed += 1;
            saveProgressCsv(root / subject / "progress.csv", capUi.progress);
            capUi.currentTaskIdx = getCurrentTaskIndex(capUi.progress);
            capUi.currentEpisode = (capUi.currentTaskIdx >= 0)
                ? (capUi.progress[capUi.currentTaskIdx].completed + 1) : 0;
            pushUiLog("Save succeeded. Next: " + (capUi.currentTaskIdx >= 0
                ? capUi.tasks[capUi.currentTaskIdx].name : "ALL DONE"));
        } else {
            pushUiLog("Save failed!");
            capUi.msg = "Save failed - data may be lost";
        }
        captureState = CaptureState::IDLE;
        recorder.reset();   // 清空已保存完毕的数据（或在 doSaveToDisk 末尾 reset）
    }
}
```

### 注意事项
- **saveThread_ 生命周期**：在 `stopIfRunning()` 和析构函数中需 `join()` saveThread_，防止野线程。
- **多次点击 Save**：`saveInProgress_` 原子标志防止重复启动，第二次点击 Save 在 SAVING 状态下会被状态机过滤（无法触达）。
- **回调线程安全**：回调在 saveThread_ 中执行，只能写原子变量 `pendingSaveResult`，不能直接修改 `capUi` 等 UI 变量（UI 变量只在主线程修改）。
- **recorder.reset() 时机**：原 `doSaveToDisk()` 末尾调用 `reset()` 清空 `buffers_`，但此时 `buffers_` 在步骤 2 已被移空，`reset()` 实质上是清空标志位。确认 `hasData_` 在 doSaveToDisk 中正确重置为 false。
- **saveThread 中的 local 生命周期**：通过 lambda 值捕获 `local`（`std::move`），确保数据在 saveThread 运行期间有效。

---

## 修改项七：深度图对齐至 RGB 像素坐标系

### 修改目标
在保存深度帧时，将原始深度图（深度相机坐标系）通过 d2c 外参变换，重映射到 RGB 相机的像素坐标系，输出与 RGB 图像分辨率相同的对齐深度图；对齐后的深度图空洞区域进行填充；保存 RGB 相机视角下的深度图（单位保持 uint16 mm/valueScale）。

### 算法原理

对深度图中每个像素 `(u_d, v_d)` 执行以下步骤：
1. 取深度值 `d`（raw），计算深度（mm）：`Z_mm = d × valueScaleMm`
2. 利用 OrbbecSDK 的 `ob::CoordinateTransformHelper::transformation2dto2d()` 将深度像素投影到 RGB 像素坐标：
   - 输入：深度相机内参、深度相机畸变、RGB 相机内参、RGB 相机畸变、d2c 变换（`rgbDepthParam`）
   - 输出：RGB 像素坐标 `(u_rgb, v_rgb)`
3. 计算 RGB 相机坐标系下的深度值（Z_rgb）：
   - 利用 d2c 旋转平移矩阵计算 3D 点在 RGB 相机坐标系下的 Z 分量：
     ```
     3D_depth = [(u_d - cx_d)/fx_d * Z_mm, (v_d - cy_d)/fy_d * Z_mm, Z_mm]  (单位 mm)
     3D_rgb   = R * 3D_depth + t  (t 单位 mm，来自 d2c_extrinsic)
     Z_rgb_mm = 3D_rgb.z
     aligned_d = round(Z_rgb_mm / valueScaleMm)  (保持与原始相同的量化单位)
     ```
   - 注意：`camera_params.json` 中 `d2c_extrinsic` 的 `translation` 单位为 **mm**，`transform.trans` 中的单位也是 mm（OrbbecSDK 内部也是 mm）。无需再乘 1000。
4. 填充 `alignedDepth(v_rgb, u_rgb) = static_cast<uint16_t>(aligned_d)`（若该位置已有值，取最近的即可，此处以覆盖方式处理，最后到达的覆盖先到达的，深度图通常连续，差异极小）。
5. 裁剪 RGB 图像范围外的点（`u_rgb < 0 || v_rgb < 0 || u_rgb >= rgbW || v_rgb >= rgbH`）。

### 空洞填充方案

深度图投影到 RGB 空间后存在空洞（多对一映射的逆问题），推荐方案：

**方案A（简单快速，推荐）：形态学膨胀**
```cpp
// 用小核多次膨胀填充单像素空洞
cv::Mat mask = (alignedDepth == 0);
cv::Mat dilated;
cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
for(int iter = 0; iter < 3; ++iter) {
    cv::Mat tmp;
    cv::dilate(alignedDepth, tmp, kernel);
    tmp.copyTo(alignedDepth, mask);   // 只在空洞处用膨胀值填充
    mask = (alignedDepth == 0);
}
```

**方案B（更精确，较慢）：最近邻插值**  
使用 `cv::inpaint()`（需要 16U 转换为 8U/float，较复杂），或构建 KD-tree 查询最近有效深度点。

推荐使用方案A，形态学膨胀 3 轮足以填充因投影产生的 1-2 像素空洞，且速度极快（毫秒级）。

### 新增函数：`alignDepthToRgb()`

在匿名 namespace 中添加：

```cpp
static cv::Mat alignDepthToRgb(const cv::Mat &depth16,
                                float valueScaleMm,
                                const OBCameraParam &rgbDepthParam,
                                int rgbW, int rgbH) {
    // depth16: CV_16UC1 原始深度图，值 × valueScaleMm = mm
    // 返回：CV_16UC1 对齐后深度图，尺寸 rgbW × rgbH，同样值 × valueScaleMm = Z_rgb_mm

    cv::Mat aligned(rgbH, rgbW, CV_16UC1, cv::Scalar(0));
    if(depth16.empty() || depth16.type() != CV_16UC1 || !(valueScaleMm > 0.0f)) {
        return aligned;
    }
    if(rgbDepthParam.depthIntrinsic.fx <= 0.0f || rgbDepthParam.rgbIntrinsic.fx <= 0.0f) {
        return aligned;
    }

    const int dW = depth16.cols;
    const int dH = depth16.rows;
    const float fx_d = rgbDepthParam.depthIntrinsic.fx;
    const float fy_d = rgbDepthParam.depthIntrinsic.fy;
    const float cx_d = rgbDepthParam.depthIntrinsic.cx;
    const float cy_d = rgbDepthParam.depthIntrinsic.cy;
    const float *R = rgbDepthParam.transform.rot;   // 3×3, row-major
    const float *t = rgbDepthParam.transform.trans; // [tx, ty, tz] in mm

    for(int v = 0; v < dH; ++v) {
        const uint16_t *row = depth16.ptr<uint16_t>(v);
        for(int u = 0; u < dW; ++u) {
            const uint16_t d = row[u];
            if(d == 0) { continue; }
            const float Z_mm = static_cast<float>(d) * valueScaleMm;
            const float X_mm = (static_cast<float>(u) - cx_d) * Z_mm / fx_d;
            const float Y_mm = (static_cast<float>(v) - cy_d) * Z_mm / fy_d;

            // 变换到 RGB 相机坐标系（单位 mm）
            const float Xr = R[0]*X_mm + R[1]*Y_mm + R[2]*Z_mm + t[0];
            const float Yr = R[3]*X_mm + R[4]*Y_mm + R[5]*Z_mm + t[1];
            const float Zr = R[6]*X_mm + R[7]*Y_mm + R[8]*Z_mm + t[2];

            if(Zr <= 0.0f) { continue; }

            // 投影到 RGB 像素坐标（利用 RGB 相机内参，不考虑畸变，近似处理）
            // 若需精确畸变校正，可调用 transformation2dto2d（但较慢）
            const float u_rgb_f = rgbDepthParam.rgbIntrinsic.fx * Xr / Zr
                                 + rgbDepthParam.rgbIntrinsic.cx;
            const float v_rgb_f = rgbDepthParam.rgbIntrinsic.fy * Yr / Zr
                                 + rgbDepthParam.rgbIntrinsic.cy;
            const int u_rgb = static_cast<int>(u_rgb_f + 0.5f);
            const int v_rgb = static_cast<int>(v_rgb_f + 0.5f);
            if(u_rgb < 0 || v_rgb < 0 || u_rgb >= rgbW || v_rgb >= rgbH) { continue; }

            const uint16_t alignedVal = static_cast<uint16_t>(Zr / valueScaleMm + 0.5f);
            aligned.at<uint16_t>(v_rgb, u_rgb) = alignedVal;
        }
    }

    // 空洞填充（形态学膨胀）
    cv::Mat mask = (aligned == 0);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    for(int iter = 0; iter < 3; ++iter) {
        cv::Mat tmp;
        cv::dilate(aligned, tmp, kernel);
        tmp.copyTo(aligned, mask);
        mask = (aligned == 0);
        if(cv::countNonZero(mask) == 0) { break; }
    }

    return aligned;
}
```

### 集成到保存流程

在 `doSaveToDisk()`（即原 `saveToDisk()` 的 I/O 部分）中，修改深度帧保存逻辑（原第 1982-1988 行）：

```cpp
// 原代码（深度直接保存）：
// saveRawMatToPng(ser.frames[picked], outPath, pngCompression);

// 新代码（深度对齐后保存）：
if(t == CollectDataType::Depth) {
    // 需要 RGB 分辨率信息
    int rgbW = 0, rgbH = 0;
    auto itRgbP = local[sn].params.find(CollectDataType::RGB);
    if(itRgbP != local[sn].params.end() && itRgbP->second.valid) {
        rgbW = itRgbP->second.width;
        rgbH = itRgbP->second.height;
    }
    float vScale = 0.0f;
    if(picked < ser.valueScale.size()) { vScale = ser.valueScale[picked]; }

    if(local[sn].rgbDepthParamValid && rgbW > 0 && rgbH > 0 && vScale > 0.0f) {
        cv::Mat aligned = alignDepthToRgb(ser.frames[picked], vScale,
                                           local[sn].rgbDepthParam, rgbW, rgbH);
        saveRawMatToPng(aligned, outPath, pngCompression);
    } else {
        // fallback：无法对齐时保存原始深度图
        saveRawMatToPng(ser.frames[picked], outPath, pngCompression);
    }
} else {
    // RGB / IR 帧保存逻辑不变
}
```

### 注意事项
- **`transform.trans` 单位确认**：OrbbecSDK 的 `OBD2CTransform.trans` 字段单位是 **mm**（与 camera_params.json 中的 `d2c_extrinsic.translation` 一致）。代码中直接使用，不再乘 1000。若实测结果偏移异常，检查此单位。
- **畸变处理简化**：上述 `alignDepthToRgb` 投影时未对 RGB 应用畸变校正（深度相机通常无畸变 k1=k2=0）。对于标定精度要求不高的场景可接受；若需精确，改用 `ob::CoordinateTransformHelper::transformation2dto2d()` 逐像素调用（速度更慢）。
- **性能考虑**：`alignDepthToRgb` 对 1280×800 深度图执行逐像素循环，约 100 万次运算，在无 SIMD 优化情况下预计 50-200ms。在后台保存线程中执行，不影响 UI 响应。
- **覆盖冲突**：多个深度像素可能映射到同一 RGB 像素（深度图与 RGB 图分辨率相似时较少发生），此处取后到达者覆盖；如需取最近深度，在赋值前判断 `alignedVal < aligned.at<uint16_t>(v_rgb, u_rgb)` 后再赋值。
- **aligned_d 溢出**：若 `Zr / valueScaleMm > 65535`（深度 > 约 65m 在 valueScale=1mm 时），会溢出。添加上界检查：`if(Zr / valueScaleMm > 65534.0f) continue;`。

---

## 修改项八：相机参数保存简化（仅保存 RGB 内参）

### 修改目标
深度对齐完成后，保存目录中的 `camera_params.json` 不再需要深度相机内参和外参，仅保留每台相机的 RGB 内参（fx, fy, cx, cy, width, height）。

### 相关代码位置
`writeParamsJson()` 静态方法（第 3074 行）。

### 新代码逻辑

完全重写 `writeParamsJson()`：

```cpp
static void writeParamsJson(const fs::path &dest,
                            const std::unordered_map<std::string, DeviceBuffer> &buffers,
                            const std::vector<CollectDataType> & /*typesSaving*/) {
    cJSON *root = cJSON_CreateObject();
    for(const auto &kv: buffers) {
        const auto &buf = kv.second;
        cJSON *camObj = cJSON_CreateObject();
        jsonAddString(camObj, "sn", kv.first);

        // 仅保存 RGB 内参
        auto itRgb = buf.params.find(CollectDataType::RGB);
        if(itRgb != buf.params.end() && itRgb->second.valid) {
            cJSON *intr = cJSON_CreateObject();
            jsonAddIntrinsic(intr, itRgb->second.intrinsic);
            cJSON_AddItemToObject(camObj, "rgb_intrinsic", intr);

            cJSON *dist = cJSON_CreateObject();
            jsonAddDistortion(dist, itRgb->second.distortion);
            cJSON_AddItemToObject(camObj, "rgb_distortion", dist);
        }
        cJSON_AddItemToObject(root, buf.camKey.c_str(), camObj);
    }
    char *printed = cJSON_Print(root);
    if(printed) {
        writeTextFile(dest / "camera_params.json", printed);
        cJSON_free(printed);
    }
    cJSON_Delete(root);
}
```

同时，`writeExtrinsicsJson()` 保持不变（继续保存多相机间外参）。

### 注意事项
- `typesSaving` 参数保留但不再使用（可标记 `(void)typesSaving;`）以保持签名兼容性。
- RGB 内参通过 `buf.params.find(CollectDataType::RGB)` 获取，确保 RGB 流在采集时已开启（`enableRgb=true` 是默认值，且深度对齐本身也要求 RGB 开启）。
- 若用户未开启 RGB 流（极少见），`itRgb` 找不到，`camObj` 中无 `rgb_intrinsic`，输出 JSON 中该相机只有 `sn` 字段——此为边界情况，可接受（打印 stderr 警告）。

---

## 修改项九：效率问题审查与修复

### 问题一：saveToDisk 阻塞 UI（已在修改项六解决）
通过异步化彻底解决。

### 问题二：latestRgbFrames() 每帧解码开销
`latestRgbFramesImpl()` 在每帧 UI 刷新时调用，若有未解码的 RGB 帧（`buf.latestRgbFrame != nullptr`），会在主线程持锁期间外执行 `visualizeObFrame()`（解码 MJPEG 等格式）。

**优化**：该函数已经将解码移出持锁范围（`pending` 模式），逻辑上是正确的；但可以进一步控制解码频率：

```cpp
// 在 run_collection() 主循环中，每隔 N 次才调用 latestRgbFrames()
// 例如保持最大 30fps 刷新：
static auto lastPreviewTime = std::chrono::steady_clock::now();
const auto now = std::chrono::steady_clock::now();
const auto previewInterval = std::chrono::milliseconds(33); // ~30fps
if(now - lastPreviewTime >= previewInterval) {
    latest = recorder.latestRgbFrames();
    lastPreviewTime = now;
}
// 否则复用上一帧 latest
```

### 问题三：recordWorkerLoop 中 MJPEG 解码在记录线程完成
`copyColorFrameToBgr()` 调用 `cv::imdecode()` 执行 MJPEG 解码，在 recordWorkerLoop（单线程）中串行执行。

**优化**：可增加 record worker 线程数量，但改动较大；更简单的方案是在预览时延迟解码（已有 `latestRgbFramesImpl()` 的 pending 机制），录制期间只存储原始帧指针，延迟到 saveToDisk 时批量解码——这已是现有架构的设计意图。当前瓶颈更多在磁盘 I/O，不在解码，故此项暂不修改。

### 问题四：stopRecordWorker() 等待时间过长

`waitRecordWorkerIdle()` 使用 300ms 超时（第 2604 行），这在 record 队列 large 时可能不够或者等待了不必要的时间。

```cpp
// 原：
recordDrainCv_.wait_for(lock, std::chrono::milliseconds(300), ...);

// 修改为：等待更长（录制结束时最多等 2s，通常 recordQueue 很快处理完）
recordDrainCv_.wait_for(lock, std::chrono::seconds(2), [&]() {
    return recordQueue_.empty() && recordInFlight_ == 0;
});
```

### 注意事项
- 异步 save 完成后，需确认 `saveThread_.join()` 在程序退出时被调用，防止 detached 线程访问已析构对象。在 `stopIfRunning()` 和 `MultiDeviceMemoryRecorder` 析构函数中添加 `if(saveThread_.joinable()) saveThread_.join();`。

---

## 完整修改顺序建议

按如下顺序实施修改，每步可单独编译验证：

1. **修改项一**：修改默认曝光值（1行改动，最简单）
2. **修改项八**：简化 `writeParamsJson()`（独立函数，无依赖）
3. **修改项三**：添加 TaskInfo/TaskProgress 结构体和任务读取函数
4. **修改项五**：修改 `saveToDisk` 签名，更新路径生成逻辑
5. **修改项六**：异步化 saveToDisk（拆分 doSaveToDisk，添加 saveThread）
6. **修改项七**：添加 `alignDepthToRgb()` 并集成到 doSaveToDisk
7. **修改项四**：重构 UI 布局 + 状态机
8. **修改项二**：添加 Ctrl+1/2/3/4 快捷键（状态机之后最后添加）
9. **修改项九**：效率优化（最后进行，不影响功能正确性）

---

## 关键边界条件汇总

| 场景 | 处理方式 |
|------|---------|
| task.json 不存在 | 进入 Capture 时报错，不切换页面 |
| task.json 含 `//` 注释 | 预处理去除注释行后 cJSON_Parse |
| 所有任务已完成（progress 全满） | currentTaskIdx=-1，Start 按钮禁用，显示"All tasks completed" |
| progress.csv 损坏/格式错误 | 回退到从头开始（completed=0），打印警告 |
| rgbDepthParam 无效时保存深度 | fallback 保存原始深度，打印警告 |
| 对齐深度值溢出 uint16_t | 跳过该像素（continue） |
| 保存线程在录制中断中途退出 | stopIfRunning() 和析构函数 join saveThread_ |
| 同名 episode 目录已存在 | create_directories 不报错但覆盖，接受此行为 |
| Ctrl 修饰符检测失败 | 快捷键静默失效，鼠标点击功能正常 |
| 多相机中某台无 RGB 流参数 | writeParamsJson 跳过 rgb_intrinsic，打印 stderr 警告 |
| 深度图无法对齐（分辨率 = 0）| fallback 保存原始深度 |
| Save 时 hasData=false | saveToDiskAsync 直接返回 false |
| SAVING 状态下强制退出（Back to Menu）| 等待 saveThread_ join 后再退出，或 detach + 忽略（选择前者，确保数据安全） |
