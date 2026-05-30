# Calibration 可视化链路设计（完全对齐 collection 实现风格）

## 1. 目标与约束

- 目标：Calibration 中所有视频可视化链路（开流、回调存帧、UI取帧显示）完全复用 collection 的实现思路与调用顺序。
- 约束：凡是可以直接使用官方 SDK API 的能力，全部用 API，不新增自定义取流路径。
- 约束：实时显示与采样解耦，实时显示由 UI 主循环每帧刷新，采样仅控制采样逻辑，不控制显示逻辑。

## 2. collection 实现全链路梳理

### 2.1 开流（官方 API）

- 入口：`MultiDeviceMemoryRecorder::start(...)`。
- 步骤：
  - 查询设备并筛选配置设备；
  - `applySyncConfig` + `splitPrimarySecondary`；
  - 为每台设备创建 `ob::Config`；
  - `config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ANY_SITUATION)`；
  - `pickVideoProfile(...)` 选择 profile；
  - `config->enableStream(profile)`；
  - `pipe->start(config, callback)` 开启回调取流。

### 2.2 回调缓存（仅缓存，不做重活）

- 入口：`onFrameSet(...)`。
- 对预览：读取 `frameSet->colorFrame()`，按 `globalTimeStampUs` 优先、必要时回退 `timeStampUs`，写入：
  - `latestRgbFrame`
  - `latestRgbFrameTsUs`
- 回调内不做 UI 绘制，不阻塞。

### 2.3 UI 侧取图（惰性解码）

- 入口：`latestRgbFramesImpl()`。
- 流程：
  - 先收集 pending 帧（`latestRgbFrame`）；
  - 对 pending 执行 `visualizeObFrame(...)`；
  - 时间戳一致时提交到 `latestRgb/latestRgbTsUs` 并 reset 帧指针；
  - 返回可显示的 `map<camKey, cv::Mat>`。

### 2.4 UI 实时显示（每帧）

- 入口：`run_collection` 主循环。
- 每帧固定顺序：
  - `waitKey(1)`；
  - `latestRgbFrames()`；
  - `drawRgbGrid(...)`；
  - `imshow(...)`。
- 这条链路没有 2 秒门限；实时性由设备输出帧率、profile、解码开销决定。

## 3. Calibration 对齐实现设计

### 3.1 数据结构（对齐 collection DeviceBuffer 思路）

- 采用 `previewBuffers_`（按 `camKey` 索引）保存：
  - `latestRgbFrame`
  - `latestRgbFrameTsUs`
  - `latestRgb`
  - `latestRgbTsUs`
- 记录活动相机键：`activeCam1_`、`activeCam2_`。

### 3.2 开流（对齐 collection startOne）

- `startPair` 中：
  - `setFrameAggregateOutputMode(ANY_SITUATION)`；
  - `pickVideoProfile(..., fps=targetFps)`；
  - `enableStream(colorProfile)`；
  - `pipe->start(config, callback)`；
  - callback 绑定 `camKey`，调用 `onPairFrameSet(camKey, fs)`。

### 3.3 回调缓存（对齐 onFrameSet 预览分支）

- `onPairFrameSet(camKey, fs)` 只做：
  - 取 `colorFrame`；
  - 计算时间戳（global 优先）；
  - 写入 `previewBuffers_[camKey].latestRgbFrame/latestRgbFrameTsUs`。

### 3.4 UI 取图与显示（对齐 latestRgbFramesImpl + drawRgbGrid）

- `latestRgbFramesFromPairImpl()` 直接采用 collection 的 pending->decode->提交逻辑。
- `updateLiveFramesFromPreview()` 每帧调用该函数并更新 `live1/live2`。
- 展示函数使用 collection 同款 `drawRgbGrid`，不再使用自定义上下分屏渲染。

### 3.5 采样与实时解耦

- `updateStreaming()` 首行执行实时链路更新（取图并准备显示）。
- `dt < 2000ms` 仅用于“是否采样”判断。
- 采样失败不影响当前帧 UI 显示。

## 4. 当前代码落地点（Calibration）

- 开流：`startPair(...)`
- 回调：`onPairFrameSet(...)`
- UI 取图：`latestRgbFramesFromPairImpl(...)`
- 实时更新：`updateLiveFramesFromPreview(...)`
- 显示：`drawRgbGrid(...)`（realtime 与 detection 两个面板）
- 采样：`captureSyncedColorFromPreview(...)` + `updateStreaming(...)` 的 2 秒门限分支

## 5. 验证标准

- 启动 chessboard 后，log 显示两路实际协商 profile（含 fps）。
- 实时 panel 按 UI 每帧刷新，不再仅在 2 秒采样时跳变。
- 2 秒节拍只影响“sample accepted/rejected”日志与样本计数，不影响画面连续性。
- 实现中不新增自定义拉帧逻辑，不使用 `waitForFrameset`。
