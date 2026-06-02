# Orbbec 采集对齐机制说明

本文档说明本仓库采集 Orbbec 数据时，RGB、Depth、IR、多相机以及鱼眼图像在时序上如何对齐，并补充同一台 Orbbec 相机内 RGB 与深度图的空间对齐方式。

主要实现文件：

- `src/sync/collection.cpp`
- `src/sync/shared_utils.cpp`
- `src/sync/shared_utils.hpp`

## 1. 总体结论

采集时的对齐分为两类：

1. 时序对齐：按时间戳把不同相机、不同模态的数据配成同一个 `frame_index`。
2. 空间对齐：保存深度图或生成彩色点云时，把 Depth 投影到 RGB 坐标系。

时序对齐不是简单按回调次数或 frame index 对齐，而是以一个参考时间戳 `centerUs` 为中心，对每一路数据找最近时间戳帧。只有时间差在容差窗口内，才组成一个完整输出帧。

## 2. SDK/硬件层同步

启动每台 Orbbec 设备时，如果启用了多个 sensor，代码会设置：

```cpp
setConfigFrameAggregateOutputMode(config, OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
enablePipelineFrameSync(rt.pipe);
```

位置：

- `src/sync/collection.cpp:2169`
- `src/sync/shared_utils.cpp:790`

含义：

- `OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE` 要求多个流聚合成同一个 `FrameSet` 后再输出。
- `pipe->enableFrameSync()` 让 SDK 尽量同步同一设备内的多路流，例如 RGB 和 Depth。

多设备采集时，如果 `cfg.enableSync` 开启，还会调用：

```cpp
enableDeviceClockSync(ctx_, 60000);
```

位置：

- `src/sync/collection.cpp:2188`

这会让多台设备做时钟同步，采集时优先使用 `globalTimeStampUs()`。如果不是多设备强同步场景，才允许退回本地 `timeStampUs()`：

```cpp
ts = frame->globalTimeStampUs();
if(ts == 0 && !requireGlobalTimestamp) {
    ts = frame->timeStampUs();
}
```

位置：

- `src/sync/shared_utils.cpp:851`

## 3. callback 阶段：只取时间戳和入队

Orbbec pipeline callback 收到 `FrameSet` 后，会按数据类型取出 frame：

- RGB: `frameSetColorFrame(frameSet)`
- Depth: `frameSetDepthFrame(frameSet)`
- IR: `frameSetFrame(frameSet, ...)`

然后读取时间戳：

```cpp
const uint64_t ts = getFrameTimestampUs(frame);
```

位置：

- `src/sync/collection.cpp:4063`

如果 `collectFps` 小于实际 stream FPS，代码会按 `collectFps` 节流，避免保存过密帧：

```cpp
if(ts > last && (ts - last) < interval) {
    continue;
}
```

位置：

- `src/sync/collection.cpp:4085`

随后把数据放入 `RecordTask` 队列：

```cpp
enqueueRecordTask(RecordTask{ deviceSn, t, ts, nextRecordSeq(deviceSn, t), std::move(detached) });
```

位置：

- `src/sync/collection.cpp:4112`

这里 `seq` 是每个相机、每种数据流独立递增的序号。它的作用不是做最终对齐，而是保证后台多线程解码后仍能按原采集顺序提交。

## 4. 后台解码和按 seq 保序

`recordWorkerLoop()` 会从 `RecordTask` 队列取任务，转换为 OpenCV 图像或保留 MJPG 编码数据，然后提交给 coordinator：

```cpp
enqueueProcessedRecord(ProcessedRecord{ ... });
```

位置：

- `src/sync/collection.cpp:3047`
- `src/sync/collection.cpp:3098`

由于解码是多线程的，不同帧可能乱序完成。coordinator 收到 `ProcessedRecord` 后，不是直接加入可配对队列，而是先放入 `bySeq`：

```cpp
state.bySeq[item.seq] = StreamPacket{ ... };
drainReadyPacketsLocked(item.sn, item.type, state);
```

位置：

- `src/sync/collection.cpp:3904`

`drainReadyPacketsLocked()` 只会从 `nextSeq` 开始连续提交：

```cpp
auto it = state.bySeq.find(state.nextSeq);
state.committed.push_back(std::move(packet));
state.nextSeq++;
```

位置：

- `src/sync/collection.cpp:2730`

因此最终参与时间配对的 `committed` 队列，是每个 `(camera, modality)` 内部按采集顺序排列的。

## 5. episode 开始时设置对齐参数

每次开始一个 episode，会根据 `collectFps` 计算时间步长：

```cpp
stepUs = 1000000.0 / collectFps;
```

位置：

- `src/sync/collection.cpp:2806`

然后计算最近邻匹配容差：

```cpp
halfWinUs = stepUs / 2;
tolUs = max(2000, stepUs / 10);
maxAbsDiffUs = halfWinUs + tolUs;
```

位置：

- `src/sync/collection.cpp:2814`

例如 `collectFps = 30` 时：

- `stepUs` 约为 33333 us
- `halfWinUs` 约为 16666 us
- `tolUs` 约为 3333 us
- `maxAbsDiffUs` 约为 19999 us，也就是约 20 ms

多视角采集时，参考相机是按 `camKey` 排序后的第一台相机：

```cpp
local.refSn = local.deviceSns.front();
```

位置：

- `src/sync/collection.cpp:2294`

参考模态是 UI 中选择的 `refType_`，默认是 RGB：

```cpp
CollectDataType refType_ = CollectDataType::RGB;
```

位置：

- `src/sync/collection.cpp:4306`

## 6. 最近邻时间戳配对

coordinator 的主循环会不断尝试生成完整 slot：

```cpp
progress = tryFinalizeOneMultiviewSlotLocked() || progress;
```

位置：

- `src/sync/collection.cpp:3945`

对齐中心 `centerUs` 来自参考相机、参考模态队列的第一帧：

```cpp
const uint64_t centerUs = itRefState->second.committed.front().tsUs;
```

位置：

- `src/sync/collection.cpp:3448`

然后对每台相机、每种需要对齐的模态，从对应 `committed` 队列中找离 `centerUs` 最近的帧：

```cpp
pickNearestPacket(itStream->second.committed, centerUs, session_.maxAbsDiffUs, picked)
```

位置：

- `src/sync/collection.cpp:3470`

`pickNearestPacket()` 内部使用 `lower_bound` 找候选帧，只比较中心时间戳前后两个候选，然后选绝对时间差更小的一个：

```cpp
auto it = std::lower_bound(items.begin(), items.end(), centerUs, ...);
chosen = (d1 <= d0) ? cand1 : cand0;
```

如果最近帧和 `centerUs` 的时间差超过 `maxAbsDiffUs`，则匹配失败：

```cpp
if(absDiff(items[chosen].tsUs, centerUs) > maxAbsDiffUs) {
    return false;
}
```

位置：

- `src/sync/collection.cpp:3270`

因此时序对齐规则可以概括为：

```text
centerUs = 参考相机参考模态的当前帧时间戳

对每一路 camera/modality:
    找 timestamp 最接近 centerUs 的帧
    如果 abs(timestamp - centerUs) <= maxAbsDiffUs:
        该帧加入当前 frame_index
    否则:
        当前 slot 缺帧，记为 missing
```

## 7. 何时 finalize 一个 slot

为了避免过早配对，代码会先判断每一路数据是否已经收到了足够晚的帧：

```cpp
if(!itStream->second.eos && itStream->second.maxTsUs < centerUs + session_.maxAbsDiffUs) {
    return false;
}
```

位置：

- `src/sync/collection.cpp:3292`

意思是：如果某一路数据还没有收到超过匹配窗口右边界的帧，就继续等。这样可以避免刚好有延迟帧还没到时就误判缺帧。

如果所有需要的数据都满足条件，才会真正尝试配对。

## 8. 成功和缺帧处理

如果所有相机、所有模态都能找到符合时间窗口的帧，则：

- `fullAligned++`
- 分配新的 `frameIndex`
- 同一个 slot 的 RGB、Depth、IR 等数据保存为同一个帧号
- 写入 `timestamps.csv`

位置：

- `src/sync/collection.cpp:3513`
- `src/sync/collection.cpp:3521`
- `src/sync/collection.cpp:3526`

如果任意一路匹配失败，则：

- `missingAligned++`
- 当前参考帧被丢弃
- 不生成完整输出帧

位置：

- `src/sync/collection.cpp:3494`

生成成功后，会清理已经不可能再被后续 slot 使用的旧帧：

```cpp
while(items.size() > 1 && items[1].tsUs <= centerUs) {
    items.pop_front();
}
```

位置：

- `src/sync/collection.cpp:3332`

## 9. timestamps.csv 记录内容

每个成功输出的 slot 会写一行 `timestamps.csv`：

- `frame_index`
- `ref_timestamp_us`
- 每台相机 RGB timestamp
- 每台相机 Depth timestamp
- fisheye timestamp
- `rgbd_max_diff_ms`
- `all_modalities_max_diff_ms`

位置：

- `src/sync/collection.cpp:2820`
- `src/sync/collection.cpp:3526`

其中 `rgbd_max_diff_ms` 和 `all_modalities_max_diff_ms` 用来检查实际对齐误差。

## 10. 鱼眼相机的时序对齐

如果启用了 fisheye，coordinator 会把鱼眼采集结果放入 `session_.fisheyeSets`：

```cpp
session_.fisheyeSets.push_back(std::move(coordFisheyeQueue_.front()));
```

位置：

- `src/sync/collection.cpp:3938`

对齐时，鱼眼图像也按 `centerUs` 找最近的 fisheye frame set：

```cpp
pickNearestFisheyeIndexLocked(centerUs)
```

位置：

- `src/sync/collection.cpp:3325`

如果 fisheye 还没有收到不早于 `centerUs` 的样本，coordinator 会继续等待：

```cpp
if(!session_.fisheyeEos && session_.fisheyeSets.back().representativeTimestampUs < centerUs) {
    return false;
}
```

位置：

- `src/sync/collection.cpp:3314`

## 11. 同机 RGB-depth 空间对齐补充

时序对齐完成后，同一 `frame_index` 下的 RGB 和 Depth 只是时间上对应。Depth 保存时还会进一步做空间对齐。

启动 stream 后，代码通过 RGB/depth 分辨率获取相机参数：

```cpp
getRgbDepthCameraParam(rt.pipe, colorW, colorH, depthW, depthH, cameraParam)
```

位置：

- `src/sync/collection.cpp:2140`

保存 Depth PNG 时，如果参数有效，会调用：

```cpp
cv::Mat alignedDepth = alignDepthToRgb(frame, valueScale, rgbDepthParam, rgbW, rgbH);
```

位置：

- `src/sync/collection.cpp:3601`

`alignDepthToRgb()` 会遍历 depth 像素，将 depth 像素通过 Orbbec SDK 坐标变换投影到 RGB 像素坐标：

```cpp
transform2dTo2d(src, depthMm, depthIntrinsic, depthDistortion, rgbIntrinsic, rgbDistortion, d2cExtrinsic, dst)
```

位置：

- `src/sync/collection.cpp:1639`
- `src/sync/shared_utils.cpp:1220`

输出的 Depth PNG 尺寸是 RGB 图像尺寸，坐标系也是 RGB 图像坐标系。若参数无效，则退回保存原始 depth。

## 12. 一句话总结

本项目采集时，先依赖 Orbbec SDK 和设备时钟做基础同步，再在软件层用参考相机参考模态的时间戳作为中心，对每一路数据做最近邻时间戳匹配；匹配成功的帧被写成同一个 `frame_index`，并在 `timestamps.csv` 中记录真实时间戳和最大误差。Depth 图像在保存阶段还会按 RGB/depth 标定参数投影到 RGB 坐标系。
