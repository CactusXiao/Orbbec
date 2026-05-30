# Fisheye Module Interface

`src/fisheyes.hpp` / `src/fisheyes.cpp` 提供了一个独立于 `collection.cpp` 的鱼眼相机模块。它的目标不是直接把逻辑写死在 collection 中，而是先定义一个稳定的数据契约，让 collection 后续只需要“选择是否启用模块、拉取样本、保存索引、做软对齐”。

## 设计目标

- 将鱼眼相机实现抽象成可插拔接口 `IFisheyeModule`
- 默认提供一个 `OpenCV VideoCapture` 实现，对应当前 Python 版 `fisheye/capture.py`
- 输出统一的 `FisheyeFrameSet` 结构，供 collection 后续接收和保存
- 保留 `timestamps.csv` 数据契约，便于和 30 Hz Orbbec 数据做最近邻软同步

## 核心类型

### 1. 配置

`FisheyeModuleConfig`

- `enabled`: 是否启用鱼眼模块
- `targetFps`: 采样频率，默认 `60`
- `maxBufferedSets`: 内存缓冲的最大样本数
- `cameras`: 鱼眼相机列表，默认两路
- `save`: 图片保存参数

`FisheyeCameraConfig`

- `cameraId`: 逻辑名称，例如 `cam0`、`cam1`
- `handRole`: 语义角色，例如 `left`、`right`
- `uniqueId`: 必填，用于绑定物理设备；推荐填 `v4l:/dev/v4l/by-id/...-video-index0` 或 `...-video-index1`
- 当前实现会先按 `uniqueId` 命中配置节点；如果该节点能打开但起流失败，会自动回退到同一物理相机的 sibling `video-index*` 节点
- `devicePath` / `preferredDeviceHint` / `deviceIndex`: 保留字段，当前检索逻辑不再使用
- `width` / `height` / `fps`: 采集配置
- `preferMjpeg`: 默认开启，优先请求 `MJPG` 以降低 USB 带宽压力

### 2. 运行时数据

`FisheyeFrame`

- `cameraId`
- `deviceIndex`
- `captureTimestampUs`
- `captureTimestampSec`
- `bgr`

`FisheyeFrameSet`

- `sequence`: 样本序号
- `representativeTimestampUs`
- `representativeTimestampSec`
- `frames`

说明：

- 一个 `FisheyeFrameSet` 表示一次“对 collection 可消费的鱼眼 RGB 样本”
- 默认实现里，每次样本由各个相机“当前最新帧”组成
- `representativeTimestampUs` 是各路鱼眼时间戳的均值，这与现有 Python 版的行为一致

### 3. 保存后索引

`FisheyeDatasetIndex`

- `cameraOrder`
- `samples`

`FisheyeSavedSample`

- `sequence`
- `representativeTimestampUs`
- `representativeTimestampSec`
- `relativePaths`

这个索引与磁盘上的 `timestamps.csv` 一一对应，后续软对齐直接使用它即可。

## 主要接口

### 1. 可插拔模块接口

`IFisheyeModule`

- `start(config, error)`
- `stop()`
- `isRunning()`
- `waitUntilReady(timeout)`
- `snapshotLatest(error)`
- `pluginId()`

接口约定：

- `start()` 只负责打开设备并启动后台读流线程
- `snapshotLatest()` 返回当前可用的一组鱼眼 RGB 样本，不负责限速
- 设备选择先按 `FisheyeCameraConfig::uniqueId` 绑定物理相机，再在同一物理设备的 `video-index*` 节点中优先尝试配置节点，必要时自动回退到 sibling 节点
- 未来如果要改成 GStreamer、厂商 SDK 或共享内存实现，只要继续实现 `IFisheyeModule` 即可，不需要改 collection 的保存逻辑

### 2. Recorder 封装

`FisheyeRecorder`

- `start(config, error)`
- `waitUntilReady(timeout)`
- `snapshotLatest(error)`
- `captureNext(error)`
- `captureFor(duration, cancel, error)`
- `bufferedSamplesCopy()`
- `takeBufferedSamples()`
- `clearBuffered()`
- `saveBufferedSession(saveRoot, indexOut, error)`
- `loadDatasetIndexCsv(csvPath, indexOut, error)`

接口约定：

- `FisheyeRecorder` 在 `IFisheyeModule` 之上增加“按 `targetFps` 抽样”和“会话保存”能力
- `snapshotLatest()` 适合预览场景，只取当前最新帧，不写入缓冲
- `captureNext()` 的行为与 Python 主循环一致：按固定节奏抽取当前最新帧组合
- `saveBufferedSession()` 会生成：
  - `<saveRoot>/cam0/*.jpg|png`
  - `<saveRoot>/cam1/*.jpg|png`
  - `<saveRoot>/timestamps.csv`

## 与 Python 版本的对应关系

当前 C++ 模块复现了 `fisheye/capture.py` 的关键行为：

- 两路相机常驻后台线程读流
- 主采样逻辑按固定 `60 Hz` 节奏取最新帧
- 代表时间戳取两路时间戳均值
- 以 `Unix 秒.微秒` 作为文件名
- 输出 `timestamps.csv`

不同点：

- C++ 版本增加了显式接口层，便于插入 collection
- C++ 版本把“采集设备”与“会话保存”拆成两层，后续更容易嵌入现有采集流程
- 默认实现会优先枚举 Linux 的视频采集节点，并优先使用 `/dev/v4l/by-id` 稳定路径，而不是依赖易变的 `/dev/videoX`
- 默认实现会优先请求 `MJPG`，以减少扩展坞/Hub 场景下的带宽不足导致的绿屏、花屏或卡死

## `timestamps.csv` 契约

写出格式如下：

```csv
frame_id,timestamp_s,cam0_file,cam1_file
0,1714012345.123456,cam0/1714012345.123456.jpg,cam1/1714012345.123456.jpg
1,1714012345.140123,cam0/1714012345.140123.jpg,cam1/1714012345.140123.jpg
```

约定：

- `timestamp_s` 对应 `FisheyeFrameSet::representativeTimestampSec`
- `cam*_file` 与 `cameraOrder` 顺序一致
- 后处理阶段可对任意 Orbbec 时间戳 `t_other_us` 调用 `findNearestFisheyeSample()` 做最近邻匹配

## collection 后续接入建议

不修改当前 `collection.cpp` 的前提下，建议后续按下面方式接入：

1. 在 collection 配置层新增 fisheye 开关和 `FisheyeModuleConfig`
2. 当 fisheye 开启时，在进入 capture 页面时启动 `FisheyeRecorder`
3. 开始录制时调用 `clearBuffered()`
4. 录制期间按 collection 自己的状态机节奏调用 `captureNext()`
5. 停止录制后，和 Orbbec 数据一起保存鱼眼图像与 `timestamps.csv`
6. 后处理阶段基于 `FisheyeDatasetIndex` 与 Orbbec 时间戳做最近邻软对齐

这种接法的好处是：

- collection 不需要知道 OpenCV 读流细节
- collection 只依赖 `FisheyeFrameSet` 和 `FisheyeDatasetIndex`
- 后续更换底层鱼眼实现时，collection 的主逻辑基本不变

## 使用示例

### 示例 1：独立运行鱼眼采集

```cpp
#include "fisheyes.hpp"

#include <chrono>
#include <iostream>

int main() {
    sync_app::FisheyeModuleConfig cfg;
    cfg.enabled = true;
    cfg.targetFps = 60;
    cfg.cameras = {
        { "cam0", "", "", "", "", -1, 1280, 720, 60, true },
        { "cam1", "", "", "", "", -1, 1280, 720, 60, true },
    };

    std::string err;
    sync_app::FisheyeRecorder recorder;
    if(!recorder.start(cfg, &err)) {
        std::cerr << err << std::endl;
        return 1;
    }
    if(!recorder.waitUntilReady(std::chrono::seconds(2))) {
        std::cerr << "fisheye cameras not ready" << std::endl;
        return 1;
    }

    if(!recorder.captureFor(std::chrono::seconds(5), nullptr, &err)) {
        std::cerr << err << std::endl;
        return 1;
    }

    sync_app::FisheyeDatasetIndex index;
    if(!recorder.saveBufferedSession("output/fisheye_run", &index, &err)) {
        std::cerr << err << std::endl;
        return 1;
    }

    recorder.stop();
    return 0;
}
```

### 示例 2：后续给 collection 用的接法

```cpp
sync_app::FisheyeRecorder fisheye;
std::string err;

if(fisheyeConfig.enabled) {
    if(!fisheye.start(fisheyeConfig, &err)) {
        throw std::runtime_error(err);
    }
    fisheye.waitUntilReady(std::chrono::seconds(2));
}

// beginRecord
fisheye.clearBuffered();

// recording loop
auto sample = fisheye.captureNext(&err);
if(sample) {
    const uint64_t tsUs = sample->representativeTimestampUs;
    // 这里可以把 tsUs 作为后续软对齐基准之一
}

// save stage
sync_app::FisheyeDatasetIndex index;
fisheye.saveBufferedSession(episodeDir / "fisheye", &index, &err);

auto match = sync_app::findNearestFisheyeSample(index, orbbecTimestampUs);
if(match) {
    // match->sample->relativePaths 即最近邻鱼眼帧文件
}
```

## 当前实现边界

- 默认实现依赖 OpenCV `VideoCapture`
- 当前只处理 RGB/BGR 图像，不做畸变校正
- `timestamps.csv` 解析使用简单逗号分割，因此默认假设路径本身不包含逗号
- 本次没有修改 `collection.cpp`，只先把接口和实现准备好
