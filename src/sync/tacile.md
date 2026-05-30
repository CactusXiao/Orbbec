# Tactile Module Interface

`src/tacile.hpp` / `src/tacile.cpp` 提供了一套和 `fisheyes` 同风格的触觉模块接口。目标是先把串口采集、标定、抽样缓冲、落盘索引这些职责拆出来，后续 `collection` 只需要决定“是否启用触觉”“何时取样”“何时保存”“如何按时间戳软对齐”。

## 设计目标

- 将触觉手套采集实现抽象成可插拔接口 `ITactileModule`
- 默认提供一个 POSIX 串口实现 `createPosixSerialTactileModule()`
- 兼容 `tactile/phlex_reader.py` 的核心协议：
  - 波特率 `115200`
  - 请求命令 `A\r\n`
  - 返回 `48` 通道、每通道 `uint16 little-endian`
  - 标定公式 `f(x) = b * c * x / (c - a * x)`
- 输出统一的 `TactileSample`，便于 `collection` 后续接收和保存
- 保持与 `fisheyes` 相同的 `timestamps.csv` 时间戳格式和语义，便于软对齐

## 核心类型

### 1. 配置

`TactileModuleConfig`

- `enabled`: 是否启用触觉模块
- `targetFps`: `TactileRecorder` 的抽样频率
- `maxBufferedSamples`: 内存缓冲的最大样本数
- `applyCalibration`: 是否应用标定
- `calibrationPath`: 标定文件路径
- `serial`: 串口参数
- `save`: 保存参数

`TactileSerialConfig`

- `portPath`: 串口路径，例如 `/dev/ttyUSB0`
- `baudRate`: 默认 `115200`
- `timeoutMs`: 单帧读取超时
- `requestCommand`: 默认 `A\r\n`
- `clearInputBufferBeforeRequest`: 每次请求前是否清空输入缓冲

`TactileSaveOptions`

- `csvFloatPrecision`: 触觉样本 CSV 中浮点值的保存精度
- `sampleDirectoryName`: 每一帧触觉样本 CSV 的子目录名，默认 `samples`

### 2. 运行时数据

`TactileFrame`

- `captureTimestampUs`
- `captureTimestampSec`
- `rawAdc`: 原始 48 通道 ADC 值
- `calibratedValues`: 标定后的 48 通道值
- `outputValues`: 当前提供给上层使用的值；当前实现等于 `calibratedValues`

`TactileSample`

- `sequence`
- `representativeTimestampUs`
- `representativeTimestampSec`
- `frame`

说明：

- 触觉模块每次只产出一帧，因此 `representativeTimestampUs` 与 `frame.captureTimestampUs` 相同
- 时间戳使用 `system_clock` 的 Unix epoch 微秒，和 `fisheyes` 完全一致
- 文件名同样使用 `seconds.microseconds` 格式

### 3. 保存后索引

`TactileDatasetIndex`

- `samples`

`TactileSavedSample`

- `sequence`
- `representativeTimestampUs`
- `representativeTimestampSec`
- `relativePath`

这个索引与磁盘上的 `timestamps.csv` 一一对应，后续软对齐可直接用 `findNearestTactileSample()`。

## 主要接口

### 1. 可插拔设备接口

`ITactileModule`

- `start(config, error)`
- `stop()`
- `isRunning()`
- `config()`
- `waitUntilReady(timeout)`
- `snapshotLatest(error)`
- `pluginId()`

接口约定：

- `start()` 只负责打开串口、加载标定并启动后台采集线程
- `snapshotLatest()` 返回当前最新触觉样本，不负责限速，也不写入缓冲
- 如果后续换成厂商 SDK、共享内存、网络转发，只需要重新实现 `ITactileModule`

### 2. Recorder 封装

`TactileRecorder`

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

- `TactileRecorder` 在 `ITactileModule` 之上增加“按 `targetFps` 抽样”“录制缓冲”“保存索引”的能力
- `snapshotLatest()` 适合预览，不写缓冲
- `captureNext()` 适合嵌入 `collection` 的录制循环
- `saveBufferedSession()` 会生成：
  - `<saveRoot>/samples/*.csv`
  - `<saveRoot>/timestamps.csv`

## 保存契约

### 1. `timestamps.csv`

格式如下：

```csv
frame_id,timestamp_s,tactile_file
0,1714012345.123456,samples/1714012345.123456.csv
1,1714012345.140123,samples/1714012345.140123.csv
```

约定：

- `timestamp_s` 的字符串格式和 `fisheyes` 完全一致，都是 `seconds.microseconds`
- `timestamp_s` 对应 `TactileSample::representativeTimestampSec`
- `tactile_file` 是相对于模块保存根目录的路径

### 2. 单帧触觉样本 CSV

每个样本文件包含以下列：

```csv
channel_index,region_id,region_name,point_id,raw_adc,calibrated_value,output_value
```

说明：

- `channel_index`: `0..47`
- `region_id`: `1..6`，依次对应 `Thumb / Index / Middle / Ring / Pinky / Palm`
- `point_id`: `1..8`
- `raw_adc`: 原始 ADC 值
- `calibrated_value`: 按标定文件公式换算后的值
- `output_value`: 当前上层消费值；当前实现与 `calibrated_value` 相同

## 标定映射说明

当前模块按照 `tactile/coeff#6x8#1-R-2.txt` 的结构解析标定参数：

- 第一列：区域编号 `1..6`
- 第二列：区域内点位编号 `1..8`
- 第三到五列：`a / b / c`
- 第六列：`rsquare`
- 第七列：有效计数

通道映射规则：

- 通道 `0..7` -> `Thumb`
- 通道 `8..15` -> `Index`
- 通道 `16..23` -> `Middle`
- 通道 `24..31` -> `Ring`
- 通道 `32..39` -> `Pinky`
- 通道 `40..47` -> `Palm`

对应点位按每组内 `1..8` 顺序映射。

## collection 后续接入建议

本次没有修改 `collection.cpp`。后续建议按 `fisheyes` 一样接入：

1. 在 collection 配置层新增 `tactile.enabled` 和 `TactileModuleConfig`
2. 当触觉开启时，在进入采集阶段前启动 `TactileRecorder`
3. 正式开始录制时调用 `clearBuffered()`
4. 录制循环中按 collection 自己的节奏调用 `captureNext()`
5. 停止录制后调用 `saveBufferedSession(episodeDir / "tactile", ...)`
6. 后处理阶段基于 `findNearestTactileSample()` 对齐 Orbbec / fisheye / tactile 时间戳

这种方式的好处是：

- `collection` 不需要知道串口协议细节
- `collection` 只依赖稳定的 `TactileSample` / `TactileDatasetIndex`
- 后续更换底层实现时，不需要重写上层保存和软对齐逻辑

## 使用示例

### 示例 1：独立录制一段触觉数据

```cpp
#include "tacile.hpp"

#include <chrono>
#include <iostream>

int main() {
    sync_app::TactileModuleConfig cfg;
    cfg.enabled = true;
    cfg.targetFps = 60;
    cfg.serial.portPath = "/dev/ttyUSB0";
    cfg.calibrationPath = "tactile/coeff#6x8#1-R-2.txt";
    cfg.applyCalibration = true;

    std::string err;
    sync_app::TactileRecorder recorder;
    if(!recorder.start(cfg, &err)) {
        std::cerr << err << std::endl;
        return 1;
    }
    if(!recorder.waitUntilReady(std::chrono::seconds(2))) {
        std::cerr << "tactile sensor not ready" << std::endl;
        return 1;
    }

    if(!recorder.captureFor(std::chrono::seconds(5), nullptr, &err)) {
        std::cerr << err << std::endl;
        return 1;
    }

    sync_app::TactileDatasetIndex index;
    if(!recorder.saveBufferedSession("output/tactile_run", &index, &err)) {
        std::cerr << err << std::endl;
        return 1;
    }

    recorder.stop();
    return 0;
}
```

### 示例 2：后续给 collection 用的接法

```cpp
sync_app::TactileRecorder tactile;
std::string err;

if(tactileConfig.enabled) {
    if(!tactile.start(tactileConfig, &err)) {
        throw std::runtime_error(err);
    }
    tactile.waitUntilReady(std::chrono::seconds(2));
}

// beginRecord
tactile.clearBuffered();

// recording loop
auto sample = tactile.captureNext(&err);
if(sample) {
    const uint64_t tsUs = sample->representativeTimestampUs;
    // tsUs 可作为后续软对齐基准之一
}

// save stage
sync_app::TactileDatasetIndex index;
tactile.saveBufferedSession(episodeDir / "tactile", &index, &err);

auto match = sync_app::findNearestTactileSample(index, orbbecTimestampUs);
if(match) {
    // match->sample->relativePath 即最近邻触觉样本文件
}
```

### 示例 3：列出可用串口

```cpp
for(const auto &port : sync_app::listAvailableTactileSerialPorts()) {
    std::cout << "device=" << port.devicePath;
    if(!port.stablePath.empty()) {
        std::cout << " stable=" << port.stablePath;
    }
    std::cout << std::endl;
}
```

## 当前实现边界

- 默认串口实现依赖 POSIX `termios`，适用于 Linux / macOS
- 当前没有把“自动清零/基线扣除”做成模块行为，先保留最稳定的原始值 + 标定值链路
- 本次只完成模块代码、构建配置和接口文档，没有修改现有 `collection` 采集逻辑
