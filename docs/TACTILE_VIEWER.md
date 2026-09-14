# Episode 查看页：Pico + 眼动 + 触觉

触觉侧栏在「Pico + 眼动」模式中与画面同步显示。重启后端并重新打开查看会话后生效。后端组件为 `viewer_service.py`、`tactile_viewer.py`、`tactile_viewer_ui.py`、`tactile_layout.py`。

## 布局与测量含义

位置以《织物电子皮肤（触觉手套）》第 11–13 页为准。左右手均按掌面显示，不用单纯镜像右手编号生成左手编号。

- 五指各 12 个压力点，按 PDF 的 4 行 × 3 列排列。
- 掌心 72 点按 5 行排列：12、15、15、15、15 点。右手首行前留 3 列空位，左手首行后留 3 列空位。
- 五个弯曲通道单独显示 ADC 数值和固定 0–255 范围条，不混入压力热图，不换算成角度或 N。
- 共展示每手 132 个压力点和 5 个弯曲通道。未映射槽位保存在原始 CSV，不虚构物理位置。

默认显示「压力 ADC」，色阶固定 0–255；左右手均可查看压力分布。上方单独显示右手中指标定区域总力 N。切换「区域力 N」时，以绿色边框标出区域，灰色测点表示没有独立逐点力值；不绘制虚假的逐点 N 热图。左手显示尚无力值标定。

总力使用已经保存的 `calibrated_region_force_n`，不在页面计算新的标定。显示越界、质量和同步状态；缺失样本清空热图和弯曲显示，不沿用前一帧。

## 格式与同步

只接受 `orbbec.touch.jq_shroom.v3` 的 `touch_manifest.json` 和完整的新采集 CSV。设备侧别必须与 `sensor_type`（1 左、2 右）一致。拒绝旧格式、缺少清单、旧 `pressure_XXX` 和逐点 `force_XXX_n` 字段；不做旧数据回退或转换。文件路径必须位于 episode 内。

根据根目录 `timestamps.csv` 的 `frame_index` 定位画面，优先使用 `touch_<设备ID>_frame_index` 精确匹配 CSV 的 `sample_index`。显式缺失匹配保持缺失。没有索引字段时可按采集时间戳选最近样本，容差为 `max(50 ms, 2 / fps)`；没有时间依据不猜测对齐。

Pico 准备阶段生成临时逐帧触觉 JSON，图像与触觉一同预取、一同更新。NaN/Infinity 转为 JSON null。该过程不修改采集数据。

## 验证与部署

`python3 -m unittest tests.test_tactile_force tests.test_tactile_viewer tests.test_viewer_service -v`

覆盖 PDF 映射、弯曲隔离、区域总力、索引和时间同步、缺帧、旧格式拒绝、路径约束，以及 Pico 输出流程。部署需要更新并重新编译采集程序，重启后端；旧程序生成的数据不会被新版查看器接收。
