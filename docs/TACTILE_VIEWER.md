# Episode 查看页：Pico + 眼动 + 触觉

在原有「Pico + 眼动」模式中加入触觉侧栏，不增加独立查看模式。重启后端并重新打开 episode 查看会话后生效。部署时需要同时包含 `task_backend/viewer_service.py`、`task_backend/tactile_viewer.py` 和 `task_backend/tactile_viewer_ui.py`。

## 显示

- 左右手切换，默认右手。右手掌面采用 `tactile/record_tactile.py` 的五指 12 点阵列和手掌 72 点布局，排除该参考脚本中的弯折、未使用通道。
- 当前参考资料仅提供右手通道位置，左手使用 #1–256 通道矩阵，不推断或镜像解剖位置。矩阵的原始 ADC 包括弯折及其他通道。
- 「力 N」直接读取已经保存的 `force_XXX_n`；「原始 ADC」读取 `raw_adc_XXX`，兼容旧数据的 `pressure_XXX`。不在查看页重新标定或将旧 ADC 标为 N。
- 显示保存的标定区域总力（不是全手总力）、同步时间差、越界及采样质量信息。未标定、无有效结果以灰色显示，与零值区分；悬停或键盘聚焦测点显示编号和数值。
- N 色阶使用对应手套整段采集的有限力值最大值，ADC 固定为 0–255；播放时不逐帧归一化。

## 数据与同步

读取 episode 下 `*/touch_manifest.json` 中的设备 ID、左右手标识和 `raw_csv`；无 manifest 时兼容 `touch/left_raw.csv`、`touch/right_raw.csv`。路径限制在 episode 内，拒绝目录穿越或指向外部的符号链接。

按根目录 `timestamps.csv` 的 `frame_index` 寻找当前画面对应行，优先用 `touch_<设备ID>_frame_index` 精确匹配原始 CSV 的 `sample_index`（不是 CSV 行号）。显式缺失匹配保持缺失。只有没有此索引字段时才按采集时间戳找最近样本，容差为 `max(50 ms, 2/fps 秒)`。存在参考时间戳时，精确索引匹配也检查时差。

未采集、无法读取或没有同步样本时显示状态，不沿用上一帧热图。没有时间依据时不按播放帧率猜测对齐。

Pico 模式准备时，在现有临时会话目录生成逐帧 `*.tactile.json`。前端将图像与对应触觉数据一起预取、一起提交显示，使用既有切帧令牌避免快速拖动和切换模式时旧请求覆盖新帧。NaN/Infinity 转为 JSON null。此流程仅读取 episode，不改写 NAS 原始数据。

## 验证

```sh
python3 -m unittest tests.test_tactile_viewer tests.test_viewer_service -v
```

覆盖稀疏采样索引、精确匹配与最近邻容差、显式缺帧、旧 ADC、无效力值、自定义设备目录、路径约束、Pico 准备和媒体输出，以及现有查看器行为。浏览器使用合成数据验证热图、左右手切换、N/ADC 切换、进度跳转与画面同步；真实 episode 的设备时间和布局需在采集环境中验收。
