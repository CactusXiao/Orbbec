# 触觉手套采集与区域力（N）

协议和位置映射以 `tactile/【矩侨精密】 织物电子皮肤（触觉手套）产.pdf`（v1.0，第 11–15 页）为准。高密度 1024 ADC 产品的协议不用于此手套。

## 通道和单位

- 串口 921600 bps；`AA 55 03 99` 后为包序号和传感器类型。第 1 包 128 个 ADC，第 2 包 128 个 ADC 加 16 字节四元数数据（w、x、y、z）。传感器类型 1 为左手、2 为右手；配置左右手与类型不一致时拒绝启动。
- 256 个位置是传输槽位，不能视为 256 个压力点。PDF 的明确映射为每手 132 个压力点、5 个弯曲通道；其余 119 个位置未定义，仍完整保存原始 ADC。PDF 标称 162 点与表格不一致，不能凭空补点。
- 左右手有各自的编号映射。PDF 编号为 1-based，串口数组下标为编号减一。代码 `src/sync/tactile_layout.hpp` 和后端 `task_backend/tactile_layout.py` 通过测试核对一致性。
- 弯曲通道只保存 ADC，没有角度或力值换算依据。按拇指、食指、中指、无名指、小指顺序，右手为 47、44、41、38、35；左手为 210、213、216、219、222。
- 配置默认 `targetFps=100` 对应说明书标称速率，仅为元数据，不是向设备下发的速率设置。采集按设备实际输出接收，实际频率应从时间戳统计；PDF 允许有线定制至 600 Hz。

## 图片标定的适用范围

保留 `tactile/微信图片_2026-09-07_165536_559.jpg` 指定的 Hill 参数：

```text
S = 右手中指 12 个通道 ADC 之和
F(N) = 18.2084 × (S / (1257.0210 - S))^(1 / 1.1685)
```

三份 CSV 的传感器编号都是 8、9、10、24、25、26、232、233、234、248、249、250，按 PDF 对应右手中指。该式输出此区域总力，不能逐点使用，也不能把全部 256 个 ADC 求和。左手相同编号会混合中指与无名指，因此左手不应用该标定。仅修改成左手中指编号也不足以证明同一组系数有效。

目前没有独立逐点标定，不计算或保存 ADC 占比分摊的逐点力。左手和其他区域保留原始 ADC，区域力没有有效标定时为 `nan`；该状态不会使左手停止采集。

零输入输出 0 N；`0 < S < 1257.0210` 按原式计算，不截断高值。超过 CSV 实测 ADC 上限时设置 `force_out_of_range=1`，仍可得到外推值；`S >= 1257.0210` 时区域力为 `nan` 且标记越界。公式的高值不代表厂家验证了对应精度。

默认读取三份 CSV，`touch.calibrationPaths` 可只配置其中任意一份；路径相对于 `src/sync/config.json`。CSV 用于校验区域编号和实测 ADC 范围，不重新拟合图片系数。换成其他通道集合、文件损坏或缺失会使启动失败，防止错误复用此曲线。

## 新数据格式（v3）

`touch/left_raw.csv`、`touch/right_raw.csv` 保留采样索引、时间戳、IMU 与质量字段，测量字段为：

| 字段 | 含义 |
| --- | --- |
| `calibrated_region_force_n` | 右手中指区域总力，单位 N；左手为 `nan` |
| `force_calibration_status` | `right_middle_region`、`invalid_force` 或 `uncalibrated_hand` |
| `force_out_of_range` | 是否超出实测 ADC 范围或公式无有效结果 |
| `raw_adc_000` … `raw_adc_255` | 全部串口槽位的原始 ADC |

不再写出 `force_000_n..force_255_n`，也不保留旧逐点估计字段。`touch_manifest.json` 使用 `orbbec.touch.jq_shroom.v3`，记录协议来源、槽位与压力/弯曲点数、公式及适用区域、每手类型和状态定义。

独立 `TactileRecorder::saveSamples` 单帧 CSV 使用相同 PDF 解剖映射，保存 `sensor_id`、`channel_kind`（pressure/bend/unmapped）、部位、部位内编号、原始 ADC、区域总力及状态。原来按每 8 个连续通道划分部位的逻辑已删除。

不兼容 v1/v2 旧数据；需重新编译采集程序并重新采集。查看器只接收 v3 清单和完整的新 CSV 字段，不转换旧 episode。

## 验证

`python3 -m unittest tests.test_tactile_force tests.test_tactile_viewer tests.test_viewer_service -v`

测试实际编译 C++ 标定与保存代码，验证原始 CSV、Hill 公式、区域限定、左右手隔离、零输入与越界、保存单位、完整 ADC，以及采集/后端的 512 个通道分类一致性。硬件上的串口输出和实际接触位置仍需现场验收。
