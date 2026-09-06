# Label / QC 界面更新（2026-09-05）

## 实际入口

采集主程序在 `src/sync/shared_utils.cpp` 中分别启动 `python -m label.main` 和 `python -m src.qc.main`。
本次调整对应 `label/app.py`、`src/qc/app.py` 和共用的 `label/theme.py`，不涉及旧的 `src/label/`。

## 字体与布局

- ego 的 X11 桌面为 1920 × 1080，原始 Tk scaling 为 0.91655，11 pt 正文约为 10 像素。
- 非 macOS 环境将过低的 Tk scaling 提升到 96 DPI（1.33333），保留更高的系统缩放值。
- 简体中文优先使用 Noto Sans CJK SC；正文 13 pt，分区标题 15 pt，加粗页标题 22 pt。
- 表格行高随字体实际行高变化，ego 为 42 像素；统一深蓝灰背景、亮色正文、蓝色主按钮、红色异常按钮和键盘焦点提示。
- Label 首页使用并排任务列表，保留本地 JSONL 入口（点击后展开）。标注页收窄进度侧栏，操作按钮自动换行。
- QC 列表增加滚动条，播放与坏帧操作区自动换行，五视图保留全部原图显示逻辑。
- 默认窗口 1480 × 920，并根据屏幕尺寸限制初始大小；支持 1040 × 680 的较小窗口。

## 验证

在 ego 的临时目录中，以实际 Tk / 字体环境运行：

- 53 项原有 Label 交互、Label 客户端、QC 裁剪、QC Worker 测试通过。
- 新增 `tests/test_frontend_ui_layout.py` 的 2 项界面测试通过：检查标题层级、行高、不同宽度下操作按钮完整可见、无重叠、QC 模式切换后视图空间。
- 检查首页、标注页、QC 任务页、五视图和坏帧模式的真实窗口截图。
- 界面预览使用示例行，不租用或提交真实任务；未进行真实采集到提交的完整硬件流程测试。

有桌面显示时可运行：

```bash
python3 -m unittest tests.test_frontend_ui_layout -v
```

无 Tk 显示环境时，界面测试会明确跳过。

## ego 应用

实际目录：`/home/ubuntu/demo/Orbbec_demo`。应用的是仅含 UI 改动的补丁，保留该主机已有的标注业务差异。
备份目录：`/home/ubuntu/demo/orbbec-ui-backup-20260905`。

重新打开 Label / QC 即可生效，无需重新编译或重启 C++ 采集进程。
