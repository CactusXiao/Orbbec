# 关节真值标注工具（label）

## 依赖
- Python 3
- `Pillow`（用于读取 JPG/PNG 等图像）

Ubuntu 如遇中文字体显示异常（方块/乱码），安装字体包：

```bash
sudo apt-get update && sudo apt-get install -y fonts-noto-cjk
```

## 启动
在仓库根目录执行：

```bash
python3 -m src.label.main
```

或直接运行：

```bash
python3 src/label/main.py
```

## 数据目录约定
在首页输入：
- 数据所在目录：采集保存根目录（例如 `/dest`）
- 受试者：例如 `s01`

程序会组合得到会话目录：`/dest/s01`。

会话目录下：
- 若不存在 `record.csv` 会自动创建，用于记录每个任务已标注到的帧号（flag）。
- 标注结果写入 `labels.json`，按 `任务名 -> 机位号 -> 帧号 -> 点列表` 的层级组织。

任务目录默认从会话目录下的子目录发现（会自动忽略 `record.csv` / `labels.json` / `camera_params.json` / `extrinsics.json` 等文件）。
