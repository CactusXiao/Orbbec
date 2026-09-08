# 七相机显示

- **QC**：保留 3×2 布局，显示 00、02、03、05、06、ego。06 与其他视角一起解码、渲染并参与完整帧缓冲检查。
- **Label 单视角**：00–06 均可查看、纠偏和保存；快捷键 1–7 对应七个机位，0 打开总览。已有六机位任务若 episode 下存在 `06/RGB`，会自动补充 06。
- **Label 总览**：显示 00、02、03、05、06、ego，隐藏 01 和 04；这两路仍可在单视角中编辑。总览全部只读，支持各格独立缩放和平移。
- **Label ego**：仅在总览显示同步 RGB，不进入单视角、标注状态、可见视角计数或保存结果。根据 `frame_index → ego_frame_index` 对齐，可复用同一 ego 帧；后台按需解码。没有 ego 数据或当前帧缺少同步映射时显示占位，不影响其他相机编辑。切换 MANO/骨架预览时 ego 保持 RGB。
- **后端 Episode Viewer**：RGB 与 MANO mesh 均显示 00–06。桌面首屏保持原有 3×2 六格尺寸，向下滚动显示 06；窄窗口保持原有 2×3 首屏布局。第七路也参与解码、渲染和同步预加载。融合点云同时纳入可用的 06。

旧六相机 episode 仍可查看，不会为不存在的 06 创建标注。MANO 渲染仍要求相应机位具有有效内外参。

验证：

```sh
python3 -m unittest tests.test_label_overview tests.test_label_ego_preview tests.test_label_view_modes tests.test_qc_worker_smoke tests.test_qc_playback_completion tests.test_viewer_service
```

Label/QC 界面测试需要 Tk 显示环境；ego 和 Viewer 数据测试无需真实相机。
