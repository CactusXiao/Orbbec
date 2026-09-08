# 单相机外参修复与七相机标定

在 `calibration` 的 `chessboard` 页签，点击 `mode: All 7 cameras` 切换为 `mode: Single camera`。

1. `Moved camera (target)` 选择被碰动的相机，支持 00–06。
2. `Unmoved camera (reference)` 选择一个未移动、已有可靠外参、且能与目标相机同时看到棋盘格的相机。参考相机不能与目标相同。
3. 设置 `valid samples per pair`（至少 3，默认沿用配置），点击 `start auto`。使用现有 11×8 内角点棋盘格，方格尺寸沿用 `calibration.chessboard.squareSize`；改变棋盘格姿态，等待有效样本采集完成。
4. 程序仅采集这一对相机，完成后自动更新 `initExtrinsicPath` 中目标相机的 RGB 外参。其他相机和附加字段保留；目标原有 RGB/depth 工厂参数保留，缺失时从设备补充。
5. 原文件备份为 `<initExtrinsicPath>.before_single_camera.bak`（下一次成功进入保存流程会覆盖此备份）。若外参文件在采样期间被其他操作修改，本次保存会被拒绝，需重新开始。

仅目标和参考相机需要在线并出现在 `devices` 配置中，无需其他五台参与采样。修改模式或相机选择会清除当前采样；`pause` / `start auto` 可保留样本继续，`restart calibration` 清空样本并重新开始。

世界坐标系沿用原文件：`T_world_to_target = T_reference_to_target × T_world_to_reference`。即使移动的是 00，也可用其他未移动相机修复 00，不会把 00 重新设为单位变换，也不会移动其他相机。参考外参必须正确；如果整组外参尚未建立，请先做全量标定。

全量模式支持 00–06，依次标定 00→01、01→02、02→03、03→04、04→05、05→06 六对，以 00 为世界坐标原点，并更新全组外参。界面进度显示为 `/6`；单相机模式为 `/1`。`ICP (all)` 是全量优化，只能在全量模式使用。

硬件无关回归检查（需 C++17、OpenCV 和 pkg-config）：

```sh
python3 -m unittest tests.test_calibration_repair -v
```

这些检查验证变换方向、00/06 修复、其他相机及元数据保留、无效输入拒绝、备份与外部修改保护。真实标定精度仍需现场用棋盘格和点云预览验证。
