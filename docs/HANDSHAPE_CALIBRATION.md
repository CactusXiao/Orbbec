# 手型标定采集入口

主菜单 `Handshape Calibration` 使用当前登录 ID 作为 subject，直接打开专用采集页。

- 本地：`<collection.savePath>/<subject>/task_handshapeCalibration/episode1`
- NAS：`<taskBackend.nas.mountPath>/<subject>/task_handshapeCalibration/episode1`
- 结果（由新版 publisher 回传）：`<subject>/shape_calibration_result/{shape.npy,scale.npy,pose_mesh.png,pose_2d.png}`

点击菜单按钮创建目录并启动相机预览。预热及外参检查沿用普通采集，Start 开始录制，Stop 等待写盘完成，Calibrate（Ctrl+3）确认并异步上传、调用固定 NAS API：

```bash
ssh synology sudo /usr/local/sbin/nas-uploader-publish --shape-calibration '<subject>/task_handshapeCalibration/episode1'
```

标定不调用普通任务领取/确认接口，不占用普通任务的次数。发布成功后向 `/api/v1/shape-calibration/published` 登记专用的第 3 步进度。

后端仅跟踪至 NAS 状态 `shape_calibrated` 且四个标定文件均已回传。随后第 3 步“自动标注 + Episode 3D”标记完成，跟踪结束，不运行普通 `optimized_pose → mano/episode` 转换，不生成 QC、review、人工标注、人工 3D 或返修任务。详情页仅显示前 3 步。任务调度层也拒绝给标定 episode 创建后续任务，而不仅仅隐藏页面。

每个 subject 使用一个固定的后端 episode，capture token 用于提交重试幂等及隔离新旧轮次，旧结果不能完成新一轮标定。
再次进入标定时仍使用同一目录，Start 会删除该 episode 的旧文件再录制，避免短视频重录后混入旧帧。同一主机同一 ID 的并发标定会被锁拒绝；其他 subject 和其他 task 不受影响。

上传期间暂停再次开始和再次提交。发布失败保留本地采集文件，可用 Retry Confirm 重试；退出时也保留已确认的数据。本地文件不会在发布后自动删除。

## 当前部署阻塞（2026-09-06）

已核对 ego 的 `/home/ubuntu/demo/Orbbec_demo` 及它连接的 synology：

- `/usr/local/sbin/nas-uploader-publish` 包装器仅接受普通 episode 与 `--manual-2d`，尚未实现 README 中的 `--shape-calibration`。
- NAS/ego 当前 producer 源码也没有 shape calibration 分支。
- README 明确同一 episode ID 的原始文件不可变，旧任务重复 publish 幂等，不会触发重新计算。

因此采集端按钮、录制和调用链已实现，但首次远端标定及已发布 episode 的重复标定尚不能在此部署完成。需要提供并部署 README 对应的新版 publisher，以及固定路径重新标定的版本/替换协议。`scripts/publish_handshape.py` 会在覆写已登记的远端 episode 前检测这个条件并报错，避免破坏正在上传的数据或误报成功。它不会擅自重置 NAS 数据库。

## 验证与恢复

```bash
python3 -m unittest tests.test_handshape_calibration -v
cmake --build build --target orbbec -j2
```

7 项测试覆盖目录覆盖范围、symlink/路径检查、并发锁、准确的 shape 参数、幂等重试、失败保留与旧发布器拒绝重新标定。ego C++ 编译通过。没有实际启动摄像头录制、发布真实 episode 或修改 NAS 状态库。

ego 备份：`/home/ubuntu/demo/orbbec-handshape-backup-20260906`，含原有 C++ 文件与主程序。正在运行的采集进程未被重启；替换后的主程序下次启动生效。


## 流程隔离更新（2026-09-06）

新增 `tests/test_handshape_workflow.py` 验证专用进度、结果回传后停止跟踪、不执行普通 3D 转换、不创建后续任务、页面止于第 3 步，以及新旧轮次隔离。更新已同步 ego；交互式运行中的后端和采集进程未被中断，重新启动两者后加载新代码。NAS publisher 的上述部署阻塞仍然独立存在。
