
进入 `Orbbec_demo`

## 1. 编译

```bash
./build.sh
```

## 2. 启动任务管理后端

```bash
python3 task_backend/server.py
```

打开：

```text
http://127.0.0.1:8765/
```

选择或创建一个 instance，点击 Start。

## 3. 启动采集程序

```bash
bash run.sh
```

## 4. 采集流程操作

- 菜单界面会显示pico客户端的连接情况。在菜单界面时，启动pico客户端，当显示连接成功时即可进入collection（需要保证ubuntu桌面侧栏有pico图标，一个长得像手机的图标，表示pico已经连接）
- 填写保存目录和subject_id，曝光时长等
- 点击select task，进入任务选择界面，每次只能选择一个任务，每个任务只能被一个subject_id选择
- 点击选择的任务，便可直接带上pico开始采集，无需重启pico客户端程序。
- episode都采集完成后重新选择task。可以中途退出，后端会维护相关状态的正确性。
