# PICO 自动连接

运行 `./run.sh` 时，会先执行与现有 `pico` 命令相同的 USB 连接准备：
通过 ADB 配置 `adb reverse tcp:50051 tcp:50051`，然后启动采集程序。
启动参数会原样传递给采集程序。

主机需安装 `adb`，头显需连接 USB 并允许 USB 调试。多个 Android 设备
同时连接时，用 `ANDROID_SERIAL` 指定头显。没有头显、未授权或连接失败
时会显示提示，采集程序仍会启动。

这一步只准备 USB 通信，不会启动头显内的应用，也不会修改采集配置中的
`ego.enabled`。头显中的串流应用仍按原有方式打开。

部署环境如果使用自定义 `run.sh` 或桌面入口，在最终启动采集进程的位置
调用共用脚本，保留原有预检查、服务启动、配置文件和参数。例如：

```bash
exec "$PROJECT_ROOT/scripts/launch_with_pico.sh" \
    "$PROJECT_ROOT/bin/orbbec" "$PROJECT_ROOT/src/sync/config.real.json" "$@"
```

只在最终启动采集进程的位置调用一次，转发到该入口的其他脚本不重复调用。
