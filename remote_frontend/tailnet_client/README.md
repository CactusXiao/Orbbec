# Tailscale 令牌接入工具

仅接入网络；QC／Label 账号认证不变。

- 支持管理员发放的可重复使用 Auth Key；接入时请求 `tag:vendor` 标签。
- 设备只能访问工作台 TCP 443；不分发 Tailscale 个人账号。
- 按用户要求读取包内 network-key.txt；缺失时使用隐藏输入。以私有临时文件交给官方 CLI，不将令牌写入命令行或日志。
- 不覆盖已绑定的其他身份，不执行强制重认证、退出账号或重置配置。
- 成功后检查 HTTPS 与未登录 401，再打开工作台；`--check` 仅检查。
- Windows、macOS 使用官方独立版客户端；Linux 需要已运行的 tailscaled 及操作权限。

```sh
go test ./...
go vet ./...
CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -trimpath -ldflags='-s -w' -o network-darwin-arm64 .
```

签名：工具未做商业代码签名；企业终端可由 IT 审核、签名后分发。
打包入口：`python3 -m remote_frontend.package_tailnet`，要求交付目录 bin 已有各平台构建。

验证：一次性带标签令牌已在独立临时节点完成入网；HTTPS 200、未登录接口 401，SSH 及后端端口不可达。
临时节点验证后退出；未替换用户电脑或实验室主节点身份。
macOS 已验证程序启动和身份保护；Windows 已交叉编译，尚未进行 Windows 终端实测。
