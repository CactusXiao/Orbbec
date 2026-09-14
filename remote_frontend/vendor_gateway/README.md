# VPN 直连启动包

外包机浏览器 → 本机 127.0.0.1:18885 启动器 → HTTPS 10.162.241.5:18884 → 实验室 127.0.0.1:18882。
启动器不使用 SSH，不执行解码、渲染或任务计算；只代理固定工作台地址并保留视频 Range 请求。
浏览器需要保留启动器进程。页面、操作和后端账号仍由既有 browser_server 提供。

## 证书与边界

- TLS 使用内嵌 server.pem 作为唯一信任根，仍验证服务器 IP、证书有效期和签名；未使用 InsecureSkipVerify。
- 私钥仅在实验室 `outsourcing-live-20260910/vpn-gateway/server.key`，不进入代码或外包包。
- 不向系统证书库安装任何 CA，不使用外包人员的 SSH 身份，也没有绕过浏览器证书警告。
- 网关只绑定指定 IPv4 地址的 18884，不绑定 0.0.0.0 或公网 IPv6。该 IP 是实验室内网地址，并非 VPN 专用接口；可路由至此地址的其他内网设备也能访问登录页。应用账号和任务授权继续执行。
- 两端都限定 Host、Origin、方法、请求体和并发数；不能转发任意目的地址。转发前删除不可信代理头。
- 后端生成的 cookie 在 HTTPS 网关增加 Secure，在只监听 loopback 的启动器去除 Secure；HttpOnly、SameSite 和有效期保留。
- 后端看到来源为 loopback，来源级登录限流仍是所有用户共享的附加限制；账号级限流独立。
- `server.pem` 的更新或 IP 变更需要重新构建外包包。不要直接续签后替换服务端证书，否则旧包会拒绝连接。
- 如果以后已有浏览器信任的企业 HTTPS 证书，可以部署标准浏览器入口省去本地启动器；更换 origin 前先提交现有草稿。

## 构建

使用 Go 标准库，无外部模块。证书是公开信息。构建前确认它对应受控的实验室服务。

```sh
go test ./...
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags='-s -w' -o gateway .
```

`remote_frontend/package_vendor.py` 从已编译二进制生成三个系统的交付 ZIP，明确列出文件，不会扫描或打包仓库。
macOS/Windows 包未进行商业代码签名；外包单位如禁止未签名程序，需要其 IT 审核/签名分发。未在 Windows 或 Linux 外包桌面实际验证。

## 服务与回退

服务为当前用户的 `orbbec-vpn-gateway.service`，只代理现有后端，不重启后端。证书及日志由用户服务管理。
网关及 Tailscale 服务依赖现有 18882 后端运行；后台服务启用不等于后端已配置自启。
停止网关使用 `systemctl --user disable --now orbbec-vpn-gateway.service`，不会中断原来的 SSH 验证入口或 native QC/Label。


`启动器 --check` 只验证 VPN/TLS 和未登录返回 401，不会申请任务或提交结果。

## Tailscale 模式

服务端：`-tailnet-origin https://HOST.TAILNET.ts.net`，仅监听 `127.0.0.1:18886`，由 Tailscale Serve 的 HTTPS 443 转发。适配器校验 Host/Origin，后端仍为 `127.0.0.1:18882`。

本机启动器：`-tailscale -no-open`，保留 `http://127.0.0.1:18885/`，通过 HTTPS 连接 `orbbec-label.tail22716c.ts.net`。使用系统信任根校验证书，连接不经过代理；原地址下的草稿和账号 Cookie 保留。只在此模式将建连和 TLS 超时设为 30 秒，不自动重试提交。

Mac 常驻服务：`~/Library/LaunchAgents/com.orbbec.tailnet-launcher.plist`；程序与日志位于 `~/.local/state/orbbec/`。须保持 Tailscale 已连接。

实验室用户服务：`orbbec-tailscaled.service`、`orbbec-tailnet-workbench.service`。网络授权只允许外包身份访问工作台 TCP 443。部署与维护见交付目录中的《内部管理说明》。
