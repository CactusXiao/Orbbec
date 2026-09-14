"""Package only compiled VPN launchers and operator instructions, never backend data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

README = '''Orbbec 外包工作台

使用步骤
1. 将压缩包完整解压到固定目录。
2. 连接公司分配的企业 VPN。
3. 双击“启动工作台”（Windows 为 .cmd，macOS 为 .command）。
   Linux 在终端运行：sh 启动工作台.sh
4. 浏览器打开后，使用项目管理员分配的个人账号登录，选择 Label 或 QC。
   账号、密码及 VPN 账号单独分配，不包含在此包中。
5. 工作期间保留启动器窗口。完成后退出工作台登录，再关闭启动器窗口。

不需要安装 Python、Go、模型、SSH 或系统证书。启动器直接通过 HTTPS 连接后端。
如浏览器没有自动打开，请手动访问 http://127.0.0.1:18885/ 。
这里的本机地址由随包启动器提供，连接的真实后端是 10.162.241.5:18884。
启动器不能用于连接其他服务器；它会核验实验室证书，连接不可信服务器时拒绝继续。

工作进度
- 固定使用同一台设备、同一个浏览器、同一个入口，不使用无痕模式。
- 提交成功提示出现前，工作仍属于当前浏览器的本地草稿。
- 断网或 VPN 中断时，不要清理浏览器数据；恢复网络后刷新或按页面提示继续。
- 新入口不会自动迁移旧入口、其他电脑或其他浏览器里的草稿。
- 没有任务：请联系管理员核查角色、任务范围及任务状态。
- 共享电脑请使用独立的系统账号；完成工作后退出登录。

常见问题
- 连接失败：先确认 VPN 正常，并将错误提示发给项目管理员，不要提供密码。
- 提示端口被占用：关闭之前打开的工作台启动器后再试；请勿随意改端口，否则本地草稿入口会变化。
- 系统阻止启动：本包尚未进行商业代码签名，请联系公司 IT 审核分发，不要关闭系统安全保护。
- 服务器证书更新后需由项目管理员分发新版启动包。

运行要求
Windows x64/ARM64；macOS Apple 芯片/Intel；Linux x64/ARM64。
上述为编译目标。已在当前 macOS 机器验证直连；Windows/Linux 桌面尚未实机验收。
浏览器需支持 IndexedDB 与 H.264 播放。页面和业务功能从后端加载，后续 UI 更新通常刷新即可。
'''

def build(binaries: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    launchers = {
        'Windows': ('启动工作台.cmd', '@echo off\r\nchcp 65001 >nul\r\ncd /d "%~dp0"\r\nset "APP=windows-amd64.exe"\r\nif /I "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "APP=windows-arm64.exe"\r\nif /I "%PROCESSOR_ARCHITEW6432%"=="ARM64" set "APP=windows-arm64.exe"\r\n"bin\\%APP%"\r\nif errorlevel 1 pause\r\n', 'windows'),
        'macOS': ('启动工作台.command', '#!/bin/sh\ncd "$(dirname "$0")" || exit 1\ncase "$(uname -m)" in arm64) arch=arm64;; x86_64) arch=amd64;; *) echo "不支持此处理器"; exit 1;; esac\n./bin/darwin-$arch\nstatus=$?\nif [ "$status" -ne 0 ]; then echo "启动失败，请将上面的提示提供给管理员。按回车关闭。"; read answer; fi\nexit "$status"\n', 'darwin'),
        'Linux': ('启动工作台.sh', '#!/bin/sh\ncd "$(dirname "$0")" || exit 1\ncase "$(uname -m)" in aarch64|arm64) arch=arm64;; x86_64) arch=amd64;; *) echo "不支持此处理器"; exit 1;; esac\nexec ./bin/linux-$arch\n', 'linux'),
    }
    results = []
    for platform, (launcher, script, system) in launchers.items():
        members = {launcher: (script.encode('utf-8'), 0o755), '使用说明.txt': (README.encode('utf-8-sig'), 0o644)}
        for arch in ['arm64', 'amd64']:
            name = f'{system}-{arch}' + ('.exe' if system == 'windows' else '')
            members['bin/' + name] = ((binaries / name).read_bytes(), 0o755)
        manifest = {'platform': platform, 'backend': 'https://10.162.241.5:18884',
                    'browser_origin': 'http://127.0.0.1:18885', 'transport': 'HTTPS, embedded server certificate, no SSH',
                    'files': {name: hashlib.sha256(data).hexdigest() for name, (data, _) in members.items()}}
        members['版本信息.json'] = ((json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode(), 0o644)
        archive = output / f'Orbbec外包工作台-{platform}.zip'
        with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
            for name, (data, mode) in members.items():
                info = ZipInfo('Orbbec外包工作台/' + name)
                info.create_system = 3
                info.external_attr = (0o100000 | mode) << 16
                info.compress_type = ZIP_DEFLATED
                z.writestr(info, data)
        results.append(archive)
    (output / 'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n' for p in results))
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binaries', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for path in build(args.binaries, args.output): print(path)
