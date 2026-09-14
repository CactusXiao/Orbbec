"""Build from an explicit allowlist, including an operator-supplied network key."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import zipfile


def package(root: Path) -> Path:
    stage = root / 'Orbbec外包工作台'
    stage.mkdir(exist_ok=True)
    text = (root / '使用说明.txt').read_text()
    parts = re.split(r'(?m)^(\d\. .+)$', text)
    if len(parts) != 13:
        raise ValueError('Expected six guide sections')
    sections = ''.join(
        '<section id="s%d"><h2>%s</h2><p>%s</p></section>' % (
            (i + 1) // 2, html.escape(parts[i]),
            html.escape(parts[i + 1].strip()).replace('\n\n', '</p><p>').replace('\n', '<br>'))
        for i in range(1, len(parts), 2))
    css = '''body{margin:0;background:#f3f5f7;color:#172435;font:16px/1.8 system-ui,sans-serif}main{max-width:880px;margin:24px auto;padding:28px;background:white;border-radius:12px}h1{font-size:27px}h2{font-size:20px;margin:28px 0 10px}p{margin:10px 0}a{color:#175c92;overflow-wrap:anywhere}nav{display:flex;flex-wrap:wrap;gap:8px 20px}.start{display:inline-block;background:#175c92;color:white;padding:8px 20px;border-radius:6px;text-decoration:none}@media(max-width:600px){main{margin:0;padding:18px}}@media print{body{background:white}main{margin:0;padding:0}nav,.start{display:none}h2{break-after:avoid}}'''
    prefix = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Orbbec 外包工作台</title><style>' + css + '</style><main><h1>Orbbec 外包工作台</h1>'
    link = '<a class="start" href="https://orbbec-label.tail22716c.ts.net/" target="_blank" rel="noopener noreferrer">打开工作台</a>'
    nav = '<nav>' + ''.join('<a href="#s%d">%s</a>' % (i, html.escape(parts[2*i-1])) for i in range(1, 7)) + '</nav>'
    (root / '使用说明.html').write_text(prefix + link + nav + sections + '</main></html>')
    (root / '打开工作台.html').write_text(prefix + '''<ol><li>安装并打开 Tailscale，无需登录 Tailscale 账号。</li><li>首次使用：运行包内对应系统的“首次接入”工具，自动读取包内令牌接入网络。</li><li>网络连接后打开工作台，QC／Label 使用自己的账号登录。</li></ol>''' + link + '<p><a href="使用说明.html">使用说明</a></p></main></html>')
    launchers = {
        '连接诊断-Windows.cmd': '''@echo off
setlocal
set "arch=amd64"
if /I "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "arch=arm64"
if /I "%PROCESSOR_ARCHITEW6432%"=="ARM64" set "arch=arm64"
"%~dp0bin\\network-windows-%arch%.exe" --diagnose
''',
        '连接诊断-macOS.command': '''#!/bin/sh
cd "$(dirname "$0")" || exit 1
case "$(uname -m)" in
  arm64) arch=arm64 ;;
  x86_64) arch=amd64 ;;
  *) exit 1 ;;
esac
exec "./bin/network-darwin-$arch" --diagnose
''',
        '首次接入-macOS.command': '''#!/bin/sh
cd "$(dirname "$0")" || exit 1
case "$(uname -m)" in
  arm64) arch=arm64 ;;
  x86_64) arch=amd64 ;;
  *) echo '不支持此处理器'; exit 1 ;;
esac
exec "./bin/network-darwin-$arch"
''',
        '首次接入-Windows.cmd': '''@echo off
setlocal
set "arch=amd64"
if /I "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "arch=arm64"
if /I "%PROCESSOR_ARCHITEW6432%"=="ARM64" set "arch=arm64"
"%~dp0bin\\network-windows-%arch%.exe"
''',
        '首次接入-Linux.sh': '''#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec ./bin/network-linux-amd64
'''
    }
    for name, value in launchers.items():
        if name.endswith('.cmd'):
            (stage / name).write_bytes(value.replace('\n', '\r\n').encode('ascii'))
        else:
            (stage / name).write_text(value)
            (stage / name).chmod(0o755)
    docs = ['打开工作台.html', '使用说明.html', '使用说明.txt']
    for name in docs:
        shutil.copy2(root / name, stage / name)
    binaries = ['bin/network-' + target for target in [
        'darwin-arm64', 'darwin-amd64', 'windows-amd64.exe', 'windows-arm64.exe', 'linux-amd64']]
    names = docs + list(launchers) + binaries
    embedded_key = (stage / 'network-key.txt').is_file()
    if embedded_key:
        if not re.fullmatch(r'tskey-auth-[A-Za-z0-9_-]{20,512}', (stage / 'network-key.txt').read_text().strip()):
            raise ValueError('Invalid bundled network key')
        names.append('network-key.txt')
    for name in names:
        if not (stage / name).is_file():
            raise FileNotFoundError(stage / name)
    result = root / 'Orbbec外包工作台-Tailscale版.zip'
    with zipfile.ZipFile(result, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(stage / name, 'Orbbec外包工作台/' + name)
    with zipfile.ZipFile(result) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == ['Orbbec外包工作台/' + n for n in names]
        for name in names:
            assert archive.read('Orbbec外包工作台/' + name) == (stage / name).read_bytes()
    state_path = root / '交付状态.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state.update(package_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
                 network_enrollment='reusable-auth-key-vendor-tag', package_files=names,
                 shared_key_embedded=embedded_key,
                 application_login='individual-workbench-account')
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2] / '交付/Tailscale配置')
    args = parser.parse_args()
    print(package(args.output))
