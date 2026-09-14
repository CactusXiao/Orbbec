"""Prepare the loopback workbench adapter after a human enrolls the Tailscale node.

Does not change tailnet grants, disable shields-up, or enable Serve/Funnel.
"""
from pathlib import Path
import datetime
import json
import re
import subprocess

root = Path.home() / '.local/share/orbbec-tailscale'
state = Path.home() / '.local/state/orbbec-tailscale'
cli = [str(root / 'tailscale'), '--socket=' + str(state / 'tailscaled.sock')]
status = json.loads(subprocess.check_output(cli + ['status', '--json'], text=True))
if status.get('BackendState') != 'Running':
    raise SystemExit('Waiting for the owner to enroll or approve this device in Tailscale.')
host = status.get('Self', {}).get('DNSName', '').rstrip('.')
if not re.fullmatch(r'[a-z0-9-]+\.[a-z0-9.-]+\.ts\.net', host):
    raise SystemExit('No valid tailnet DNS name yet; enable MagicDNS and retry.')
origin = 'https://' + host
unit = Path.home() / '.config/systemd/user/orbbec-tailnet-workbench.service'
if unit.exists():
    stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    (state / ('workbench-unit-before-' + stamp)).write_bytes(unit.read_bytes())
unit.write_text(f'''[Unit]
Description=Orbbec workbench adapter for Tailscale Serve
After=orbbec-tailscaled.service

[Service]
ExecStart={root}/workbench-adapter -tailnet-origin {origin}
Restart=on-failure
RestartSec=3
UMask=0077
NoNewPrivileges=true

[Install]
WantedBy=default.target
''')
subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
subprocess.run(['systemctl', '--user', 'enable', '--now', unit.name], check=True)
(state / 'workbench-origin.json').write_text(json.dumps({'origin': origin, 'adapter': '127.0.0.1:18886', 'upstream': '127.0.0.1:18882', 'serve_enabled_by_this_script': False}, indent=2))
print('Loopback adapter ready:', origin)
print('Next: verify restrictive tailnet grants, enable HTTPS, configure Serve to 127.0.0.1:18886, and then allow incoming traffic.')
