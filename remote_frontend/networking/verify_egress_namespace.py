#!/usr/bin/env python3
"""Run as root on the configured Linux host; uses an isolated test namespace."""
import subprocess,json,os
name='orbbec-egress-check'
created=False
def run(*args,ok=True):return subprocess.run(args,check=ok,text=True,capture_output=True)
def n(*args,ok=True):return run('ip','netns','exec',name,*args,ok=ok)
script=['/usr/local/libexec/orbbec-tailscale-egress','--interface','testwired','--uid','1000','--port','56935']
try:
 run('ip','netns','add',name);created=True
 for iface,peer,subnet,metric in [('testwired','wiredpeer','192.0.2',600),('testwifi','wifipeer','198.51.100',50)]:
  n('ip','link','add',iface,'type','veth','peer','name',peer)
  n('ip','link','set',iface,'up');n('ip','link','set',peer,'up')
  n('ip','addr','add',subnet+'.2/24','dev',iface)
  n('ip','route','add','default','via',subnet+'.1','dev',iface,'metric',str(metric))
 n(*script)
 def route(dest,protocol,port=56935):
  return json.loads(n('ip','-j','route','get',dest,'ipproto',protocol,'sport',str(port),'dport','41641','uid','1000').stdout)[0]['dev']
 assert route('203.0.113.9','udp')=='testwired'
 assert route('203.0.113.9','tcp')=='testwifi'
 assert route('203.0.113.9','udp',56936)=='testwifi'
 assert route('198.51.100.9','udp')=='testwifi'
 n(*script);assert route('203.0.113.9','udp')=='testwired'
 n('ip','link','set','testwired','down');n(*script)
 assert route('203.0.113.9','udp')=='testwifi'
 n('ip','link','set','testwired','up');n(*script)
 assert route('203.0.113.9','udp')=='testwifi'
 n('ip','route','replace','default','via','192.0.2.1','dev','testwired','metric','600');n(*script)
 assert route('203.0.113.9','udp')=='testwired'
 n(*script,'--remove');assert route('203.0.113.9','udp')=='testwifi'
 n('ip','rule','add','priority','10441','table','main')
 assert n(*script,ok=False).returncode!=0
 print('PASS: UDP isolation, unchanged TCP/other UDP/LAN routes, repeat apply, link loss fallback, recovery, removal, unrelated rule protection')
finally:
 if created:run('ip','netns','del',name,ok=False)
