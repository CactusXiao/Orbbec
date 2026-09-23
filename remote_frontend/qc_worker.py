"""Capture-host QC agent and direct, capability-protected HTTPS media origin.

Expose the loopback HTTP port using Tailscale Serve. No workflow DB or Label
runtime is started here. One low-priority child owns one complete episode.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import secrets
import signal
import subprocess
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .qc_dispatch import valid_capability
from .preview_server import byte_range


def export_inputs(source, stream):
    """Transfer source bytes only; omit depth and workflow output, never decode."""
    source = source.resolve()
    with tarfile.open(fileobj=stream, mode='w|', dereference=True) as archive:
        for path in sorted(source.rglob('*')):
            rel = path.relative_to(source)
            if not path.is_file():
                continue
            if any(p.lower() in {'depth','qc','workflow','manual_2d','manual_joints_vis','.cache'} for p in rel.parts):
                continue
            if not path.resolve().is_relative_to(source):
                raise ValueError('QC input escapes episode root')
            archive.add(path, arcname=str(Path('subject/task/episode')/rel), recursive=False)
        for name in ('shape.npy','scale.npy'):
            path = source.parents[1]/name
            if path.is_file():
                archive.add(path, arcname='subject/'+name, recursive=False)


def extract_inputs(stream, target, limit):
    total = 0
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            rel = Path(member.name)
            if rel.is_absolute() or '..' in rel.parts or not member.isfile() or not rel.parts or rel.parts[0] != 'subject':
                raise ValueError('Unsafe QC input archive')
            total += member.size
            if total > limit:
                raise ValueError('QC input exceeds disk budget')
            path = target/rel
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src, path.open('wb') as dst:
                shutil.copyfileobj(src, dst, 1024*1024)
    return total


def request(config, secret, path, body=None, timeout=20):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(config['backend_origin']+path, data=data, headers={
        'Authorization': 'Bearer '+secret, 'Content-Type': 'application/json',
        'X-Orbbec-Request': '1', 'X-QC-Worker': config['worker_id']})
    return urlopen(req, timeout=timeout)


def render_job(folder, config):
    # Hard process affinity also bounds ffmpeg, Mesa, BLAS and MANO children.
    os.nice(int(config.get('nice', 10)))
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, available[-int(config.get('cpu_count', 8)):])
    from .browser_media import BrowserMedia
    started = time.monotonic()
    job = json.loads((folder/'job.json').read_text())
    secret = Path(config['secret_file']).read_text().strip()
    target = folder/'input'
    target.mkdir(parents=True, exist_ok=True)
    with request(config, secret, f"/internal/qc/input/{job['id']}/{job['generation']}", timeout=120) as stream:
        size = extract_inputs(stream, target, min(int(config.get('max_input_bytes', 20*1024**3)),
            shutil.disk_usage(folder).free - int(config.get('min_free_bytes', 30*1024**3))))
    payload = dict(job['payload'], episode_uri='nas://qc/subject/task/episode')
    class Batch:
        mounts = {'nas://qc': str(target)}
        def session(self, sid):
            if sid != job['id']:
                raise ValueError('Invalid QC job')
            return {'role': 'qc', 'payload': payload}
        def check(self, item):
            pass
    media = BrowserMedia(Batch(), folder/'media', config, slots=threading.Semaphore(1))
    def stop(*_):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        media.cancel(job['id'])
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    transferred = time.monotonic()
    while True:
        if shutil.disk_usage(folder).free < int(config.get('min_free_bytes', 30*1024**3)):
            media.cancel(job['id'])
            raise RuntimeError('QC stopped to preserve capture disk reserve')
        state = media.status(job['id'])
        state['timings'] = dict(input_bytes=size, transfer_seconds=round(transferred-started, 3),
                                total_seconds=round(time.monotonic()-started, 3))
        temp = folder/'status.tmp'
        temp.write_text(json.dumps(state))
        temp.replace(folder/'status.json')
        if state.get('complete') or state.get('error'):
            break
        time.sleep(1)


def process_tree(pid):
    """Track only this job's descendants, including children spawned by threads."""
    found, pending = {}, [pid]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        try:
            root = Path('/proc')/str(current)
            found[current] = root.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]
            for task in root.joinpath('task').iterdir():
                pending.extend(int(p) for p in (task/'children').read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            pass
    return found


def kill_tracked(tracked):
    for pid, started in reversed(list(tracked.items())):
        try:
            now = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
            if now == started:
                os.kill(pid, signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError):
            pass


class Agent:
    def __init__(self, config, config_path):
        self.config, self.config_path = config, config_path
        self.secret = Path(config['secret_file']).read_text().strip()
        self.root = Path(config['state_dir'])
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.jobs, self.lock = {}, threading.RLock()
        self.boot_id = secrets.token_hex(16)
        self.last_contact = 0
        self.lease_seconds = 30
        self.stop = threading.Event()

    def folder(self, job):
        return self.root/(job['id']+'-'+job['generation'])

    def reports(self):
        result = {}
        for key, job in self.jobs.items():
            path = self.folder(job)/'status.json'
            state = json.loads(path.read_text()) if path.exists() else {'ready': False, 'progress': {'input': {'status': 'transferring'}}}
            if job['process'].poll() is not None and not state.get('complete') and not state.get('error'):
                state['error'] = 'QC worker process stopped; check worker log and retry'
            result[key] = {'generation': job['generation'], 'state': state}
        return result

    def halt(self, job):
        process = job['process']
        if process.poll() is None:
            tracked = process_tree(process.pid)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            # Renderer subprocesses create separate process groups. Reap the
            # exact tracked descendants even when their parent exited promptly.
            for pid, started in list(tracked.items()):
                current = process_tree(pid)
                if current.get(pid) == started:
                    tracked.update(current)
            kill_tracked(tracked)
            process.wait(timeout=5)
        job['log'].close()
        status = self.folder(job)/'status.json'
        if status.exists() and json.loads(status.read_text()).get('error'):
            errors = self.root/'errors'
            errors.mkdir(exist_ok=True)
            shutil.copyfile(self.folder(job)/'worker.log', errors/(self.folder(job).name+'.log'))

    def loop(self):
        while not self.stop.is_set():
            try:
                free = shutil.disk_usage(self.root).free
                memory = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
                available = int(memory['MemAvailable'].strip().split()[0])*1024
                load = os.getloadavg()[0]
                headroom = float(self.config.get('max_load_average', max(1, (os.cpu_count() or 1)-int(self.config.get('cpu_count', 8)))))
                accepting = load <= headroom and free >= int(self.config.get('min_free_bytes', 30*1024**3)) and available >= int(self.config.get('min_available_memory', 4*1024**3)) and not (self.root/'pause').exists()
                with self.lock:
                    reports = self.reports()
                with request(self.config, self.secret, '/internal/qc/poll', dict(worker_id=self.config['worker_id'], boot_id=self.boot_id,
                        accepting=accepting, jobs=reports, resources=dict(free_bytes=free, available_memory=available, load_average=load, max_load_average=headroom))) as response:
                    assignments = json.load(response)
                self.last_contact = time.monotonic()
                self.lease_seconds = assignments['lease_seconds']
                desired = {j['id']: j for j in assignments['jobs']}
                with self.lock:
                    for key, job in list(self.jobs.items()):
                        if key not in desired or desired[key]['generation'] != job['generation']:
                            del self.jobs[key]  # Revoke access before waiting for child shutdown.
                            self.halt(job)
                            shutil.rmtree(self.folder(job), ignore_errors=True)
                    for key, job in desired.items():
                        if key in self.jobs:
                            continue
                        if any(j['process'].poll() is None for j in self.jobs.values()):
                            continue
                        folder = self.folder(job)
                        # A restarted daemon never adopts an unverifiable prior child/result.
                        shutil.rmtree(folder, ignore_errors=True)
                        folder.mkdir(mode=0o700)
                        (folder/'job.json').write_text(json.dumps(job))
                        log = (folder/'worker.log').open('w')
                        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                   NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1')
                        process = subprocess.Popen([sys.executable, '-m', 'remote_frontend.qc_worker',
                            '--config', str(self.config_path), '--job', str(folder)], stdout=log, stderr=subprocess.STDOUT, env=env)
                        self.jobs[key] = dict(job, process=process, log=log)
                # Prune unreferenced outputs after restart, bounded by a retention interval.
                for folder in self.root.iterdir():
                    if folder.name != 'errors' and folder.is_dir() and folder.name not in {self.folder(j).name for j in self.jobs.values()} and time.time()-folder.stat().st_mtime > 600:
                        shutil.rmtree(folder, ignore_errors=True)
            except Exception as exc:
                print('QC heartbeat:', type(exc).__name__, getattr(exc, 'code', ''), flush=True)  # Never log tokens or URLs.
            if time.monotonic()-self.last_contact > self.lease_seconds:
                with self.lock:
                    for job in self.jobs.values():
                        self.halt(job)
                    self.jobs.clear()
            self.stop.wait(2)

    def media_path(self, parts):
        if len(parts) < 5 or parts[0] != 'media':
            raise ValueError('Invalid media path')
        _, key, generation, token, *tail = parts
        if not valid_capability(self.secret, key, generation, token):
            raise PermissionError('Media capability expired')
        with self.lock:
            job = self.jobs.get(key)
            if (not job or job['generation'] != generation or
                    time.monotonic()-self.last_contact > self.lease_seconds):
                raise PermissionError('QC assignment expired')
            folder = self.folder(job)/'media'/key/'layered-v1'
            if tail == ['preview.mp4']:
                return folder/'preview.mp4', 'video/mp4'
            if len(tail) == 2 and tail[0] == 'chunks' and tail[1].endswith('.mp4') and tail[1][:-4].isdigit():
                return folder/'chunks'/tail[1], 'video/mp4'
            if len(tail) == 3 and tail[0] in ('frames','raw_frames') and tail[2].isdigit():
                payload = job['payload']
                if tail[1] not in list(payload.get('cameras', []))+['ego'] or int(tail[2]) not in payload.get('frames', []):
                    raise PermissionError('Frame outside assignment')
                return folder/tail[0]/tail[1]/(tail[2]+'.jpg'), 'image/jpeg'
            raise ValueError('Invalid media resource')


class MediaHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def end_headers(self):
        origin = self.headers.get('Origin')
        if origin in self.server.agent.config['browser_origins']:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
            self.send_header('Access-Control-Expose-Headers', 'Content-Range, Accept-Ranges, Content-Length')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Methods','GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers','Range')
        self.send_header('Content-Length','0')
        self.end_headers()

    def do_GET(self):
        try:
            path, kind = self.server.agent.media_path(urlsplit(self.path).path.strip('/').split('/'))
            size = path.stat().st_size
        except PermissionError:
            return self.send_error(403, 'QC media authorization expired')
        except (ValueError, FileNotFoundError):
            return self.send_error(404, 'QC media not ready')
        try:
            start, end = byte_range(self.headers.get('Range',''), size)
        except ValueError:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        try:
            self.send_response(206 if self.headers.get('Range') else 200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length',str(end-start+1))
            self.send_header('Accept-Ranges','bytes')
            if self.headers.get('Range'):
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            with path.open('rb') as stream:
                stream.seek(start)
                left = end-start+1
                while left:
                    data = stream.read(min(left, 65536))
                    if not data:
                        break
                    self.wfile.write(data)
                    left -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.job:
        try:
            render_job(args.job, config)
        except Exception as exc:
            (args.job/'status.tmp').write_text(json.dumps({'ready':False, 'error':str(exc)}))
            (args.job/'status.tmp').replace(args.job/'status.json')
            raise
        return
    agent = Agent(config, args.config.resolve())
    server = ThreadingHTTPServer(('127.0.0.1', int(config.get('port',18900))), MediaHandler)
    server.agent = agent
    threading.Thread(target=agent.loop, daemon=True).start()
    def stop(*_):
        agent.stop.set()
        for job in list(agent.jobs.values()):
            agent.halt(job)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
