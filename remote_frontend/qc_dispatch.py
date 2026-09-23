"""Episode-grained QC scheduling. This module never decodes or renders media."""
from __future__ import annotations
import copy
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit

from .batch import reject, digest
from task_backend.workflow_store import now_iso
from .browser_label import browser_task


def capability(secret, job, generation, expires):
    body = f'{job}:{generation}:{expires}'
    return f'{expires}.' + hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()


def valid_capability(secret, job, generation, token):
    try:
        expires = int(token.split('.')[0])
        return time.time() < expires <= time.time() + 660 and hmac.compare_digest(
            token, capability(secret, job, generation, expires))
    except (ValueError, IndexError):
        return False


class QCDispatch:
    def __init__(self, root, config, *, clock=time.time):
        self.root, self.clock = Path(root), clock
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'qc-dispatch.json'
        self.secret = Path(config['secret_file']).read_text().strip()
        if len(self.secret) < 32:
            raise ValueError('QC worker secret must contain at least 32 characters')
        self.allowed = config['workers']
        for origin in self.allowed.values():
            p = urlsplit(origin)
            if p.scheme != 'https' or not p.hostname or p.path or p.query or p.fragment or p.username or p.password:
                raise ValueError('QC worker origins must be explicit HTTPS origins')
        self.timeout = int(config.get('heartbeat_timeout', 30))
        self.viewer_ttl = int(config.get('viewer_ttl', 180))
        self.lock = threading.RLock()
        self.jobs = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.workers = {}
        # Process restart fences every old assignment. No local fallback.
        for job in self.jobs.values():
            job.update(worker=None, generation=secrets.token_hex(16), state={'ready': False})

    def authenticate(self, header):
        if not hmac.compare_digest(header or '', 'Bearer ' + self.secret):
            reject('Worker authentication required', 403)

    def save(self):
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.jobs))
        temp.chmod(0o600)
        temp.replace(self.path)

    def sweep(self):
        now = self.clock()
        for key, job in list(self.jobs.items()):
            job['viewers'] = {s: t for s, t in job['viewers'].items() if now - t < self.viewer_ttl}
            if not job['viewers']:
                del self.jobs[key]
            elif job['worker'] and now - self.workers.get(job['worker'], {}).get('seen', 0) > self.timeout:
                job.update(worker=None, generation=secrets.token_hex(16), state={'ready': False})

    def status(self, batch, sid):
        item = batch.session(sid)
        if item['role'] != 'qc':
            raise ValueError('Only QC episodes can be dispatched')
        # Existing sessions must still own an active workflow lease.
        job_state = batch.store.get_job(item['job_id']) or {}
        if (job_state.get('lease_owner') != item['owner'] or job_state.get('status') not in ('leased', 'running')
                or (job_state.get('lease_until') and job_state['lease_until'] <= now_iso())):
            return {'ready': False, 'phase': 'released'}
        key = digest([item['payload']['episode_id'], item['revision']])[:32]
        with self.lock:
            self.sweep()
            if key not in self.jobs:
                batch.check(item)
                task = browser_task(item['payload'], mounts=batch.mounts, role='qc')
                self.jobs[key] = dict(id=key, episode_id=item['payload']['episode_id'],
                    payload=dict(item['payload'], frames=list(task.frames), cameras=list(task.cameras)),
                    source=str(task.episode_dir()), viewers={},
                    worker=None, generation=secrets.token_hex(16), state={'ready': False})
            job = self.jobs[key]
            job['viewers'][sid] = self.clock()
            self.save()
            state = copy.deepcopy(job['state'])
            state.update(distributed=True, phase=('queued' if not job['worker'] else
                         'failed' if state.get('error') else 'complete' if state.get('complete') else 'preparing'),
                         worker_id=job['worker'], assignment=key + ':' + job['generation'])
            if job['worker']:
                token = capability(self.secret, key, job['generation'], int(time.time()) + 600)
                state['media_base'] = f"{self.allowed[job['worker']]}/media/{key}/{job['generation']}/{token}"
            return state

    def cancel(self, batch, sid):
        batch.session(sid)
        with self.lock:
            for job in self.jobs.values():
                job['viewers'].pop(sid, None)
            self.sweep()
            self.save()

    def retry(self, batch, sid):
        batch.check(batch.session(sid))
        with self.lock:
            for job in self.jobs.values():
                if sid in job['viewers'] and job['state'].get('error'):
                    job.update(worker=None, generation=secrets.token_hex(16), state={'ready': False})
            self.save()
        self.status(batch, sid)
        return {'ok': True}

    def poll(self, body):
        worker = body.get('worker_id')
        if worker not in self.allowed:
            reject('Unregistered QC worker', 403)
        reports = body.get('jobs', {})
        if not isinstance(reports, dict) or len(reports) > 100:
            reject('Invalid worker report', 400)
        with self.lock:
            self.sweep()  # Fence late heartbeats before accepting their results.
            boot = body.get('boot_id', '')
            previous = self.workers.get(worker)
            if previous and previous.get('boot_id') != boot:
                for job in self.jobs.values():
                    if job['worker'] == worker:
                        job.update(worker=None, generation=secrets.token_hex(16), state={'ready': False})
            self.workers[worker] = {'seen': self.clock(), 'boot_id': boot, 'resources': body.get('resources', {})}
            for key, report in reports.items():
                job = self.jobs.get(key)
                if job and job['worker'] == worker and report.get('generation') == job['generation']:
                    state = report.get('state', {})
                    if isinstance(state, dict):
                        job['state'] = {k: v for k, v in state.items() if k in {
                            'ready','complete','error','progress','cameras','chunks','prepared','total','codec','layout','timings'}}
            held = [j for j in self.jobs.values() if j['worker'] == worker]
            busy = any(not (j['state'].get('complete') or j['state'].get('error')) for j in held)
            # One episode at a time per capture host; workers may decline new work under pressure.
            if body.get('accepting') is True and not busy:
                pending = next((j for j in self.jobs.values() if j['worker'] is None), None)
                if pending:
                    pending['worker'] = worker
            result = [{k: copy.deepcopy(j[k]) for k in ('id','generation','payload')}
                      for j in self.jobs.values() if j['worker'] == worker]
            self.save()
            return {'jobs': result, 'lease_seconds': self.timeout}

    def input_source(self, worker, key, generation):
        with self.lock:
            self.sweep()
            job = self.jobs.get(key)
            if not job or job['worker'] != worker or job['generation'] != generation:
                reject('QC assignment expired', 409)
            return Path(job['source'])


class RoutedMedia:
    """Labels use the original media object. QC never reaches it."""
    def __init__(self, local, dispatch):
        self.local, self.dispatch, self.batch = local, dispatch, local.batch

    def qc(self, sid):
        return self.batch.session(sid)['role'] == 'qc'

    def status(self, sid):
        return self.dispatch.status(self.batch, sid) if self.qc(sid) else self.local.status(sid)

    def directory(self, sid):
        if self.qc(sid):
            reject('QC media is served directly by the assigned capture worker', 409)
        return self.local.directory(sid)

    def cancel(self, sid):
        return self.dispatch.cancel(self.batch, sid) if self.qc(sid) else self.local.cancel(sid)

    def retry(self, sid):
        return self.dispatch.retry(self.batch, sid) if self.qc(sid) else self.local.retry(sid)

    def ensure_ego(self, sid):
        if self.qc(sid):
            reject('QC media is remote', 409)
        return self.local.ensure_ego(sid)
