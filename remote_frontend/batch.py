"""Opt-in browser drafts/commit adapter; desktop clients never import this module."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import secrets
import threading
import uuid

import numpy as np

from label.storage import correction_task_from_backend_payload
from mano.joint_order import SMPLX_MANO_SKELETON_EDGES, SMPLX_MANO_JOINT_NAMES
from src.qc.state_store import normalize_ranges, normalize_segments
from src.qc.report import build_qc_result, write_qc_report, write_ego_pose_qc_report
from task_backend.job_service import JobService
from task_backend.workflow_store import WorkflowStore, now_iso
from task_backend.workflow_models import WorkflowError


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def reject(message, status=409):
    raise WorkflowError(status, message)


@contextmanager
def preserve_files(paths):
    """Restore published files if validation/workflow/SQLite commit raises.

    Filesystem writes and SQLite cannot form a single power-loss transaction;
    deterministic paths and the same submission body make crash retries safe.
    """
    originals = {p: p.read_bytes() if p.is_file() else None for p in paths}
    try:
        yield
    except BaseException:
        for path, value in originals.items():
            if value is None:
                path.unlink(missing_ok=True)
            else:
                temporary = path.with_name(path.name + ".rollback-" + secrets.token_hex(4))
                temporary.write_bytes(value)
                temporary.replace(path)
        raise


class _Connection:
    """Reuse the outer write transaction for existing store methods' BEGINs."""
    def __init__(self, connection):
        self.connection = connection

    def execute(self, sql, *args):
        if sql.strip().upper() == "BEGIN IMMEDIATE":
            return self.connection.execute("SELECT 1")
        return self.connection.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self.connection, name)


class BatchStore(WorkflowStore):
    def __init__(self, path):
        self.local = threading.local()
        # WorkflowStore.initialize() performs production restart recovery and
        # resets some stage controls. A sidecar must NEVER run those migrations.
        self.db_path = Path(path).expanduser().resolve()
        if not self.db_path.is_file():
            raise ValueError("browser adapter requires an existing workflow database")
        with self.connect() as conn:
            if conn.execute("PRAGMA user_version").fetchone()[0] != 5:
                raise ValueError("unsupported workflow schema; start the normal backend first")
            conn.execute("""CREATE TABLE IF NOT EXISTS browser_sessions (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, job_id TEXT NOT NULL,
                revision TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS browser_receipts (
                session_id TEXT PRIMARY KEY, submission_id TEXT NOT NULL,
                digest TEXT NOT NULL, receipt TEXT NOT NULL)""")

    @contextmanager
    def connect(self):
        active = getattr(self.local, "connection", None)
        if active is not None:
            yield _Connection(active)
        else:
            with super().connect() as conn:
                yield conn

    @contextmanager
    def transaction(self):
        if getattr(self.local, "connection", None) is not None:
            raise RuntimeError("nested batch transaction")
        with super().connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self.local.connection = conn
            try:
                yield conn
            finally:
                self.local.connection = None


class BatchService:
    def __init__(self, db, mounts, operator, *, roles=None, tasks=None):
        self.store = BatchStore(db)
        self.service = JobService(self.store, nas_mounts=mounts)
        self.mounts = mounts
        self.operator = operator
        self.roles = set(roles if roles is not None else ["label", "qc"])
        self.tasks = set(tasks if tasks is not None else ["*"])

    def available(self, role):
        if role not in {"label", "qc"}:
            reject("无效任务类型", 400)
        if role not in self.roles:
            reject("账号没有此类任务权限", 403)
        kind = "manual_label" if role == "label" else "qc"
        with self.store.connect() as conn:
            rows = conn.execute("""SELECT j.job_id, j.episode_id, e.task_name, e.episode_index
                FROM jobs j JOIN episodes e ON e.episode_id=j.episode_id
                WHERE j.type=? AND (j.status='queued' OR (j.status IN ('leased','running')
                AND (j.lease_until IS NULL OR j.lease_until<=?))) ORDER BY j.created_at""", (kind, now_iso())).fetchall()
        items = [dict(row) for row in rows if "*" in self.tasks or row["task_name"] in self.tasks]
        for item in items:
            episode = self.store.get_episode(item["episode_id"]) or {}
            segments = self.store.segments_for_episode(item["episode_id"])
            item.update(subject_id=episode.get("subject_id", ""), segments=len(segments),
                frames=sum(self.service._segment_frame_count(segment) for segment in segments) if role == "label" else episode.get("frame_count", 0),
                first_start_frame=min((segment["start_frame"] for segment in segments), default=0))
        return sorted(items, key=lambda item: (item["task_name"], item["subject_id"], item["episode_index"] or 0))

    def authorize_job(self, role, job_id):
        if role not in self.roles:
            reject("账号没有此类任务权限", 403)
        with self.store.connect() as conn:
            row = conn.execute("SELECT j.type, e.task_name FROM jobs j JOIN episodes e USING(episode_id) WHERE j.job_id=?", (job_id,)).fetchone()
        if not row or row["type"] != ("manual_label" if role == "label" else "qc"):
            reject("任务不存在或类型不匹配", 404)
        if "*" not in self.tasks and row["task_name"] not in self.tasks:
            reject("该任务不在账号的授权范围内", 403)

    def context(self, job_id):
        job = self.store.get_job(job_id)
        if not job:
            reject("任务不存在", 404)
        return (self.service.enrich_manual_label_job(job) if job["type"] == "manual_label"
                else self.service.enrich_job(job))

    def revision(self, context):
        # Heartbeats do not change the source revision. Updated upstream artifacts,
        # segment ranges, calibration, visibility or MANO files do.
        payload = dict(context["payload"])
        payload.pop("frame_decisions", None)
        task = correction_task_from_backend_payload(payload, mounts=self.mounts)
        root = task.episode_dir()
        signatures = []
        for name in ("mano/episode", "optimized_pose", "joints_vis", "camera_params.json", "extrinsics.json", "ego_pose.json"):
            path = root / name
            files = sorted(path.rglob("*")) if path.is_dir() else [path]
            for file in files:
                if file.is_file():
                    stat = file.stat()
                    signatures.append((str(file.relative_to(root)), stat.st_size, stat.st_mtime_ns))
        job = context["job"]
        return digest([job["job_id"], job.get("attempt"), payload, context.get("artifacts"), signatures])

    def lease(self, role, job_id):
        if role not in {"label", "qc"}:
            reject("无效任务类型", 400)
        sid = uuid.uuid4().hex
        owner = f"browser:{self.operator}:{sid}"
        with self.store.transaction() as conn:
            self.authorize_job(role, job_id)
            body = dict(job_id=job_id, lease_owner=owner, lease_seconds=600)
            context = (self.service.lease_label_episode(body) if role == "label"
                       else self.service.lease_job(body, forced_type="qc"))
            revision = self.revision(context)
            saved = {"role": role, "payload": context["payload"], "operator": self.operator}
            conn.execute("INSERT INTO browser_sessions VALUES (?,?,?,?,?,?)", (
                sid, owner, context["job"]["job_id"], revision, json.dumps(saved), now_iso()))
        return self.session(sid)

    def session(self, sid):
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM browser_sessions WHERE id=?", (sid,)).fetchone()
        if not row:
            reject("浏览器任务不存在", 404)
        item = dict(row)
        item.update(json.loads(item.pop("payload")))
        if item["operator"] != self.operator:
            reject("该草稿属于其他操作员", 403)
        self.authorize_job(item["role"], item["job_id"])
        return item

    def manifest(self, sid):
        item = self.session(sid)
        task = correction_task_from_backend_payload(item["payload"], mounts=self.mounts)
        episode = self.store.get_episode(item["payload"]["episode_id"]) or {}
        job = self.store.get_job(item["job_id"]) or {}
        return dict(workflow_version=3, subject_id=episode.get("subject_id", ""),
                    segments=len(self.store.segments_for_episode(item["payload"]["episode_id"])),
                    first_start_frame=min(task.frames, default=0),
                    lease_until=job.get("lease_until", ""), released=bool(item.get("released")),
                    id=sid, role=item["role"], revision=item["revision"], job_id=item["job_id"],
                    episode_id=item["payload"]["episode_id"], operator=item["operator"],
                    task_name=episode.get("task_name"), episode_index=episode.get("episode_index"),
                    frames=task.frames, cameras=task.cameras, fps=30,
                    qc_segments=list(item["payload"].get("segments") or []),
                    skeleton_edges=SMPLX_MANO_SKELETON_EDGES, joint_names=SMPLX_MANO_JOINT_NAMES)

    def check(self, item, *, allow_expired=False):
        context = self.context(item["job_id"])
        job = context["job"]
        if job["status"] not in {"leased", "running"} or job.get("lease_owner") != item["owner"]:
            reject("任务已结束或已分配给其他人；本地草稿保留，请勿覆盖新结果")
        if not allow_expired and (job.get("lease_until") or "") <= now_iso():
            reject("任务租约已到期，请先恢复连接续租；草稿已保留")
        if self.revision(context) != item["revision"]:
            reject("源数据版本已变化；草稿保留，需要重新核对任务")
        return context

    def heartbeat(self, sid):
        with self.store.transaction():
            item = self.session(sid)
            # Reconnect can reclaim only its unchanged owner, under the same DB lock
            # used by desktop leasing. It never takes a task from another owner.
            self.check(item, allow_expired=True)
            self.service.heartbeat_job(item["job_id"], dict(lease_owner=item["owner"], lease_seconds=600, status="running"))
        return {"ok": True}

    def release(self, sid):
        with self.store.transaction() as conn:
            item = self.session(sid)
            job = self.store.get_job(item["job_id"])
            if job["status"] in {"succeeded", "failed", "canceled"}:
                return {"ok": True}
            if item.get("released") and job["status"] == "queued":
                return {"ok": True}
            if job.get("lease_owner") != item["owner"]:
                reject("任务已归属其他人，不能释放")
            self.service.release_job(item["job_id"], {"reason": "operator_left_browser"})
            conn.execute("UPDATE browser_sessions SET payload=json_set(payload, '$.released', json('true')) WHERE id=?", (sid,))
        return {"ok": True}

    def resume(self, sid):
        with self.store.transaction() as conn:
            item = self.session(sid)
            context = self.context(item["job_id"])
            job = context["job"]
            if job.get("lease_owner") == item["owner"] and job["status"] in {"leased", "running"}:
                self.check(item, allow_expired=True)
                self.service.heartbeat_job(item["job_id"], dict(lease_owner=item["owner"], lease_seconds=600, status="running"))
            else:
                if not item.get("released") or job["status"] != "queued" or self.revision(context) != item["revision"]:
                    reject("任务已被重新分配或版本变化，无法恢复；本机草稿保留")
                body = dict(job_id=item["job_id"], lease_owner=item["owner"], lease_seconds=600)
                context = (self.service.lease_label_episode(body) if item["role"] == "label" else self.service.lease_job(body, forced_type="qc"))
                conn.execute("UPDATE browser_sessions SET revision=?, payload=json_set(payload, '$.released', json('false')) WHERE id=?", (self.revision(context), sid))
        return self.manifest(sid)

    def record_frames(self, sid, body):
        with self.store.transaction():
            item = self.session(sid)
            self.check(item)
            if item["role"] != "label":
                reject("仅标注任务可记录帧判定", 400)
            return self.store.record_label_frames(job_id=item["job_id"], operator_id=self.operator, lease_owner=item["owner"],
                frames=body.get("frames"), decision=body.get("decision"))

    def validate(self, item, result):
        task = correction_task_from_backend_payload(item["payload"], mounts=self.mounts)
        frames, cameras = set(task.frames), set(task.cameras)
        progress = result.get("confirmed" if item["role"] == "label" else "reviewed", [])
        if (not isinstance(progress, list) or any(type(f) is not int for f in progress)
                or not set(progress).issubset(frames) or len(progress) != len(set(progress))):
            reject("确认帧格式错误或超出任务范围", 400)
        if item["role"] == "label":
            no_error = result.get("no_error_frames", [])
            if (not isinstance(no_error, list) or any(type(f) is not int for f in no_error)
                    or not set(no_error).issubset(set(progress))):
                reject("无错误帧必须属于已确认的标注帧", 400)
            if set(result.get("confirmed", [])) != frames:
                reject("请确认所有标注帧", 400)
            samples = result.get("samples", {})
            expected = {f"{f}:{c}" for f in frames for c in cameras}
            if set(samples) != expected:
                reject("标注帧或相机不完整／超出任务范围", 400)
            for sample in samples.values():
                points = np.asarray(sample.get("points"), dtype=float)
                visible = np.asarray(sample.get("visible"))
                if points.shape != (2, 21, 2) or not np.isfinite(points).all() or np.abs(points).max() > 1e7:
                    reject("关节坐标格式错误", 400)
                if visible.shape != (2, 21) or not np.isin(visible, [0, 1]).all():
                    reject("关节可见性格式错误", 400)
        else:
            if not isinstance(result.get("bad_episode", False), bool):
                reject("整段异常格式错误", 400)
            if "playback_complete" in result and type(result["playback_complete"]) is not bool:
                reject("播放完成状态无效", 400)
            complete = result.get("playback_complete") if "playback_complete" in result else frames.issubset(set(result.get("reviewed", [])))
            if not result.get("bad_episode") and not complete:
                reject("当前 Episode 尚未完成一次播放，暂不能提交", 400)
            for field in ("bad_ranges", "ego_ranges"):
                ranges = result.get(field, [])
                if not isinstance(ranges, list):
                    reject("区间格式错误", 400)
                for pair in ranges:
                    if (not isinstance(pair, list) or len(pair) != 2 or
                        any(type(f) is not int for f in pair) or pair[0] > pair[1] or
                        pair[1] - pair[0] > len(frames) or not set(range(pair[0], pair[1]+1)).issubset(frames)):
                        reject("质检区间超出任务范围", 400)
                segment_field = "bad_segments" if field == "bad_ranges" else "ego_segments"
                if segment_field not in result:
                    continue  # Old drafts predate primary-camera selection.
                segments = result[segment_field]
                if not isinstance(segments, list):
                    reject("主要错误视角区间格式错误", 400)
                for segment in segments:
                    if not isinstance(segment, dict):
                        reject("主要错误视角区间格式错误", 400)
                    a, b = segment.get("start_frame"), segment.get("end_frame")
                    if (type(a) is not int or type(b) is not int or a > b or b - a > len(frames)
                        or not set(range(a, b + 1)).issubset(frames)):
                        reject("质检区间超出任务范围", 400)
                    if segment.get("primary_camera") and segment["primary_camera"] not in task.cameras + ["ego"]:
                        reject("主要错误视角无效", 400)
                if normalize_ranges([(s["start_frame"], s["end_frame"]) for s in segments]) != normalize_ranges(ranges):
                    reject("主要错误视角与坏帧区间不一致", 400)
        return task

    def submit(self, sid, body):
        # Receipt, job transition and downstream queue insertion commit together.
        # A lost HTTP acknowledgement can be retried with the same immutable body.
        submission_id = body.get("submission_id")
        if not isinstance(submission_id, str) or not 8 <= len(submission_id) <= 100:
            reject("缺少提交编号", 400)
        checksum = digest(body)
        with ExitStack() as publications, self.store.transaction() as conn:
            item = self.session(sid)
            old = conn.execute("SELECT * FROM browser_receipts WHERE session_id=?", (sid,)).fetchone()
            if old:
                if old["submission_id"] != submission_id or old["digest"] != checksum:
                    reject("本任务已有另一份提交；保留草稿并核对回执")
                return json.loads(old["receipt"])
            self.check(item)
            if body.get("revision") != item["revision"]:
                reject("提交版本与领取版本不一致")
            result = body.get("result")
            if not isinstance(result, dict):
                reject("结果格式错误", 400)
            task = self.validate(item, result)
            root, payload = task.episode_dir(), item["payload"]
            # Older deployed desktops encode hidden joints as (-1, -1). Keep
            # their downstream consumers compatible without modifying label/.
            legacy_visibility = not hasattr(task, "correction_visibility_dir")
            visibility_dir = getattr(task, "correction_visibility_dir",
                                     f"manual_joints_vis/segments/{item['job_id']}")
            paths = [root / "workflow/final_3d_sources.json"]
            if item["role"] == "label":
                paths.extend(root / directory / camera / f"{frame:05d}.npy"
                             for directory in (task.correction_dir, visibility_dir)
                             for frame in task.frames for camera in task.cameras)
            else:
                paths.extend([root / "qc/qc_report.json", root / "ego/ego_pose_qc.json"])
            publications.enter_context(preserve_files(paths))
            artifacts = []
            if item["role"] == "label":
                for key, sample in result["samples"].items():
                    frame, camera = key.split(":")
                    points = np.asarray(sample["points"], dtype=np.float32).copy()
                    if legacy_visibility:
                        points[~np.asarray(sample["visible"], dtype=bool)] = -1
                    for directory, value, dtype in (
                        (task.correction_dir, points, np.float32),
                        (visibility_dir, sample["visible"], np.uint8),
                    ):
                        path = root / directory / camera / f"{int(frame):05d}.npy"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temp = path.with_name(path.name + "." + secrets.token_hex(4) + ".tmp")
                        try:
                            with temp.open("wb") as stream:
                                np.save(stream, np.asarray(value, dtype=dtype), allow_pickle=False)
                            temp.replace(path)
                        finally:
                            temp.unlink(missing_ok=True)
                for kind, directory in (("manual_2d", task.correction_dir), ("manual_joints_vis", visibility_dir)):
                    if legacy_visibility and kind == "manual_joints_vis":
                        continue  # Legacy backends do not recognize this artifact kind.
                    artifacts.append(dict(kind=kind, uri=payload["episode_uri"].rstrip("/")+"/"+directory,
                                          metadata=dict(scope="episode", frames=task.frames, cameras=task.cameras,
                                                        operator_id=self.operator)))
                no_error = set(result.get("no_error_frames", []))
                for decision, frames in (("no_error", sorted(no_error)),
                                         ("corrected", sorted(set(task.frames) - no_error))):
                    if frames:
                        self.store.record_label_frames(job_id=item["job_id"], operator_id=self.operator, lease_owner=item["owner"],
                                                       frames=frames, decision=decision)
                completed_result = dict(operator_id=self.operator, frames_completed=task.frames,
                                        no_error_frames=sorted(no_error))
            else:
                bad_ranges = normalize_ranges(result.get("bad_ranges", []), max_gap_frames=5)
                ego_ranges = normalize_ranges(result.get("ego_ranges", []), max_gap_frames=5)
                completed_result = build_qc_result(episode_id=payload["episode_id"], worker_id=self.operator,
                    bad_ranges=bad_ranges, bad_episode=result.get("bad_episode", False),
                    segments=normalize_segments(result["bad_segments"]) if "bad_segments" in result else None)
                completed_result["operator_id"] = self.operator
                write_qc_report(episode_dir=root, result=completed_result, bad_ranges=[] if result.get("bad_episode") else bad_ranges, sample_interval=10)
                write_ego_pose_qc_report(episode_dir=root, episode_id=payload["episode_id"], worker_id=self.operator,
                    operator_id=self.operator, bad_ranges=ego_ranges,
                    segments=normalize_segments(result["ego_segments"]) if "ego_segments" in result else None)
                artifacts.append(dict(kind="qc_report", uri=payload["episode_uri"].rstrip("/")+"/qc/qc_report.json"))
            self.service.complete_job(item["job_id"], dict(result=completed_result, artifacts=artifacts))
            receipt = dict(accepted=True, session_id=sid, submission_id=submission_id,
                           job_id=item["job_id"], digest=checksum, accepted_at=now_iso())
            conn.execute("INSERT INTO browser_receipts VALUES (?,?,?,?)", (sid, submission_id, checksum, json.dumps(receipt)))
            return receipt
