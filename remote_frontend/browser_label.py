"""Browser Label camera contract, including timestamp-aligned Ego RGB."""
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import numpy as np
from label.storage import CorrectionTask, correction_task_from_backend_payload, find_frame_path
from label.video_frames import ensure_decoded_rgb_frames
from label.mano_view import ManoViewRuntime, CameraParams, load_episode_cameras, triangulate_hands


class BrowserCorrectionTask(CorrectionTask):
    def __post_init__(self):
        super().__post_init__()
        if (self.episode_dir() / 'ego' / 'RGB').is_dir():
            object.__setattr__(self, 'cameras', [*self.cameras, 'ego'])


def browser_task(payload, *, mounts, role='label'):
    task = correction_task_from_backend_payload(payload, mounts=mounts)
    return BrowserCorrectionTask(**asdict(task)) if role == 'label' else task


def decode_label(task, payload, *, cache_root, stop_event=None):
    target = Path(cache_root) / 'aligned'
    prepared = replace(task, rgb_path_template=str(target.resolve() / '{camera}' / '{frame:05d}.jpg'))
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError('Label decoding stopped')
    # The session already prepares full-resolution, reference-aligned frames.
    # In particular, rebuilding EgoPreview seeks through the H.265 video again.
    # Reuse these exact inputs for subsequent tracking and mesh requests.
    if all((target / c / f'{f:05d}.jpg').is_file() for c in task.cameras for f in task.frames):
        return prepared
    # Decode fixed cameras separately: Ego uses reference -> ego_frame_index.
    fixed = CorrectionTask(**asdict(task))
    if any(not (target / c / f'{f:05d}.jpg').is_file() for c in fixed.cameras for f in task.frames):
        decoded = ensure_decoded_rgb_frames(fixed, payload, cache_root=cache_root, stop_event=stop_event)
        for camera in fixed.cameras:
            (target / camera).mkdir(parents=True, exist_ok=True)
            for frame in task.frames:
                source = find_frame_path(decoded.episode_dir(), camera, frame, decoded.rgb_path_template)
                if source is None:
                    raise ValueError(f'Missing RGB frame {camera}/{frame}')
                dest = target / camera / f'{frame:05d}.jpg'
                if not dest.is_file():
                    dest.unlink(missing_ok=True)  # Repair a dangling cache symlink.
                    dest.symlink_to(source.resolve())
    missing_ego = [f for f in task.frames if not (target / 'ego' / f'{f:05d}.jpg').is_file()]
    if 'ego' in task.cameras and missing_ego:
        from label.ego_preview import EgoPreview
        preview = EgoPreview(replace(task, frames=missing_ego))
        try:
            while not preview.done_event.wait(.1):
                if stop_event is not None and stop_event.is_set():
                    raise InterruptedError('Label decoding stopped')
            if preview.error:
                raise ValueError(preview.error)
            (target / 'ego').mkdir(parents=True, exist_ok=True)
            for frame in missing_ego:
                source = preview.path(frame)
                if source is None:
                    raise ValueError(f'帧 {frame} 缺少同步 Ego RGB，无法准备标注')
                dest = target / 'ego' / f'{frame:05d}.jpg'
                temporary = dest.with_suffix('.tmp')
                shutil.copyfile(source, temporary)
                temporary.replace(dest)
        finally:
            preview.close()
    return prepared


class BrowserLabelRuntime(ManoViewRuntime):
    def __init__(self, frame=0):
        super().__init__()
        self.frame = frame
        self.ego_calibration = {}

    def ego_camera(self, episode_dir):
        from src.qc.mesh_renderer import load_ego_camera, load_ego_extrinsics
        key = str(episode_dir)
        if key not in self.ego_calibration:
            self.ego_calibration[key] = (*load_ego_camera(episode_dir), load_ego_extrinsics(episode_dir))
        k, dist, size, transforms = self.ego_calibration[key]
        transform = transforms[self.frame]
        return CameraParams(k=k.astype(float), dist=np.zeros(5), r=transform[:3,:3], t=transform[:3,3]), dist.astype(float), size

    def project_mano_frame(self, *, frame_idx, **kwargs):
        self.frame = frame_idx
        return super().project_mano_frame(frame_idx=frame_idx, **kwargs)

    def project_skeleton(self, *, episode_dir, cam_id, joints_3d):
        if cam_id != 'ego':
            return super().project_skeleton(episode_dir=episode_dir, cam_id=cam_id, joints_3d=joints_3d)
        import cv2
        cam, dist, size = self.ego_camera(episode_dir)
        xyz = np.asarray(joints_3d, dtype=float).reshape(-1, 3) @ cam.r.T + cam.t
        valid = np.isfinite(xyz).all(axis=1) & (xyz[:,2] > 0)
        uv = np.full((42,2), -1.)
        if valid.any():
            uv[valid] = cv2.fisheye.projectPoints(xyz[valid].reshape(-1,1,3), np.zeros(3), np.zeros(3), cam.k, dist)[0].reshape(-1,2)
        valid &= (uv[:,0] >= 0) & (uv[:,0] < size[0]) & (uv[:,1] >= 0) & (uv[:,1] < size[1])
        return uv.reshape(2,21,2).tolist(), valid.reshape(2,21).tolist()

    def build_skeleton(self, *, episode_dir, camera_ids, view_states):
        if 'ego' not in camera_ids:
            return super().build_skeleton(episode_dir=episode_dir, camera_ids=camera_ids, view_states=view_states)
        import cv2
        cameras = load_episode_cameras(episode_dir, [c for c in camera_ids if c != 'ego'])
        cam, dist, _ = self.ego_camera(episode_dir)
        points, visible = view_states['ego']
        # Convert fisheye pixels into pinhole pixels before common triangulation.
        uv = cv2.fisheye.undistortPoints(np.asarray(points, dtype=float).reshape(-1,1,2), cam.k, dist, P=cam.k)
        cameras['ego'] = cam
        return triangulate_hands(cameras, {**view_states, 'ego': (uv.reshape(2,21,2).tolist(), visible)})
