#!/usr/bin/env python3

import json
import math
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import mediapipe as mp
import numpy as np


HEADER = struct.Struct("<IQ")

MAX_TRACKS = 2
TRACK_TTL_FRAMES = 12
GROUP_DISTANCE_M = 0.12
TRACK_MATCH_DISTANCE_M = 0.15
MAX_JOINTS = 21
KEY_JOINTS = (0, 5, 9, 13, 17)
MIN_DETECTION_LANDMARKS = 6
AVG_REPROJ_ERROR_MAX_PX = 14.0
POINT_REPROJ_OUTLIER_MAX_PX = 20.0
SIDE_HYSTERESIS_THRESHOLD = 0.15
LANDMARK_NORMALIZED_MARGIN = 0.18


def read_exact(stream, size: int) -> Optional[bytes]:
    buf = bytearray(size)
    view = memoryview(buf)
    offset = 0
    while offset < size:
        chunk = stream.read(size - offset)
        if not chunk:
            return None
        view[offset : offset + len(chunk)] = chunk
        offset += len(chunk)
    return bytes(buf)


def read_message():
    header = read_exact(sys.stdin.buffer, HEADER.size)
    if header is None:
        return None, None
    json_size, payload_size = HEADER.unpack(header)
    meta_raw = read_exact(sys.stdin.buffer, json_size)
    if meta_raw is None:
        return None, None
    payload = b""
    if payload_size:
        payload = read_exact(sys.stdin.buffer, payload_size)
        if payload is None:
            return None, None
    return json.loads(meta_raw.decode("utf-8")), payload


def write_message(obj: Dict):
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(HEADER.pack(len(data), 0))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def normalize_handedness(label: str) -> str:
    # MediaPipe Hands assumes a mirrored selfie image, so third-person cameras need a swap.
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def opposite_side(side: str) -> str:
    return "Right" if side == "Left" else "Left"


class LowPassFilter:
    def __init__(self):
        self.initialized = False
        self.value = 0.0

    def reset(self):
        self.initialized = False
        self.value = 0.0

    def apply(self, value: float, alpha: float) -> float:
        if not self.initialized:
            self.value = value
            self.initialized = True
            return value
        self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value


class OneEuroFilter1D:
    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.03, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.last_timestamp = None
        self.last_raw = None
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.last_timestamp = None
        self.last_raw = None
        self.x_filter.reset()
        self.dx_filter.reset()

    def apply(self, timestamp_s: float, value: float) -> float:
        if self.last_timestamp is None or self.last_raw is None:
            self.last_timestamp = timestamp_s
            self.last_raw = value
            self.x_filter.apply(value, 1.0)
            self.dx_filter.apply(0.0, 1.0)
            return value

        dt = max(1e-6, timestamp_s - self.last_timestamp)
        dx = (value - self.last_raw) / dt
        edx = self.dx_filter.apply(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        filtered = self.x_filter.apply(value, self._alpha(cutoff, dt))
        self.last_timestamp = timestamp_s
        self.last_raw = value
        return filtered


class OneEuroFilterVec3:
    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.03, d_cutoff: float = 1.0):
        self.filters = [
            OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff),
            OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff),
            OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff),
        ]

    def reset(self):
        for f in self.filters:
            f.reset()

    def apply(self, timestamp_s: float, value: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self.filters[i].apply(timestamp_s, float(value[i])) for i in range(3)],
            dtype=np.float64,
        )


@dataclass
class HandInstance:
    camera_id: str
    score: float
    side_vote: Optional[str]
    projection: np.ndarray
    joints_orig: List[Optional[Tuple[float, float]]]
    joints_scaled: List[Optional[Tuple[float, float]]]
    bbox: Tuple[float, float, float, float]
    palm_center_orig: Tuple[float, float]
    palm_center_scaled: Tuple[float, float]
    palm_world: Optional[np.ndarray]


@dataclass
class HandGroup:
    instances: List[HandInstance] = field(default_factory=list)
    camera_ids: set = field(default_factory=set)
    palm_world: Optional[np.ndarray] = None
    left_vote: float = 0.0
    right_vote: float = 0.0
    preferred_track_id: Optional[int] = None

    def add(self, inst: HandInstance):
        self.instances.append(inst)
        self.camera_ids.add(inst.camera_id)
        if inst.side_vote == "Left":
            self.left_vote += inst.score
        elif inst.side_vote == "Right":
            self.right_vote += inst.score
        palm_samples = [x.palm_world for x in self.instances if x.palm_world is not None]
        self.palm_world = np.mean(np.stack(palm_samples, axis=0), axis=0) if palm_samples else None


@dataclass
class TrackState:
    track_id: int
    side_label: Optional[str] = None
    side_score: float = 0.0
    last_seen_frame: int = -100000
    visible: bool = False
    filtered_joints: List[Optional[np.ndarray]] = field(default_factory=lambda: [None] * MAX_JOINTS)
    filtered_palm_center: Optional[np.ndarray] = None
    joint_filters: List[OneEuroFilterVec3] = field(default_factory=lambda: [OneEuroFilterVec3() for _ in range(MAX_JOINTS)])
    palm_filter: OneEuroFilterVec3 = field(default_factory=OneEuroFilterVec3)

    def is_active(self, frame_index: int) -> bool:
        return (frame_index - self.last_seen_frame) <= TRACK_TTL_FRAMES

    def reset(self):
        self.side_label = None
        self.side_score = 0.0
        self.last_seen_frame = -100000
        self.visible = False
        self.filtered_joints = [None] * MAX_JOINTS
        self.filtered_palm_center = None
        for filt in self.joint_filters:
            filt.reset()
        self.palm_filter.reset()


class HandGtWorker:
    def __init__(self):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.25,
            min_tracking_confidence=0.25,
        )
        self._frame_index = 0
        self._tracks = [TrackState(track_id=i) for i in range(MAX_TRACKS)]
        self._last_time = None
        self._smoothed_fps = 0.0

    def _projection_matrix(self, cam: Dict) -> np.ndarray:
        intr = cam["intrinsic"]
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        r = np.array(cam["Rcw"], dtype=np.float64).reshape(3, 3)
        t = np.array(cam["tcw"], dtype=np.float64).reshape(3, 1)
        return k @ np.hstack((r, t))

    @staticmethod
    def _mean_xy(points: Sequence[Optional[Tuple[float, float]]]) -> Optional[np.ndarray]:
        valid_points = [np.asarray(point, dtype=np.float64) for point in points if point is not None]
        if not valid_points:
            return None
        return np.mean(np.stack(valid_points, axis=0), axis=0)

    @staticmethod
    def _fuse_world_points(points: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        if not points:
            return None
        if len(points) == 1:
            return np.asarray(points[0], dtype=np.float64)

        arr = np.asarray(points, dtype=np.float64)
        center = np.median(arr, axis=0)
        dists = np.linalg.norm(arr - center, axis=1)
        median_dist = float(np.median(dists))
        threshold = max(0.05, 2.5 * median_dist)
        inliers = arr[dists <= threshold]
        if inliers.size == 0:
            inliers = arr
        return np.mean(inliers, axis=0)

    @staticmethod
    def _project_world_to_image(world_xyz: np.ndarray, projection: np.ndarray) -> Optional[np.ndarray]:
        homog = np.asarray([world_xyz[0], world_xyz[1], world_xyz[2], 1.0], dtype=np.float64)
        proj = projection @ homog
        if abs(float(proj[2])) < 1e-8:
            return None
        return np.asarray([float(proj[0] / proj[2]), float(proj[1] / proj[2])], dtype=np.float64)

    def _track_instance_cost(self, track: TrackState, inst: HandInstance) -> Optional[float]:
        all_errors: List[float] = []
        anchor_errors: List[float] = []
        for joint_index, xyz in enumerate(track.filtered_joints):
            if xyz is None:
                continue
            pt = inst.joints_orig[joint_index]
            if pt is None:
                continue
            proj_xy = self._project_world_to_image(xyz, inst.projection)
            if proj_xy is None:
                continue
            err = float(np.linalg.norm(proj_xy - np.asarray(pt, dtype=np.float64)))
            all_errors.append(err)
            if joint_index in KEY_JOINTS:
                anchor_errors.append(err)

        if len(anchor_errors) >= 3:
            cost = float(np.median(np.asarray(anchor_errors, dtype=np.float64)))
        elif len(all_errors) >= 6:
            cost = float(np.median(np.asarray(all_errors, dtype=np.float64)))
        elif len(all_errors) >= 3:
            cost = float(np.mean(np.asarray(all_errors, dtype=np.float64))) + 10.0
        elif track.filtered_palm_center is not None:
            palm_proj = self._project_world_to_image(track.filtered_palm_center, inst.projection)
            if palm_proj is None:
                return None
            cost = float(np.linalg.norm(palm_proj - np.asarray(inst.palm_center_orig, dtype=np.float64))) + 20.0
        else:
            return None

        if inst.side_vote in ("Left", "Right") and track.side_label in ("Left", "Right") and inst.side_vote != track.side_label:
            confidence = abs(track.side_score)
            if confidence > 0.55:
                cost += 42.0
            elif confidence > 0.25:
                cost += 20.0
            else:
                cost += 8.0
        return cost

    def _detect_camera(self, cam: Dict, payload: bytes) -> List[HandInstance]:
        rgb_width = int(cam["rgb_width"])
        rgb_height = int(cam["rgb_height"])
        rgb_offset = int(cam["rgb_offset"])
        rgb_size = int(cam["rgb_size"])
        rgb_scale_x = float(cam.get("rgb_scale_x", 1.0))
        rgb_scale_y = float(cam.get("rgb_scale_y", 1.0))
        projection = self._projection_matrix(cam)

        bgr = np.frombuffer(payload, dtype=np.uint8, count=rgb_size, offset=rgb_offset).reshape((rgb_height, rgb_width, 3))
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])

        result = self._hands.process(rgb)
        landmarks_list = result.multi_hand_landmarks or []
        handedness_list = result.multi_handedness or []

        detections: List[HandInstance] = []
        for idx, hand_landmarks in enumerate(landmarks_list):
            side_vote = None
            score = 0.0
            if idx < len(handedness_list) and handedness_list[idx].classification:
                cls = handedness_list[idx].classification[0]
                side_vote = normalize_handedness(cls.label)
                score = float(cls.score)

            joints_orig: List[Optional[Tuple[float, float]]] = []
            joints_scaled: List[Optional[Tuple[float, float]]] = []
            xs = []
            ys = []
            xs_scaled = []
            ys_scaled = []
            valid_count = 0
            for lm in hand_landmarks.landmark:
                valid = (-LANDMARK_NORMALIZED_MARGIN) <= lm.x <= (1.0 + LANDMARK_NORMALIZED_MARGIN) and (-LANDMARK_NORMALIZED_MARGIN) <= lm.y <= (1.0 + LANDMARK_NORMALIZED_MARGIN)
                if valid:
                    u_scaled = float(lm.x) * float(rgb_width)
                    v_scaled = float(lm.y) * float(rgb_height)
                    u_orig = u_scaled * rgb_scale_x
                    v_orig = v_scaled * rgb_scale_y
                    joints_scaled.append((u_scaled, v_scaled))
                    joints_orig.append((u_orig, v_orig))
                    xs.append(u_orig)
                    ys.append(v_orig)
                    xs_scaled.append(u_scaled)
                    ys_scaled.append(v_scaled)
                    valid_count += 1
                else:
                    joints_scaled.append(None)
                    joints_orig.append(None)

            if valid_count < MIN_DETECTION_LANDMARKS:
                continue

            palm_scaled_xy = self._mean_xy([joints_scaled[key] for key in KEY_JOINTS if joints_scaled[key] is not None])
            palm_orig_xy = self._mean_xy([joints_orig[key] for key in KEY_JOINTS if joints_orig[key] is not None])
            if palm_scaled_xy is None or palm_orig_xy is None:
                palm_scaled_xy = np.asarray(
                    [0.5 * (min(xs_scaled) + max(xs_scaled)), 0.5 * (min(ys_scaled) + max(ys_scaled))],
                    dtype=np.float64,
                )
                palm_orig_xy = np.asarray(
                    [0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))],
                    dtype=np.float64,
                )

            detections.append(
                HandInstance(
                    camera_id=str(cam["camera_id"]),
                    score=score,
                    side_vote=side_vote,
                    projection=projection,
                    joints_orig=joints_orig,
                    joints_scaled=joints_scaled,
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    palm_center_orig=(float(palm_orig_xy[0]), float(palm_orig_xy[1])),
                    palm_center_scaled=(float(palm_scaled_xy[0]), float(palm_scaled_xy[1])),
                    palm_world=None,
                )
            )

        detections.sort(key=lambda det: det.score, reverse=True)
        return detections[:2]

    @staticmethod
    def _point_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return float("inf")
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _group_dominant_side(group: HandGroup) -> Optional[str]:
        if group.left_vote <= 1e-6 and group.right_vote <= 1e-6:
            return None
        if group.left_vote > group.right_vote:
            return "Left"
        if group.right_vote > group.left_vote:
            return "Right"
        return None

    @staticmethod
    def _sides_compatible(first: Optional[str], second: Optional[str]) -> bool:
        if first not in ("Left", "Right") or second not in ("Left", "Right"):
            return True
        return first == second

    def _build_frame_groups(self, instances: Sequence[HandInstance]) -> List[HandGroup]:
        groups: List[HandGroup] = []
        active_tracks = [
            (idx, track)
            for idx, track in enumerate(self._tracks)
            if track.is_active(self._frame_index) and track.filtered_palm_center is not None
        ]
        if active_tracks:
            groups_by_track = {}
            leftovers = []
            instances_by_camera: Dict[str, List[HandInstance]] = {}
            for inst in instances:
                instances_by_camera.setdefault(inst.camera_id, []).append(inst)

            for camera_instances in instances_by_camera.values():
                candidates = []
                for inst_idx, inst in enumerate(camera_instances):
                    bbox_w = max(1.0, float(inst.bbox[2] - inst.bbox[0]))
                    bbox_h = max(1.0, float(inst.bbox[3] - inst.bbox[1]))
                    distance_threshold = max(140.0, 1.15 * max(bbox_w, bbox_h))
                    for track_idx, track in active_tracks:
                        if not self._sides_compatible(inst.side_vote, track.side_label):
                            continue
                        cost = self._track_instance_cost(track, inst)
                        if cost is None or cost > distance_threshold:
                            continue
                        candidates.append((cost, inst_idx, track_idx))

                candidates.sort(key=lambda item: item[0])
                used_inst = set()
                used_track = set()
                for _, inst_idx, track_idx in candidates:
                    if inst_idx in used_inst or track_idx in used_track:
                        continue
                    used_inst.add(inst_idx)
                    used_track.add(track_idx)
                    inst = camera_instances[inst_idx]
                    group = groups_by_track.get(track_idx)
                    if group is None:
                        group = HandGroup(preferred_track_id=track_idx)
                        groups_by_track[track_idx] = group
                    group.add(inst)

                for inst_idx, inst in enumerate(camera_instances):
                    if inst_idx not in used_inst:
                        leftovers.append(inst)

            groups.extend(groups_by_track.values())
            instances = leftovers

        valid_instances = [inst for inst in instances if inst.palm_world is not None]
        valid_instances.sort(key=lambda inst: inst.score, reverse=True)
        for inst in valid_instances:
            best_group = None
            best_dist = float("inf")
            for group in groups:
                if inst.camera_id in group.camera_ids or group.palm_world is None:
                    continue
                if not self._sides_compatible(inst.side_vote, self._group_dominant_side(group)):
                    continue
                dist = self._point_distance(inst.palm_world, group.palm_world)
                if dist < best_dist:
                    best_dist = dist
                    best_group = group
            if best_group is not None and best_dist < GROUP_DISTANCE_M:
                best_group.add(inst)
            elif len(groups) < MAX_TRACKS:
                new_group = HandGroup()
                new_group.add(inst)
                groups.append(new_group)
        if not instances:
            return groups

        fallback_instances = sorted(instances, key=lambda inst: inst.score, reverse=True)
        for inst in fallback_instances:
            best_group = None
            best_cost = float("inf")
            for group in groups:
                if inst.camera_id in group.camera_ids:
                    continue
                group_side = self._group_dominant_side(group)
                if not self._sides_compatible(inst.side_vote, group_side):
                    continue
                if group.preferred_track_id is not None:
                    track_idx = int(group.preferred_track_id)
                    if 0 <= track_idx < len(self._tracks):
                        if not self._sides_compatible(inst.side_vote, self._tracks[track_idx].side_label):
                            continue
                        cost = self._track_instance_cost(self._tracks[track_idx], inst)
                        threshold = max(150.0, 1.2 * max(1.0, float(inst.bbox[2] - inst.bbox[0]), float(inst.bbox[3] - inst.bbox[1])))
                        if cost is None or cost > threshold:
                            continue
                        if cost < best_cost:
                            best_cost = cost
                            best_group = group
                        continue
                best_group = group
                break

            if best_group is not None:
                best_group.add(inst)
            elif len(groups) < MAX_TRACKS:
                group = HandGroup()
                group.add(inst)
                groups.append(group)
        return groups

    def _expire_stale_tracks(self):
        for track in self._tracks:
            if self._frame_index - track.last_seen_frame > TRACK_TTL_FRAMES:
                track.reset()

    def _assign_groups_to_tracks(self, groups: Sequence[HandGroup]) -> List[Tuple[int, HandGroup]]:
        assignments: List[Tuple[int, HandGroup]] = []
        pairs = []
        used_groups = set()
        used_tracks = set()
        for group_idx, group in enumerate(groups):
            if group.preferred_track_id is None:
                continue
            track_idx = int(group.preferred_track_id)
            if track_idx < 0 or track_idx >= len(self._tracks):
                continue
            if group_idx in used_groups or track_idx in used_tracks:
                continue
            if not self._tracks[track_idx].is_active(self._frame_index):
                continue
            assignments.append((track_idx, group))
            used_groups.add(group_idx)
            used_tracks.add(track_idx)

        active_track_ids = [
            idx
            for idx, track in enumerate(self._tracks)
            if track.is_active(self._frame_index) and idx not in used_tracks
        ]
        for group_idx, group in enumerate(groups):
            if group_idx in used_groups:
                continue
            group_side = self._group_dominant_side(group)
            for track_idx in active_track_ids:
                if not self._sides_compatible(group_side, self._tracks[track_idx].side_label):
                    continue
                dist = self._point_distance(group.palm_world, self._tracks[track_idx].filtered_palm_center)
                if dist < TRACK_MATCH_DISTANCE_M:
                    pairs.append((dist, group_idx, track_idx))
        pairs.sort(key=lambda item: item[0])

        for _, group_idx, track_idx in pairs:
            if group_idx in used_groups or track_idx in used_tracks:
                continue
            assignments.append((track_idx, groups[group_idx]))
            used_groups.add(group_idx)
            used_tracks.add(track_idx)

        free_tracks = [idx for idx, track in enumerate(self._tracks) if idx not in used_tracks and not track.is_active(self._frame_index)]
        for group_idx, group in enumerate(groups):
            if group_idx in used_groups or group.palm_world is None or not free_tracks:
                continue
            track_idx = free_tracks.pop(0)
            self._tracks[track_idx].reset()
            assignments.append((track_idx, group))
            used_groups.add(group_idx)
            used_tracks.add(track_idx)

        remaining_active_tracks = [idx for idx in active_track_ids if idx not in used_tracks]
        for group_idx, group in enumerate(groups):
            if group_idx in used_groups or group.palm_world is not None:
                continue

            voted_side = None
            dominant_side = self._group_dominant_side(group)
            if dominant_side is not None and abs(group.left_vote - group.right_vote) >= SIDE_HYSTERESIS_THRESHOLD:
                voted_side = dominant_side

            candidate_tracks = []
            for track_idx in remaining_active_tracks:
                track_side = self._tracks[track_idx].side_label
                if voted_side is not None and track_side in ("Left", "Right") and track_side != voted_side:
                    continue
                candidate_tracks.append(track_idx)

            if len(candidate_tracks) == 1:
                track_idx = candidate_tracks[0]
                assignments.append((track_idx, group))
                used_groups.add(group_idx)
                used_tracks.add(track_idx)
                remaining_active_tracks = [idx for idx in remaining_active_tracks if idx != track_idx]

        free_tracks = [idx for idx, track in enumerate(self._tracks) if idx not in used_tracks and not track.is_active(self._frame_index)]
        for group_idx, group in enumerate(groups):
            if group_idx in used_groups or not free_tracks:
                continue
            track_idx = free_tracks.pop(0)
            self._tracks[track_idx].reset()
            assignments.append((track_idx, group))
            used_groups.add(group_idx)
            used_tracks.add(track_idx)
        return assignments

    def _resolve_group_side(self, track: TrackState, group: HandGroup) -> Tuple[str, float]:
        left_score = group.left_vote
        right_score = group.right_vote
        total = left_score + right_score
        frame_vote = ((left_score - right_score) / total) if total > 1e-6 else 0.0
        margin = abs(frame_vote)

        if total <= 1e-6:
            track.side_score *= 0.92
        else:
            if track.side_label == "Left":
                frame_vote = max(frame_vote, -0.35)
            elif track.side_label == "Right":
                frame_vote = min(frame_vote, 0.35)
            track.side_score = max(-1.0, min(1.0, 0.88 * track.side_score + 0.55 * frame_vote))

        if track.side_label is None:
            if total <= 1e-6:
                return ("Left", margin)
            return ("Left" if frame_vote >= 0.0 else "Right", margin)

        if track.side_label == "Left":
            if track.side_score < -0.45:
                return ("Right", margin)
            return ("Left", margin)

        if track.side_score > 0.45:
            return ("Left", margin)
        return ("Right", margin)

    def _enforce_unique_sides(self, assignments: List[Dict]):
        if len(assignments) != 2:
            return
        a = assignments[0]
        b = assignments[1]
        if a["side"] != b["side"]:
            return

        prev_a = a["track"].side_label
        prev_b = b["track"].side_label
        if prev_a in ("Left", "Right") and prev_b in ("Left", "Right") and prev_a != prev_b:
            a["side"] = prev_a
            b["side"] = prev_b
            return
        if prev_a in ("Left", "Right") and prev_a != a["side"]:
            a["side"] = prev_a
            b["side"] = opposite_side(prev_a)
            return
        if prev_b in ("Left", "Right") and prev_b != b["side"]:
            b["side"] = prev_b
            a["side"] = opposite_side(prev_b)
            return

        stronger = a if a["margin"] >= b["margin"] else b
        weaker = b if stronger is a else a
        weaker["side"] = opposite_side(stronger["side"])

    def _attach_ungrouped_instances(self, assignments: List[Dict], instances: Sequence[HandInstance]):
        if not assignments:
            return

        grouped_ids = {id(inst) for item in assignments for inst in item["group"].instances}
        leftovers = [inst for inst in instances if id(inst) not in grouped_ids and inst.palm_world is None]
        leftovers.sort(key=lambda inst: inst.score, reverse=True)

        for inst in leftovers:
            candidates = []
            for item in assignments:
                group = item["group"]
                if inst.camera_id in group.camera_ids:
                    continue
                if not self._sides_compatible(inst.side_vote, self._group_dominant_side(group)):
                    continue
                track = item["track"]
                if not self._sides_compatible(inst.side_vote, track.side_label):
                    continue
                cost = self._track_instance_cost(track, inst)
                threshold = max(150.0, 1.2 * max(1.0, float(inst.bbox[2] - inst.bbox[0]), float(inst.bbox[3] - inst.bbox[1])))
                if cost is None or cost > threshold:
                    continue
                candidates.append((cost, item))

            if not candidates:
                continue

            candidates.sort(key=lambda pair: pair[0])
            if len(candidates) >= 2 and (candidates[1][0] - candidates[0][0]) < 18.0:
                continue
            candidates[0][1]["group"].add(inst)
            grouped_ids.add(id(inst))

    @staticmethod
    def _compute_residuals(xyz: np.ndarray, observations: Sequence[Tuple[float, float, np.ndarray]]) -> List[float]:
        homog = np.asarray([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
        residuals = []
        for u, v, p in observations:
            proj = p @ homog
            if abs(float(proj[2])) < 1e-8:
                return []
            pu = float(proj[0] / proj[2])
            pv = float(proj[1] / proj[2])
            residuals.append(math.hypot(pu - u, pv - v))
        return residuals

    @staticmethod
    def _triangulate_dlt(observations: Sequence[Tuple[float, float, np.ndarray]]):
        a_rows = []
        for u, v, p in observations:
            a_rows.append(u * p[2, :] - p[0, :])
            a_rows.append(v * p[2, :] - p[1, :])
        a = np.asarray(a_rows, dtype=np.float64)
        _, _, vh = np.linalg.svd(a, full_matrices=False)
        x = vh[-1, :]
        if abs(float(x[3])) < 1e-8:
            return None
        xyz = x[:3] / x[3]
        positive_depth = 0
        homog = np.asarray([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
        for _, _, p in observations:
            proj = p @ homog
            if abs(float(proj[2])) < 1e-8:
                return None
            if proj[2] > 0.0:
                positive_depth += 1
        if positive_depth < 2:
            return None
        return xyz

    def _triangulate_joint(self, observations: Sequence[Tuple[float, float, np.ndarray]]):
        if len(observations) < 2:
            return None

        xyz_all = self._triangulate_dlt(observations)
        if xyz_all is None:
            return None
        residuals = self._compute_residuals(xyz_all, observations)
        if not residuals:
            return None
        median_residual = float(np.median(np.asarray(residuals, dtype=np.float64)))
        filtered_obs = []
        for obs, residual in zip(observations, residuals):
            if residual > POINT_REPROJ_OUTLIER_MAX_PX:
                continue
            if median_residual > 1e-6 and residual > 2.5 * median_residual:
                continue
            filtered_obs.append(obs)
        if len(filtered_obs) < 2:
            return None

        xyz_refined = self._triangulate_dlt(filtered_obs)
        if xyz_refined is None:
            return None
        refined_residuals = self._compute_residuals(xyz_refined, filtered_obs)
        if not refined_residuals:
            return None
        avg_residual = float(sum(refined_residuals) / len(refined_residuals))
        if avg_residual > AVG_REPROJ_ERROR_MAX_PX:
            return None
        return xyz_refined, avg_residual

    def _build_hand_output(self, track: TrackState, group: HandGroup, side: str, timestamp_s: float) -> Dict:
        joints_out = [{"index": joint_index, "valid": False, "reproj_error_px": 0.0} for joint_index in range(MAX_JOINTS)]
        raw_positions: List[Optional[np.ndarray]] = [None] * MAX_JOINTS
        reproj_errors: List[float] = []

        for joint_index in range(MAX_JOINTS):
            observations = []
            for inst in group.instances:
                pt = inst.joints_orig[joint_index]
                if pt is None:
                    continue
                observations.append((float(pt[0]), float(pt[1]), inst.projection))

            solved = self._triangulate_joint(observations)
            if solved is None:
                continue

            xyz, reproj_error = solved
            raw_positions[joint_index] = xyz
            reproj_errors.append(reproj_error)
            joints_out[joint_index] = {
                "index": joint_index,
                "valid": True,
                "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "reproj_error_px": float(reproj_error),
            }

        valid_joint_count = 0
        avg_reproj_error = float(sum(reproj_errors) / len(reproj_errors)) if reproj_errors else float("inf")
        for joint_index, xyz in enumerate(raw_positions):
            if xyz is not None:
                filtered = track.joint_filters[joint_index].apply(timestamp_s, xyz)
                track.filtered_joints[joint_index] = filtered
                joints_out[joint_index]["xyz"] = [float(filtered[0]), float(filtered[1]), float(filtered[2])]
                valid_joint_count += 1

        visible = valid_joint_count > 0
        center_world = self._fuse_world_points([xyz for xyz in raw_positions if xyz is not None])
        if center_world is not None:
            track.filtered_palm_center = track.palm_filter.apply(timestamp_s, center_world)

        track.side_label = side
        track.last_seen_frame = self._frame_index
        track.visible = visible

        return {
            "track_id": track.track_id,
            "side": side,
            "visible": visible,
            "valid_joint_count": int(valid_joint_count),
            "avg_reproj_error_px": 0.0 if not math.isfinite(avg_reproj_error) else float(avg_reproj_error),
            "joints": joints_out,
        }

    def process_frame_batch(self, meta: Dict, payload: bytes) -> Dict:
        self._frame_index += 1
        frame_timestamp_us = int(meta.get("timestamp_us", 0))
        timestamp_s = frame_timestamp_us * 1e-6 if frame_timestamp_us > 0 else time.perf_counter()

        all_instances: List[HandInstance] = []
        for cam in meta.get("cameras", []):
            all_instances.extend(self._detect_camera(cam, payload))

        self._expire_stale_tracks()
        groups = self._build_frame_groups(all_instances)
        matched = self._assign_groups_to_tracks(groups)

        assignments = []
        for track_idx, group in matched:
            track = self._tracks[track_idx]
            side, margin = self._resolve_group_side(track, group)
            assignments.append({"track": track, "group": group, "side": side, "margin": margin})
        self._enforce_unique_sides(assignments)
        self._attach_ungrouped_instances(assignments, all_instances)

        hands = [self._build_hand_output(item["track"], item["group"], item["side"], timestamp_s) for item in assignments]
        hands.sort(key=lambda hand: int(hand.get("track_id", 0)))
        visible_hands = sum(1 for hand in hands if hand["visible"])

        now = time.perf_counter()
        if self._last_time is not None:
            dt = max(1e-6, now - self._last_time)
            instant = 1.0 / dt
            self._smoothed_fps = instant if self._smoothed_fps <= 0.0 else (0.8 * self._smoothed_fps + 0.2 * instant)
        self._last_time = now

        if visible_hands > 0:
            status = f"{visible_hands} visible hand(s)"
        elif hands:
            status = "partial hand tracking"
        else:
            status = "no hands detected"

        return {
            "ok": True,
            "frame_id": int(meta.get("frame_id", 0)),
            "timestamp_us": frame_timestamp_us,
            "fps": float(self._smoothed_fps),
            "status": status,
            "visible_hands": int(visible_hands),
            "hands": hands,
        }


def main():
    worker = HandGtWorker()
    while True:
        meta, payload = read_message()
        if meta is None:
            break
        if meta.get("type") == "shutdown":
            write_message({"ok": True, "status": "shutdown", "visible_hands": 0, "hands": []})
            break
        try:
            response = worker.process_frame_batch(meta, payload or b"")
        except Exception as exc:  # noqa: BLE001
            response = {
                "ok": False,
                "frame_id": int(meta.get("frame_id", 0)),
                "timestamp_us": int(meta.get("timestamp_us", 0)),
                "fps": 0.0,
                "status": f"worker error: {type(exc).__name__}",
                "visible_hands": 0,
                "hands": [],
            }
        write_message(response)


if __name__ == "__main__":
    main()