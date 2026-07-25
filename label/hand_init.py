from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


Point = Tuple[float, float]
HandPoints = List[List[Point]]

_HAND_COUNT = 2
_JOINT_COUNT = 21


class MediaPipeHandInitializer:
    def __init__(self) -> None:
        self._hands = None

    def ensure_available(self) -> None:
        self._ensure_hands()

    def close(self) -> None:
        if self._hands is None:
            return
        try:
            self._hands.close()
        except Exception:
            pass
        self._hands = None

    def points_from_image(self, image_path: Optional[Path]) -> HandPoints:
        points = self.empty_points()
        if image_path is None or not image_path.exists() or not image_path.is_file():
            return points

        hands = self._ensure_hands()
        try:
            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
                width, height = rgb.size
                image = np.asarray(rgb)
            result = hands.process(image)
        except Exception:
            return points

        landmarks = getattr(result, "multi_hand_landmarks", None) or []
        handedness = getattr(result, "multi_handedness", None) or []
        scores = [-1.0 for _ in range(_HAND_COUNT)]
        for hand_landmarks, hand_info in zip(landmarks, handedness):
            hand_idx = self._hand_index(hand_info)
            if hand_idx is None:
                continue
            score = self._hand_score(hand_info)
            if score < scores[hand_idx]:
                continue
            scores[hand_idx] = score
            detected = list(hand_landmarks.landmark)[:_JOINT_COUNT]
            points[hand_idx] = [
                (
                    self._clamp(float(lm.x) * width, width),
                    self._clamp(float(lm.y) * height, height),
                )
                for lm in detected
            ]
            if len(points[hand_idx]) < _JOINT_COUNT:
                points[hand_idx].extend([(0.0, 0.0) for _ in range(_JOINT_COUNT - len(points[hand_idx]))])
        return points

    @staticmethod
    def empty_points() -> HandPoints:
        return [[(0.0, 0.0) for _ in range(_JOINT_COUNT)] for _ in range(_HAND_COUNT)]

    def _ensure_hands(self):
        if self._hands is not None:
            return self._hands
        try:
            import mediapipe as mp
        except Exception as exc:
            raise RuntimeError("Scratch mode requires `mediapipe`. Install dependencies with `pip install -r requirements.txt`.") from exc

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
        )
        return self._hands

    @staticmethod
    def _hand_index(hand_info) -> Optional[int]:
        classes = getattr(hand_info, "classification", None) or []
        if not classes:
            return None
        label = (getattr(classes[0], "label", "") or "").strip().lower()
        # MediaPipe handedness is for mirrored/selfie input. The label images here are not mirrored,
        # so swap labels before returning the app's canonical order: hand 0 left, hand 1 right.
        if label == "left":
            return 1
        if label == "right":
            return 0
        return None

    @staticmethod
    def _hand_score(hand_info) -> float:
        classes = getattr(hand_info, "classification", None) or []
        if not classes:
            return 0.0
        try:
            return float(getattr(classes[0], "score", 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _clamp(value: float, limit: int) -> float:
        return max(0.0, min(float(max(0, limit - 1)), float(value)))
