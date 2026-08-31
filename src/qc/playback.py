from __future__ import annotations


def playback_target_position(
    *, start_position: int, elapsed_seconds: float, fps: float, last_position: int
) -> int:
    """Return the wall-clock playback position, skipping display frames when late."""
    elapsed_frames = max(
        1,
        int(max(0.0, float(elapsed_seconds)) * max(1.0, float(fps)) + 1e-9),
    )
    return min(int(last_position), int(start_position) + elapsed_frames)
