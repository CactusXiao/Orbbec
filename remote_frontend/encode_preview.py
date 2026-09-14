"""Encode a synchronized six-view QC cache into a browser preview.

This is an evaluation tool, not the original video or an annotation artifact.
Frame index N is preserved as video time N / fps. HQ stills remain server-side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from PIL import Image, ImageOps, ImageDraw


def encode(cache: Path, output: Path, frames: int, fps: int = 30, bitrate: str = "3M", *, frame_ids=None, camera_ids=None, fragmented=False) -> dict:
    cameras = camera_ids or ["00", "02", "03", "05", "06", "ego"]
    frame_ids = list(range(frames)) if frame_ids is None else list(frame_ids)
    if len(frame_ids) != frames or len(cameras) > 6:
        raise ValueError("invalid frame/camera mapping")
    sources = [[cache / ("mesh_preview" if c == "ego" else "mesh") / c / f"{f:05d}.jpg"
                for c in cameras] for f in frame_ids]
    missing = next((p for group in sources for p in group if not p.is_file()), None)
    if missing:
        raise ValueError(f"QC render is not ready: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", "1536x864", "-framerate", str(fps),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-threads", "4", "-pix_fmt", "yuv420p", "-bf", "0", "-b:v", bitrate, "-maxrate", bitrate,
               "-bufsize", bitrate, "-g", str(fps), "-keyint_min", str(fps), "-sc_threshold", "0",
               "-profile:v", "high", "-level:v", "4.0",
               "-movflags", "+frag_keyframe+empty_moov+default_base_moof" if fragmented else "+faststart", str(output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        try:
            for frame, group in enumerate(sources):
                mosaic = Image.new("RGB", (1536, 864), "#101722")
                for idx, path in enumerate(group):
                    with Image.open(path) as source:
                        tile = ImageOps.contain(source.convert("RGB"), (512, 408))
                    x, y = idx % 3 * 512, idx // 3 * 432
                    mosaic.paste(tile, (x + (512 - tile.width) // 2, y + 24 + (408 - tile.height) // 2))
                    ImageDraw.Draw(mosaic).text((x + 8, y + 5), f"{cameras[idx]} | frame {frame_ids[frame]}", fill="white")
                process.stdin.write(mosaic.tobytes())
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("preview encoding failed")
    seconds = time.monotonic() - started
    result = {"frames": frames, "fps": fps, "duration_seconds": frames / fps,
              "width": 1536, "height": 864, "encode_seconds": seconds,
              "encode_fps": frames / seconds, "bytes": output.stat().st_size,
              "average_mbps": output.stat().st_size * 8 / (frames / fps) / 1e6,
              "target_bitrate": bitrate, "encoder": "libx264/veryfast/4 threads"}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--bitrate", default="3M")
    a = p.parse_args()
    print(json.dumps(encode(a.cache, a.output, a.frames, bitrate=a.bitrate), indent=2))
