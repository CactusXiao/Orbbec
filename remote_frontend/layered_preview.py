"""One CFR video carries matched RGB/composited planes; no dual-player drift."""
from pathlib import Path
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from PIL import Image, ImageOps

WIDTH, PLANE_HEIGHT = 1536, 864

def layout(cameras):
    return {'version': 1, 'width': WIDTH, 'height': PLANE_HEIGHT * 2,
        'tiles': [{'camera': c, 'raw': [i % 3 * 512, i // 3 * 432 + 24, 512, 408],
                   'rendered': [i % 3 * 512, i // 3 * 432 + 24 + PLANE_HEIGHT, 512, 408]}
                  for i, c in enumerate(cameras)]}

def content_layout(cameras, paths):
    """Describe image pixels only, excluding padding inside packed video cells.

    Match Pillow's own contain rounding on the encoder host without decoding RGB.
    This also supplies crop metadata for previously prepared video caches.
    """
    result = layout(cameras)
    result['content_bounds'] = True
    for part in result['tiles']:
        with Image.open(paths[part['camera']]) as source:
            source.draft('RGB', (512, 408))
            with Image.new('1', source.size) as shape:
                with ImageOps.contain(shape, (512, 408)) as fitted:
                    width, height = fitted.size
        dx, dy = (512-width)//2, (408-height)//2
        for key in ('raw', 'rendered'):
            x, y, _, _ = part[key]
            part[key] = [x+dx, y+dy, width, height]
    return result

def raw_frame(cache, camera, frame):
    for ext in ('jpg', 'png'):
        p = Path(cache) / camera / f'{frame:05d}.{ext}'
        if p.is_file(): return p
    raise FileNotFoundError(f'RGB not ready: {camera}/{frame}')

def pack_frame(cache, cameras, frame):
    canvas = Image.new('RGB', (WIDTH, PLANE_HEIGHT * 2), '#202d3e')
    for i, camera in enumerate(cameras):
        with Image.open(raw_frame(cache, camera, frame)) as raw, Image.open(Path(cache)/'mesh'/camera/f'{frame:05d}.jpg') as rendered:
            if raw.size != rendered.size:
                raise ValueError(f'Raw/MANO dimensions differ: {camera}/{frame}')
            for plane, source in enumerate((raw, rendered)):
                # Decode JPEG at preview scale; full resolution copies remain available for explicit inspection.
                source.draft("RGB", (512, 408))
                tile = ImageOps.contain(source.convert('RGB'), (512, 408))
                x = i % 3 * 512 + (512-tile.width)//2
                y = i // 3 * 432 + 24 + (408-tile.height)//2 + plane*PLANE_HEIGHT
                canvas.paste(tile, (x, y))
    return canvas

def encode_layered(cache, output, frames, cameras, fps=30, bitrate='6M'):
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    command = ['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
        '-pixel_format','rgb24','-video_size',f'{WIDTH}x{PLANE_HEIGHT*2}',
        '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast',
        '-threads','4','-pix_fmt','yuv420p','-bf','0','-b:v',bitrate,'-maxrate',bitrate,
        '-bufsize',bitrate,'-g',str(fps),'-keyint_min',str(fps),'-sc_threshold','0',
        '-profile:v','high','-level:v','5.0','-movflags',
        '+frag_keyframe+empty_moov+default_base_moof',str(output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        try:
            # Pillow releases the GIL during decode/resize. Bound in-flight frames
            # so long episodes never accumulate raw RGB images in RAM.
            with ThreadPoolExecutor(max_workers=4) as workers:
                pending, remaining = deque(), iter(frames)
                for frame in list(frames)[:4]:
                    next(remaining)
                    pending.append(workers.submit(pack_frame, cache, cameras, frame))
                while pending:
                    image = pending.popleft().result()
                    process.stdin.write(image.tobytes())
                    image.close()
                    frame = next(remaining, None)
                    if frame is not None: pending.append(workers.submit(pack_frame, cache, cameras, frame))
        finally: process.stdin.close()
        if process.wait() != 0: raise RuntimeError('Layered video encoding failed')
    info = dict(frames=len(frames), bytes=output.stat().st_size, encode_seconds=time.monotonic()-started,
                fps=fps, codec='avc1.640032', layout=content_layout(cameras, {c: raw_frame(cache, c, frames[0]) for c in cameras}), target_bitrate=bitrate)
    output.with_suffix('.json').write_text(json.dumps(info))
    return info
