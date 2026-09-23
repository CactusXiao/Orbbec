"""One CFR video carries matched RGB/composited planes; no dual-player drift."""
from pathlib import Path
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from PIL import Image, ImageOps

WIDTH, PLANE_HEIGHT = 1536, 864
RESAMPLING = getattr(Image, 'Resampling', Image)

def layout(cameras, profile='legacy'):
    if profile != 'legacy':
        if profile not in ('detail960', 'detail1280'):
            raise ValueError('Unknown QC encoding profile')
        rgb = [c for c in cameras if c != 'ego']
        ego_width = 960 if profile == 'detail960' else 1280
        ego_height = ego_width * 3 // 4
        left_width = 1280 if rgb else 0
        plane = max(((len(rgb)+1)//2)*400, ego_height if 'ego' in cameras else 0)
        plane = (plane+15)//16*16
        width = left_width + (ego_width if 'ego' in cameras else 0)
        tiles = []
        for c in cameras:
            if c == 'ego':
                box = [left_width, (plane-ego_height)//2, ego_width, ego_height]
            else:
                i = rgb.index(c)
                box = [i%2*640, i//2*400, 640, 400]
            tiles.append(dict(camera=c, raw=box, rendered=[box[0],box[1]+plane,*box[2:]]))
        return dict(version=2, profile=profile, width=width, height=plane*2, tiles=tiles)
    return {'version': 1, 'width': WIDTH, 'height': PLANE_HEIGHT * 2,
        'tiles': [{'camera': c, 'raw': [i % 3 * 512, i // 3 * 432 + 24, 512, 408],
                   'rendered': [i % 3 * 512, i // 3 * 432 + 24 + PLANE_HEIGHT, 512, 408]}
                  for i, c in enumerate(cameras)]}

def content_layout(cameras, paths, profile='legacy'):
    """Describe image pixels only, excluding padding inside packed video cells.

    Match Pillow's own contain rounding on the encoder host without decoding RGB.
    This also supplies crop metadata for previously prepared video caches.
    """
    result = layout(cameras, profile)
    result['content_bounds'] = True
    for part in result['tiles']:
        with Image.open(paths[part['camera']]) as source:
            box = tuple(part['raw'][2:])
            source.draft('RGB', box)
            with Image.new('1', source.size) as shape:
                with ImageOps.contain(shape, box) as fitted:
                    width, height = fitted.size
        dx, dy = (box[0]-width)//2, (box[1]-height)//2
        for key in ('raw', 'rendered'):
            x, y, _, _ = part[key]
            part[key] = [x+dx, y+dy, width, height]
    return result

def raw_frame(cache, camera, frame):
    for ext in ('jpg', 'png'):
        p = Path(cache) / camera / f'{frame:05d}.{ext}'
        if p.is_file(): return p
    raise FileNotFoundError(f'RGB not ready: {camera}/{frame}')

def pack_frame(cache, cameras, frame, profile='legacy'):
    geometry = layout(cameras, profile)
    canvas = Image.new('RGB', (geometry['width'], geometry['height']), '#202d3e')
    for part in geometry['tiles']:
        camera = part['camera']
        with Image.open(raw_frame(cache, camera, frame)) as raw, Image.open(Path(cache)/'mesh'/camera/f'{frame:05d}.jpg') as rendered:
            if raw.size != rendered.size:
                raise ValueError(f'Raw/MANO dimensions differ: {camera}/{frame}')
            for key, source in zip(('raw','rendered'), (raw, rendered)):
                # Decode JPEG at preview scale; full resolution copies remain available for explicit inspection.
                x, y, w, h = part[key]
                source.draft('RGB', (w,h))
                tile = ImageOps.contain(source.convert('RGB'), (w,h),
                    method=RESAMPLING.BICUBIC if profile=='legacy' else RESAMPLING.LANCZOS)
                canvas.paste(tile, (x+(w-tile.width)//2, y+(h-tile.height)//2))
                tile.close()
    return canvas

def encode_layered(cache, output, frames, cameras, fps=30, bitrate=None, *, profile='legacy', encoder='libx264'):
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    geometry = layout(cameras, profile)
    bitrate = bitrate or ('6M' if profile=='legacy' else '7M')
    if encoder not in ('libx264', 'h264_nvenc'):
        raise ValueError('Unsupported QC video encoder')
    if encoder == 'h264_nvenc':
        encoding = ['-c:v',encoder,'-preset','p7','-tune','hq','-rc','vbr','-cq','19',
                    '-b:v','0','-multipass','fullres','-spatial-aq','1','-temporal-aq','1']
    else:
        encoding = ['-c:v',encoder,'-preset','veryfast' if profile=='legacy' else 'medium','-threads','4']
        encoding += ['-b:v',bitrate] if profile=='legacy' else ['-crf','19']
    level = '5.0' if profile=='legacy' else '5.1'
    command = ['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
        '-pixel_format','rgb24','-video_size',f"{geometry['width']}x{geometry['height']}",
        '-framerate',str(fps),'-i','pipe:0','-an',*encoding,
        '-pix_fmt','yuv420p','-bf','0','-maxrate',bitrate,
        '-bufsize',bitrate,'-g',str(fps),'-keyint_min',str(fps),'-sc_threshold','0',
        '-profile:v','high','-level:v',level,'-movflags',
        '+frag_keyframe+empty_moov+default_base_moof',str(output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        try:
            # Pillow releases the GIL during decode/resize. Bound in-flight frames
            # so long episodes never accumulate raw RGB images in RAM.
            with ThreadPoolExecutor(max_workers=4) as workers:
                pending, remaining = deque(), iter(frames)
                for frame in list(frames)[:4]:
                    next(remaining)
                    pending.append(workers.submit(pack_frame, cache, cameras, frame, profile))
                while pending:
                    image = pending.popleft().result()
                    process.stdin.write(image.tobytes())
                    image.close()
                    frame = next(remaining, None)
                    if frame is not None: pending.append(workers.submit(pack_frame, cache, cameras, frame, profile))
        finally: process.stdin.close()
        if process.wait() != 0: raise RuntimeError('Layered video encoding failed')
    info = dict(frames=len(frames), bytes=output.stat().st_size, encode_seconds=time.monotonic()-started,
                fps=fps, codec='avc1.640032' if level=='5.0' else 'avc1.640033',
                layout=content_layout(cameras, {c: raw_frame(cache, c, frames[0]) for c in cameras}, profile),
                target_bitrate=bitrate, profile=profile, encoder=encoder)
    output.with_suffix('.json').write_text(json.dumps(info))
    return info
