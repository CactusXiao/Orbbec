#!/usr/bin/env python3
import argparse
import datetime as _dt
import grp
import json
import os
import platform
import pwd
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path


def decode_bytes(data):
    return data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else str(data)


def tail_text(text, limit=12000):
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def run_command(args, timeout=15):
    started = time.time()
    result = {
        "cmd": args,
        "found": shutil.which(args[0]) is not None,
        "returncode": None,
        "duration_sec": None,
        "stdout": "",
        "stderr": "",
        "timeout": False,
    }
    if not result["found"]:
        result["stderr"] = f"command not found: {args[0]}"
        result["duration_sec"] = round(time.time() - started, 3)
        return result
    try:
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        result["returncode"] = proc.returncode
        result["stdout"] = decode_bytes(proc.stdout)
        result["stderr"] = decode_bytes(proc.stderr)
    except subprocess.TimeoutExpired as exc:
        result["timeout"] = True
        result["stdout"] = decode_bytes(exc.stdout or b"")
        result["stderr"] = decode_bytes(exc.stderr or b"") + f"\nTIMEOUT after {timeout}s"
    except Exception as exc:
        result["stderr"] = repr(exc)
    result["duration_sec"] = round(time.time() - started, 3)
    return result


def read_text_file(path, limit=200000):
    p = Path(path)
    try:
        data = p.read_bytes()
    except Exception as exc:
        return {"path": str(p), "ok": False, "error": repr(exc)}
    if len(data) > limit:
        data = data[:limit] + b"\n...TRUNCATED...\n"
    return {"path": str(p), "ok": True, "text": decode_bytes(data)}


def username(uid):
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def groupname(gid):
    try:
        return grp.getgrgid(gid).gr_name
    except KeyError:
        return str(gid)


def current_user_info():
    groups = []
    for gid in os.getgroups():
        groups.append({"gid": gid, "name": groupname(gid)})
    return {
        "uid": os.getuid(),
        "user": username(os.getuid()),
        "euid": os.geteuid(),
        "effective_user": username(os.geteuid()),
        "groups": groups,
        "id_command": run_command(["id"], timeout=5),
    }


def render_nodes_info():
    nodes = []
    dri = Path("/dev/dri")
    by_path = dri / "by-path"
    symlink_targets = {}
    if by_path.exists():
        for item in sorted(by_path.iterdir()):
            try:
                target = (by_path / os.readlink(item)).resolve()
            except Exception:
                try:
                    target = item.resolve()
                except Exception:
                    target = None
            if target:
                symlink_targets.setdefault(str(target), []).append(str(item))

    for node in sorted(dri.glob("renderD*")) if dri.exists() else []:
        try:
            st = node.stat()
            mode = stat.filemode(st.st_mode)
            info = {
                "path": str(node),
                "mode": mode,
                "uid": st.st_uid,
                "user": username(st.st_uid),
                "gid": st.st_gid,
                "group": groupname(st.st_gid),
                "readable": os.access(node, os.R_OK),
                "writable": os.access(node, os.W_OK),
                "by_path": symlink_targets.get(str(node.resolve()), []),
            }
        except Exception as exc:
            info = {"path": str(node), "error": repr(exc)}
        nodes.append(info)
    return nodes


def ffmpeg_encoders_set(encoders_text):
    encoders = set()
    for line in encoders_text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].startswith("hevc"):
            encoders.add(parts[1])
    return encoders


def make_bgr_frame(width, height, frame_index):
    row = bytearray(width * 3)
    for x in range(width):
        pos = x * 3
        row[pos + 0] = (x + frame_index * 3) & 0xFF
        row[pos + 1] = ((x >> 1) + frame_index * 5) & 0xFF
        row[pos + 2] = ((x >> 2) + frame_index * 7) & 0xFF
    return bytes(row) * height


def run_ffmpeg_pipe_test(name, cmd, output_path, width, height, frames, timeout):
    started = time.time()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.unlink()
    except FileNotFoundError:
        pass

    result = {
        "name": name,
        "cmd": cmd,
        "output_file": str(output_path),
        "returncode": None,
        "duration_sec": None,
        "stdout": "",
        "stderr": "",
        "timeout": False,
        "broken_pipe": False,
        "output_size_bytes": 0,
        "success": False,
        "ffprobe": None,
    }

    if shutil.which("ffmpeg") is None:
        result["stderr"] = "ffmpeg not found"
        result["duration_sec"] = round(time.time() - started, 3)
        return result

    proc = None
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        for frame_idx in range(frames):
            proc.stdin.write(make_bgr_frame(width, height, frame_idx))
        proc.stdin.close()
        stdout = proc.stdout.read() if proc.stdout else b""
        stderr = proc.stderr.read() if proc.stderr else b""
        proc.wait(timeout=timeout)
        result["returncode"] = proc.returncode
        result["stdout"] = decode_bytes(stdout)
        result["stderr"] = decode_bytes(stderr)
    except BrokenPipeError:
        result["broken_pipe"] = True
        if proc is not None:
            try:
                stdout, stderr = proc.communicate(timeout=3)
                result["stdout"] = decode_bytes(stdout)
                result["stderr"] = decode_bytes(stderr)
                result["returncode"] = proc.returncode
            except Exception as exc:
                result["stderr"] += "\n" + repr(exc)
    except subprocess.TimeoutExpired:
        result["timeout"] = True
        if proc is not None:
            proc.kill()
            stdout, stderr = proc.communicate()
            result["stdout"] = decode_bytes(stdout)
            result["stderr"] = decode_bytes(stderr) + f"\nTIMEOUT after {timeout}s"
            result["returncode"] = proc.returncode
    except Exception as exc:
        result["stderr"] += "\n" + repr(exc)
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    result["duration_sec"] = round(time.time() - started, 3)
    if output_path.exists():
        result["output_size_bytes"] = output_path.stat().st_size
    result["success"] = (result["returncode"] == 0 and result["output_size_bytes"] > 0)
    if result["success"] and shutil.which("ffprobe"):
        result["ffprobe"] = run_command([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_frames",
            "-of", "json",
            str(output_path),
        ], timeout=10)
    return result


def build_test_commands(args, out_dir, encoders, render_nodes):
    width_height = f"{args.width}x{args.height}"
    common = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", width_height,
        "-r", str(args.fps),
        "-i", "-",
        "-an",
    ]
    tests = []

    if "hevc_vaapi" in encoders:
        for node in render_nodes:
            dev = node.get("path")
            if not dev:
                continue
            out_file = out_dir / f"test_hevc_vaapi_{Path(dev).name}.h265"
            cmd = common + [
                "-vaapi_device", dev,
                "-vf", "format=nv12,hwupload",
                "-c:v", "hevc_vaapi",
                "-qp", str(args.quality),
                "-f", "hevc",
                str(out_file),
            ]
            tests.append((f"hevc_vaapi:{dev}", cmd, out_file))

    if "hevc_qsv" in encoders:
        out_file = out_dir / "test_hevc_qsv_current_command.h265"
        cmd = common + [
            "-vf", "format=nv12",
            "-c:v", "hevc_qsv",
            "-preset", args.preset,
            "-global_quality", str(args.quality),
            "-f", "hevc",
            str(out_file),
        ]
        tests.append(("hevc_qsv:current_collection_style", cmd, out_file))

    if args.include_v4l2m2m and "hevc_v4l2m2m" in encoders:
        out_file = out_dir / "test_hevc_v4l2m2m.h265"
        cmd = common + [
            "-vf", "format=nv12",
            "-c:v", "hevc_v4l2m2m",
            "-q:v", str(args.quality),
            "-f", "hevc",
            str(out_file),
        ]
        tests.append(("hevc_v4l2m2m", cmd, out_file))

    if args.include_nvidia and "hevc_nvenc" in encoders:
        out_file = out_dir / "test_hevc_nvenc.h265"
        cmd = common + [
            "-c:v", "hevc_nvenc",
            "-preset", args.preset,
            "-cq", str(args.quality),
            "-pix_fmt", "yuv420p",
            "-f", "hevc",
            str(out_file),
        ]
        tests.append(("hevc_nvenc", cmd, out_file))

    if args.include_software and "libx265" in encoders:
        out_file = out_dir / "test_libx265.h265"
        cmd = common + [
            "-c:v", "libx265",
            "-preset", args.preset,
            "-crf", str(args.quality),
            "-threads", str(max(1, args.software_threads)),
            "-f", "hevc",
            str(out_file),
        ]
        tests.append(("libx265:software_baseline", cmd, out_file))

    return tests


def load_current_config():
    config_path = Path("src/sync/config.json")
    if not config_path.exists():
        return None
    try:
        cfg = json.loads(config_path.read_text())
        return {"path": str(config_path), "save": cfg.get("save")}
    except Exception as exc:
        return {"path": str(config_path), "error": repr(exc)}


def recommendation_from_tests(tests):
    vaapi = [t for t in tests if t["success"] and t["name"].startswith("hevc_vaapi:")]
    qsv = [t for t in tests if t["success"] and t["name"].startswith("hevc_qsv:")]
    if vaapi:
        best = vaapi[0]
        dev = best["name"].split(":", 1)[1]
        return {
            "status": "usable",
            "preferred_path": "VAAPI",
            "reason": "hevc_vaapi succeeded with the same bgr24->nv12->hwupload style used by collection.cpp.",
            "suggested_save_config": {
                "rgbEncoding": "h265",
                "h265EncoderMode": "hardware",
                "h265Codec": "hevc_vaapi",
                "h265HwDevice": dev,
                "h265Preset": "",
                "h265Crf": 18,
            },
        }
    if qsv:
        return {
            "status": "usable",
            "preferred_path": "QSV",
            "reason": "hevc_qsv succeeded with the current collection-style command.",
            "suggested_save_config": {
                "rgbEncoding": "h265",
                "h265EncoderMode": "hardware",
                "h265Codec": "hevc_qsv",
                "h265Preset": "medium",
                "h265Crf": 18,
            },
        }
    return {
        "status": "not_ready",
        "preferred_path": None,
        "reason": "No integrated-GPU H265 encode test succeeded. Check /dev/dri permissions, ffmpeg VAAPI/QSV support, and installed VA drivers.",
    }


def write_text_report(report, path):
    rec = report.get("recommendation", {})
    lines = []
    lines.append("H265 iGPU probe report")
    lines.append("=" * 72)
    lines.append(f"created_at: {report.get('created_at')}")
    lines.append("")
    lines.append("Recommendation")
    lines.append("-" * 72)
    lines.append(json.dumps(rec, indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("Render nodes")
    lines.append("-" * 72)
    for node in report["system"].get("render_nodes", []):
        lines.append(json.dumps(node, ensure_ascii=False))
    lines.append("")
    lines.append("FFmpeg HEVC encoder tests")
    lines.append("-" * 72)
    for test in report.get("encode_tests", []):
        lines.append(f"name: {test['name']}")
        lines.append(f"success: {test['success']} rc={test['returncode']} size={test['output_size_bytes']} duration={test['duration_sec']}s")
        lines.append("cmd: " + " ".join(test["cmd"]))
        if test.get("ffprobe"):
            lines.append("ffprobe: " + tail_text(test["ffprobe"].get("stdout", ""), 2000).strip())
        if test.get("stderr"):
            lines.append("stderr_tail:")
            lines.append(tail_text(test["stderr"], 4000).strip())
        lines.append("")
    lines.append("Key command outputs are in the JSON report.")
    Path(path).write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Probe integrated-GPU H265 encoding support for the Orbbec collection pipeline.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--quality", type=int, default=23)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--software-threads", type=int, default=4)
    parser.add_argument("--include-software", action="store_true", help="Also run a short libx265 CPU baseline test.")
    parser.add_argument("--include-nvidia", action="store_true", help="Also run NVENC if present. Disabled by default.")
    parser.add_argument("--include-v4l2m2m", action="store_true", help="Also run hevc_v4l2m2m if present.")
    return parser.parse_args()


def main():
    args = parse_args()
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"h265_igpu_probe_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_encoders = run_command(["ffmpeg", "-hide_banner", "-encoders"], timeout=20)
    encoders = ffmpeg_encoders_set(ffmpeg_encoders.get("stdout", "") + "\n" + ffmpeg_encoders.get("stderr", ""))
    render_nodes = render_nodes_info()

    commands = {
        "uname": run_command(["uname", "-a"], timeout=5),
        "lscpu": run_command(["lscpu"], timeout=10),
        "lspci_display": run_command(["lspci", "-Dnnk"], timeout=10),
        "lsusb_tree": run_command(["lsusb", "-t"], timeout=10),
        "lsmod": run_command(["lsmod"], timeout=10),
        "ffmpeg_version": run_command(["ffmpeg", "-hide_banner", "-version"], timeout=10),
        "ffmpeg_hwaccels": run_command(["ffmpeg", "-hide_banner", "-hwaccels"], timeout=10),
        "ffmpeg_encoders": ffmpeg_encoders,
        "dpkg_video_packages": run_command([
            "dpkg-query", "-W",
            "vainfo",
            "intel-media-va-driver",
            "intel-media-va-driver-non-free",
            "i965-va-driver",
            "i965-va-driver-shaders",
            "mesa-va-drivers",
            "libva2",
            "libva-drm2",
            "libmfx1",
            "libvpl2",
            "onevpl-tools",
        ], timeout=10),
        "nvidia_smi": run_command(["nvidia-smi"], timeout=10),
    }

    encoder_help = {}
    for encoder in ["hevc_vaapi", "hevc_qsv", "hevc_nvenc", "hevc_v4l2m2m", "libx265"]:
        if encoder in encoders:
            encoder_help[encoder] = run_command(["ffmpeg", "-hide_banner", "-h", f"encoder={encoder}"], timeout=15)

    vainfo = {}
    for node in render_nodes:
        dev = node.get("path")
        if dev:
            vainfo[dev] = run_command(["vainfo", "--display", "drm", "--device", dev], timeout=15)

    tests = []
    for name, cmd, out_file in build_test_commands(args, out_dir, encoders, render_nodes):
        tests.append(run_ffmpeg_pipe_test(name, cmd, out_file, args.width, args.height, args.frames, args.timeout))

    report = {
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "probe_args": vars(args),
        "cwd": str(Path.cwd()),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "current_collection_h265_logic": {
            "hardware_default_codec_when_h265Codec_empty": "hevc_nvenc",
            "vaapi_command_shape": "bgr24 rawvideo input -> -vaapi_device <h265HwDevice> -vf format=nv12,hwupload -c:v hevc_vaapi -qp <h265Crf>",
            "qsv_command_shape": "bgr24 rawvideo input -> -vf format=nv12 -c:v hevc_qsv -global_quality <h265Crf>",
            "nvenc_command_shape": "bgr24 rawvideo input -> -c:v hevc_nvenc -cq <h265Crf> -pix_fmt yuv420p",
        },
        "current_config": load_current_config(),
        "system": {
            "user": current_user_info(),
            "environment": {
                key: os.environ.get(key)
                for key in ["LIBVA_DRIVER_NAME", "LIBVA_DRIVERS_PATH", "VDPAU_DRIVER", "DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE"]
            },
            "render_nodes": render_nodes,
        },
        "commands": commands,
        "encoder_help": encoder_help,
        "vainfo": vainfo,
        "available_hevc_encoders": sorted(encoders),
        "encode_tests": tests,
    }
    report["recommendation"] = recommendation_from_tests(tests)

    json_path = out_dir / "h265_igpu_probe.json"
    txt_path = out_dir / "h265_igpu_probe.txt"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_text_report(report, txt_path)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {txt_path}")
    print(json.dumps(report["recommendation"], indent=2, ensure_ascii=False))
    return 0 if report["recommendation"].get("status") == "usable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
