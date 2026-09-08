"""Choose the existing tracking environment before opening the Label UI."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def tracking_python() -> Path | None:
    explicit = os.environ.get("LABEL_PYTHON")
    if not explicit and importlib.util.find_spec("torch") is not None:
        return None
    roots = [Path(sys.prefix)]
    if os.environ.get("CONDA_EXE"):
        roots.append(Path(os.environ["CONDA_EXE"]).parent.parent)
    if Path(sys.prefix).parent.name == "envs":
        roots.append(Path(sys.prefix).parent.parent)
    candidates = ([Path(explicit).expanduser()] if explicit else
                  [root / "envs" / "track" / "bin" / "python" for root in roots])
    for python in dict.fromkeys(candidates):
        if not python.is_file() or python.resolve() == Path(sys.executable).resolve():
            continue
        try:
            result = subprocess.run(
                [str(python), "-c", "import torch, numpy, tkinter, PIL"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return python
        if explicit:
            raise RuntimeError(f"LABEL_PYTHON cannot load Label dependencies: {python}\n"
                               + result.stderr.decode(errors="replace"))
    return None


def ensure_tracking_environment() -> None:
    python = tracking_python()
    if python is not None:
        print(f"Label: using tracking environment {python}", flush=True)
        entry = Path(__file__).with_name("main.py")
        os.execv(str(python), [str(python), str(entry), *sys.argv[1:]])
