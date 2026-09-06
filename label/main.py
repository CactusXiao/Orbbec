from __future__ import annotations

def _ensure_utf8_env() -> None:
    import locale
    import os
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    try:
        if locale.getpreferredencoding(False).lower() != "utf-8":
            os.environ.setdefault("LC_ALL", "C.UTF-8")
            os.environ.setdefault("LANG", "C.UTF-8")
            locale.setlocale(locale.LC_ALL, "")
    except Exception:
        pass


def main() -> None:
    _ensure_utf8_env()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the Label launch config JSON.")
    args = parser.parse_args()

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from frontend_runtime import SingleInstance

    with SingleInstance("label") as instance:
        if not instance.acquired:
            return
        try:
            from .app import LabelToolApp
            from .env_config import load_label_config
        except Exception:
            from app import LabelToolApp
            from env_config import load_label_config

        app = LabelToolApp(load_label_config(args.config))
        instance.attach(app)
        app.mainloop()


if __name__ == "__main__":
    main()
