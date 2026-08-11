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
    try:
        from .app import QcWorkerApp
        from .config import load_qc_config
    except Exception:
        from app import QcWorkerApp  # type: ignore
        from config import load_qc_config  # type: ignore

    app = QcWorkerApp(load_qc_config())
    app.mainloop()


if __name__ == "__main__":
    main()

