from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse


def local_uri_from_path(path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return "local://" + str(p.resolve())


def path_from_local_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme != "local":
        raise ValueError(f"not a local URI: {uri}")
    if parsed.netloc and parsed.path:
        path = "/" + parsed.netloc + parsed.path
    elif parsed.netloc:
        path = parsed.netloc
    else:
        path = parsed.path
    return str(Path(unquote(path)).expanduser())


def uri_join(base_uri: str, *parts: str) -> str:
    base = str(base_uri or "").rstrip("/")
    clean_parts = [str(part).strip("/") for part in parts if str(part or "").strip("/")]
    if not clean_parts:
        return base
    return base + "/" + "/".join(clean_parts)
