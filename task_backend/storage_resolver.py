from __future__ import annotations

def uri_join(base_uri: str, *parts: str) -> str:
    base = str(base_uri or "").rstrip("/")
    clean_parts = [str(part).strip("/") for part in parts if str(part or "").strip("/")]
    if not clean_parts:
        return base
    return base + "/" + "/".join(clean_parts)
