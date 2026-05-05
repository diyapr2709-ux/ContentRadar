"""
task1/utils/sanitizer.py — single security layer.
Threats addressed: NUL/control chars, Unicode bidi, whitespace, size bombs, URL schemes.
Complexity: O(N) per sanitisation — single str.translate pass.
"""

from __future__ import annotations
import re
from urllib.parse import urlparse

_BANNED = (
    list(range(0x00, 0x09)) + [0x0B, 0x0C] + list(range(0x0E, 0x20))
    + [0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
       0x2066, 0x2067, 0x2068, 0x2069]
)
_TABLE = str.maketrans("", "", "".join(chr(c) for c in _BANNED))
_MULTI_SPACE   = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
MAX_FIELD_LEN = 50_000


def clean_text(value: object, max_len: int = MAX_FIELD_LEN) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.translate(_TABLE)
    value = _MULTI_SPACE.sub(" ", value)
    value = _MULTI_NEWLINE.sub("\n\n", value)
    return value.strip()[:max_len]


def clean_list(values: object, max_items: int = 10) -> list[str]:
    if not isinstance(values, list):
        return []
    return [clean_text(v) for v in values[:max_items] if v is not None]


def validate_url(url: object) -> bool:
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        p = urlparse(url.strip())
        return p.scheme in ("http", "https") and bool(p.netloc) and len(url) < 2048
    except Exception:
        return False