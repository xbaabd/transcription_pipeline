"""Serializers for downstream consumers: JSON for services, SRT/VTT for players."""

from __future__ import annotations

import json
from pathlib import Path


def write_json(result: dict, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _ts(seconds: float, ms_sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}{ms_sep}{ms:03d}"


def write_srt(result: dict, path: str | Path) -> None:
    lines: list[str] = []
    for i, seg in enumerate(result["segments"], start=1):
        lines += [str(i), f"{_ts(seg['start'], ',')} --> {_ts(seg['end'], ',')}", seg["text"], ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_vtt(result: dict, path: str | Path) -> None:
    lines = ["WEBVTT", ""]
    for seg in result["segments"]:
        lines += [f"{_ts(seg['start'], '.')} --> {_ts(seg['end'], '.')}", seg["text"], ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
