"""Audio ingestion.

Every file entering the pipeline goes through this module: the real
container/codec is detected with ffprobe (extensions are not trusted), then
the audio is decoded to a single canonical format -- 16 kHz mono PCM WAV --
so nothing downstream has to care what format the caller uploaded.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "ffprobe")

# Whisper-family models are trained on 16 kHz mono; anything else gets
# resampled internally anyway, so we normalize once at the boundary instead.
TARGET_SAMPLE_RATE = 16_000


class IngestError(Exception):
    """Raised when a file is missing, unreadable, or has no usable audio."""


def probe(path: str | Path) -> dict:
    """Return metadata for the first audio stream of ``path``.

    Fails fast on corrupt or non-audio input so bad files never reach the
    model (where errors are slower and harder to attribute).
    """
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"file not found: {path}")

    cmd = [
        FFPROBE, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise IngestError(f"unreadable or corrupt file: {path} ({result.stderr.strip()})")

    meta = json.loads(result.stdout)
    audio_streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise IngestError(f"no audio stream found in: {path}")

    stream = audio_streams[0]
    fmt = meta.get("format", {})
    return {
        "container": fmt.get("format_name"),
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "duration": float(fmt.get("duration") or stream.get("duration") or 0.0),
    }


def normalize(path: str | Path, out_dir: str | Path | None = None) -> Path:
    """Decode any input to the canonical 16 kHz mono PCM WAV.

    One conversion at the boundary means zero format-specific code later:
    anything ffmpeg can decode (mp3, m4a, flac, ogg, even video containers)
    just works.
    """
    path = Path(path)
    out_dir = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="stt_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / f"{path.stem}.16k.wav"

    cmd = [
        FFMPEG, "-y", "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise IngestError(f"ffmpeg failed to decode {path}: {result.stderr.strip()}")
    return out_wav
