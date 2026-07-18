"""Long-file handling: silence-aware chunking + parallel transcription.

Whisper copes with arbitrary length by sliding a 30 s window sequentially, so
wall-clock time grows linearly with duration. For long recordings we instead:

1. detect silences with ffmpeg's silencedetect filter,
2. cut chunks at silence midpoints near a target length (never mid-word --
   fixed-interval cuts split syllables and wreck accuracy at the seams),
3. transcribe chunks on a small worker pool, retrying failed chunks
   individually so one bad chunk doesn't lose a two-hour file,
4. stitch segments back together, shifting each chunk's timestamps by its
   position in the original recording.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .ingest import FFMPEG, IngestError

CHUNK_TARGET_S = 300.0  # aim for ~5 min chunks
CHUNK_MAX_S = 480.0     # hard cap when no usable silence is found near target
SILENCE_DB = -35        # threshold for "this counts as silence"
SILENCE_MIN_S = 0.35    # pauses shorter than this aren't safe cut points

_SILENCE_START = re.compile(r"silence_start:\s*([\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([\d.]+)")


def detect_silences(wav_path: str | Path) -> list[tuple[float, float]]:
    """Return (start, end) of silent stretches, via a single fast decode pass."""
    cmd = [
        FFMPEG, "-v", "info", "-i", str(wav_path),
        "-af", f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN_S}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise IngestError(f"silence detection failed: {result.stderr.strip()[:300]}")

    starts = [float(m) for m in _SILENCE_START.findall(result.stderr)]
    ends = [float(m) for m in _SILENCE_END.findall(result.stderr)]
    return list(zip(starts, ends))


def plan_chunks(duration: float,
                silences: list[tuple[float, float]],
                target_s: float = CHUNK_TARGET_S,
                max_s: float = CHUNK_MAX_S) -> list[tuple[float, float]]:
    """Return (start, end) chunk boundaries cutting at silence midpoints.

    Walks forward from 0; for each chunk it prefers the silence midpoint
    closest to ``start + target_s``, and falls back to a hard cut at
    ``start + max_s`` only when the audio has no usable pause (e.g. music).
    """
    midpoints = [(s + e) / 2 for s, e in silences]
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while duration - start > max_s:
        ideal = start + target_s
        candidates = [m for m in midpoints if start + target_s / 2 < m <= start + max_s]
        cut = min(candidates, key=lambda m: abs(m - ideal)) if candidates else start + max_s
        chunks.append((start, cut))
        start = cut
    chunks.append((start, duration))
    return chunks


def slice_chunk(wav_path: str | Path, start: float, end: float, out_dir: Path) -> Path:
    """Losslessly extract [start, end) into its own WAV (stream copy, no re-encode)."""
    out = out_dir / f"chunk_{start:09.2f}.wav"
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(wav_path),
        "-c", "copy", str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise IngestError(f"chunk slice failed at {start:.1f}s: {result.stderr.strip()[:300]}")
    return out


def transcribe_chunked(transcriber, wav_path: str | Path, duration: float,
                       language: str | None = None, max_workers: int = 3,
                       target_s: float = CHUNK_TARGET_S,
                       max_s: float = CHUNK_MAX_S) -> dict:
    """Chunk a long normalized WAV, transcribe in parallel, stitch results."""
    silences = detect_silences(wav_path)
    bounds = plan_chunks(duration, silences, target_s=target_s, max_s=max_s)

    work_dir = Path(tempfile.mkdtemp(prefix="stt_chunks_"))
    chunk_files = [slice_chunk(wav_path, s, e, work_dir) for s, e in bounds]

    def job(chunk_file: Path, offset: float) -> dict:
        try:
            return transcriber.transcribe(chunk_file, offset=offset, language=language)
        except Exception:
            # one retry per chunk; a transient failure shouldn't sink the file
            return transcriber.transcribe(chunk_file, offset=offset, language=language)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(job, f, start) for f, (start, _end) in zip(chunk_files, bounds)]
        results = [f.result() for f in futures]  # submission order == chronological order

    segments = []
    for res in results:
        segments.extend(res["segments"])
    for i, seg in enumerate(segments, start=1):
        seg["id"] = i

    return {
        "language": results[0]["language"] if results else None,
        "duration": duration,
        "chunks": [{"start": round(s, 2), "end": round(e, 2)} for s, e in bounds],
        "segments": segments,
    }
