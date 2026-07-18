"""Transcription pipeline CLI.

    python transcribe.py samples/sample_short.wav -o out.json --srt

Accepts anything ffmpeg can decode, transcribes it with faster-whisper, and
writes timestamped segments as JSON (plus optional SRT/VTT).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pipeline import ingest, output
from pipeline.transcriber import Transcriber


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file to timestamped text.")
    parser.add_argument("input",
                        help="audio file (wav, mp3, m4a, flac... anything ffmpeg decodes)")
    parser.add_argument("-o", "--out", default=None,
                        help="output JSON path (default: <input>.json)")
    parser.add_argument("--srt", action="store_true", help="also write an SRT subtitle file")
    parser.add_argument("--vtt", action="store_true", help="also write a WebVTT file")
    parser.add_argument("--model", default="small",
                        help="whisper model size: tiny/base/small/medium/large-v3 (default: small)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--language", default=None,
                        help="force language code e.g. 'en' (default: auto-detect)")
    args = parser.parse_args()

    src = Path(args.input)
    try:
        meta = ingest.probe(src)
    except ingest.IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"input: {src.name} | container={meta['container']} codec={meta['codec']} "
          f"{meta['sample_rate']} Hz x{meta['channels']}ch | {meta['duration']:.1f}s")

    wav = ingest.normalize(src)
    transcriber = Transcriber(model_size=args.model, device=args.device)

    t0 = time.time()
    result = transcriber.transcribe(wav, language=args.language)
    elapsed = time.time() - t0

    result["source"] = {"file": src.name, **meta}
    result["text"] = " ".join(seg["text"] for seg in result["segments"])

    out_json = Path(args.out) if args.out else src.with_suffix(".json")
    output.write_json(result, out_json)
    print(f"wrote {out_json}")
    if args.srt:
        srt_path = out_json.with_suffix(".srt")
        output.write_srt(result, srt_path)
        print(f"wrote {srt_path}")
    if args.vtt:
        vtt_path = out_json.with_suffix(".vtt")
        output.write_vtt(result, vtt_path)
        print(f"wrote {vtt_path}")

    speed = meta["duration"] / max(elapsed, 0.01)
    print(f"{len(result['segments'])} segments | {meta['duration']:.1f}s audio "
          f"in {elapsed:.1f}s ({speed:.1f}x realtime)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
