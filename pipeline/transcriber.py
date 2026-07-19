"""Speech-to-text via faster-whisper (CTranslate2 port of OpenAI Whisper).

Chosen over the reference openai-whisper package for ~4x faster CPU inference
with int8 quantization, and over hosted APIs so the pipeline runs fully
offline with no per-minute cost and no upload size limits.
"""

from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size: str = "small", device: str = "cpu",
                 compute_type: str = "int8", num_workers: int = 1,
                 cpu_threads: int = 3):
        # small/int8 runs ~2x realtime on a laptop CPU, which is enough here.
        # On a CUDA box: device="cuda", compute_type="float16", model "large-v3".
        # num_workers > 1 lets concurrent transcribe() calls run in parallel
        # instead of queueing on one worker (used by the chunked path).
        # cpu_threads is capped per worker: unbounded workers * threads
        # oversubscribes every logical CPU and can freeze the host machine.
        self.model = WhisperModel(model_size, device=device,
                                  compute_type=compute_type,
                                  num_workers=num_workers, cpu_threads=cpu_threads)

    def transcribe(self, wav_path: str | Path, offset: float = 0.0,
                   language: str | None = None) -> dict:
        """Transcribe a normalized WAV and return timestamped segments.

        ``offset`` shifts every timestamp by the chunk's position within the
        original recording -- this is what keeps timestamps globally correct
        when a long file is split into chunks and each chunk starts at 0.
        """
        segments, info = self.model.transcribe(
            str(wav_path),
            language=language,
            vad_filter=True,  # skip non-speech regions; big win on real-world audio
        )
        out = []
        for seg in segments:  # lazy generator: inference happens as we iterate
            out.append({
                "id": seg.id,
                "start": round(seg.start + offset, 2),
                "end": round(seg.end + offset, 2),
                "text": seg.text.strip(),
            })
        return {"language": info.language, "duration": info.duration, "segments": out}
