"""Thin HTTP service over the transcription pipeline.

    uvicorn api:app --port 8000

Long transcriptions don't fit a request/response cycle, so the API is async:
POST returns a job id immediately, the client polls GET for status/result.
Job state lives in an in-process dict and work runs on a small thread pool --
deliberately minimal for this exercise; in production the store would be
Redis/Postgres and the workers a proper queue (Celery, SQS...) so jobs
survive restarts and scale horizontally.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile

from pipeline import chunker, ingest
from pipeline.transcriber import Transcriber

CHUNK_THRESHOLD_S = 600.0

app = FastAPI(title="Transcription Pipeline")

_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=1)  # one file at a time; chunker parallelizes within it
_transcriber: Transcriber | None = None


def _get_transcriber() -> Transcriber:
    # lazy singleton: model load takes seconds, so do it once and reuse
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber(num_workers=3)
    return _transcriber


def _process(job_id: str, path: Path, duration: float) -> None:
    job = _jobs[job_id]
    job["status"] = "processing"
    try:
        wav = ingest.normalize(path)
        transcriber = _get_transcriber()
        if duration > CHUNK_THRESHOLD_S:
            result = chunker.transcribe_chunked(transcriber, wav, duration)
        else:
            result = transcriber.transcribe(wav)
        result["text"] = " ".join(seg["text"] for seg in result["segments"])
        job.update(status="done", result=result)
    except Exception as exc:
        job.update(status="failed", error=str(exc))
    finally:
        shutil.rmtree(path.parent, ignore_errors=True)


@app.post("/transcriptions", status_code=202)
async def create_transcription(file: UploadFile):
    tmp_dir = Path(tempfile.mkdtemp(prefix="stt_upload_"))
    dest = tmp_dir / (file.filename or "upload")
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        meta = ingest.probe(dest)  # reject non-audio before accepting the job
    except ingest.IngestError as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc))

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "queued", "file": file.filename, "audio": meta}
    _executor.submit(_process, job_id, dest, meta["duration"])
    return {"job_id": job_id, "status": "queued", "status_url": f"/transcriptions/{job_id}"}


@app.get("/transcriptions/{job_id}")
async def get_transcription(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    return job
