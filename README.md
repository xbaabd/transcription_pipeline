# Transcription Pipeline

A speech-to-text pipeline that takes any audio file, transcribes it with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), and returns
timestamped segments as JSON (plus SRT/WebVTT for players and subtitle
workflows). Runs fully offline on CPU - no hosted API, no per-minute cost,
no upload size limits.

Built as a take-home exercise; the same layering (normalize at ingest,
silence-aware chunking, async job API) is what I use in production for
post-call transcription at ConviqAI.

## Setup

Requires Python 3.10+ and ffmpeg on PATH (`winget install Gyan.FFmpeg` /
`apt install ffmpeg` / `brew install ffmpeg`).

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use bin/activate on Linux/macOS
pip install -r requirements.txt
```

The Whisper model (~460 MB for `small`) downloads automatically on first run.

## Usage

CLI:

```bash
python transcribe.py samples/sample_short.wav --srt
python transcribe.py interview.mp3 -o out.json --model medium --language en
python transcribe.py podcast.m4a --workers 2      # long files chunk automatically
```

Service:

```bash
uvicorn api:app --port 8000

curl -X POST -F file=@podcast.mp3 http://localhost:8000/transcriptions
# -> {"job_id": "...", "status": "queued", "status_url": "/transcriptions/..."}
curl http://localhost:8000/transcriptions/<job_id>
# -> {"status": "done", "result": {"language": "en", "segments": [...], ...}}
```

Output shape:

```json
{
  "language": "en",
  "duration": 10.69,
  "segments": [
    {"id": 1, "start": 0.0,  "end": 5.16, "text": "Hello, this is a short test recording..."},
    {"id": 2, "start": 5.16, "end": 9.96, "text": "It contains two sentences spoken clearly..."}
  ],
  "text": "Hello, this is a short test recording... It contains two sentences spoken clearly..."
}
```

Real outputs from the test runs below are committed under [`examples/`](examples/).

## How it works

```
input file (anything ffmpeg decodes)
  |
  v
ingest.probe      ffprobe: detect real container/codec, reject non-audio early
  |
  v
ingest.normalize  ffmpeg: decode once to canonical 16 kHz mono PCM WAV
  |
  v
<= 10 min ------------------------> > 10 min
  |                                   |
  v                                   v
single pass                         chunker: cut at silence midpoints,
faster-whisper                      transcribe chunks on a worker pool,
  |                                 shift timestamps by chunk offset, stitch
  |                                   |
  v                                   v
output: JSON / SRT / WebVTT with per-segment start/end times
```

## Design decisions

### Handling different audio formats

There is no per-format code anywhere in the pipeline. Ingest probes the file
with ffprobe to find out what it actually is (extensions lie), then decodes
it once to 16 kHz mono PCM - the format Whisper models are trained on and
would resample to internally anyway. Everything downstream sees exactly one
format. MP3, M4A/AAC, FLAC, OGG and WAV are covered by the same code path
(all five verified, see `examples/`), and corrupt or non-audio files fail
fast at the boundary with a clear error instead of deep inside the model.

### Handling long audio files

Whisper transcribes a 30-second sliding window sequentially, so wall-clock
time grows linearly and a long file would also make an HTTP request time out.
The measures taken:

- **Never load the whole file:** ffmpeg streams the decode; chunk extraction
  is a stream-copy (`-c copy`), not a re-encode.
- **Cut at silences, not at fixed intervals.** ffmpeg's `silencedetect`
  finds pauses in one fast pass; chunks aim for ~5 minutes and cut at the
  silence midpoint nearest the target. Fixed-interval cuts land mid-word and
  damage accuracy at every seam. A hard cap (8 min) covers audio with no
  usable pauses.
- **Parallelize and stitch.** Chunks transcribe on a small worker pool, each
  with an independent retry. Every chunk's timestamps start at zero, so each
  segment is shifted by its chunk's start offset before stitching - that is
  what keeps timestamps globally correct across seams (verified monotonic on
  every test run).
- **Bound the resources.** Workers and per-worker CPU threads are both
  capped (2 workers x 3 threads by default). An early version left threads
  unbounded; on a 6-core laptop, parallel workers each grabbing every
  logical core starved the host badly enough to force a reboot mid-run.
  A transcription service should never be able to take down its own host.

In the API, long jobs return `202` with a job id immediately and the client
polls - a transcription that takes minutes has no business inside a single
request/response cycle.

### Model and runtime choice

`small` with int8 quantization on CPU is the default here: it runs about
2x realtime on a laptop with no GPU requirements, which fits a take-home
that anyone should be able to clone and run. The quality/speed dial is one
flag (`--model medium`, `--device cuda`); production would run `large-v3`
in float16 on GPU behind the same interface. faster-whisper (CTranslate2)
was chosen over the reference openai-whisper implementation for ~4x faster
CPU inference at the same accuracy.

### Deliberate simplifications

The API keeps job state in a process-local dict and one worker thread. For
production: Redis/Postgres for job state, a real queue (Celery/SQS) so jobs
survive restarts and scale horizontally, word-level timestamps
(`word_timestamps=True`) and speaker diarization as options, and object
storage for uploads instead of temp dirs.

## Measured results (Windows laptop, 6-core i7, CPU only, int8, model `small`)

| Test | Input | Result |
|---|---|---|
| Short file | 10.7 s WAV | 2 segments, ~2x realtime |
| Same content, 4 more formats | MP3 / M4A / FLAC / OGG | identical transcription, same code path |
| Non-audio file | requirements.txt | rejected at ingest with clear error |
| Controlled chunking test | 6 min TTS, 40 numbered sentences | 9 chunks, all cuts at silence midpoints, timestamps monotonic, sentences in order |
| Chunked vs single pass | same 6 min file | 211 s parallel vs 328 s single pass (~35% faster) |
| Long real speech | 38.7 min LibriVox audiobook MP3 (64 kbps, 24 kHz stereo) | 8 chunks, 541 segments, 451 s wall clock = 5.1x realtime (2 workers), timestamps monotonic across all seams |

To reproduce the long-file test:

```bash
curl -L -o samples/longform_pp47.mp3 https://archive.org/download/pride_and_prejudice_librivox/prideandprejudice_47-48_austen_64kb.mp3
python transcribe.py samples/longform_pp47.mp3 --srt
```

Short fixtures are generated with the Windows TTS engine
(`scripts/make_sample.ps1`) so the repo ships no third-party audio.
