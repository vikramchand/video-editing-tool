# Video Understanding

A local-first, self-hostable video understanding API. Give it a short phone
video; get back a structured description of what happens in it — transcript,
scenes, people, objects, actions, highlights, and concrete editing suggestions —
plus the ability to render an edited cut from those suggestions.

Everything runs on your machine. No OpenAI, no Anthropic, no Gemini, no AWS, no
API keys, no video leaving the host.

```bash
curl -X POST http://localhost:8000/v1/videos -F "file=@my_video.mp4"
# {"video_id": "abc123", "status": "queued", "filename": "my_video.mp4"}

curl http://localhost:8000/v1/videos/abc123
# {"video_id": "abc123", "status": "completed", "result": { ... }}
```

---

## Contents

1. [What this is](#what-this-is)
2. [Why local-first](#why-local-first)
3. [Architecture](#architecture)
4. [Quick start](#quick-start-under-10-minutes)
5. [Installation](#installation)
6. [Ollama setup and model installation](#ollama-setup-and-model-installation)
7. [Running the API](#running-the-api)
8. [Running the CLI](#running-the-cli)
9. [Web UI](#web-ui)
10. [API reference](#api-reference)
11. [Editing](#editing)
12. [Configuration](#configuration)
13. [Mac notes and model sizing](#mac-notes-and-model-sizing)
14. [Docker](#docker)
15. [Testing](#testing)
16. [Project layout](#project-layout)
17. [Design decisions](#design-decisions)
18. [Roadmap](#roadmap)

---

## What this is

This is the **video understanding layer** that could sit underneath an editor
like Captions.ai — not a clone of one. The goal is to take raw footage and
produce a machine-readable representation rich enough that an editor (human or
automated) can act on it.

Given a video where a person walks into a kitchen, talks to the camera, opens a
refrigerator, takes out a drink and walks outside, you get back something like:

```json
{
  "summary": "A person walks into a kitchen and greets the camera, then opens
              the refrigerator and takes out a bottled drink...",
  "scenes": [
    {
      "id": 2,
      "start": 8.4,
      "end": 18.7,
      "description": "The person opens a stainless steel refrigerator and removes a bottled drink.",
      "people": ["one adult in a grey hoodie"],
      "objects": ["refrigerator", "bottled drink"],
      "actions": ["opening", "reaching", "removing"],
      "importance": 0.82
    }
  ],
  "highlights": [
    { "start": 8.4, "end": 18.7, "reason": "The main action of the video.", "score": 0.82 }
  ],
  "edit_suggestions": [
    { "start": 0.0, "end": 2.1, "action": "remove", "reason": "2.1s of silence before the first word." }
  ]
}
```

A complete example is in [`examples/sample_output.json`](examples/sample_output.json).

**What works today:** the full pipeline (upload → FFmpeg → Whisper → scene
detection → vision model → reasoning model → structured JSON), the REST API, a
CLI, a web UI, and an editing MVP that removes segments, concatenates the
remainder and burns in captions.

**What is deliberately not here:** authentication, cloud storage, distributed
queues, Kubernetes, payments, B-roll generation. Those belong to later versions,
and the [roadmap](#roadmap) explains where each one plugs in.

---

## Why local-first

1. **Video is personal.** Phone footage contains faces, homes, children and
   locations. The default should be that it never leaves the machine.
2. **Cost per minute is the wrong shape.** Frame-by-frame cloud vision billing
   makes experimentation expensive. Local inference makes the marginal cost of
   re-running a video zero.
3. **Apple Silicon is genuinely capable.** A 7B VLM and an 8B reasoning model
   both run comfortably on an M-series Mac with 16 GB.
4. **It has to work offline**, on a plane, in a workshop, with no account.

Local-first is a starting point, not a ceiling. Every model sits behind a
Protocol (see [Design decisions](#design-decisions)), so moving a stage to a GPU
box or a hosted endpoint is a configuration change plus one new class.

---

## Architecture

```
                            VIDEO
                              │
                              ▼
                        ┌───────────┐
                        │  FFmpeg   │  validate, probe, demux
                        └─────┬─────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
              AUDIO                       VIDEO
                │                           │
                ▼                           ▼
        ┌───────────────┐        ┌────────────────────┐
        │    Whisper    │        │  Scene detection   │
        │ (faster-      │        │  PySceneDetect →   │
        │  whisper, CPU)│        │  ffmpeg → uniform  │
        └───────┬───────┘        └─────────┬──────────┘
                │                          │
                │                          ▼
                │               ┌──────────────────────┐
                │               │ Representative frames│
                │               │  ~3 per scene, capped│
                │               └─────────┬────────────┘
                │                         │
                │                         ▼
                │               ┌──────────────────────┐
                │               │   Vision LLM         │
                │               │   (Ollama, Qwen-VL)  │
                │               └─────────┬────────────┘
                │                         │
                └────────────┬────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  Structured representation   │
              │  metadata + transcript +     │
              │  per-scene visual analysis   │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  Reasoning LLM (Ollama)      │
              │  reasons over structure,     │
              │  never over raw pixels       │
              └──────────────┬───────────────┘
                             │
        ┌────────────┬───────┴────────┬─────────────┐
        ▼            ▼                ▼             ▼
     Summary     Highlights    Edit suggestions   Scenes
        └────────────┴────────────────┴─────────────┘
                             │
                             ▼
                     JSON API  /  CLI  /  Web UI
                             │
                             ▼
                    FFmpeg rendering (editing MVP)
```

### The frame-sampling optimisation

The single most important design decision. **Never send every frame to the
vision model.**

```
60-second video @ 30fps = 1,800 frames
            ↓  scene detection
        10 scenes
            ↓  2-4 frames per scene
        20-40 frames
            ↓
        Vision model
```

A 7B VLM takes roughly 2-6 seconds per frame on an M-series Mac. At 1,800 frames
that is over an hour; at 30 frames it is one to three minutes. `MAX_FRAMES` caps
the total for the whole video regardless of how many scenes are found, and when
the cap binds, frames are redistributed in proportion to scene duration so long
scenes keep coverage while every scene keeps at least one frame.

```yaml
video:
  max_duration_seconds: 180
  frames_per_scene: 3
  max_frames: 100
```

---

## Quick start (under 10 minutes)

### The 60-second version (no models needed)

You can see the whole pipeline run — real FFmpeg, real scene detection, real
frame sampling, real JSON — before installing a single model, using the built-in
mock providers.

```bash
git clone https://github.com/vikramchand/video-editing-tool.git
cd video-editing-tool

python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Requires ffmpeg: brew install ffmpeg
TRANSCRIPTION_PROVIDER=mock VISION_PROVIDER=mock LLM_PROVIDER=mock \
  video-understand path/to/any_video.mp4
```

You get the full report and `output/any_video.json`. Now swap in real models.

### The full version

```bash
# 1. System dependency
brew install ffmpeg

# 2. Ollama on the host (this is what reaches the GPU)
brew install ollama
ollama serve &                  # leave running

# 3. Models (~9 GB total; this is the slow step)
ollama pull qwen2.5vl:7b        # vision
ollama pull qwen3:8b            # reasoning

# 4. The project, with local Whisper and better scene detection
pip install -e ".[all,dev]"

# 5. Confirm everything is wired up
video-understand check

# 6. Process a video
video-understand my_video.mp4

# 7. Or run the API and the web UI
uvicorn video_understanding.api.app:app --reload
open http://localhost:8000
```

`video-understand check` tells you exactly what is missing and how to fix it
before you spend time on a video:

```
  hardware         Apple Silicon (arm64)
  ffmpeg           ffmpeg version 7.1
  ollama host      http://localhost:11434

  ok transcription  faster-whisper 'small' (cpu/int8, not yet loaded)
  ok vision         ollama at http://localhost:11434, model 'qwen2.5vl:7b' available
  !! llm            ollama is running but model 'qwen3:8b' is not installed
                    (run: ollama pull qwen3:8b)
```

---

## Installation

Requires **Python 3.12+** and **FFmpeg**.

```bash
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Debian/Ubuntu
```

| Install | Command | What you get |
|---|---|---|
| Base | `pip install -e .` | API, CLI, UI, FFmpeg pipeline, Ollama providers |
| + Whisper | `pip install -e ".[whisper]"` | Local speech-to-text |
| + Scenes | `pip install -e ".[scenes]"` | PySceneDetect content-aware detection |
| Everything | `pip install -e ".[all,dev]"` | The above plus the test suite |

Both extras are optional on purpose:

- Without `whisper`, transcription is skipped and recorded in `warnings`.
- Without `scenes`, scene detection falls back to FFmpeg's own `scene` filter,
  which needs no extra dependencies and works well on hard cuts.

Neither missing package stops a video from being processed.

---

## Ollama setup and model installation

Ollama serves the vision and reasoning models over HTTP on port 11434.

```bash
brew install ollama    # or download from https://ollama.com
ollama serve

ollama pull qwen2.5vl:7b
ollama pull qwen3:8b

ollama list            # verify
curl http://localhost:11434/api/tags
```

### Why Ollama runs on the host

**Docker Desktop on macOS cannot give a container access to the Mac's GPU.**
Containers run inside a Linux VM with no Metal passthrough, so an Ollama
instance inside the API image would be pinned to CPU and several times slower.

```
Mac
 ├── Ollama ................... host process, uses Metal GPU
 │    └── qwen2.5vl:7b, qwen3:8b
 │
 └── Docker
      └── Video Understanding API ──HTTP──> host.docker.internal:11434
```

So the app talks to Ollama over HTTP, wherever it lives:

| Where the API runs | `OLLAMA_HOST` |
|---|---|
| Natively on the Mac | `http://localhost:11434` |
| In Docker | `http://host.docker.internal:11434` |
| On another machine | `http://<that-host>:11434` |

Whisper is different: it runs on CPU regardless, so it is installed *inside* the
container with no penalty.

---

## Running the API

```bash
uvicorn video_understanding.api.app:app --reload --port 8000
# or
video-understand serve --port 8000
```

Then open http://localhost:8000 for the UI, or http://localhost:8000/docs for
interactive API documentation.

### Example session

```bash
# Health, including per-provider status
curl http://localhost:8000/health

# Upload; returns immediately
curl -X POST http://localhost:8000/v1/videos -F "file=@my_video.mp4"
# {"video_id":"a1b2c3d4e5f6","status":"queued","filename":"my_video.mp4"}

# Poll progress
curl http://localhost:8000/v1/videos/a1b2c3d4e5f6/status
# {"status":"processing","stage":"analyzing_frames","progress":0.7}

# Fetch the result
curl http://localhost:8000/v1/videos/a1b2c3d4e5f6 | jq .result.summary

# Subtitles
curl -O http://localhost:8000/v1/videos/a1b2c3d4e5f6/transcript.srt

# Render an edit, then download it
curl -X POST http://localhost:8000/v1/videos/a1b2c3d4e5f6/render \
  -H 'Content-Type: application/json' \
  -d '{"apply_removals": true, "burn_captions": true}'

curl -O -J http://localhost:8000/v1/videos/a1b2c3d4e5f6/render/download
```

---

## Running the CLI

```bash
video-understand my_video.mp4
```

1. Processes the video, showing each stage as it happens.
2. Prints a human-readable report.
3. Writes the full JSON to `output/my_video.json`.

```
  [################--------]  70%  analyzing frames    analysed scene 5/8    24.1s

==============================================================================
  Grabbing a Drink Before Heading Outside
==============================================================================
  Duration 0:47 (47.2s)   Resolution 1080x1920 (portrait)   FPS 30.0

SUMMARY
  A person walks into a kitchen and greets the camera, then opens the
  refrigerator and takes out a bottled drink...

SCENES (4)
     0.0s -    8.4s **  0.65  A person walks into a kitchen holding a phone
           people: one adult in a grey hoodie
           objects: kitchen counter, phone
     8.4s -   18.7s *** 0.82  The person opens a refrigerator and removes a drink

HIGHLIGHTS (3)
     8.4s -   18.7s score 0.82  The main action of the video.

EDIT SUGGESTIONS (6)
      0.0s -    2.1s remove   2.1s of silence before the first word.
     13.7s -   18.7s zoom     Punch in on the drink being removed.
```

| Command | Purpose |
|---|---|
| `video-understand VIDEO` | Analyse a video |
| `video-understand VIDEO --render` | Also render an edit with removals applied |
| `video-understand VIDEO --render --captions` | Render with burned-in captions |
| `video-understand VIDEO --highlights-only --render` | Render a highlight reel |
| `video-understand VIDEO -o out.json` | Choose the JSON output path |
| `video-understand VIDEO --quiet` | Only print the saved path |
| `video-understand check` | Verify FFmpeg and model availability |
| `video-understand serve` | Run the API server |

---

## Web UI

Served at `http://localhost:8000` by the API itself — three static files
(`index.html`, `app.css`, `app.js`), no build step, no `npm install`, no CDN.
It works with the network unplugged, which is the point of a local-first tool.

The layout is a three-column workspace: **library** on the left, **player and
timeline** in the middle, **inspector** on the right.

**Library.** Every upload ever made, read from SQLite, newest first. Poster
frame, duration, status and scene count per row; live progress bars for jobs
still running; search across filename, title, summary and topics; filter by
status; sort by date, duration or name; paginated. Selecting a video deep-links
to it (`#/v/<id>`), so a video is bookmarkable, reloadable and back-buttonable.

**Player.** The video takes whatever height the window allows. Custom transport:
play/pause, ±5 s, scene-to-scene stepping, speed, volume, fullscreen, and a
caption overlay driven by the transcript. Keyboard shortcuts throughout
(`space`, `J`/`K`/`L`, `,`/`.` for single frames, `[`/`]` for scenes, `0`–`9`
to jump, `?` for the list).

**Timeline.** Five aligned tracks under a time ruler: a filmstrip of real
stills, scenes coloured and shaded by importance, speech, highlights, and edit
suggestions coloured by action. Click or drag anywhere to scrub, hover for a
tooltip with the range and the model's description, zoom to 8× for tighter
work. The playhead, the scene readout, the active transcript line and the
highlighted scene card all follow playback.

**Detail tabs.** Transcript (searchable, follows playback, click to seek),
scenes as a card grid with a still per scene, highlights and edit suggestions
with scored meters, extracted people/objects/actions, and the raw JSON.

**Inspector.** Summary, topics, full media specs, provider attribution,
degradation warnings, and the export panel — which previews the exact result of
the current options (`10.0s → 4.0s · 3 highlights · 1 removal applied`) before
you spend FFmpeg time on it.

Light and dark themes follow the system and can be toggled; the choice
persists.

A React + Vite frontend was the alternative; static files were chosen because
they remove a build toolchain, a second dev server and a CORS story from a
project whose value is in the pipeline.

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service and per-provider health |
| `GET` | `/v1/config` | Active configuration |
| `POST` | `/v1/videos` | Upload a video (multipart `file`), returns `202` |
| `GET` | `/v1/videos` | List videos (`limit`, `offset`, `include_results`, `q`, `status`) |
| `GET` | `/v1/library` | Browse the library: cards, totals and per-status counts |
| `GET` | `/v1/videos/{id}` | Status plus the full result when complete |
| `GET` | `/v1/videos/{id}/status` | Lightweight progress poll |
| `GET` | `/v1/videos/{id}/source` | The original upload (ranged, inline) |
| `GET` | `/v1/videos/{id}/thumbnail` | Poster frame, generated once and cached |
| `GET` | `/v1/videos/{id}/frame?t=&w=` | A still at any timestamp (cached) |
| `GET` | `/v1/videos/{id}/transcript.srt` | Transcript as SRT |
| `POST` | `/v1/videos/{id}/render` | Render an edited video |
| `GET` | `/v1/videos/{id}/render/download` | Download the rendered edit |
| `DELETE` | `/v1/videos/{id}` | Delete the video, result and files |

### Status values

```
queued → processing → completed
                    ↘ failed
```

`stage` reports the pipeline step in progress, and `progress` is a 0-1 fraction:

```
pending → validating → extracting_metadata → extracting_audio → transcribing
  → detecting_scenes → extracting_frames → analyzing_frames → reasoning → done
```

### Response shape

```jsonc
{
  "video_id": "abc123",
  "status": "completed",
  "stage": "done",
  "progress": 1.0,
  "result": {
    "duration": 47.2,
    "summary": "...",
    "title": "...",
    "topics": ["..."],
    "metadata": { "width": 1080, "height": 1920, "fps": 29.97, "has_audio": true },
    "transcript": [{ "start": 2.1, "end": 5.8, "text": "...", "confidence": 0.94 }],
    "scenes": [{
      "id": 1, "start": 0.0, "end": 8.4,
      "description": "...",
      "people": ["..."], "objects": ["..."], "actions": ["..."],
      "environment": "...", "importance": 0.65,
      "keyframes": [1.4, 4.2, 7.0],
      "transcript_text": "..."
    }],
    "highlights": [{ "start": 8.4, "end": 18.7, "reason": "...", "score": 0.82 }],
    "edit_suggestions": [{ "start": 0.0, "end": 2.1, "action": "remove", "reason": "..." }],
    "people": ["..."], "objects": ["..."], "actions": ["..."],
    "providers": { "vision": "ollama:qwen2.5vl:7b", "reasoning_source": "llm" },
    "warnings": []
  }
}
```

Two fields worth knowing about:

- **`warnings`** records non-fatal degradation. If Ollama was unreachable, you
  still get a result — built from heuristics — and a warning saying so. A
  partial answer beats a 500.
- **`providers`** records which model produced each layer, so a result is
  reproducible and explainable after the fact.

### Errors

| Code | Meaning |
|---|---|
| `400` | Invalid video (unreadable, empty, too long) |
| `404` | Unknown `video_id` |
| `409` | The video is not in a state that allows this (e.g. rendering before completion) |
| `413` / `415` | File too large / unsupported type |
| `503` | A local model backend was unreachable |

---

## Editing

The editing MVP proves the understanding layer produces *actionable* output.
Given edit suggestions, it renders a real video with FFmpeg.

Supported today: **removing segments**, **concatenating the remainder**, and
**burning captions**. `zoom`, `broll` and `trim` are produced by the analysis and
returned in the API, but are not yet rendered — the seam for them is
`compute_keep_segments` → `render_edit`.

```bash
curl -X POST http://localhost:8000/v1/videos/{id}/render \
  -H 'Content-Type: application/json' \
  -d '{"apply_removals": true, "burn_captions": true, "highlights_only": false}'
```

| Option | Effect |
|---|---|
| `apply_removals` | Cut every `remove` suggestion and concatenate what is left |
| `burn_captions` | Burn the transcript into the video |
| `highlights_only` | Keep only the highlighted windows (a highlight reel) |
| `edit_suggestions` | Supply your own decisions instead of the generated ones |

Two implementation details that matter:

- **Caption timings are remapped onto the edited timeline.** Cutting 0-4s means
  speech at 6s must appear at 2s in the output. Rendering is therefore two
  passes — cut, then burn — and speech straddling a cut is clipped to the part
  that survives.
- **Cutting uses an FFmpeg filter graph** (`trim`/`atrim` + `concat`), not
  `-ss` with stream copy. Stream copy can only cut on keyframes, which makes
  edits drift by up to a second from where the analysis said they were.

An edit that would remove the entire video is rejected with `400` rather than
silently producing an empty file.

---

## Configuration

Everything model-related is environment-driven. **No model name is hard-coded
anywhere in the codebase.** Copy `.env.example` to `.env` and edit.

```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
VISION_PROVIDER=ollama
VISION_MODEL=qwen2.5vl:7b
TRANSCRIPTION_PROVIDER=whisper
WHISPER_MODEL=small
```

Precedence: **defaults < `config.yaml` < `.env` < environment variables**.

The YAML form uses nested sections that flatten to the same names
(`video.frames_per_scene` → `VIDEO_FRAMES_PER_SCENE`); see
[`config.yaml.example`](config.yaml.example).

### Settings that change behaviour most

| Variable | Default | Effect |
|---|---|---|
| `VIDEO_FRAMES_PER_SCENE` | `3` | Frames sent per scene. The main quality/speed dial. |
| `VIDEO_MAX_FRAMES` | `100` | Hard ceiling for the whole video. |
| `VIDEO_MAX_DURATION_SECONDS` | `180` | Uploads longer than this are rejected. |
| `SCENE_DETECTOR` | `auto` | `auto`, `pyscenedetect`, `ffmpeg`, `uniform`. |
| `SCENE_THRESHOLD` | `27.0` | PySceneDetect sensitivity; lower finds more cuts. |
| `WORKER_COUNT` | `1` | Concurrent jobs. Raise only if you have RAM to spare. |
| `FAIL_ON_PROVIDER_ERROR` | `false` | `true` makes any model failure fail the job. |
| `KEEP_INTERMEDIATE_FILES` | `false` | Keep extracted audio and frames for debugging. |

Provider values `mock` and `null` are available for every stage: `mock` gives
deterministic fake output (the demo mode above), `null` skips the stage.

---

## Mac notes and model sizing

The MVP targets Apple Silicon. Hardware is detected at startup and reported by
`video-understand check` and `/health`.

- **Apple Silicon (M1-M4):** Ollama uses the Metal GPU. Whisper runs on CPU with
  int8 quantisation — there is no Metal backend for faster-whisper, and int8 CPU
  is still comfortably faster than realtime for `small`.
- **Intel Mac:** everything runs on CPU. Use the 16 GB row below regardless of
  RAM, and expect roughly 3-5× the processing time.
- **No NVIDIA GPU is required anywhere.**

### Recommended models by RAM

| RAM | Vision | Reasoning | Whisper | ~Time for 60s of video |
|---|---|---|---|---|
| **16 GB** | `qwen2.5vl:3b` | `qwen3:4b` | `base` | 2-4 min |
| **32 GB** | `qwen2.5vl:7b` | `qwen3:8b` | `small` | 3-6 min |
| **64 GB+** | `qwen2.5vl:32b` | `qwen3:14b` | `medium` | 6-15 min |

Start smaller than you think. The frame-sampling budget dominates runtime far
more than model size does — halving `VIDEO_FRAMES_PER_SCENE` is a bigger speedup
than dropping a model tier, and usually costs less quality.

For 16 GB machines specifically:

```env
VISION_MODEL=qwen2.5vl:3b
LLM_MODEL=qwen3:4b
WHISPER_MODEL=base
VIDEO_FRAMES_PER_SCENE=2
VIDEO_MAX_FRAMES=40
```

Vision analysis is the bottleneck in every configuration. If a run feels slow,
lower the frame budget before changing anything else.

---

## Docker

```bash
docker compose up
```

The image contains the API, FFmpeg and Whisper. **It does not contain Ollama** —
see [Why Ollama runs on the host](#why-ollama-runs-on-the-host). Start Ollama on
the host first:

```bash
ollama serve
ollama pull qwen2.5vl:7b
ollama pull qwen3:8b

docker compose up
```

`data/` and `output/` are bind-mounted, so uploads, the SQLite database and
rendered edits survive container restarts. On Linux hosts, `extra_hosts` maps
`host.docker.internal` to the gateway; on Docker Desktop it already resolves.

If Docker on macOS gets in your way, **run natively instead** — it is the
better-supported path on a Mac, and the only difference is `OLLAMA_HOST`:

```bash
pip install -e ".[all]"
uvicorn video_understanding.api.app:app --port 8000
```

PySceneDetect is not installed in the image by default; it pulls in a full
OpenCV build for a modest gain over the FFmpeg detector. The Dockerfile has the
lines to enable it, commented out.

---

## Testing

```bash
pytest                      # everything
pytest -k "not ffmpeg"      # skip tests that shell out to ffmpeg
pytest --cov=video_understanding
```

**No test requires a model to be running.** Every AI stage is exercised through
mock providers or scripted fakes. Tests that genuinely need FFmpeg are marked
and skip cleanly when it is absent; they build their own fixture videos.

Coverage spans: metadata extraction, upload validation, audio extraction, scene
detection and boundary post-processing, the frame-sampling budget, transcript
normalisation and SRT output, model-response parsing (including the malformed
JSON shapes small models actually emit), Pydantic validation and coercion,
edit-decision generation, transcript remapping across cuts, FFmpeg rendering,
the job store under concurrent writes, the library read model (search
escaping, filters, sorting, schema migration and backfill), source-path
resolution across hosts, graceful degradation when a provider fails, the CLI,
and every API endpoint.

The suite also validates the shipped example JSON against the live schema, so
the documentation cannot silently drift from the models.

---

## Project layout

```
video_understanding/
├── api/
│   ├── app.py              FastAPI factory, error handlers, UI mount
│   ├── dependencies.py     Shared app state
│   └── routes/             videos.py, library.py, health.py
├── core/
│   ├── config.py           Settings, precedence, hardware detection
│   ├── models.py           Every Pydantic model — the contract
│   └── errors.py           Exception hierarchy
├── video/
│   ├── ffmpeg_utils.py     The only place subprocess touches ffmpeg
│   ├── metadata.py         ffprobe + upload validation
│   ├── audio.py            16 kHz mono extraction
│   ├── scenes.py           Three detectors + shared post-processing
│   ├── frames.py           The frame-budget optimisation
│   ├── thumbnails.py       Poster frames and cached stills for the UI
│   └── rendering.py        Editing MVP
├── ai/
│   ├── prompts.py          Vision and reasoning prompts
│   ├── transcription.py    Transcript normalisation, SRT
│   ├── vision.py           Per-scene analysis, fused with speech
│   ├── reasoning.py        Structured representation → editorial output
│   └── providers/          base.py (Protocols), ollama.py, whisper.py,
│                           mock.py, registry.py
├── jobs/
│   ├── pipeline.py         The pipeline. One implementation, shared.
│   └── processor.py        Thread-pool job queue
├── storage/
│   ├── database.py         SQLite job store and library read model
│   └── files.py            Resolving a job back to its file on disk
├── cli/main.py             CLI
└── web/                    index.html, app.css, app.js
```

---

## Design decisions

**Providers are Protocols, not base classes.** Each stage depends on a
structural interface:

```python
class VisionProvider(Protocol):
    name: str
    model: str
    def analyze_frames(self, frames: list[str], context: str = "") -> SceneAnalysis: ...
    def health(self) -> tuple[bool, str]: ...
```

Nothing downstream knows whether it is talking to a local VLM or a hosted GPU
endpoint. Adding a backend means writing a class and adding one line to the
registry.

**The reasoning model never sees pixels.** It receives the structured
representation — metadata, transcript, per-scene analysis — as text. This keeps
expensive multimodal work confined to a bounded set of sampled frames, and lets
the reasoning model be swapped independently of the vision one.

**Model output is treated as untrusted input.** A local 8B model will emit
`<think>` blocks, markdown fences, trailing commas, `"significance": 85`,
`"action": "delete"`, and highlights ending at 900s in a 47-second video. All of
that is parsed defensively, coerced where the intent is unambiguous, and clamped
against the real duration before it reaches the API. The parsing tests are a
catalogue of real failures.

**Degrade rather than fail.** If Ollama is down, you get a heuristic summary and
a warning, not a 500. If Whisper is missing, you get scenes without a transcript.
`FAIL_ON_PROVIDER_ERROR=true` opts into strictness. A silent video is a valid
input, not an error.

**Scene detection has a floor that cannot fail.** PySceneDetect → FFmpeg →
uniform chunking. The last one is arithmetic, so this stage always produces
something.

**The library is a second read model over the same rows.** The full analysis
lives in one `result_json` blob, but the fields a card needs — title, duration,
dimensions, scene and highlight counts, topics — are written to their own
columns when a result is saved. Drawing a page of the library would otherwise
mean parsing and validating fifty result documents, and searching it would mean
`LIKE` over JSON. `save_result` is the only writer, so the denormalisation
cannot drift; opening an older database file adds the columns and backfills
them from the blobs already stored.

**A job resolves to its file, not to a recorded path.** `source_path` is an
absolute path, which stops being true the moment the same data directory is
opened from somewhere else — a video uploaded through the container
(`/app/data/uploads/x.mp4`) and then browsed from a server on the host. Uploads
are named after the video id, so the file can always be found again from the id
alone. Old rows stay playable instead of becoming dead links.

**Cutting is one ffmpeg process per segment, not one filter graph.** The
elegant version — a `trim` per segment feeding `concat` — is a trap: the trims
share an input pad, and on ffmpeg 7.x the first branch to end tears the graph
down, so the command exits 0 having written only one of the segments. Segments
are encoded individually with identical settings and joined with `-c copy`,
which is lossless, frame-accurate, and cannot fail that way.

**Threads, not a broker.** The work is dominated by subprocess and HTTP waits,
which release the GIL. A `ThreadPoolExecutor` plus SQLite is the honest fit for
one machine; nothing outside `JobProcessor` knows how jobs are scheduled, so
replacing it later is a local change.

---

## Roadmap

**V2 — better understanding**
GPU inference for the vision stage; parallel frame analysis; better scene
detection (transitions and slow pans, not just hard cuts); an embedding index
over scene descriptions for semantic search — "find every moment where someone
opens a door"; automatic highlight reels.
*Seam:* new `VisionProvider` implementations; `VideoUnderstanding` already
carries per-scene text ready to embed.

**V3 — AI editing**
Rendering the `zoom`, `trim` and `broll` actions the analysis already emits;
B-roll generation and insertion; music and sound-effect selection; social
formatting presets (YouTube / TikTok / Reels aspect ratios and safe areas).
*Seam:* `compute_keep_segments` → `render_edit`; `EditAction` is a closed set,
so adding an action is a compile-time-visible change.

**V4 — hosted**
Multi-user accounts, cloud object storage, a real job queue, GPU workers,
horizontally scalable inference.
*Seam:* `JobProcessor` becomes a queue client; `Database` gains a Postgres
backend; storage paths move behind an interface.

---

## Licence

Apache-2.0.
