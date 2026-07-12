# Phase 4 — FastAPI serving + Docker

## Architecture

One FastAPI app, one loaded model instance. The base Qwen3-4B model is
loaded once at startup; both the extraction and summarization LoRA adapters
are attached to that same instance as named adapters (`app.py`'s
`ModelManager.load()`), and each request switches which one is active via
PEFT's `set_adapter()` — or, for `/ask`, runs the model with no adapter at
all via `disable_adapter()`, the same "plain base model" behavior established
in `src/rag/qa.py`. This is deliberately different from how `evaluate.py`
and the notebooks load models (a fresh base-model-plus-one-adapter copy each
time) — that pattern is fine for offline batch jobs running one task at a
time, but a server fielding requests for three different behaviors needs all
of them available at once without holding three separate 8GB+ model copies
in memory. It's also the standard shape real multi-LoRA serving systems
(e.g. vLLM's multi-LoRA support) are built around.

## Before you run this: get the trained artifacts onto this machine

`data/processed/` and `outputs/` are gitignored on purpose — clinical data
and model binaries don't belong in git — which means a fresh clone of this
repo (including on your Mac, for the Docker test) does NOT have them. They
only exist wherever training/indexing actually ran: your Google Drive, via
Colab. Before building the image, copy these down from
`/content/drive/MyDrive/Clinical_Report_Assistant/` (via the Google Drive
desktop app, or downloading the folders manually) into the same relative
paths in your local clone:

- `outputs/extraction-Qwen3-4B/final_adapter/`
- `outputs/summarization-Qwen3-4B/final_adapter/`
- `data/processed/reports.jsonl`
- `data/processed/qdrant_index/`

`docker-compose.yml` mounts `./outputs` and `./data/processed` into the
container read-only — the app reads the same artifacts you already trained,
nothing gets baked into the image itself.

## Running it

```bash
docker compose up --build
```

First build downloads the full quantization/serving stack (torch,
transformers, bitsandbytes, etc.) — expect it to take a while. Once it's up:

```bash
curl http://localhost:8000/health
```

## GPU vs. CPU — set expectations before you test

Docker Desktop on a Mac has no NVIDIA GPU passthrough, so this runs CPU-only
by default. `ModelManager.load()` detects this and falls back to plain fp32
on CPU (bitsandbytes' 4-bit kernels are CUDA-only) — correct, but slow.
**This test is about verifying the container, dependencies, and API actually
work end-to-end — not about generation speed.** Expect requests to take
noticeably longer than they did in Colab on a GPU; that's expected, not a
bug. If you ever deploy this on an actual GPU host (a cloud VM with the
NVIDIA Container Toolkit), add a `deploy.resources.reservations.devices` GPU
block to `docker-compose.yml` to get real throughput.

## Testing each endpoint

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"report_text": "The heart is mildly enlarged. Lungs are clear. Impression: Mild cardiomegaly."}'

curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"findings": "The heart is mildly enlarged. Lungs are clear."}'

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What have we typically found in similar cases of mild cardiomegaly with clear lungs?"}'
```

FastAPI also serves interactive docs at `http://localhost:8000/docs` —
useful for poking at the API by hand without constructing curl commands.

## What hasn't been verified yet

Everything in `src/serving/` was written and reviewed carefully, and its
syntax has been checked, but **the FastAPI app itself has not actually been
run** during development — `fastapi`/`pydantic`/`torch`/etc. aren't
installed in the sandbox this project was built in, and it has no network
access to install them (same limitation that's applied to every GPU-
dependent file in this project — `train.py`, the notebooks' GPU cells, and
so on). `tests/test_serving.py`'s Pydantic validation tests are also
unexecuted for the same reason. This is the first time any of this code
actually runs — treat the first `docker compose up` as a debugging pass,
same as the very first training run back in Phase 2, and bring back whatever
errors come up.
