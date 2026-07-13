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
by default. `ModelManager.load()` detects this and falls back to bf16 on CPU
(bitsandbytes' 4-bit kernels are CUDA-only; fp32 was the first attempt but
OOM'd a memory-capped container at ~16GB just for the weights — bf16 halves
that) — correct, but slow. **This test is about verifying the container,
dependencies, and API actually work end-to-end — not about generation
speed.** Model loading alone took 10-13 minutes in practice; `/ask` with its
full 1536-token budget can take 20-40+ minutes on CPU (use the
`max_new_tokens` override on `/ask` for a fast smoke test instead — see
below). If you ever deploy this on an actual GPU host (a cloud VM with the
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
  -d '{"question": "What have we typically found in similar cases of mild cardiomegaly with clear lungs?", "max_new_tokens": 100}'
```

`/ask` accepts an optional `max_new_tokens` override specifically for fast
CPU smoke tests — a small value (e.g. 100) will almost always cut generation
off mid-reasoning, but that's fine: `parse_think_output()` (see
`src/rag/qa.py`) detects the unclosed `<think>` block and returns an honest
"ran out of tokens" message instead of garbage, which is enough to confirm
the whole retrieve→generate→respond pipeline works. Leave it unset (or set
it high) for a real, complete answer — expect a long wait on CPU.

FastAPI also serves interactive docs at `http://localhost:8000/docs` —
useful for poking at the API by hand without constructing curl commands.

## Monitoring

`/metrics` exposes Prometheus-format counters and histograms: total requests
and latency per endpoint (`http_requests_total`, `http_request_duration_seconds`
— the histogram specifically so p50/p95 can be computed at query time), and
tokens generated per adapter (`llm_tokens_generated_total`). Every request
also gets a structured log line (`request method=... path=... status=...
duration_ms=...`). No Prometheus server or Grafana dashboard is actually
deployed in this project — that was a deliberate scope decision (see
`k8s/README.md`) — but the endpoint is real and scrapeable, and
`k8s/deployment.yaml` carries the standard `prometheus.io/scrape` annotations
documenting how it would be wired up.

```bash
curl http://localhost:8000/metrics
```

## What's been verified vs. what hasn't

Every endpoint (`/health`, `/extract`, `/summarize`, `/ask`) has been run for
real and confirmed working — both via `docker compose up` on CPU and, later,
via a real `kubectl apply` on a local kind cluster (see `k8s/README.md`).
Real bugs surfaced and got fixed in the process: an OOM from loading in fp32
on CPU (fixed by switching to bf16), a Qdrant crash from a read-only volume
mount (Qdrant's local mode needs to write a `.lock` file even for read
queries), and `/ask` truncating mid-reasoning on a too-small `max_new_tokens`
(fixed with a higher default plus the request-level override above). What
has **not** been verified: the `/metrics` endpoint and the monitoring
middleware added in Phase 6, since they were written after the last real
test run — same disclosure as everything else in this project that's new
and untested, treat the first request against a freshly rebuilt image as a
debugging pass.
