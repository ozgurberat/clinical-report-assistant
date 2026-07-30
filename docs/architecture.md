# Architecture notes

This file tracks the actual, final technical decisions per phase — what got
built, not the original candidate list.

## Phase 1 — Data pipeline

- Source: Open-i / Indiana University Chest X-ray dataset (~3,955 reports).
- `src/data/download_openi.py` — official NLM bulk files, Kaggle fallback.
- `src/data/preprocess.py` — XML → structured JSONL; drops empty/near-empty
  findings and a rare duplicate-content export artifact (see `data/README.md`).
- Deterministic 80/10/10 train/val/test split (`random.Random(seed=42)`),
  computed once and shared by both fine-tuning tasks.

## Phase 2 — Fine-tuning

- Base model: **Qwen3-4B** (Apache 2.0, no gating, instruction-tuned
  checkpoint) — not Llama 3.2/Qwen2.5/Phi-3-mini, the original candidates.
  Apache 2.0 avoided the license-acceptance friction of Llama; 4B is small
  enough to iterate on fast, big enough to reliably learn a narrow format.
- QLoRA: bitsandbytes NF4 4-bit quantization + PEFT LoRA adapters
  (r=16, alpha=32, all 7 major linear layers targeted), TRL's `SFTTrainer`
  with `assistant_only_loss=True` for completion-only loss masking.
- Two tasks, two separate adapters sharing one frozen base: extraction
  (raw text → structured JSON) and summarization (findings → impression).
- Precision auto-detected per GPU (`bf16` on A100/Ampere+, `fp16` on
  T4/Turing) — not a fixed config value.
- Tracking: MLflow (not W&B) — local file-backed, one experiment, one run
  per task.
- Compute: Google Colab (free T4 initially, Colab Pro A100 later), not
  RunPod/Vast.ai.
- Results: full 343-example held-out test set — see the Results table in
  the root README.

## Phase 3 — RAG

- Vector DB: **Qdrant, local/embedded mode** (`qdrant-client`, no server
  process) — not FAISS, not a self-hosted Qdrant service. Chosen as the
  actual "vector database" skill (a real client/server-shaped API) rather
  than a pure in-process similarity-search library, while still avoiding the
  operational overhead of running a real server for local development.
- Orchestration: **hand-rolled** (`src/rag/build_index.py`,
  `retrieve.py`, `qa.py`) — not LangChain. Deliberately dropped after
  repeated API drift with other fast-moving libraries (TRL) earlier in the
  project; every retrieve → build-context → generate step is plain, visible
  code instead of a framework abstraction.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` — small,
  CPU-friendly, 384-dim.
- Generation for RAG-QA: the **plain base Qwen3-4B model, no adapter**
  (PEFT's `disable_adapter()`), with thinking mode left **on** — the
  opposite choice from extraction/summarization, since synthesizing across
  several retrieved documents is exactly the multi-step task thinking mode
  is meant for, and this path never touches an untrained adapter.

## Phase 4 — Serving

- FastAPI, not a raw inference-server wrapper — three endpoints
  (`/extract`, `/summarize`, `/ask`) plus `/health` and (Phase 6) `/metrics`.
- Inference backend: **`transformers` + `peft` directly**, not vLLM/TGI. One
  base model loaded once at startup with both LoRA adapters attached as
  named adapters, switched per-request via `set_adapter()`/
  `disable_adapter()` — the same shape real multi-LoRA serving systems
  (e.g. vLLM's multi-LoRA support) are built around, without adding vLLM
  itself as a dependency.
- Quantization: bitsandbytes 4-bit NF4 on GPU; automatic bf16-on-CPU
  fallback when no CUDA device is present (bitsandbytes' 4-bit kernels are
  CUDA-only) — added specifically so the same code runs correctly, if
  slowly, in a GPU-less Docker/Kubernetes test environment.
- Containerized with Docker; `docker-compose.yml` for local orchestration.
  Trained artifacts are mounted in as volumes, never baked into the image
  (they're gitignored — see `src/serving/README.md`).

## Phase 5 — Orchestration & CI/CD

- Kubernetes manifests validated via **kind**, not minikube — runs as plain
  Docker containers on the same Docker Desktop setup already used for
  Phase 4, no separate VM/hypervisor.
- Host artifacts reach pods via a two-hop bridge: kind's `extraMounts`
  (host → kind node), then `hostPath` volumes (node → pod).
- GitHub Actions: offline unit tests (existing), extended with a Docker
  build-validation job and a static Kubernetes manifest lint (`kubeconform`)
  — not a live deploy step, which was intentionally left out of scope.

## Phase 6 — Monitoring

- Structured request/latency/error logging — done.
- `/metrics`: Prometheus-format request counts, a latency histogram
  (enables p50/p95 at query time), and a tokens-generated counter labeled by
  adapter — done.
- Prometheus + Grafana deployment — **deliberately not built**. Scoping
  decision, not an oversight: the `/metrics` endpoint and
  `prometheus.io/scrape` annotations already demonstrate the instrumentation
  skill; standing up the full observability stack would have meant another
  full deploy-debug-wait cycle better spent on the ML side of the project.
