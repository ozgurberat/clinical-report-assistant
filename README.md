# Clinical Report Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model: Qwen3-4B](https://img.shields.io/badge/base%20model-Qwen3--4B-purple.svg)](https://huggingface.co/Qwen/Qwen3-4B)

A fine-tuned, RAG-augmented LLM system for **structured extraction, summarization, and grounded question-answering over radiology reports**. Built on real clinical NLP data, served through a single quantized model with swappable LoRA adapters, containerized with Docker, and deployed to Kubernetes with Prometheus-format monitoring.

The whole system runs on one 4-bit-quantized **Qwen3-4B** base model. Two LoRA adapters handle the narrow, format-specific tasks (extraction, summarization); the same base model with no adapter, given retrieved evidence from a vector index, handles open-ended QA.

> 📖 **For the full engineering story** — why each decision was made, what broke, and how it got fixed — see [`docs/write-up.md`](docs/write-up.md).

## What it does

The service exposes three capabilities over radiology reports:

- **Extract** — turn raw, unstructured dictation into structured JSON (`comparison`, `indication`, `findings`, `impression`, `diagnosis`).
- **Summarize** — condense a report's findings into a concise clinical impression.
- **Ask** — answer open-ended questions by retrieving genuinely similar past cases and grounding the answer in them, with cited report IDs, rather than guessing.

## Architecture

```
                 ┌─────────────────────┐
                 │   Open-i / IU CXR    │
                 │  radiology reports   │
                 └──────────┬───────────┘
                             │  clean + structure
                             ▼
                 ┌─────────────────────┐
                 │   Processed corpus   │
                 │  (JSONL: findings /  │
                 │  impression / dx)    │
                 └──────┬───────┬───────┘
                        │       │
          fine-tuning   │       │  indexing
            (QLoRA)     │       │  (embeddings)
                        ▼       ▼
        ┌───────────────────┐ ┌──────────────────┐
        │  Fine-tuned LLM   │ │   Vector DB       │
        │   (Qwen3-4B,      │ │ (Qdrant, local/   │
        │   QLoRA adapters) │ │  embedded mode)   │
        └─────────┬─────────┘ └────────┬──────────┘
                   │                    │
                   └─────────┬──────────┘
                              ▼
                   ┌────────────────────┐
                   │   RAG orchestrator │
                   │  (hand-rolled:     │
                   │  qdrant-client +   │
                   │ sentence-transf.)  │
                   └──────────┬─────────┘
                              ▼
                   ┌────────────────────┐
                   │  FastAPI service   │
                   │ (transformers+PEFT,│
                   │  4-bit NF4, 1 base │
                   │ model + 2 adapters)│
                   └──────────┬─────────┘
                              ▼
                Docker Compose → Kubernetes (kind)
                              │
                              ▼
              Monitoring (structured logs + /metrics,
              Prometheus-format, scrape-annotated)
```

See [`docs/architecture.md`](docs/architecture.md) for the key technical decision behind each component.

## Results

Metrics are computed on the full **343-example held-out test set** per task by [`src/finetuning/evaluate.py`](src/finetuning/evaluate.py).

| Task | Metric | Score |
|---|---|---|
| **Extraction** | Field F1 — comparison / indication / findings / impression | 0.968 / 0.963 / 0.969 / 0.962 |
| **Extraction** | Diagnosis-list F1 (set overlap over MeSH tags) | 0.799 |
| **Extraction** | Whole-object exact match (all 5 fields byte-identical) | 0.615 |
| **Extraction** | JSON parse-failure rate | 0.029 |
| **Summarization** | ROUGE-1 / ROUGE-2 / ROUGE-L (vs. ground-truth impression) | 0.694 / 0.592 / 0.682 |
| **Summarization** | Exact match | 0.350 |

The model reliably captures the right content (96–97% per-field overlap) even when it doesn't reproduce it byte-for-byte. Summarization's low exact-match is expected and healthy for an abstractive task — near-1.0 would suggest memorization, not generalization.

## Key engineering decisions

- **One base model, two adapters, switched per request.** The serving layer loads a single Qwen3-4B instance with both LoRA adapters attached simultaneously and switches the active one per request via PEFT's `set_adapter()` / `disable_adapter()` — rather than holding three multi-gigabyte model copies in memory. This is the same pattern production multi-LoRA serving systems (e.g. vLLM's multi-LoRA support) are built around.
- **Retrieval and fine-tuning solve different problems.** The extraction and summarization adapters can't answer open-ended questions — verified directly, not assumed. RAG-QA therefore runs on the plain base model with retrieved evidence, and with Qwen3's thinking mode left *on* (the opposite of the single-step extraction/summarization tasks, where an empty reasoning block was pure waste).
- **Data quality is checked before training, not after a run fails.** An EDA pass caught and fixed two real issues — a de-identification placeholder token and a duplicate-content export artifact — both protected by regression tests before the first training run.
- **Hardware-aware precision.** Training auto-detects bf16 vs. fp16 support per GPU, so the same code runs correctly on a Colab T4 and an A100 without a config change.
- **Testable without the ML stack.** Heavy ML imports (`torch`, `transformers`, `peft`, `qdrant-client`) are deferred inside the functions that use them, so all pure logic — text building, F1/ROUGE scoring, JSON-parse fallbacks, Pydantic schemas — stays unit-testable in a plain Python environment.

## Tech stack

**ML / fine-tuning:** PyTorch · Hugging Face Transformers · PEFT (QLoRA) · TRL · bitsandbytes (4-bit NF4) · MLflow
**RAG:** Qdrant (embedded mode) · Sentence-Transformers
**Serving & ops:** FastAPI · Pydantic · Docker · Kubernetes (kind) · GitHub Actions CI · Prometheus-client

## Quickstart

```bash
git clone https://github.com/ozgurberat/clinical-report-assistant.git
cd clinical-report-assistant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Build the dataset:

```bash
# Download raw reports from the official Open-i source
python -m src.data.download_openi --out data/raw

# Parse XML reports into a structured JSONL corpus
python -m src.data.preprocess --raw data/raw --out data/processed
```

Fine-tune, index, and serve (fine-tuning and indexing require a GPU — see the module READMEs):

```bash
python -m src.finetuning.train --task extraction
python -m src.finetuning.train --task summarization
python -m src.rag.build_index --processed data/processed

# Serve locally, or via Docker Compose / Kubernetes
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
docker compose up --build
kubectl apply -f k8s/
```

### API

```bash
# Structured extraction
curl -s localhost:8000/extract -H 'Content-Type: application/json' \
  -d '{"report_text": "Comparison: none. Findings: The heart size is mildly enlarged. Lungs are clear. Impression: Mild cardiomegaly, otherwise unremarkable."}'
# -> {"comparison": "...", "indication": "...", "findings": "...", "impression": "...", "diagnosis": ["cardiomegaly"]}

# Summarization
curl -s localhost:8000/summarize -H 'Content-Type: application/json' \
  -d '{"findings": "The heart size is mildly enlarged. Lungs are clear. No effusion."}'
# -> {"impression": "Mild cardiomegaly with otherwise clear lungs."}

# Grounded QA over retrieved past cases
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "What is typically found in cases of mild cardiomegaly with clear lungs?", "top_k": 5}'
# -> {"answer": "...", "reasoning": "...", "sources": ["CXR123", "CXR456"]}
```

Operational endpoints: `GET /health` (readiness + adapter/CUDA status) and `GET /metrics` (Prometheus-format request counts, a latency histogram, and tokens generated per adapter).

> **Note:** trained LoRA adapters and the Qdrant index are gitignored — they're model/clinical artifacts that only exist wherever training and indexing ran. See [`src/serving/README.md`](src/serving/README.md) for how to place them before starting the service.

## Dataset

**[Open-i / Indiana University Chest X-ray dataset](https://openi.nlm.nih.gov)** — openly licensed radiology reports from the NLM Open-i service, ~3,955 reports paired with ~7,470 images. Chosen over MIMIC-CXR specifically to avoid a credentialing/CITI-training delay. Each report XML contains structured sections (Comparison, Indication, Findings, Impression) plus MeSH/RadLex-coded diagnoses.

- Official bulk files: [openi.nlm.nih.gov/imgs/collections](https://openi.nlm.nih.gov/imgs/collections/)
- Mirror: [Kaggle — Chest X-rays (Indiana University)](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)

See [`data/README.md`](data/README.md) for the data card and processing details. This repository does not redistribute patient data.

## Repository tour

```
clinical-report-assistant/
├── src/
│   ├── data/         # download + XML→JSONL preprocessing
│   ├── finetuning/   # QLoRA training, prompt formatting, evaluation
│   ├── rag/          # vector indexing, retrieval, grounded QA
│   └── serving/      # FastAPI app (multi-adapter) + /metrics monitoring
├── configs/          # YAML configs (data, training, RAG)
├── notebooks/        # exploratory data analysis
├── docker/           # Dockerfile
├── k8s/              # Kubernetes manifests (kind)
├── tests/            # offline unit tests (no GPU required)
├── docs/             # architecture notes + full engineering write-up
└── .github/workflows # CI (tests, image build, manifest lint)
```

## Limitations & future work

Deliberately scoped out, and honest about it:

- **No controlled base-model baseline** for the extraction/summarization metrics. The reported numbers are the fine-tuned model's absolute performance; an apples-to-apples "base model on the same narrow tasks" comparison hasn't been run.
- **RAG-QA is validated qualitatively**, not with a formal retrieval metric (precision@k) yet.
- **Serving latency/throughput are not yet benchmarked.** The `/metrics` histogram is in place to measure p50/p95, but no load test has been run.
- **Monitoring stops at a real `/metrics` endpoint** plus Kubernetes scrape annotations — a live Prometheus/Grafana stack is intentionally not deployed. The goal was to demonstrate knowing *what* to instrument.
- **Deployment is single-node `kind`**, not a cloud cluster.

## License

MIT — see [LICENSE](LICENSE). The underlying Open-i dataset has its own usage terms; see [NLM Open-i](https://openi.nlm.nih.gov).
