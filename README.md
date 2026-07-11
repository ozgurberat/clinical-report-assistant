# Clinical Report Assistant

A fine-tuned, RAG-augmented LLM system for structured extraction, summarization, and question-answering over radiology reports — built on real clinical NLP data, served via quantized inference, containerized, and orchestrated with Kubernetes.

> **Status:** Phase 1 (data pipeline) and Phase 2 (QLoRA fine-tuning) done. Full test-set metrics in progress. See [Roadmap](#roadmap) for what's done vs. planned.

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
        │  (Llama/Qwen/Phi) │ │  (FAISS / Qdrant) │
        └─────────┬─────────┘ └────────┬──────────┘
                   │                    │
                   └─────────┬──────────┘
                              ▼
                   ┌────────────────────┐
                   │   RAG orchestrator │
                   │   (LangChain)      │
                   └──────────┬─────────┘
                              ▼
                   ┌────────────────────┐
                   │  Quantized serving │
                   │  (vLLM / TGI)      │
                   └──────────┬─────────┘
                              ▼
                   ┌────────────────────┐
                   │   FastAPI service  │
                   └──────────┬─────────┘
                              ▼
                Docker Compose → Kubernetes (minikube/kind)
                              │
                              ▼
                   Monitoring (logging / Prometheus+Grafana)
```

*(Diagram will be replaced with a rendered image once Phase 4 architecture is finalized.)*

## Dataset

**Primary:** [Open-i / Indiana University Chest X-ray dataset](https://openi.nlm.nih.gov) — openly licensed radiology reports and images from the NLM Open-i service, ~3,955 reports paired with ~7,470 images. No credentialing/CITI training required (chosen over MIMIC-CXR specifically to avoid that delay).

- Official bulk files: `NLMCXR_reports.tgz` (XML reports), `NLMCXR_png.tgz` (images) — https://openi.nlm.nih.gov/imgs/collections/
- Mirror: [Kaggle — Chest X-rays (Indiana University)](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
- Mirror: [Academic Torrents — XML reports](https://academictorrents.com/details/66450ba52ba3f83fbf82ef9c91f2bde0e845aba9) / [PNG images](https://academictorrents.com/details/5a3a439df24931f410fac269b87b050203d9467d)

Each report XML contains structured sections (Comparison, Indication, Findings, Impression) plus MeSH/RadLex-coded diagnoses. See [`data/README.md`](data/README.md) for the data card and processing details.

**Optional supplement (documented as synthetic-augmented if used):** template-generated synthetic reports for volume, and/or n2c2/i2b2 clinical NLP datasets for auxiliary tasks.

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Data acquisition, cleaning, repo scaffolding | 🟢 Done |
| 2 | QLoRA fine-tuning (field extraction + summarization), tracked in MLflow | 🟡 Fine-tuned + qualitatively validated; full test-set metrics in progress |
| 3 | RAG layer — vector DB (FAISS/Qdrant) + LangChain retrieval | ⚪ Planned |
| 4 | FastAPI + quantized serving (vLLM/TGI), Dockerized, docker-compose | ⚪ Planned |
| 5 | Kubernetes manifests (minikube/kind) + GitHub Actions CI/CD | ⚪ Planned |
| 6 | Monitoring (logging, optional Prometheus+Grafana) + write-up | ⚪ Planned |

Project is modular — a stop after Phase 4 still yields a complete fine-tuning + RAG + Docker deliverable.

## Repo structure

```
clinical-report-assistant/
├── data/
│   ├── raw/            # untouched downloaded files (gitignored)
│   ├── processed/      # cleaned JSONL/CSV corpora (gitignored)
│   └── README.md       # data card
├── src/
│   ├── data/            # download + preprocessing scripts
│   ├── finetuning/       # QLoRA training scripts (Phase 2)
│   ├── rag/              # indexing + retrieval (Phase 3)
│   ├── serving/           # FastAPI app + inference server config (Phase 4)
│   └── monitoring/        # logging/metrics (Phase 6)
├── configs/              # YAML configs (data, training, serving)
├── notebooks/            # exploratory analysis
├── docker/               # Dockerfiles + docker-compose
├── k8s/                  # Kubernetes manifests
├── tests/                # unit tests
├── docs/                 # architecture notes, write-up drafts
└── .github/workflows/    # CI/CD pipelines
```

## Setup

```bash
git clone <this-repo>
cd clinical-report-assistant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Data pipeline

```bash
# Download raw reports + images from the official Open-i source
python -m src.data.download_openi --out data/raw

# Parse XML reports into a structured JSONL corpus
python -m src.data.preprocess --raw data/raw --out data/processed
```

See [`src/data/download_openi.py`](src/data/download_openi.py) and [`src/data/preprocess.py`](src/data/preprocess.py) for details, and [`data/README.md`](data/README.md) for the resulting schema.

## Results

*(Populated as phases complete.)*

| Metric | Baseline | Fine-tuned | Notes |
|---|---|---|---|
| Field extraction F1 | — | — | Findings / Impression / Diagnosis |
| Summarization ROUGE-L | — | — | |
| RAG retrieval precision@k | — | — | |
| Inference latency (p50/p95) | — | — | Pre- vs. post-quantization |
| Throughput (req/s) | — | — | vLLM/TGI benchmark |

## License

MIT — see [LICENSE](LICENSE). Note: underlying Open-i dataset has its own usage terms; see [NLM Open-i](https://openi.nlm.nih.gov) for details. This repo does not redistribute patient data.
