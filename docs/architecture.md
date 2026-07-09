# Architecture notes

_Placeholder — expand as each phase lands._

## Phase 2 — Fine-tuning

- Base model candidates: Llama 3.2 3B, Qwen2.5 3B/7B, Phi-3-mini.
- QLoRA via Hugging Face PEFT + TRL; 4-bit NF4 quantization via bitsandbytes.
- Tasks: (a) structured field extraction (findings/impression/diagnosis), (b) summarization/QA over reports.
- Experiment tracking: MLflow (or W&B) — compare LoRA rank, learning rate, quantization settings.
- Compute: Colab/Kaggle free-tier GPU, or RunPod/Vast.ai spot instances (~$0.20–0.50/hr) if needed.

## Phase 3 — RAG

- Vector DB: FAISS for local dev, Qdrant (self-hosted via Docker) for the "production" path.
- Orchestration: LangChain (or LlamaIndex) for retrieval + prompt assembly.
- Indexed corpus: larger reference set of reports/findings so generation can cite/retrieve similar prior cases.

## Phase 4 — Serving

- FastAPI wrapper around the model.
- Inference backend: vLLM or Hugging Face TGI for quantized serving.
- Benchmark: latency/throughput before vs. after quantization — logged as a concrete results table entry.
- Containerized with Docker; multi-service via docker-compose (API, model server, vector DB).

## Phase 5 — Orchestration & CI/CD

- Kubernetes manifests derived from docker-compose, validated locally via kind/minikube.
- GitHub Actions: test → build image → (optional) deploy.
- Optional public demo: Hugging Face Spaces or Fly.io/Render free tier.

## Phase 6 — Monitoring

- Baseline: structured request/latency/error logging.
- Stretch: Prometheus + Grafana dashboards.
