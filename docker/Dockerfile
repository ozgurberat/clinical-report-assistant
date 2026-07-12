# Serving image for the Clinical Report Assistant FastAPI app.
#
# Build context is the repo root (see docker-compose.yml), since the app
# imports across src/finetuning, src/rag, and src/serving.
#
# Trained artifacts (LoRA adapters, the Qdrant index) are NOT baked into this
# image — they're gitignored clinical/model data that only exists wherever
# training/indexing actually ran. They're mounted in at runtime instead (see
# docker-compose.yml's volumes) — see src/serving/README.md for how to get
# them onto whatever machine runs this container.

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first, separately from copying the source, so
# `docker build` can cache this layer across rebuilds that only touch code.
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src/ src/
COPY configs/ configs/
COPY pyproject.toml .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
