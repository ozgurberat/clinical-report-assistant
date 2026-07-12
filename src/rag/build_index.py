"""Build a searchable vector index over the full report corpus, for RAG retrieval.

Unlike the fine-tuning splits (train/val/test), this indexes EVERY report in
data/processed/reports.jsonl. RAG retrieval simulates searching a real
historical case archive, not a training set — there's no "held-out" concept
here, since nothing is being trained. Each report is embedded once, offline,
and stored in a local (embedded, no-server) Qdrant collection; queries at
question-time embed once more and search this same index.

Usage:
    python -m src.rag.build_index --processed data/processed

Produces a local Qdrant collection on disk at
data/processed/<configs/rag.yaml qdrant.path>, alongside the rest of the
processed corpus so it persists in Drive across Colab sessions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "rag.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_embedding_text(record: dict) -> str:
    """Turn one report record into the single text blob that gets embedded.

    Similarity search should match on CLINICAL CONTENT — what was found and
    concluded — so this uses comparison/indication/findings/impression, the
    same de-identification-cleaned fields prompt_format.py already relies on
    for fine-tuning. Empty sections are skipped rather than embedding empty
    labels, so a report missing "comparison" doesn't get a literal "Comparison:"
    with nothing after it baked into its embedding."""
    parts = []
    for label, key in [
        ("Comparison", "comparison"),
        ("Indication", "indication"),
        ("Findings", "findings"),
        ("Impression", "impression"),
    ]:
        value = (record.get(key) or "").strip()
        if value:
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


def build_payload(record: dict) -> dict:
    """What gets stored alongside each vector, retrievable at query time
    without needing to re-open reports.jsonl."""
    return {
        "report_id": record["report_id"],
        "comparison": record.get("comparison", ""),
        "indication": record.get("indication", ""),
        "findings": record.get("findings", ""),
        "impression": record.get("impression", ""),
        "diagnosis": record.get("mesh_major", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Heavy deps deferred to keep build_embedding_text/build_payload testable
    # without sentence-transformers/qdrant-client installed — same pattern as
    # src/finetuning/evaluate.py.
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer

    reports_path = args.processed / "reports.jsonl"
    if not reports_path.exists():
        raise SystemExit(f"Expected {reports_path} to exist. Run src/data/preprocess.py first.")

    records = [json.loads(line) for line in open(reports_path)]
    print(f"[index] Loaded {len(records)} reports from {reports_path}")

    texts, payloads = [], []
    skipped = 0
    for rec in records:
        text = build_embedding_text(rec)
        if not text:
            skipped += 1
            continue
        texts.append(text)
        payloads.append(build_payload(rec))
    print(f"[index] {len(texts)} reports have embeddable content ({skipped} skipped, all sections empty)")

    model_name = cfg["embedding"]["model_name"]
    print(f"[index] Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)

    print(f"[index] Embedding {len(texts)} reports...")
    vectors = embedder.encode(texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True)
    vector_dim = vectors.shape[1]
    print(f"[index] Produced {vectors.shape[0]} vectors of dimension {vector_dim}")

    index_path = args.processed / cfg["qdrant"]["path"]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(index_path))

    collection_name = cfg["qdrant"]["collection_name"]
    # recreate_collection() is deprecated in current qdrant-client — do the
    # delete-then-create explicitly so rerunning this script is idempotent.
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(id=i, vector=vectors[i].tolist(), payload=payloads[i])
        for i in range(len(payloads))
    ]
    client.upsert(collection_name=collection_name, points=points)

    print(f"[done] Indexed {len(points)} reports into '{collection_name}' at {index_path}")


if __name__ == "__main__":
    main()
