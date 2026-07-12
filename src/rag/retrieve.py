"""Query the RAG vector index for the most similar historical reports.

This is the "R" (retrieval) in RAG on its own, with no LLM involved yet —
just: embed a query the same way build_index.py embedded every report, ask
Qdrant for the nearest stored vectors, and return what it finds. Worth
running and reading the output by eye before wiring any generation on top,
same reasoning as inspecting the fine-tuning data before ever training on it.

Usage (as a library, e.g. from the RAG-QA layer later):
    from src.rag.retrieve import retrieve_similar
    results = retrieve_similar("mild cardiomegaly, clear lungs", top_k=5)

Usage (quick CLI check):
    python -m src.rag.retrieve --query "mild cardiomegaly, clear lungs" --top-k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "rag.yaml"

# Module-level cache so repeated calls (e.g. from separate notebook cells or
# multiple queries in a loop) don't reload the embedding model or reopen the
# Qdrant index every single time.
_embedder = None
_client = None
_cfg = None


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_clients(processed: Path, config_path: Path):
    global _embedder, _client, _cfg
    if _embedder is None:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        _cfg = load_config(config_path)
        _embedder = SentenceTransformer(_cfg["embedding"]["model_name"])
        index_path = processed / _cfg["qdrant"]["path"]
        _client = QdrantClient(path=str(index_path))
    return _embedder, _client, _cfg


def retrieve_similar(
    query_text: str,
    top_k: int | None = None,
    processed: Path = Path("data/processed"),
    config_path: Path = DEFAULT_CONFIG,
) -> list[dict]:
    """Embed query_text and return the top_k most similar reports, each as
    {report_id, comparison, indication, findings, impression, diagnosis, score}.
    score is cosine similarity — higher means more similar, 1.0 is identical."""
    embedder, client, cfg = _get_clients(processed, config_path)
    k = top_k or cfg["retrieval"]["top_k"]

    query_vector = embedder.encode(query_text, convert_to_numpy=True).tolist()
    # search() is deprecated in current qdrant-client in favor of the more
    # general query_points(); with_payload defaults to False, so it must be
    # requested explicitly or every hit comes back with an empty payload.
    response = client.query_points(
        collection_name=cfg["qdrant"]["collection_name"],
        query=query_vector,
        limit=k,
        with_payload=True,
    )

    results = []
    for point in response.points:
        result = dict(point.payload)
        result["score"] = point.score
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    results = retrieve_similar(args.query, top_k=args.top_k, processed=args.processed)
    print(f"[retrieve] Top {len(results)} matches for: {args.query!r}\n")
    for i, r in enumerate(results, 1):
        print(f"--- #{i} (score={r['score']:.4f}, report_id={r['report_id']}) ---")
        print(f"Findings:   {r['findings']}")
        print(f"Impression: {r['impression']}")
        print(f"Diagnosis:  {r['diagnosis']}")
        print()


if __name__ == "__main__":
    main()
