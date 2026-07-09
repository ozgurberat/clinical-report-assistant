"""Parse raw Open-i XML radiology reports into a clean, structured corpus.

Usage:
    python -m src.data.preprocess --raw data/raw/reports --out data/processed

Produces:
    data/processed/reports.jsonl   one structured record per report
    data/processed/reports.csv     flattened view of the same data
    data/processed/splits.json     train/val/test report-id splits
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data.yaml"

SECTION_LABELS = ("COMPARISON", "INDICATION", "FINDINGS", "IMPRESSION")


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_report(xml_path: Path) -> dict | None:
    """Parse a single Open-i report XML file into a flat dict, or None if unreadable."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    root = tree.getroot()

    sections = {label.lower(): "" for label in SECTION_LABELS}
    for abstract_text in root.iter("AbstractText"):
        label = (abstract_text.get("Label") or "").upper()
        if label in SECTION_LABELS:
            sections[label.lower()] = (abstract_text.text or "").strip()

    mesh_major = [
        (m.text or "").strip().lower()
        for m in root.iter("major")
        if m.text and m.text.strip()
    ]

    image_ids = [
        img.get("id") for img in root.iter("parentImage") if img.get("id")
    ]

    report_id = xml_path.stem

    return {
        "report_id": report_id,
        "comparison": sections["comparison"],
        "indication": sections["indication"],
        "findings": sections["findings"],
        "impression": sections["impression"],
        "mesh_major": mesh_major,
        "image_ids": image_ids,
    }


def build_splits(report_ids: list[str], cfg: dict) -> dict:
    split_cfg = cfg["split"]
    rng = random.Random(split_cfg["seed"])
    ids = list(report_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * split_cfg["train"])
    n_val = int(n * split_cfg["val"])

    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Dir containing report XML files (searched recursively).")
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(args.raw.rglob("*.xml"))
    if not xml_files:
        raise SystemExit(
            f"No XML files found under {args.raw}. "
            "Run download_openi.py first (see src/data/download_openi.py)."
        )

    records = []
    dropped = 0
    min_chars = cfg["preprocessing"]["min_findings_chars"]
    for xml_path in xml_files:
        rec = parse_report(xml_path)
        if rec is None or len(rec["findings"]) < min_chars:
            dropped += 1
            continue
        records.append(rec)

    jsonl_path = args.out / "reports.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    csv_path = args.out / "reports.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["report_id", "comparison", "indication", "findings", "impression", "mesh_major", "image_ids"],
        )
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            row["mesh_major"] = ";".join(rec["mesh_major"])
            row["image_ids"] = ";".join(rec["image_ids"])
            writer.writerow(row)

    splits = build_splits([r["report_id"] for r in records], cfg)
    with open(args.out / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print(f"[done] Parsed {len(records)} reports ({dropped} dropped as empty/invalid).")
    print(f"        -> {jsonl_path}")
    print(f"        -> {csv_path}")
    print(f"        -> {args.out / 'splits.json'} "
          f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})")


if __name__ == "__main__":
    main()
