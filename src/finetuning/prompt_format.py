"""Turn the cleaned Open-i corpus (data/processed/reports.jsonl) into
chat-formatted training examples for two QLoRA fine-tuning tasks:

  extraction     — raw unlabeled report text -> structured JSON
                   (comparison/indication/findings/impression/diagnosis)
  summarization  — findings (+ indication/comparison) -> impression

Usage:
    python -m src.finetuning.prompt_format --processed data/processed

Produces, per task, per split:
    data/processed/finetune_extraction_train.jsonl
    data/processed/finetune_extraction_val.jsonl
    data/processed/finetune_extraction_test.jsonl
    data/processed/finetune_summarization_train.jsonl
    data/processed/finetune_summarization_val.jsonl
    data/processed/finetune_summarization_test.jsonl

Each line is a chat-message-format example:
    {"report_id": ..., "messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
compatible with tokenizer.apply_chat_template() / TRL's SFTTrainer.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Open-i's de-identification placeholder (ages, dates, etc. get swept into
# literal "XXXX" tokens). Left as-is, a fine-tuned model would learn to
# output "XXXX" verbatim, which looks broken — swap in a cleaner marker.
DEIDENT_PATTERN = re.compile(r"XXXX")
REDACTED = "[REDACTED]"

# MeSH tags that are administrative/quality-control metadata, not real
# diagnoses (found during EDA — see data/README.md).
ADMIN_MESH_TAGS = {"no indexing", "technical quality of image unsatisfactory"}

SYSTEM_EXTRACTION = (
    "You are a radiology report assistant. Given the raw text of a chest "
    "X-ray report, extract its structured sections and diagnosis as JSON "
    "with keys: comparison, indication, findings, impression, diagnosis."
)

SYSTEM_SUMMARIZATION = (
    "You are a radiology report assistant. Given the findings section of a "
    "chest X-ray report (and indication/comparison when available), write "
    "the impression: a concise clinical summary."
)


def clean_deidentified(text: str) -> str:
    return DEIDENT_PATTERN.sub(REDACTED, text)


def clean_diagnosis(mesh_major: list[str]) -> list[str]:
    return [term for term in mesh_major if term not in ADMIN_MESH_TAGS]


def build_extraction_example(record: dict) -> dict | None:
    """Reconstruct an unlabeled 'raw dictation' input from the parsed
    sections, and train the model to recover the structured JSON."""
    section_values = [
        clean_deidentified(record[key].strip())
        for key in ("comparison", "indication", "findings", "impression")
        if record[key].strip()
    ]
    raw_text = " ".join(section_values)
    if not raw_text:
        return None

    target = {
        "comparison": clean_deidentified(record["comparison"].strip()),
        "indication": clean_deidentified(record["indication"].strip()),
        "findings": clean_deidentified(record["findings"].strip()),
        "impression": clean_deidentified(record["impression"].strip()),
        "diagnosis": clean_diagnosis(record["mesh_major"]),
    }

    return {
        "report_id": record["report_id"],
        "messages": [
            {"role": "system", "content": SYSTEM_EXTRACTION},
            {"role": "user", "content": raw_text},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
    }


def build_summarization_example(record: dict) -> dict | None:
    """findings (+ optional indication/comparison) -> impression.
    Skipped when impression is empty — no valid target to train on
    (6 such reports in the corpus, see data/README.md)."""
    impression = clean_deidentified(record["impression"].strip())
    if not impression:
        return None

    findings = clean_deidentified(record["findings"].strip())
    indication = clean_deidentified(record["indication"].strip())
    comparison = clean_deidentified(record["comparison"].strip())

    user_lines = []
    if indication:
        user_lines.append(f"Indication: {indication}")
    if comparison:
        user_lines.append(f"Comparison: {comparison}")
    user_lines.append(f"Findings: {findings}")

    return {
        "report_id": record["report_id"],
        "messages": [
            {"role": "system", "content": SYSTEM_SUMMARIZATION},
            {"role": "user", "content": "\n".join(user_lines)},
            {"role": "assistant", "content": impression},
        ],
    }


TASK_BUILDERS = {
    "extraction": build_extraction_example,
    "summarization": build_summarization_example,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    reports_path = args.processed / "reports.jsonl"
    splits_path = args.processed / "splits.json"
    if not reports_path.exists() or not splits_path.exists():
        raise SystemExit(
            f"Expected {reports_path} and {splits_path} to exist. "
            "Run src/data/preprocess.py first."
        )

    records = [json.loads(line) for line in open(reports_path)]
    by_id = {r["report_id"]: r for r in records}
    splits = json.loads(splits_path.read_text())

    for task_name, builder in TASK_BUILDERS.items():
        for split_name, report_ids in splits.items():
            examples = []
            skipped = 0
            for report_id in report_ids:
                rec = by_id.get(report_id)
                if rec is None:
                    continue
                example = builder(rec)
                if example is None:
                    skipped += 1
                    continue
                examples.append(example)

            out_path = args.processed / f"finetune_{task_name}_{split_name}.jsonl"
            with open(out_path, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

            print(
                f"[{task_name:13s}/{split_name:5s}] {len(examples):4d} examples "
                f"({skipped} skipped, no valid target) -> {out_path}"
            )


if __name__ == "__main__":
    main()
