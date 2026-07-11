"""Full test-set evaluation for the fine-tuned extraction and summarization models.

The qualitative check in the notebook only reads 3 examples by eye — good for
catching gross failures, useless for a results table. This runs every held-out
test example through the fine-tuned adapter and computes real metrics:

- extraction: per-field token-F1 (comparison/indication/findings/impression),
  set-F1 on the diagnosis list, and whole-target exact-match / JSON-parse
  failure rate.
- summarization: ROUGE-1/2/L F-measure against the ground-truth impression,
  plus exact-match rate.

Usage:
    python -m src.finetuning.evaluate --task extraction
    python -m src.finetuning.evaluate --task summarization

Requires a GPU (loads the base model + adapter) — run in Colab, same as
train.py. Add --limit N to sanity-check on a handful of examples before
committing to the full run.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Heavy ML imports (torch/peft/transformers/rouge_score) are deferred into the
# functions that need them rather than imported at module level. This keeps
# token_f1/set_f1 importable and unit-testable in any plain Python environment
# (no GPU, no ML stack) — see tests/test_evaluate.py.

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "Qwen/Qwen3-4B"
THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def load_finetuned(task: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, f"outputs/{task}-Qwen3-4B/final_adapter")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 256) -> str:
    import torch

    # Only system + user go in — the model generates the assistant turn.
    prompt = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return THINK_PATTERN.sub("", text).strip()


def token_f1(pred: str, gold: str) -> float:
    """Word-overlap F1 (SQuAD-style) — partial credit for near-miss text fields,
    unlike exact-match which would score a one-word difference as a total miss."""
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    gold_counts: dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    pred_counts: dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    overlap = sum(min(c, pred_counts.get(t, 0)) for t, c in gold_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def set_f1(pred: list[str], gold: list[str]) -> float:
    """Set-overlap F1 for the diagnosis list, where order doesn't matter and
    partial overlap (2 of 3 correct tags) should score better than a full miss."""
    pred_set, gold_set = set(pred), set(gold)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    overlap = len(pred_set & gold_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def evaluate_extraction(model, tokenizer, examples: list[dict]) -> dict:
    fields = ["comparison", "indication", "findings", "impression"]
    field_scores: dict[str, list[float]] = {f: [] for f in fields}
    diagnosis_scores = []
    exact_matches = 0
    parse_failures = 0

    for i, ex in enumerate(examples):
        gold_raw = ex["messages"][2]["content"]
        gold = json.loads(gold_raw)
        pred_text = generate(model, tokenizer, ex["messages"])

        if pred_text == gold_raw:
            exact_matches += 1

        try:
            pred = json.loads(pred_text)
        except json.JSONDecodeError:
            parse_failures += 1
            # A parse failure means every field scores 0 for this example —
            # still counts toward the averages, it's a real failure mode.
            pred = {}

        for field in fields:
            field_scores[field].append(token_f1(str(pred.get(field, "")), str(gold.get(field, ""))))
        diagnosis_scores.append(set_f1(pred.get("diagnosis", []), gold.get("diagnosis", [])))

        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(examples)}", flush=True)

    n = len(examples)
    metrics = {
        "n_examples": n,
        "exact_match_rate": exact_matches / n,
        "json_parse_failure_rate": parse_failures / n,
        "diagnosis_f1": sum(diagnosis_scores) / n,
    }
    for field in fields:
        metrics[f"{field}_f1"] = sum(field_scores[field]) / n
    return metrics


def evaluate_summarization(model, tokenizer, examples: list[dict]) -> dict:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    exact_matches = 0

    for i, ex in enumerate(examples):
        gold = ex["messages"][2]["content"]
        pred = generate(model, tokenizer, ex["messages"])
        if pred.strip() == gold.strip():
            exact_matches += 1
        result = scorer.score(gold, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(examples)}", flush=True)

    n = len(examples)
    metrics = {"n_examples": n, "exact_match_rate": exact_matches / n}
    for key, vals in scores.items():
        metrics[f"{key}_f"] = sum(vals) / n
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["extraction", "summarization"], required=True)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--limit", type=int, default=None, help="Only evaluate the first N test examples."
    )
    args = parser.parse_args()

    with open(args.processed / f"finetune_{args.task}_test.jsonl") as f:
        examples = [json.loads(line) for line in f]
    if args.limit:
        examples = examples[: args.limit]

    print(f"[eval] Loading fine-tuned {args.task} model...", flush=True)
    model, tokenizer = load_finetuned(args.task)

    print(f"[eval] Generating + scoring {len(examples)} test examples...", flush=True)
    if args.task == "extraction":
        metrics = evaluate_extraction(model, tokenizer, examples)
    else:
        metrics = evaluate_summarization(model, tokenizer, examples)

    print(f"\n[eval] Results for {args.task} ({len(examples)} test examples):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    out_path = Path("outputs") / f"{args.task}-Qwen3-4B" / "test_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[done] Saved to {out_path}")


if __name__ == "__main__":
    main()
