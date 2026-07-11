# Phase 2 — QLoRA fine-tuning

## Prompt formatting (done)

`prompt_format.py` converts `data/processed/reports.jsonl` + `splits.json` into
chat-formatted train/val/test JSONL files for two tasks:

- **extraction** — raw unlabeled report text -> structured JSON (comparison/
  indication/findings/impression/diagnosis). Simulates realistic unstructured
  dictation input, since the source XML already comes pre-split.
- **summarization** — findings (+ indication/comparison when present) -> impression.

```bash
python -m src.finetuning.prompt_format --processed data/processed
```

Base model: **Qwen3-4B** (Apache 2.0, no gating, strong small-model instruction
following).

## Training (done — both tasks trained on A100, adapters saved)

`train.py` loads Qwen3-4B in 4-bit (bitsandbytes NF4), applies a LoRA adapter
(peft), and trains with TRL's `SFTTrainer` using `assistant_only_loss=True`
(only the assistant's response contributes to the loss). Tracked in MLflow.
`load_best_model_at_end=True` (by `eval_loss`) so the saved adapter is the
best epoch, not just whatever ran last.

```bash
python -m src.finetuning.train --task extraction
python -m src.finetuning.train --task summarization
```

Config in `configs/training.yaml` (LoRA rank/alpha/dropout, learning rate,
epochs, quantization settings). Run via `notebooks/02_finetune.ipynb` — requires
a GPU runtime, unlike the CPU-only exploration notebook. Compute dtype
(bf16/fp16) auto-detects from the actual GPU at runtime.

Both adapters have been qualitatively spot-checked (`notebooks/02_finetune.ipynb`,
section 7): extraction hits exact-match on held-out examples, summarization
produces clinically-equivalent paraphrases of the ground-truth impression.

## Evaluation (in progress)

`evaluate.py` runs the full held-out test set (343 examples/task) through each
fine-tuned adapter and computes real metrics, rather than eyeballing a handful
of examples:

- **extraction** — per-field token-F1, diagnosis-list set-F1, whole-target
  exact-match rate, JSON-parse failure rate.
- **summarization** — ROUGE-1/2/L F-measure, exact-match rate.

```bash
python -m src.finetuning.evaluate --task extraction
python -m src.finetuning.evaluate --task summarization
```

Results are printed and saved to `outputs/<task>-Qwen3-4B/test_metrics.json`.
The pure scoring functions (`token_f1`, `set_f1`) are unit-tested in
`tests/test_evaluate.py` without needing a GPU.
