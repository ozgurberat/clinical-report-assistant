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
following). QLoRA training script — not yet implemented.
