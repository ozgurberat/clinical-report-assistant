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

## Training (done, not yet run to completion)

`train.py` loads Qwen3-4B in 4-bit (bitsandbytes NF4), applies a LoRA adapter
(peft), and trains with TRL's `SFTTrainer` using completion-only loss masking
(only the assistant's response contributes to the loss — see the response
template detection in `find_response_template`). Tracked in MLflow.

```bash
python -m src.finetuning.train --task extraction
python -m src.finetuning.train --task summarization
```

Config in `configs/training.yaml` (LoRA rank/alpha/dropout, learning rate,
epochs, quantization settings). Run via `notebooks/02_finetune.ipynb` — requires
a GPU runtime, unlike the CPU-only exploration notebook.
