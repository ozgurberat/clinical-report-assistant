"""QLoRA fine-tuning for the clinical report assistant.

Trains a LoRA adapter on top of a 4-bit quantized base model (Qwen3-4B by
default) for one of the two tasks (extraction, summarization), using the
chat-formatted JSONL files produced by prompt_format.py. Tracks each run
in MLflow.

Usage:
    python -m src.finetuning.train --task extraction
    python -m src.finetuning.train --task summarization

Requires a GPU — run this in Colab (Runtime > Change runtime type > T4 GPU),
not in a CPU-only environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "training.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pick_precision() -> tuple[torch.dtype, bool, bool]:
    """Choose bf16 vs fp16 based on what the actual GPU supports, rather than
    a static config flag. T4 (Turing, Colab free tier) lacks native bf16
    tensor core support and runs it unaccelerated; A100/L4/H100 (Ampere or
    newer, e.g. Colab Pro) support bf16 natively and it's the better choice
    there (matches the dtype most LLMs are pretrained in, avoids fp16's
    occasional gradient-scaling overflow issues). Returns
    (compute_dtype, use_bf16, use_fp16)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[train] GPU: {gpu_name} — bf16 natively supported, using bf16.")
        return torch.bfloat16, True, False
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[train] GPU: {gpu_name} — no native bf16 support, using fp16.")
    return torch.float16, False, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["extraction", "summarization"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    compute_dtype, use_bf16, use_fp16 = pick_precision()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_files = {
        "train": str(args.processed / f"finetune_{args.task}_train.jsonl"),
        "validation": str(args.processed / f"finetune_{args.task}_val.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)

    run_name = f"{args.task}-{cfg['model']['name'].split('/')[-1]}"
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    sft_config = SFTConfig(
        output_dir=str(Path(cfg["training"]["output_dir"]) / run_name),
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy=cfg["training"]["eval_strategy"],
        save_strategy=cfg["training"]["save_strategy"],
        load_best_model_at_end=True,  # keep the best-eval checkpoint, not just whatever ran last
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_fp16,
        seed=cfg["training"]["seed"],
        max_length=cfg["model"]["max_seq_length"],
        assistant_only_loss=True,  # only the assistant's reply contributes to the loss
        report_to="mlflow",
        run_name=run_name,
    )

    # Dataset already has a "messages" column in chat format (see prompt_format.py) —
    # SFTTrainer detects conversational data and applies the model's own chat
    # template internally, so no manual formatting_func is needed here.
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "task": args.task,
                "base_model": cfg["model"]["name"],
                "lora_r": cfg["lora"]["r"],
                "lora_alpha": cfg["lora"]["lora_alpha"],
                "lora_dropout": cfg["lora"]["lora_dropout"],
                "learning_rate": cfg["training"]["learning_rate"],
                "epochs": cfg["training"]["num_train_epochs"],
                "train_examples": len(dataset["train"]),
                "val_examples": len(dataset["validation"]),
            }
        )
        trainer.train()

    adapter_dir = Path(cfg["training"]["output_dir"]) / run_name / "final_adapter"
    trainer.save_model(str(adapter_dir))
    print(f"[done] Adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
