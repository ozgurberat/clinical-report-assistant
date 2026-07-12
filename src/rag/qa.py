"""RAG-QA: retrieve similar past reports, then generate a grounded answer.

Unlike the extraction and summarization fine-tunes, this uses the PLAIN BASE
Qwen3-4B model with NO adapter attached at all. Neither fine-tune is suited
to open-ended question answering (see the earlier adapter-vs-base
experiment, where both tried to force an unrelated question into their
trained output shape) — the base instruct model is already capable of
synthesizing a good answer once handed real, retrieved evidence, with no
additional training required.

One deliberate difference from generate() in evaluate.py: thinking mode is
left ON here, not suppressed. Extraction and summarization are simple,
single-step transformations our fine-tunes never modeled a reasoning trace
for, so an empty <think></think> block was pure waste there. RAG-QA is the
opposite case — synthesizing across several retrieved documents to reach one
conclusion is exactly the kind of multi-step task thinking mode exists for,
and this call never touches a fine-tuned adapter, so there's no risk of an
empty, untrained thinking block — the base model's reasoning behavior here
is the real thing it was actually trained to do.

Usage:
    python -m src.rag.qa --question "..." --show-reasoning
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.rag.retrieve import retrieve_similar

BASE_MODEL = "Qwen/Qwen3-4B"
THINK_PATTERN = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)

SYSTEM_QA = (
    "You are a radiology assistant answering questions using ONLY the "
    "retrieved past case reports provided below as context. Reference "
    "specific report IDs when you rely on them. If the retrieved cases "
    "don't contain enough information to answer confidently, say so rather "
    "than guessing."
)


def build_context(retrieved: list[dict]) -> str:
    """Turn the retrieved reports into the reference-material block pasted
    into the prompt — one case per block, report_id first so the model has
    something concrete to cite."""
    blocks = []
    for r in retrieved:
        lines = [f"[Report {r['report_id']}] (similarity={r['score']:.2f})"]
        if r.get("indication"):
            lines.append(f"Indication: {r['indication']}")
        if r.get("comparison"):
            lines.append(f"Comparison: {r['comparison']}")
        lines.append(f"Findings: {r['findings']}")
        lines.append(f"Impression: {r['impression']}")
        if r.get("diagnosis"):
            lines.append(f"Diagnosis: {', '.join(r['diagnosis'])}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def load_base_model():
    """Load Qwen3-4B quantized, with no adapter attached — see module
    docstring for why this is deliberately different from evaluate.py's
    load_finetuned()."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def parse_think_output(text: str) -> tuple[str, str]:
    """Split a model's raw decoded output into (reasoning, answer).

    Three cases:
    1. A complete <think>...</think> block followed by the real answer —
       the normal case.
    2. Text that starts with <think> but never closes it — generation hit
       max_new_tokens while still reasoning, so there's no real answer at
       all. Surfaced explicitly rather than silently returning the raw,
       unclosed trace as if it were the answer (this happened in practice
       with the original max_new_tokens=512 default — see generate_answer).
    3. No <think> tag at all — just return the text as-is.

    Pulled out as its own pure function (no model/tokenizer involved) so this
    parsing logic is unit-testable without a GPU — see tests/test_qa.py."""
    match = THINK_PATTERN.search(text)
    if match:
        return match.group(1).strip(), THINK_PATTERN.sub("", text).strip()
    if text.lstrip().startswith("<think>"):
        reasoning = text.split("<think>", 1)[1].strip()
        answer = (
            "[No final answer — generation ran out of tokens while still "
            "reasoning. Raise max_new_tokens and retry.]"
        )
        return reasoning, answer
    return "", text.strip()


def generate_answer(
    model, tokenizer, question: str, context: str, max_new_tokens: int = 1536
) -> tuple[str, str]:
    """Returns (reasoning, answer) via parse_think_output() on the model's
    raw decoded output.

    max_new_tokens has been raised twice now: 512 wasn't enough headroom for
    reasoning over several retrieved documents plus a conclusion (generation
    hit the limit mid reasoning-block, no closing </think>, no answer at
    all); 1024 got further — a complete reasoning trace and most of a
    well-structured answer — but still truncated mid-sentence in the closing
    references section. 1536 gives real headroom past both observed failure
    points rather than nudging the ceiling up by the smallest amount that
    happened to work once."""
    import torch

    user_content = f"Similar past cases, most relevant first:\n\n{context}\n\nQuestion: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_QA},
        {"role": "user", "content": user_content},
    ]
    # enable_thinking=True explicitly — see module docstring for why this is
    # the opposite choice from generate() in evaluate.py.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    return parse_think_output(text)


def answer_question(
    question: str,
    top_k: int | None = None,
    processed: Path = Path("data/processed"),
) -> dict:
    """End-to-end RAG-QA: retrieve -> build context -> generate. Returns
    {question, answer, reasoning, sources, retrieved}."""
    retrieved = retrieve_similar(question, top_k=top_k, processed=processed)
    context = build_context(retrieved)

    model, tokenizer = load_base_model()
    reasoning, answer = generate_answer(model, tokenizer, question, context)

    import torch

    del model
    torch.cuda.empty_cache()

    return {
        "question": question,
        "answer": answer,
        "reasoning": reasoning,
        "sources": [r["report_id"] for r in retrieved],
        "retrieved": retrieved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--show-reasoning", action="store_true", help="Also print the model's <think> trace."
    )
    args = parser.parse_args()

    result = answer_question(args.question, top_k=args.top_k, processed=args.processed)

    print(f"[qa] Question: {result['question']}\n")
    print(f"[qa] Retrieved sources: {result['sources']}\n")
    if args.show_reasoning and result["reasoning"]:
        print("[qa] Model's reasoning trace:")
        print(result["reasoning"])
        print()
    print("[qa] Answer:")
    print(result["answer"])


if __name__ == "__main__":
    main()
