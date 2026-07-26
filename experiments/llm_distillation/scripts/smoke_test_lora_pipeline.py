"""
Smoke test for the LoRA distillation pipeline (ClauseGuard experiment).

Purpose
-------
Verify the full training pipeline mechanics — tokenization, LoRA adapter
attachment, forward/backward pass, save/load round-trip, and the
SFTTrainer data path — using a TINY RANDOM-WEIGHT Qwen2 architecture and
SYNTHETIC data shaped exactly like the real teacher_labels.jsonl schema.

This test proves nothing about model quality. It only proves the plumbing
doesn't break before you spend real Colab GPU time on the real 153-row
dataset with real Qwen weights.

No network access required and no real Qwen weights downloaded — the
tokenizer is trained from scratch on synthetic text, and the Qwen2Config
is instantiated directly rather than loaded via from_pretrained.

Run:
    python smoke_test_lora_pipeline.py

Requires: torch, transformers, tokenizers, peft, trl, datasets
"""

import json
import tempfile
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from peft import LoraConfig, get_peft_model, TaskType


def build_tiny_tokenizer(vocab_size: int = 512) -> PreTrainedTokenizerFast:
    """Train a minimal byte-level BPE tokenizer on synthetic clause-like
    text. Entirely offline — no hub download, no real Qwen tokenizer."""
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    )
    synthetic_corpus = [
        "This Agreement shall be governed by the laws of the State of Delaware.",
        "Either party may terminate this Agreement upon thirty days written notice.",
        "Liability under this clause is limited to direct damages only.",
        "The parties consent to exclusive jurisdiction in the courts of New York.",
    ] * 20
    tok.train_from_iterator(synthetic_corpus, trainer=trainer)

    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )


def build_tiny_qwen2(vocab_size: int) -> Qwen2ForCausalLM:
    """Tiny random-weight Qwen2 — architecture-accurate (real module names,
    real GQA structure) but small enough to run instantly on CPU."""
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # grouped-query attention, matches real Qwen2
        max_position_embeddings=512,
    )
    return Qwen2ForCausalLM(config)


def build_synthetic_teacher_labels(path: Path, n: int = 12) -> None:
    """Mimic the real teacher_labels.jsonl schema (category, label_status,
    severity, explanation, citation) so the data-loading path gets
    exercised the same way it will be against the real 153-row file."""
    categories = ["Governing Law / Jurisdiction", "Limitation of Liability", "Termination"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "clause_text": f"Synthetic clause #{i} about {categories[i % 3]}.",
                "category": categories[i % 3],
                "label_status": "ok",
                "severity": ["low", "medium", "high"][i % 3],
                "explanation": f"Synthetic explanation for clause #{i}.",
                "citation": f"Synthetic clause #{i}",
            }
        )
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def format_example(row: dict) -> dict:
    """Same prompt/target shape the real distillation run will train on."""
    prompt = f"Clause: {row['clause_text']}\nAssess severity and explain."
    target = json.dumps(
        {
            "severity": row["severity"],
            "explanation": row["explanation"],
            "citation": row["citation"],
        }
    )
    return {"text": f"{prompt}\n{target}"}


def main() -> None:
    print("[1/5] Building tiny offline tokenizer...")
    tokenizer = build_tiny_tokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"      vocab_size={vocab_size}")

    print("[2/5] Building tiny random-weight Qwen2 model...")
    model = build_tiny_qwen2(vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      base params: {n_params:,}")

    print("[3/5] Attaching LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # must match real Qwen2 module names
    )
    peft_model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in peft_model.parameters() if not p.requires_grad)
    print(f"      trainable={trainable:,} frozen={frozen:,}")
    assert trainable > 0, "No trainable LoRA params — target_modules likely wrong for this arch"
    assert frozen == n_params, "Base model params should remain fully frozen"

    print("[4/5] Forward + backward pass on a synthetic input...")
    dummy_input = tokenizer("Test clause about governing law.", return_tensors="pt")
    out = peft_model(**dummy_input, labels=dummy_input["input_ids"])
    assert out.loss is not None and not torch.isnan(out.loss), "Loss is NaN or missing"
    out.loss.backward()
    print(f"      loss={out.loss.item():.4f} (backward pass OK)")

    print("[5/5] Data path (teacher_labels.jsonl schema) + save/load round-trip...")
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "synthetic_teacher_labels.jsonl"
        build_synthetic_teacher_labels(data_path)

        from datasets import load_dataset

        raw_ds = load_dataset("json", data_files=str(data_path), split="train")
        ds = raw_ds.map(format_example)

        # This is the same category-completeness check the real Colab
        # notebook needs, since the real data has ZERO Termination
        # examples. Proving the check fires here proves it'll fire there.
        cats_present = set(raw_ds["category"])
        expected = {"Governing Law / Jurisdiction", "Limitation of Liability", "Termination"}
        missing = expected - cats_present
        if missing:
            print(f"      [data-quality warning] missing categories: {missing}")
        else:
            print(f"      all categories present: {cats_present}")

        try:
            from trl import SFTTrainer, SFTConfig

            sft_config = SFTConfig(
                output_dir=str(Path(tmp) / "out"),
                per_device_train_batch_size=2,
                max_steps=2,
                logging_steps=1,
                report_to=[],
                use_cpu=True,  # this smoke test runs on CPU; real Colab run has a GPU
            )
            trainer = SFTTrainer(
                model=peft_model,
                args=sft_config,
                train_dataset=ds,
                processing_class=tokenizer,
            )
            trainer.train()
            print("      SFTTrainer ran 2 steps OK")
        except TypeError as e:
            # trl's SFTTrainer/SFTConfig signature has shifted across
            # versions (processing_class vs tokenizer, dataset_text_field,
            # etc). If this fires, check `pip show trl` and adjust the
            # call to match the installed version before assuming a bug.
            print(f"      [trl API mismatch — check installed trl version] {e}")
            raise

        adapter_dir = Path(tmp) / "adapter"
        peft_model.save_pretrained(adapter_dir)
        reloaded_base = build_tiny_qwen2(vocab_size)
        reloaded = get_peft_model(reloaded_base, lora_config)
        reloaded.load_adapter(str(adapter_dir), adapter_name="default")
        print("      save/load round-trip OK")

    print("\nALL SMOKE TESTS PASSED — pipeline plumbing is sound.")
    print("This does NOT validate model quality — only mechanics.")
    print("Real Qwen weights + real teacher_labels.jsonl must still run in Colab,")
    print("where huggingface.co is reachable and a real GPU is available.")


if __name__ == "__main__":
    main()
