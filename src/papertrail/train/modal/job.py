"""Fine-tune Nemotron 3 Nano Omni as a vision LoRA on Modal (A100-80GB).

This trains the model to emit the `ExtractedExpense` JSON from a document image,
using the synthetic dataset produced by `papertrail.train.data.build_dataset`.

------------------------------------------------------------------------------
NOT runnable in this repo's local env — it needs a Modal account + GPU and the
pinned `finetune` extra. This file is the job definition; you run it on Modal.
------------------------------------------------------------------------------

One-time setup:
    pip install modal && modal token new
    modal secret create huggingface HF_TOKEN=hf_xxx     # write token
    modal volume create papertrail-data
    modal volume create papertrail-out
    # upload the dataset you built locally:
    modal volume put papertrail-data data/synthetic /synthetic

Day-1 de-risk spike (tiny, fast — gates vision-LoRA vs text-LoRA fallback):
    modal run -m papertrail.train.modal.job --max-samples 10 --epochs 1 \
        --push-repo "<you>/papertrail-nemotron-lora-spike"

Full run:
    modal run -m papertrail.train.modal.job --epochs 3 \
        --push-repo "<you>/papertrail-nemotron-lora"

After training, pull the adapter:
    modal volume get papertrail-out /lora ./outputs/lora

GGUF + mmproj conversion for llama.cpp is intentionally NOT automated here: that
round-trip is the Day-1 spike's gate and may require model-specific tooling. Do
it manually against the merged 16-bit weights this job writes to /out/merged.
"""

from __future__ import annotations

import modal

# The TRAINABLE base — the safetensors repo, not the GGUF (GGUF is inference-only).
# Confirmed from Unsloth's Nemotron Omni docs examples. 4-bit is loaded on the fly.
MODEL_NAME = "unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning"

# Pinned to match pyproject's [finetune] extra — Nemotron is sensitive to these.
FINETUNE_PACKAGES = [
    "torch==2.7.1",
    "triton>=3.3.0",
    "transformers==4.56.2",
    "torchvision>=0.22.0",
    "datasets==4.3.0",
    "mamba_ssm==2.2.5",
    "causal_conv1d==1.5.2",
    "unsloth",
    "unsloth_zoo",
    "peft",
    "trl",
    "accelerate",
    "pillow",
]

app = modal.App("papertrail-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(*FINETUNE_PACKAGES)
)

data_vol = modal.Volume.from_name("papertrail-data", create_if_missing=True)
out_vol = modal.Volume.from_name("papertrail-out", create_if_missing=True)

DATA_DIR = "/data/synthetic"
OUT_DIR = "/out"


def _load_conversations(manifest_path: str, splits: set[str], max_samples: int | None):
    """Read the manifest and build Unsloth vision-chat conversations."""
    import json
    import os

    from PIL import Image

    base = os.path.dirname(manifest_path)
    convos = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] not in splits:
                continue
            img = Image.open(os.path.join(base, row["image"])).convert("RGB")
            convos.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": row["instruction"]},
                                {"type": "image", "image": img},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": row["response"]}],
                        },
                    ]
                }
            )
            if max_samples is not None and len(convos) >= max_samples:
                break
    return convos


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=4 * 60 * 60,
    volumes={"/data": data_vol, "/out": out_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def train(
    epochs: int = 3,
    max_samples: int | None = None,
    push_repo: str | None = None,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
) -> str:
    import os

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel, UnslothVisionDataCollator, is_bf16_supported

    manifest = os.path.join(DATA_DIR, "manifest.jsonl")
    convos = _load_conversations(manifest, splits={"train"}, max_samples=max_samples)
    print(f"loaded {len(convos)} training conversations")
    dataset = Dataset.from_list(convos)

    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_rank,
        lora_alpha=lora_rank,
        lora_dropout=0.0,
        bias="none",
        random_state=42,
    )
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=os.path.join(OUT_DIR, "checkpoints"),
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,
        ),
    )

    stats = trainer.train()
    print(
        f"train_runtime={stats.metrics.get('train_runtime')}s "
        f"loss={stats.metrics.get('train_loss')}"
    )

    # LoRA adapter (small) + merged 16-bit (for the manual GGUF+mmproj step).
    lora_dir = os.path.join(OUT_DIR, "lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    model.save_pretrained_merged(os.path.join(OUT_DIR, "merged"), tokenizer)
    out_vol.commit()

    if push_repo:
        token = os.environ["HF_TOKEN"]
        model.push_to_hub(push_repo, token=token)
        tokenizer.push_to_hub(push_repo, token=token)
        print(f"pushed adapter -> https://huggingface.co/{push_repo}")

    return f"done: {len(convos)} samples, {epochs} epochs"


@app.local_entrypoint()
def main(
    epochs: int = 3,
    max_samples: int = 0,
    push_repo: str = "",
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
) -> None:
    result = train.remote(
        epochs=epochs,
        max_samples=max_samples or None,
        push_repo=push_repo or None,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
    )
    print(result)
