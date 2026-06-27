#!/usr/bin/env python3
"""
Kimi-VL-A3B-Thinking-2506 — Full Fine-Tuning (no LoRA)
Hardware: 64GB VRAM GPU, 32GB RAM
"""

import subprocess, sys, os, json, requests

def run(cmd, **kwargs):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

# ─── 0. ENV CHECK ─────────────────────────────────────────────────────────────
run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")

# ─── 1. INSTALL ───────────────────────────────────────────────────────────────
run("pip install -q 'llamafactory[torch,metrics]' deepspeed requests")

# ─── 2. LOGIN ─────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
WANDB_KEY = os.getenv("WANDB_API_KEY", "")

if HF_TOKEN:
    run(f"huggingface-cli login --token {HF_TOKEN}")
if WANDB_KEY:
    run(f"wandb login {WANDB_KEY}")

# ─── 3. DOWNLOAD & CONVERT DATASET ───────────────────────────────────────────
DATASET_URL = (
    "https://huggingface.co/datasets/thetrillioniar/"
    "claude-sonnet-4.6-opus-4.8-mythos-5-fable-5-openai-finetuning-dataset"
    "/resolve/main/opts/opt1.jsonl"
)

ROLE_MAP = {"system": "system", "user": "human", "assistant": "gpt"}

print("\nDownloading dataset...")
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
resp = requests.get(DATASET_URL, headers=headers)
resp.raise_for_status()

converted = []
for line in resp.text.strip().split("\n"):
    if not line.strip():
        continue
    rec = json.loads(line)
    messages = rec.get("messages", [])
    conversations = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            content = " ".join(c["text"] for c in content if c.get("type") == "text")
        mapped = ROLE_MAP.get(role)
        if mapped:
            conversations.append({"from": mapped, "value": content})
    if conversations:
        converted.append({"conversations": conversations})

print(f"Converted {len(converted)} records")

os.makedirs("data", exist_ok=True)
with open("data/opt1_sharegpt.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False)

dataset_info = {
    "kimi_vl_sft": {
        "file_name": "opt1_sharegpt.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system",
        },
    }
}
with open("data/dataset_info.json", "w") as f:
    json.dump(dataset_info, f, indent=2)

print("Dataset ready in ./data/")

# ─── 4. WRITE TRAINING CONFIG (full fine-tune, no LoRA) ──────────────────────
train_config = """
### model
model_name_or_path: moonshotai/Kimi-VL-A3B-Thinking-2506
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full          # full fine-tuning, all weights updated

### dataset
dataset: kimi_vl_sft
dataset_dir: ./data
template: kimi_vl
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: ./kimi-vl-full-out
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true

### training
per_device_train_batch_size: 2
gradient_accumulation_steps: 4    # effective batch = 8
learning_rate: 2.0e-5             # lower LR for full fine-tuning vs LoRA
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
gradient_checkpointing: true      # saves VRAM during backward pass
ddp_timeout: 180000000

### deepspeed (ZeRO-2 for single GPU, ZeRO-3 for multi-GPU)
deepspeed: ds_config.json
"""

with open("train_full.yaml", "w") as f:
    f.write(train_config.strip())

# ─── 5. WRITE DEEPSPEED CONFIG ────────────────────────────────────────────────
# ZeRO-2: shards optimizer states + gradients across GPU memory — good for single 64GB GPU
ds_config = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "offload_optimizer": {
            "device": "cpu",        # offload optimizer states to CPU RAM (you have 32GB)
            "pin_memory": True
        }
    },
    "bf16": {
        "enabled": True
    },
    "steps_per_print": 10,
    "wall_clock_breakdown": False
}

with open("ds_config.json", "w") as f:
    json.dump(ds_config, f, indent=2)

print("Configs written: train_full.yaml + ds_config.json")

# ─── 6. TRAIN ─────────────────────────────────────────────────────────────────
run("llamafactory-cli train train_full.yaml")

print("\n✅ Done! Full model saved to ./kimi-vl-full-out")
print("Upload with:")
print("  huggingface-cli upload thetrillioniar/Kimi-VL-A3B-Thinking-2506-Fabel5-SFT ./kimi-vl-full-out")
