#!/usr/bin/env python3
"""
Kimi-VL-A3B-Thinking-2506 — QLoRA Fine-Tuning (4-bit)
Hardware: 2x Tesla T4 (15GB each), 30GB CPU RAM
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
run("pip install -q llamafactory bitsandbytes requests accelerate")

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

# ─── 4. WRITE TRAINING CONFIG (QLoRA — 4-bit, 2x T4) ────────────────────────
train_config = """
### model
model_name_or_path: moonshotai/Kimi-VL-A3B-Thinking-2506
trust_remote_code: true

### quantization — loads model in 4-bit (~8GB), fits on single T4
quantization_bit: 4
quantization_method: bitsandbytes

### method
stage: sft
do_train: true
finetuning_type: lora

### lora
lora_target: all
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05

### dataset
dataset: kimi_vl_sft
dataset_dir: ./data
template: kimi_vl
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: ./kimi-vl-qlora-out
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true

### training — conservative settings for T4
per_device_train_batch_size: 1
gradient_accumulation_steps: 8    # effective batch = 8
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.05
fp16: true                        # T4 doesn't support bf16, use fp16
gradient_checkpointing: true
ddp_timeout: 180000000
"""

with open("train_qlora.yaml", "w") as f:
    f.write(train_config.strip())

print("Config written to train_qlora.yaml")

# ─── 5. TRAIN (multi-GPU with torchrun for 2x T4) ────────────────────────────
run("torchrun --nproc_per_node=2 --master_port=29500 "
    "$(which llamafactory-cli) train train_qlora.yaml")

# ─── 6. MERGE LORA INTO BASE MODEL ───────────────────────────────────────────
merge_config = """
model_name_or_path: moonshotai/Kimi-VL-A3B-Thinking-2506
trust_remote_code: true
adapter_name_or_path: ./kimi-vl-qlora-out
finetuning_type: lora
template: kimi_vl
export_dir: ./kimi-vl-merged
export_size: 5
export_device: cpu
export_legacy_format: false
"""

with open("merge_qlora.yaml", "w") as f:
    f.write(merge_config.strip())

print("\nMerging LoRA adapter into base model...")
run("llamafactory-cli export merge_qlora.yaml")

print("\n✅ Done! Merged model saved to ./kimi-vl-merged")
print("Upload with:")
print("  huggingface-cli upload thetrillioniar/Kimi-VL-A3B-Thinking-2506-Fabel5 ./kimi-vl-merged")
