"""
Convert OpenAI fine-tuning format (opt1.jsonl) to LLaMA-Factory ShareGPT format.

Input format (OpenAI):
  {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Output format (ShareGPT):
  {"conversations": [{"from": "system", "value": "..."}, {"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
"""

import json
import requests
import sys
from pathlib import Path

DATASET_URL = "https://huggingface.co/datasets/thetrillioniar/claude-sonnet-4.6-opus-4.8-mythos-5-fable-5-openai-finetuning-dataset/resolve/main/opts/opt1.jsonl"
OUTPUT_FILE = "opt1_sharegpt.json"

ROLE_MAP = {
    "system": "system",
    "user": "human",
    "assistant": "gpt",
}


def download_dataset(url: str) -> list[dict]:
    print(f"Downloading dataset from:\n  {url}")
    headers = {"Authorization": f"Bearer {__import__('os').getenv('HF_TOKEN', '')}"}
    resp = requests.get(url, headers=headers, stream=True)
    resp.raise_for_status()

    records = []
    for line in resp.iter_lines():
        if line:
            records.append(json.loads(line))
    print(f"Downloaded {len(records)} records")
    return records


def convert_record(record: dict) -> dict | None:
    """Convert a single OpenAI-format record to ShareGPT format."""
    messages = record.get("messages", [])
    if not messages:
        return None

    conversations = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Handle content as list (OpenAI multimodal format)
        if isinstance(content, list):
            text_parts = [c["text"] for c in content if c.get("type") == "text"]
            content = " ".join(text_parts)

        mapped_role = ROLE_MAP.get(role)
        if mapped_role is None:
            print(f"  Warning: unknown role '{role}', skipping message")
            continue

        conversations.append({"from": mapped_role, "value": content})

    if not conversations:
        return None

    return {"conversations": conversations}


def main():
    # Download
    try:
        records = download_dataset(DATASET_URL)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Try setting HF_TOKEN env var if dataset is gated.")
        sys.exit(1)

    # Convert
    converted = []
    skipped = 0
    for i, rec in enumerate(records):
        result = convert_record(rec)
        if result:
            converted.append(result)
        else:
            skipped += 1
            print(f"  Skipped record {i}: {rec}")

    print(f"\nConverted: {len(converted)} records")
    print(f"Skipped:   {skipped} records")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {OUTPUT_FILE}")

    # Preview first record
    if converted:
        print("\n--- First Record Preview ---")
        for turn in converted[0]["conversations"]:
            print(f"  [{turn['from']}]: {turn['value'][:100]}...")


if __name__ == "__main__":
    main()
