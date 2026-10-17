#!/usr/bin/env bash
set -euo pipefail
export PATH="/root/miniconda3/bin:$PATH"
cd /root/autodl-tmp/Agent+SFT

python3 - <<'PY'
import json
data = json.load(open("merged_train_final_multiturn_v2.clean.json", encoding="utf-8"))
with open("smoke_train_24.jsonl", "w", encoding="utf-8") as f:
    for item in data[:24]:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print("smoke samples:", 24)
PY

python3 train_qwen_last_assistant_lora.py \
  --train_file ./smoke_train_24.jsonl \
  --model_name_or_path ./qwen3-0_6b \
  --output_dir ./smoke_lora_out \
  --tools_file ./all_tools.json \
  --max_seq_length 6144 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --warmup_ratio 0.03 \
  --logging_steps 1 \
  --save_steps 1000 \
  --lr_scheduler_type cosine \
  --bf16 \
  --gradient_checkpointing \
  --local_files_only 2>&1 | tee smoke_0_6b_console.log
