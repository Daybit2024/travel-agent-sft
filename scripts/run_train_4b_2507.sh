#!/usr/bin/env bash
set -euo pipefail
export PATH="/root/miniconda3/bin:$PATH"
cd /root/autodl-tmp/Agent+SFT

TRAIN_FILE="./merged_train_final_multiturn_v2.clean.json"
MODEL_PATH="/root/autodl-tmp/Agent+SFT/qwen3-4b-instruct-2507"
OUTPUT_DIR="/root/autodl-tmp/Agent+SFT/qwen3-4b-instruct-2507_lora_last_assistant"

mkdir -p "${OUTPUT_DIR}"

python3 train_qwen_last_assistant_lora.py \
  --train_file "${TRAIN_FILE}" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --tools_file ./all_tools.json \
  --max_seq_length 6144 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --save_steps 500 \
  --save_total_limit 3 \
  --lr_scheduler_type cosine \
  --bf16 \
  --gradient_checkpointing \
  --log_file ./train_4b_log.txt \
  --local_files_only 2>&1 | tee train_4b_console.log
