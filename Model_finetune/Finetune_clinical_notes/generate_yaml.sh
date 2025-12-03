#!/bin/bash

# ==========================
# Usage:
# bash generate_yaml.sh MODEL_NAME DATASET NUM_EPOCHS LEARNING_RATE BATCH_SIZE
# Example:
# bash generate_yaml.sh meta-llama/Meta-Llama-3-8B clinical_notes 1 1e-6 16
# ==========================

MODEL_NAME=$1
DATASET=$2
NUM_EPOCHS=$3
LEARNING_RATE=$4
BATCH_SIZE=$5

# Parameters
VALID_SPLIT=valid
WEIGHT_DECAY=0.1
BLOCK_SIZE=2048
GRADIENT_ACCUMULATION=1
EARLY_STOPPING=1
MIXED_PRECISION=bf16
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.0

# ================ Generate RUN_ID =================
SHORT_MODEL_NAME=$(basename "$MODEL_NAME")
RUN_ID="${SHORT_MODEL_NAME}_lr${LEARNING_RATE}_bs${BATCH_SIZE}"
RUN_ID="${RUN_ID//./}"
RUN_ID="${RUN_ID//-/}"

echo "RUN_ID = $RUN_ID"

# ================ Generate filled_sft.yml =================
cat <<EOF > filled_sft.yml
task: llm-sft
base_model: /home/jupyter/20000360102458359xu/LingfeiQian/saved_models/${MODEL_NAME}
project_name: ${RUN_ID}
log: tensorboard
backend: local-cli

data:
  path: /home/jupyter/20000360102458359xu/LingfeiQian/saved_dataset/${DATASET}
  train_split: train
  valid_split: ${VALID_SPLIT}
  chat_template: chatml
  column_mapping:
    text_column: conversations

params:
  text_column: conversations

  block_size: ${BLOCK_SIZE}
  model_max_length: ${BLOCK_SIZE}

  epochs: ${NUM_EPOCHS}
  batch_size: ${BATCH_SIZE}
  lr: ${LEARNING_RATE}
  weight_decay: ${WEIGHT_DECAY}
  early_stopping: ${EARLY_STOPPING}

  save_strategy: epoch
  save_total_limit: 20

  peft: true
  lora_r: ${LORA_R}
  lora_alpha: ${LORA_ALPHA}
  lora_dropout: ${LORA_DROPOUT}
  quantization: int4
  target_modules: all-linear

  optimizer: adamw_torch
  scheduler: cosine
  gradient_accumulation: ${GRADIENT_ACCUMULATION}
  mixed_precision: ${MIXED_PRECISION}
  padding: right
EOF

echo "filled_sft.yml has been generated!"
