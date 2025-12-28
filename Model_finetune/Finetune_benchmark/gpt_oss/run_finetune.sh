#!/bin/bash
#SBATCH --partition=gpu            # Use GPU partition
#SBATCH --gpus=h200:1              # Request 1 A100 GPU
#SBATCH --cpus-per-task=8          # Request 8 CPU cores
#SBATCH --mem=150G                  # Request 80 GiB memory
#SBATCH --time=48:00:00            # Max runtime 12 hours
#SBATCH --job-name=ft_medqa        # Job name

# Set environment variables
export HUGGINGFACE_HUB_TOKEN="xx"
export HF_HOME="xx"
export TRITON_CACHE_DIR="xx"
export TORCH_HOME="xx"
# export CUDA_HOME="xx"
# export PATH="$CUDA_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
mkdir -p $TRITON_CACHE_DIR $TORCH_HOME

# ============================================
# Configuration - Modify these as needed
# ============================================

# Base model to fine-tune (change this to your desired model)
# Examples:
#   - "meta-llama/Meta-Llama-3-8B"
#   - "meta-llama/Llama-2-7b-hf"
#   - "EleutherAI/gpt-neox-20b"
#   - "mistralai/Mistral-7B-v0.1"
MODEL_PATH="openai/gpt-oss-20b"

# Output directory for the fine-tuned model
OUTPUT_DIR="xx"

# Training parameters
NUM_EPOCHS=3
BATCH_SIZE=16
GRADIENT_ACCUMULATION=1
LEARNING_RATE=1e-5
MAX_SEQ_LENGTH=512

# LoRA parameters
LORA_R=16
LORA_ALPHA=64

# ============================================
# Run fine-tuning
# ============================================

echo "============================================"
echo "Fine-tuning on MedQA-USMLE"
echo "============================================"
echo "Base model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "============================================"

python Model_finetune/Finetune_benchmark/gpt_oss/gpt_oss_finetune.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --no_quantize \
    --run_name "gpt-oss-20b_medqa"

echo "============================================"
echo "Fine-tuning complete!"
echo "============================================"

