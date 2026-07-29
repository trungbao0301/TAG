#!/usr/bin/env bash
set -euo pipefail

DREAMER4_ROOT="${DREAMER4_ROOT:-$HOME/dreamer4}"
DATA_DIRS="${DREAMER4_TOKENIZER_DATA_DIRS:?Set DREAMER4_TOKENIZER_DATA_DIRS to preprocessed shard dirs, separated by spaces.}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_DISABLED="${WANDB_DISABLED:-true}"

export CUDA_VISIBLE_DEVICES WANDB_MODE WANDB_DISABLED

cd "$DREAMER4_ROOT/dreamer4"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "WANDB_MODE=$WANDB_MODE"
echo "MASTER_PORT=$MASTER_PORT"
echo "DREAMER4_TOKENIZER_DATA_DIRS=$DATA_DIRS"

/home/tbt589/bin/micromamba run -n dreamer4 torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  train_tokenizer.py \
  --data_dirs $DATA_DIRS \
  "$@"
