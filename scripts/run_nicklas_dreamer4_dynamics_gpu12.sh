#!/usr/bin/env bash
set -euo pipefail

DREAMER4_ROOT="${DREAMER4_ROOT:-$HOME/dreamer4}"
RAW_DIRS="${DREAMER4_DYNAMICS_RAW_DIRS:?Set DREAMER4_DYNAMICS_RAW_DIRS to raw data dirs, separated by spaces.}"
FRAME_DIRS="${DREAMER4_DYNAMICS_FRAME_DIRS:?Set DREAMER4_DYNAMICS_FRAME_DIRS to preprocessed shard dirs, separated by spaces.}"
TOKENIZER_CKPT="${DREAMER4_TOKENIZER_CKPT:-$DREAMER4_ROOT/dreamer4/logs/tokenizer_ckpts/latest.pt}"
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
echo "DREAMER4_DYNAMICS_RAW_DIRS=$RAW_DIRS"
echo "DREAMER4_DYNAMICS_FRAME_DIRS=$FRAME_DIRS"
echo "DREAMER4_TOKENIZER_CKPT=$TOKENIZER_CKPT"

/home/tbt589/bin/micromamba run -n dreamer4 torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  train_dynamics.py \
  --data_dirs $RAW_DIRS \
  --frame_dirs $FRAME_DIRS \
  --tokenizer_ckpt "$TOKENIZER_CKPT" \
  --use_actions \
  "$@"
