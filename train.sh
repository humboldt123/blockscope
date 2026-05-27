#!/usr/bin/env bash
# train.sh — generic PuM experiment launcher (DDP + AMP + precomputed frames)
#
# Usage:
#   bash train.sh                                             # viblock2 finetune, 6-GPU default
#   EXPERIMENT=vis SCRIPT=train.py bash train.sh              # different experiment
#   N_GPUS=1 DEVICE=cuda:2 bash train.sh                      # single GPU
#   UNFREEZE_N=4 EXP_NAME=viblock2_ablation bash train.sh     # override hyperparams
#   EXTRA_ARGS="--my_arg value" bash train.sh                 # forward extra args to script
#
# Brev-specific paths (update if environment changes):
#   BREV_ROOT  /home/vvm33/blockscope
#   Python     /home/nvidia/miniconda3/bin/python
#   Data       /data/vvm33/

set -euo pipefail

# ── Experiment ────────────────────────────────────────────────────────────────
EXPERIMENT="${EXPERIMENT:-viblock2}"
SCRIPT="${SCRIPT:-train_finetune.py}"

# ── Paths ─────────────────────────────────────────────────────────────────────
BREV_ROOT="${BREV_ROOT:-/home/vvm33/blockscope}"
PYTHON_BIN="/home/nvidia/miniconda3/bin/python"
TORCHRUN_BIN="/home/nvidia/miniconda3/bin/torchrun"

CONFIG="${CONFIG:-${BREV_ROOT}/pum/experiments/${EXPERIMENT}/config.yaml}"
CHECKPOINT="${CHECKPOINT:-}"  # empty = train from scratch
LOG_DIR="${LOG_DIR:-/home/vvm33/logs}"

# ── Hyperparameters ───────────────────────────────────────────────────────────
EXP_NAME="${EXP_NAME:-${EXPERIMENT}}"
EPOCHS="${EPOCHS:-30}"
N_GPUS="${N_GPUS:-6}"
DEVICE="${DEVICE:-cuda:0}"          # only used when N_GPUS=1
MASTER_PORT="${MASTER_PORT:-29500}"

# Fine-tuning params — only forwarded when SCRIPT=train_finetune.py
UNFREEZE_N="${UNFREEZE_N:-2}"
HEAD_LR="${HEAD_LR:-1e-4}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"

# Extra args forwarded verbatim to the training script
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ── Experiment identity ───────────────────────────────────────────────────────
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
GIT_SHA="$(cd "${BREV_ROOT}" && git rev-parse --short HEAD 2>/dev/null || echo 'nogit')"

if [ "${N_GPUS}" -gt 1 ]; then
    GPU_TAG="ddp${N_GPUS}gpu"
else
    GPU_ID="${DEVICE##*:}"
    GPU_TAG="gpu${GPU_ID}"
fi

# Encode fine-tuning hyperparams in run name only when relevant
FINETUNE_TAG=""
if [ "${SCRIPT}" = "train_finetune.py" ]; then
    FINETUNE_TAG="__unfrz${UNFREEZE_N}__hlr${HEAD_LR}__blr${BACKBONE_LR}"
fi

RUN_NAME="${EXP_NAME}__${GPU_TAG}${FINETUNE_TAG}__e${EPOCHS}__${TIMESTAMP}__${GIT_SHA}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

# ── Banner ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  EXPERIMENT   : ${EXPERIMENT}"
echo "  SCRIPT       : ${SCRIPT}"
echo "  EXP_NAME     : ${EXP_NAME}"
echo "  RUN_NAME     : ${RUN_NAME}"
echo "  n_gpus       : ${N_GPUS}"
echo "  CUDA_VISIBLE : ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "  epochs       : ${EPOCHS}"
echo "  config       : ${CONFIG}"
echo "  checkpoint   : ${CHECKPOINT:-<none>}"
echo "  git sha      : ${GIT_SHA}"
echo "  log          : ${LOG_FILE}"
echo "  started      : $(date -u)"
echo "================================================================"

cd "${BREV_ROOT}"

# ── Build args ────────────────────────────────────────────────────────────────
COMMON_ARGS=(
    "pum/experiments/${EXPERIMENT}/${SCRIPT}"
    --config     "${CONFIG}"
    --epochs     "${EPOCHS}"
    --run_name   "${RUN_NAME}"
)

[ -n "${CHECKPOINT}" ] && COMMON_ARGS+=(--checkpoint "${CHECKPOINT}")

if [ "${SCRIPT}" = "train_finetune.py" ]; then
    COMMON_ARGS+=(
        --unfreeze_n  "${UNFREEZE_N}"
        --head_lr     "${HEAD_LR}"
        --backbone_lr "${BACKBONE_LR}"
    )
fi

# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS}" ] && COMMON_ARGS+=(${EXTRA_ARGS})

# ── Launch ────────────────────────────────────────────────────────────────────
if [ "${N_GPUS}" -gt 1 ]; then
    PYTHONPATH="${BREV_ROOT}" \
    "${TORCHRUN_BIN}" \
        --nproc_per_node="${N_GPUS}" \
        --master_port="${MASTER_PORT}" \
        "${COMMON_ARGS[@]}" \
        2>&1 | tee "${LOG_FILE}"
else
    PYTHONPATH="${BREV_ROOT}" \
    "${PYTHON_BIN}" \
        "${COMMON_ARGS[@]}" \
        --device "${DEVICE}" \
        2>&1 | tee "${LOG_FILE}"
fi

echo "================================================================"
echo "  done: $(date -u)"
echo "  log : ${LOG_FILE}"
echo "================================================================"
