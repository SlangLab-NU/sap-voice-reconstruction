#!/bin/bash
#SBATCH --job-name=vtn_train_mg
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:a4500:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=07:55:00
#SBATCH --signal=B:USR1@300
#SBATCH --dependency=singleton
#SBATCH --mail-user=aa.mohan@northeastern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --open-mode=append
#SBATCH --output=/projects/aanchan/slurm/logs/vtn_train_mg_%j.log
#SBATCH --error=/projects/aanchan/slurm/logs/vtn_train_mg_%j.log
# =============================================================================
# VTN training on BOTH A4500s via DistributedDataParallel (torchrun, 2 procs),
# inside sap_data_prep.sif. multigpu QOS requires >=2 GPUs/job. Same chained
# self-resume as the single-GPU script: disk is the checkpoint (trainer resumes
# from $EXP/checkpoints/latest.pt), USR1 -> resubmit once (unless $EXP/DONE),
# singleton serializes rounds. --batch-size is PER-GPU (effective 2x).
# Submit:  sbatch cluster/slurm/train_vtn_multigpu.slurm
# =============================================================================
set -uo pipefail

SELF=/projects/aanchan/sap-voice-reconstruction/cluster/slurm/train_vtn_multigpu.slurm
REPO=/projects/aanchan/sap-voice-reconstruction
SIF=/projects/aanchan/sap_data_prep.sif
CACHE=/projects/aanchan/cache
MANIFESTS=/projects/aanchan/data/manifests
EXP=${VTN_EXP:-/projects/aanchan/exp/vtn_mg_run1}
MAX_STEPS=${VTN_MAX_STEPS:-60000}
TRAIN_SPLIT=${VTN_TRAIN_SPLIT:-train}
BATCH=${VTN_BATCH:-8}            # per-GPU
NUM_WORKERS=${VTN_NUM_WORKERS:-8}
MAX_DURATION=${VTN_MAX_DURATION:-12.0}
VAL_EVERY=${VTN_VAL_EVERY:-2000}
CKPT_EVERY=${VTN_CKPT_EVERY:-2000}
LOG_EVERY=${VTN_LOG_EVERY:-50}
REDUCTION=${VTN_REDUCTION:-2}
LAMBDA_STOP=${VTN_LAMBDA_STOP:-5.0}
POS_WEIGHT=${VTN_POS_WEIGHT:-50.0}
STOP_WINDOW=${VTN_STOP_WINDOW:-8}
LAMBDA_GUIDED=${VTN_LAMBDA_GUIDED:-100.0}
SS_MAX_PROB=${VTN_SS_MAX_PROB:-0.30}
EXTRA=()
[ -n "${VTN_MAX_TRAIN_IDS:-}" ] && EXTRA+=(--max-train-ids "$VTN_MAX_TRAIN_IDS")
[ "${VTN_NO_SS:-0}" = "1" ] && EXTRA+=(--no-scheduled-sampling)

resubmitted=0
on_usr1() {
  if [ "$resubmitted" -eq 0 ] && [ ! -f "$EXP/DONE" ]; then
    resubmitted=1
    echo "[$(date)] USR1: wall approaching -> resubmitting ${SLURM_JOB_NAME} (resumes from latest.pt)"
    sbatch --export=ALL "$SELF" || echo "WARN: resubmit failed"
  fi
}
trap on_usr1 USR1

mkdir -p "$CACHE"/{home,torch,xdg,numba,mpl} "$EXP" /projects/aanchan/slurm/logs
echo "[$(date)] start job=$SLURM_JOB_ID exp=$EXP max_steps=$MAX_STEPS (2x A4500 DDP)"

apptainer exec --nv --no-home --cleanenv \
  --bind /projects/aanchan:/projects/aanchan \
  --env HOME="$CACHE/home" \
  --env PYTHONPATH="$REPO" \
  --env TORCH_HOME="$CACHE/torch" \
  --env XDG_CACHE_HOME="$CACHE/xdg" \
  --env NUMBA_CACHE_DIR="$CACHE/numba" \
  --env MPLCONFIGDIR="$CACHE/mpl" \
  "$SIF" \
  torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    "$REPO/scripts/train_vtn.py" \
    --manifest-dir "$MANIFESTS" \
    --exp-dir "$EXP" \
    --train-split "$TRAIN_SPLIT" \
    --device cuda \
    --batch-size "$BATCH" \
    --num-workers "$NUM_WORKERS" \
    --max-duration "$MAX_DURATION" \
    --max-steps "$MAX_STEPS" \
    --log-every "$LOG_EVERY" --val-every "$VAL_EVERY" --ckpt-every "$CKPT_EVERY" --val-batches 30 \
    --d-model 256 --nhead 4 --enc-layers 4 --dec-layers 4 --ffn 1024 \
    --reduction-factor "$REDUCTION" \
    --lambda-stop "$LAMBDA_STOP" --pos-weight "$POS_WEIGHT" --stop-window "$STOP_WINDOW" \
    --lambda-guided "$LAMBDA_GUIDED" --ss-max-prob "$SS_MAX_PROB" \
    --resume auto "${EXTRA[@]}" &
wait $!
echo "[$(date)] training process exited (rc=$?)"
