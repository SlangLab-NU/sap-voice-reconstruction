#!/bin/bash
#SBATCH --job-name=vtn_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4500:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=11:55:00
#SBATCH --signal=B:USR1@300
#SBATCH --dependency=singleton
#SBATCH --mail-user=aa.mohan@northeastern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --open-mode=append
#SBATCH --output=/projects/aanchan/slurm/logs/vtn_train_%j.log
#SBATCH --error=/projects/aanchan/slurm/logs/vtn_train_%j.log
# =============================================================================
# VTN direct-VC training — single A4500 on the gpu partition, run inside the
# existing sap_data_prep.sif container (torch 2.1.2+cu121 + torchaudio + lhotse;
# our package via PYTHONPATH). CHAINED + SELF-RESUMING:
#   - disk is the checkpoint: trainer auto-resumes from $EXP/checkpoints/latest.pt
#   - USR1 (5 min before the wall) -> resubmit this job once (unless $EXP/DONE)
#   - singleton + fixed job-name -> the next round waits for this one to end
# Stops when the trainer reaches --max-steps (writes $EXP/DONE) or on scancel.
# Submit:  sbatch cluster/slurm/train_vtn.slurm
# Override:  VTN_EXP=/path VTN_MAX_STEPS=80000 sbatch ... (via --export)
# =============================================================================
set -uo pipefail

SELF=/projects/aanchan/sap-voice-reconstruction/cluster/slurm/train_vtn.slurm
REPO=/projects/aanchan/sap-voice-reconstruction
SIF=/projects/aanchan/sap_data_prep.sif
CACHE=/projects/aanchan/cache
MANIFESTS=/projects/aanchan/data/manifests
EXP=${VTN_EXP:-/projects/aanchan/exp/vtn_run1}
MAX_STEPS=${VTN_MAX_STEPS:-60000}
TRAIN_SPLIT=${VTN_TRAIN_SPLIT:-train}
BATCH=${VTN_BATCH:-8}
NUM_WORKERS=${VTN_NUM_WORKERS:-8}
MAX_DURATION=${VTN_MAX_DURATION:-12.0}
VAL_EVERY=${VTN_VAL_EVERY:-2000}
CKPT_EVERY=${VTN_CKPT_EVERY:-2000}
LOG_EVERY=${VTN_LOG_EVERY:-50}
# optional: VTN_MAX_TRAIN_IDS to cap the training set (sanity runs)
EXTRA=()
[ -n "${VTN_MAX_TRAIN_IDS:-}" ] && EXTRA+=(--max-train-ids "$VTN_MAX_TRAIN_IDS")

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
echo "[$(date)] start job=$SLURM_JOB_ID exp=$EXP max_steps=$MAX_STEPS"

apptainer exec --nv --no-home --cleanenv \
  --bind /projects/aanchan:/projects/aanchan \
  --env HOME="$CACHE/home" \
  --env PYTHONPATH="$REPO" \
  --env TORCH_HOME="$CACHE/torch" \
  --env XDG_CACHE_HOME="$CACHE/xdg" \
  --env NUMBA_CACHE_DIR="$CACHE/numba" \
  --env MPLCONFIGDIR="$CACHE/mpl" \
  "$SIF" \
  python3 "$REPO/scripts/train_vtn.py" \
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
    --resume auto "${EXTRA[@]}" &
wait $!
echo "[$(date)] training process exited (rc=$?)"
