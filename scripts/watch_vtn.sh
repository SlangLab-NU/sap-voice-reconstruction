#!/bin/bash
# Babysit the VTN training chain: show job status, tail recent loss, and — if the
# chain has died without finishing — resubmit it. Safe to run on a loop / via watch.
#
#   bash scripts/watch_vtn.sh                 # one status report
#   watch -n 60 bash scripts/watch_vtn.sh     # live
#   RESURRECT=1 bash scripts/watch_vtn.sh     # also resubmit if dead & not DONE
set -uo pipefail

JOB=vtn_train
EXP=${VTN_EXP:-/projects/aanchan/exp/vtn_run1}
SELF=/projects/aanchan/sap-voice-reconstruction/cluster/slurm/train_vtn.slurm

echo "=== $(date) ==="
echo "--- squeue (${JOB}) ---"
squeue --name="$JOB" -o "%.10i %.12j %.8T %.10M %.12l %R" 2>/dev/null

running=$(squeue --name="$JOB" -h -o "%i" 2>/dev/null | wc -l)
done=0; [ -f "$EXP/DONE" ] && done=1

echo "--- last train/val metrics ($EXP/metrics.jsonl) ---"
if [ -f "$EXP/metrics.jsonl" ]; then
  tail -n 6 "$EXP/metrics.jsonl"
else
  echo "(no metrics yet)"
fi
echo "--- checkpoints ---"
ls -1t "$EXP/checkpoints" 2>/dev/null | head -3 || echo "(none yet)"
echo "status: jobs_in_queue=$running done=$done"

if [ "$running" -eq 0 ] && [ "$done" -eq 0 ]; then
  echo "!! chain not running and not DONE."
  if [ "${RESURRECT:-0}" = "1" ]; then
    echo ">> resubmitting $SELF"
    sbatch --export=ALL "$SELF"
  else
    echo ">> rerun with RESURRECT=1 to resubmit, or: sbatch $SELF"
  fi
fi
