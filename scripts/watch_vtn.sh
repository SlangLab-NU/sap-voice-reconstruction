#!/bin/bash
# Babysit the VTN training chain: show job status, tail recent loss, and — if the
# chain has died without finishing — resubmit it.
#
#   bash scripts/watch_vtn.sh                  # one status report
#   bash scripts/watch_vtn.sh 300              # loop every 300s (Ctrl-C to stop)
#   RESURRECT=1 bash scripts/watch_vtn.sh 300  # loop AND resubmit if dead & not DONE
#   # persistent, survives logout:
#   nohup env RESURRECT=1 VTN_JOB=vtn_train_mg VTN_SLURM=.../train_vtn_multigpu.slurm \
#         bash scripts/watch_vtn.sh 300 > "$VTN_EXP/babysit.log" 2>&1 &
set -uo pipefail

JOB=${VTN_JOB:-vtn_train_mg}
EXP=${VTN_EXP:-/projects/aanchan/exp/vtn_mg_run1}
SELF=${VTN_SLURM:-/projects/aanchan/sap-voice-reconstruction/cluster/slurm/train_vtn_multigpu.slurm}
INTERVAL=${1:-0}   # 0 = single report; >0 = loop every N seconds

report() {
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
}

if [ "$INTERVAL" -gt 0 ] 2>/dev/null; then
  while true; do
    report
    [ -f "$EXP/DONE" ] && { echo "DONE — stopping babysitter."; break; }
    sleep "$INTERVAL"
  done
else
  report
fi
