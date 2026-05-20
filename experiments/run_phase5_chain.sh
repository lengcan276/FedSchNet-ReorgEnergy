#!/bin/bash
# Phase 5 unattended chain.
# Waits for the pc2_multiseed tmux session (Step 2) to exit, then runs:
#   Step 3 + 4: E65, E68, E69, E70 multiseed
#   Step 5:     paired stats tests on all multiseed predictions
#
# Launch via:
#   tmux new -s pc2_phase5_chain -d 'bash experiments/run_phase5_chain.sh'

LOGDIR=logs/pc2_multiseed
PY=/vol1/home/lengcan/cleng/miniconda3/envs/H-CAAN/bin/python
mkdir -p "$LOGDIR"

CHAIN_LOG="$LOGDIR/chain_$(date +%Y%m%d_%H%M%S).log"

{
echo "[chain] start: $(date)"
echo "[chain] waiting for tmux session pc2_multiseed to exit..."

# Poll every 60s. Note: tmux has-session returns 0 if exists, nonzero if not.
while tmux has-session -t pc2_multiseed 2>/dev/null; do
    sleep 60
done
echo "[chain] pc2_multiseed exited at $(date)"

# Step 3 + 4: E65 + E68/E69/E70 multiseed
LOG_S34="$LOGDIR/step3_4_$(date +%Y%m%d_%H%M%S).log"
echo "[chain] === Step 3+4 begin -> $LOG_S34 ==="
"$PY" experiments/run_phase5_step3_4.py 2>&1 | tee "$LOG_S34"
RC34=${PIPESTATUS[0]}
echo "[chain] Step 3+4 exit code $RC34 at $(date)"

# Step 5: stats. Always run -- partial data is informative even if Step 3+4 had failures.
LOG_S5="$LOGDIR/step5_$(date +%Y%m%d_%H%M%S).log"
echo "[chain] === Step 5 begin -> $LOG_S5 ==="
"$PY" experiments/run_phase5_step5_stats.py 2>&1 | tee "$LOG_S5"
RC5=${PIPESTATUS[0]}
echo "[chain] Step 5 exit code $RC5 at $(date)"

echo "[chain] all Phase 5 steps complete at $(date)"
echo "[chain] outputs:"
ls -la results/pc2_batch1_multiseed_summary.csv \
       results/pc2_batch1_multiseed_predictions.json \
       results/pc2_phase5_ablation_summary.csv \
       results/pc2_phase5_ablation_predictions.json \
       results/pc2_phase5_stats.csv \
       results/pc2_phase5_stats.md 2>&1 | head -10
} 2>&1 | tee "$CHAIN_LOG"
