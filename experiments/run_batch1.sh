#!/bin/bash
# Batch 1 sequential runner for PC²-FedReorg.
# Order:  E60 -> E61 -> E62 -> E63 -> E66.
# Skipped: E64 (MOON, Batch 2), E65 (D->C placeholder), E67-E73 (Batch 2).
# Each experiment writes to its own log under logs/pc2_batch1/.

LOGDIR=logs/pc2_batch1
PY=/vol1/home/lengcan/cleng/miniconda3/envs/H-CAAN/bin/python
mkdir -p "$LOGDIR"

EXPERIMENTS=(E60 E61 E62 E63 E66)

START_STAMP=$(date +%Y%m%d_%H%M%S)
SESSION_LOG="$LOGDIR/_session_${START_STAMP}.log"

{
echo "===================================="
echo "PC2-FedReorg Batch 1"
echo "Start: $(date)"
echo "Hostname: $(hostname)"
echo "Python: $PY"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "===================================="
} | tee -a "$SESSION_LOG"

for exp in "${EXPERIMENTS[@]}"; do
    LOG="$LOGDIR/${exp}_$(date +%Y%m%d_%H%M%S).log"
    {
        echo
        echo "----- $exp start at $(date) -----"
        echo "Log: $LOG"
    } | tee -a "$SESSION_LOG"

    "$PY" experiments/run_all.py --exp "$exp" > "$LOG" 2>&1
    rc=$?

    echo "----- $exp exit code $rc at $(date) -----" | tee -a "$SESSION_LOG"

    # Per user constraint: if E66 (the only PC² entry) fails, stop subsequent
    # PC² tasks. E66 is last in this batch so nothing to stop, but reflect
    # the contract anyway in case the order changes later.
    if [ "$exp" = "E66" ] && [ $rc -ne 0 ]; then
        echo "E66 failed; per user constraint, halting subsequent PC² tasks." \
            | tee -a "$SESSION_LOG"
    fi
done

{
echo
echo "===================================="
echo "Batch 1 done: $(date)"
echo "===================================="
} | tee -a "$SESSION_LOG"
