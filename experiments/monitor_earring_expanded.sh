#!/bin/bash
# Monitor earring expanded training and auto-submit inference

TRAIN_JOB=13318861
INF_SCRIPT=/home/users/ntu/song0304/code/LoRAEdit/scripts/inference_earring_expanded.pbs
LOG=/scratch/users/ntu/song0304/loraedit_exp/logs/train_earring_expanded.log
MONITOR_LOG=/scratch/users/ntu/song0304/loraedit_exp/logs/earring_expanded_monitor.log

echo "=== Monitor started: $(date) ===" > "$MONITOR_LOG"
echo "Watching job $TRAIN_JOB" >> "$MONITOR_LOG"

while true; do
    STATUS=$(qstat "$TRAIN_JOB" 2>/dev/null | tail -1 | awk '{print $5}')
    if [ -z "$STATUS" ]; then
        echo "[$(date)] Job $TRAIN_JOB finished" >> "$MONITOR_LOG"

        if grep -q "TRAINING COMPLETE" "$LOG" 2>/dev/null; then
            echo "[$(date)] Training SUCCESS - submitting inference" >> "$MONITOR_LOG"
            eval "$(/app/apps/miniforge3/25.3.1/condabin/conda shell.bash hook)"
            INF_JOB=$(qsub "$INF_SCRIPT")
            echo "[$(date)] Inference submitted: $INF_JOB" >> "$MONITOR_LOG"
        else
            echo "[$(date)] Training may have FAILED. Last lines:" >> "$MONITOR_LOG"
            tail -5 "$LOG" >> "$MONITOR_LOG" 2>/dev/null
        fi
        break
    else
        LAST_LOSS=$(grep "loss:" "$LOG" 2>/dev/null | tail -1)
        echo "[$(date)] Job $TRAIN_JOB status=$STATUS $LAST_LOSS" >> "$MONITOR_LOG"
    fi
    sleep 300
done

echo "=== Monitor ended: $(date) ===" >> "$MONITOR_LOG"
