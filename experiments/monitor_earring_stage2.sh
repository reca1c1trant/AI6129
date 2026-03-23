#!/bin/bash
# Monitor Stage 2 training job and auto-submit inference when done.
# Usage: nohup bash monitor_earring_stage2.sh <JOB_ID> &

TRAIN_JOB_ID="$1"
TRAIN_LOG="/scratch/users/ntu/song0304/loraedit_exp/logs/train_earring_stage2.log"
INFERENCE_PBS="/home/users/ntu/song0304/code/LoRAEdit/scripts/inference_earring_stage2.pbs"

if [ -z "$TRAIN_JOB_ID" ]; then
    echo "Usage: $0 <TRAINING_JOB_ID>"
    exit 1
fi

echo "[Monitor] Watching training job: ${TRAIN_JOB_ID}"
echo "[Monitor] Will auto-submit inference when training completes."

while true; do
    # Check if job is still running
    JOB_STATE=$(qstat -f "${TRAIN_JOB_ID}" 2>/dev/null | grep "job_state" | awk '{print $3}')

    if [ -z "${JOB_STATE}" ]; then
        echo "[Monitor] Job ${TRAIN_JOB_ID} no longer in queue. Checking log..."

        if grep -q "TRAINING COMPLETE" "${TRAIN_LOG}" 2>/dev/null; then
            echo "[Monitor] Training completed successfully!"
            echo "[Monitor] Submitting inference job..."
            INF_JOB=$(qsub "${INFERENCE_PBS}")
            echo "[Monitor] Inference job submitted: ${INF_JOB}"
            echo "[$(date)] Training done. Inference submitted: ${INF_JOB}" >> /scratch/users/ntu/song0304/loraedit_exp/logs/monitor_stage2.log
            exit 0
        else
            echo "[Monitor] WARNING: Training may have failed. Check log: ${TRAIN_LOG}"
            exit 1
        fi
    fi

    echo "[Monitor] Job state: ${JOB_STATE} ($(date '+%H:%M'))"
    sleep 300  # Check every 5 minutes
done
