#!/bin/bash
SERVER="root@65.109.75.13"
LOCAL_DIR="$HOME/Projects/liver-us-nnunet-baselines/results"
INSTANCE_HOSTNAME="bold-mind-glows-fin-01"

echo "Polling for training completion..."
while true; do
    if ssh $SERVER "grep -q 'All training runs complete' ~/liver-us-nnunet-baselines/efficiency_study.log 2>/dev/null"; then
        echo "Training complete! Starting download..."
        break
    fi
    echo "$(date): Still training..."
    sleep 300
done

scp -r $SERVER:~/nnUNet_results $LOCAL_DIR/
scp -r $SERVER:~/nnUNet_preprocessed $LOCAL_DIR/
scp $SERVER:~/liver-us-nnunet-baselines/efficiency_study.log $LOCAL_DIR/

echo "Download complete."

verda vm delete $INSTANCE_HOSTNAME --force
echo "Instance deleted."
