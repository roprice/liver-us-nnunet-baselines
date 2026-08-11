#!/bin/bash
INSTANCE_HOSTNAME="clear-life-begins-fin-01"

# Poll until no scp process is running
echo "Waiting for download to finish..."
while pgrep -f "scp.*65.109.75.28" > /dev/null 2>&1; do
    echo "$(date): Still downloading..."
    sleep 120
done

echo "$(date): Download complete. Deleting instance..."
verda vm delete $INSTANCE_HOSTNAME --force
echo "Instance deleted. Volumes preserved."
