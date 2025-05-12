#!/bin/bash

# Default time for interactive session (in case you don't provide one)
DEFAULT_TIME="2:00:00"

# Default partition and account (adjust according to your use case)
DEFAULT_PARTITION="mb-a30"
DEFAULT_ACCOUNT="joycelab-niall"
DEFAULT_QOS="interactive"

# Function to request compute resources (Tier 1 - Lowest)
request_low() {
    echo "Requesting Low Resource Tier: 1 GPU, 32GB RAM, 2 hours"
    salloc --job-name=contrastive_gpu \
           --time=2:00:00 \
           --gres=gpu:1 \
           --mem=32G \
           --partition=mb-a30 \
           --account=joycelab-niall \
           --qos=interactive
}

# Function to request compute resources (Tier 2 - Medium)
request_medium() {
    echo "Requesting Medium Resource Tier: 2 GPUs, 64GB RAM, 4 hours"
    salloc --job-name=contrastive_gpu \
           --time=4:00:00 \
           --gres=gpu:2 \
           --mem=64G \
           --partition=mb-a30 \
           --account=joycelab-niall \
           --qos=interactive
}

# Function to request compute resources (Tier 3 - High)
request_high() {
    echo "Requesting High Resource Tier: 4 GPUs, 128GB RAM, 8 hours"
    salloc --job-name=contrastive_gpu \
           --time=8:00:00 \
           --gres=gpu:4 \
           --mem=128G \
           --partition=mb-a30 \
           --account=joycelab-niall \
           --qos=interactive
}

# Usage message
usage() {
    echo "Usage: $0 {low|medium|high}"
    echo "low    - Request low-tier resources (1 GPU, 32GB RAM, 2 hours)"
    echo "medium - Request medium-tier resources (2 GPUs, 64GB RAM, 4 hours)"
    echo "high   - Request high-tier resources (4 GPUs, 128GB RAM, 8 hours)"
}

# Main logic to choose tier
if [ $# -ne 1 ]; then
    usage
    exit 1
fi

case $1 in
    low)
        request_low
        ;;
    medium)
        request_medium
        ;;
    high)
        request_high
        ;;
    *)
        usage
        exit 1
        ;;
esac

