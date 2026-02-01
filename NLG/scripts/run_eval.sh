#!/bin/bash

# Mistral Evaluation Script
# Evaluates LoRA adapters on Lots-of-LoRAs benchmark tasks

# Default values
TASK=""
DATASET_SOURCE="iid"
CUSTOM_DATASET=""
ADAPTER_PATH=""
OUTPUT_DIR="./results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --dataset_source)
            DATASET_SOURCE="$2"
            shift 2
            ;;
        --custom_dataset)
            CUSTOM_DATASET="$2"
            shift 2
            ;;
        --adapter_path)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$TASK" ]; then
    echo "Error: --task is required"
    echo ""
    echo "Usage: $0 --task TASK [--dataset_source iid|ood|custom] [--custom_dataset DATASET] [--adapter_path PATH] [--output_dir DIR]"
    echo ""
    echo "Examples:"
    echo "  # Evaluate IID task"
    echo "  $0 --task task076 --dataset_source iid"
    echo ""
    echo "  # Evaluate OOD task"
    echo "  $0 --task task280 --dataset_source ood"
    echo ""
    echo "  # Evaluate with custom dataset"
    echo "  $0 --task mytask --dataset_source custom --custom_dataset MyOrg/my_dataset"
    exit 1
fi

# Build command
CMD="python mistral_eval.py --task $TASK --dataset_source $DATASET_SOURCE"

if [ -n "$CUSTOM_DATASET" ]; then
    CMD="$CMD --custom_dataset $CUSTOM_DATASET"
fi

if [ -n "$ADAPTER_PATH" ]; then
    CMD="$CMD --adapter_path $ADAPTER_PATH"
fi

CMD="$CMD --output_dir $OUTPUT_DIR"

echo "Running: $CMD"
eval $CMD
