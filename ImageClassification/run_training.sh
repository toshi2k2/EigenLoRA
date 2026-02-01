

set -e  # Exit on error

# Default configuration
METHOD="${1:-lora}"
DATASET="${2:-CIFAR100}"
DATA_ROOT="${DATA_ROOT:-./data}"
SAVE_PATH="${SAVE_PATH:-./checkpoints}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-5e-6}"
RANK="${RANK:-8}"
SUBSET_SIZE="${SUBSET_SIZE:-10}"

# Print configuration
echo "=============================================="
echo "Training Configuration"
echo "=============================================="
echo "Method:      $METHOD"
echo "Dataset:     $DATASET"
echo "Data Root:   $DATA_ROOT"
echo "Save Path:   $SAVE_PATH"
echo "Epochs:      $EPOCHS"
echo "Batch Size:  $BATCH_SIZE"
echo "LR:          $LR"
echo "Rank:        $RANK"
echo "Subset Size: $SUBSET_SIZE"
echo "=============================================="

# Base command
CMD="python train_vit.py \
    --method $METHOD \
    --dataset $DATASET \
    --data_root $DATA_ROOT \
    --save_path $SAVE_PATH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --rank $RANK \
    --subset_size $SUBSET_SIZE \
    --use_wandb"

# Method-specific configurations
case $METHOD in
    lora)
        echo "Training with LoRA..."
        $CMD
        ;;
    
    vera)
        echo "Training with VeRA..."
        # VeRA requires the sampled subsets from LoRA for consistency
        LORA_SUBSETS="${SAVE_PATH}/lora/${DATASET}/sampled_subsets.txt"
        if [ ! -f "$LORA_SUBSETS" ]; then
            echo "Error: VeRA requires LoRA subsets file at $LORA_SUBSETS"
            echo "Please train LoRA first to generate class subsets."
            exit 1
        fi
        $CMD --sampled_subsets_path "lora/${DATASET}/sampled_subsets.txt"
        ;;
    
    elora)
        echo "Training with EigenLoRA..."
        # EigenLoRA requires LoRA checkpoints for eigenvector computation
        LORA_DIR="${SAVE_PATH}/lora/${DATASET}/model_checkpoints"
        LORA_SUBSETS="lora/${DATASET}/sampled_subsets.txt"
        
        if [ ! -d "$LORA_DIR" ]; then
            echo "Error: EigenLoRA requires LoRA checkpoints at $LORA_DIR"
            echo "Please train LoRA first."
            exit 1
        fi
        
        ELORA_COMPONENTS="${ELORA_COMPONENTS:-8}"
        
        $CMD \
            --sampled_subsets_path "$LORA_SUBSETS" \
            --lora_dict_directory "$LORA_DIR" \
            --elora_components $ELORA_COMPONENTS \
            ${3:-}  # Pass additional args like --elora_holdout
        ;;
    
    none)
        echo "Training with full fine-tuning..."
        $CMD
        ;;
    
    *)
        echo "Unknown method: $METHOD"
        echo "Supported methods: lora, vera, elora, none"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
