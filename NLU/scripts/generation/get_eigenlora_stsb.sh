#!/bin/bash

# Script to compute EigenLoRA components for STS-B task
# Uses CoLA and QNLI as source LoRA adapters

python get_eigenlora.py \
  --source_lora_paths "ankit-vaidya19/cola_lora_r_8" "ankit-vaidya19/qnli_lora_r_8" \
  --source_lora_names cola qnli \
  --target_task_name stsb \
  --num_labels 1 \
  --model_name_or_path roberta-base \
  --eigenlora_r 8 \
  --num_eigenvector_components 32 \
  --num_gram_schmidt_components 32 \
  --loading_source_index 0 \
  --output_dir ./stsb_eigenlora \
  --save_eigenlora_components \
  --save_eigenlora_loadings
