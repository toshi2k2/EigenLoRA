#!/bin/bash

# Generate EigenLoRA and Reconstructed LoRA for 'toy_face' style
# Uses default CiroN2022 source LoRAs

python get_eigenlora.py \
  --target_lora_path "CiroN2022/toy-face" \
  --target_lora_name "toy_face" \
  --num_eigenvector_components 32 \
  --compute_reconstruction \
  --output_dir "./output"
