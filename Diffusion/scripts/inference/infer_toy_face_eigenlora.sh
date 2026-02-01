#!/bin/bash

# Generate image using EigenLoRA for 'toy_face'

python sdxl_inference.py \
  --lora_path "./output/toy_face_eigenlora" \
  --weight_name "weights_sdxl.safetensors" \
  --use_eigenlora \
  --prompt "toy_face of a red headed man with a beard and blue eyes" \
  --output_path "image_eigenlora.png"
