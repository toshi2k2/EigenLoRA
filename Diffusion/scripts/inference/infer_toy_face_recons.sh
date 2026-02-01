#!/bin/bash

# Generate image using Reconstructed LoRA for 'toy_face'

python sdxl_inference.py \
  --lora_path "./output/toy_face_recons" \
  --weight_name "weights_sdxl.safetensors" \
  --prompt "toy_face of a red headed man with a beard and blue eyes" \
  --output_path "image_recons.png"
