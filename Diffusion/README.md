# SDXL inference using EigenLoRA

## Adapting for SDXL
Our experiments are run on 1 NVIDIA A5000 GPU card. The results may vary due to different GPU models, drivers, CUDA SDK versions, floating-point precisions, and random seeds. 


<h1 align="center"> 
    <image src="../imgs/diffusion.png"/>
</h1>

## Steps to reproduce our results

### 1. Navigate to Diffusion directory
```bash
cd Diffusion/
```

### 2. Generate EigenLoRA and Reconstructed LoRAs

Generate EigenLoRA initialization and reconstructed LoRAs for a target style using a set of source LoRAs.

**Example: Generating 'toy_face' EigenLoRA**

```bash
sh ./scripts/generation/get_eigenlora_toy_face.sh
```

This will save:
- EigenLoRA in `./output/toy_face_eigenlora/weights_sdxl.safetensors`
- Reconstructed LoRA in `./output/toy_face_recons/weights_sdxl.safetensors`

**Using Custom Source LoRAs (Manual):**

```bash
python get_eigenlora.py \
  --source_lora_paths path/to/lora1 path/to/lora2 \
  --source_lora_names name1 name2 \
  --target_lora_path path/to/target \
  --target_lora_name target_name \
  --num_eigenvector_components 32 \
  --output_dir "./output"
```

### 3. Generate Images

Generate images using the created EigenLoRA or reconstructed LoRA adapters.

**Using Reconstructed LoRA:**

```bash
sh ./scripts/inference/infer_toy_face_recons.sh
```

**Using EigenLoRA:**

```bash
sh ./scripts/inference/infer_toy_face_eigenlora.sh
```

## File Structure

```
Diffusion/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── get_eigenlora.py        # Script to compute EigenLoRA components
├── sdxl_inference.py       # Script for image generation
├── utils.py                # Utility functions for EigenLoRA computation
└── scripts/
    ├── generation/         # Scripts for creating EigenLoRAs
    └── inference/          # Scripts for image generation
```


