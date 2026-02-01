# NLG Experiments with EigenLoRA

## Mistral Instruction-Following with EigenLoRA

Our experiments are run on NVIDIA A5000 GPU cards. The results may vary due to different GPU models, drivers, CUDA SDK versions, floating-point precisions, and random seeds.

<h1 align="center"> 
    <image src="../imgs/lots_of_loras.png"/>
</h1>

## Data Splits

The experiments use two sets of tasks from the Lots-of-LoRAs benchmark:

- **IID (In-Distribution)**: 10 tasks used for training and in-distribution evaluation
- **OOD (Out-of-Distribution)**: 10 held-out tasks for zero-shot evaluation

## Steps to reproduce our results

### 1. Navigate to NLG directory
```bash
cd NLG/
```

### 2. Generate EigenLoRA or Reconstructed LoRAs

Generate EigenLoRA or reconstructed LoRA adapters for target tasks:

**Reconstruction (default):**
```bash
python get_eigenlora.py \
  --source_lora_config train.json \
  --target_lora_config train_subset.json \
  --num_components 256 \
  --output_type reconstruction \
  --output_dir ./output
```

**EigenLoRA (components + loadings):**
```bash
python get_eigenlora.py \
  --source_lora_config train.json \
  --target_lora_config eval.json \
  --num_components 256 \
  --output_type eigenlora \
  --output_dir ./output
```

Output will be saved as:
- Reconstruction: `./output/{task}_recons/`
- EigenLoRA: `./output/{task}_eigenlora/`

### 3. Run Evaluation

Use the shell script or Python directly:

**Using shell script:**
```bash
# Evaluate IID task
sh ./scripts/run_eval.sh --task task076 --dataset_source iid

# Evaluate OOD task
sh ./scripts/run_eval.sh --task task280 --dataset_source ood

# Evaluate with custom dataset
sh ./scripts/run_eval.sh --task mytask --dataset_source custom --custom_dataset MyOrg/my_dataset
```

**Using Python directly:**
```bash
python mistral_eval.py \
  --task task076 \
  --dataset_source iid \
  --adapter_path ./output/task076_recons \
  --output_dir ./results
```

### 4. Compute ROUGE-L Scores

```bash
# Compute scores for IID results
python rouge_scorer.py --results_dir ./results/iid

# Compute scores for OOD results
python rouge_scorer.py --results_dir ./results/ood
```

