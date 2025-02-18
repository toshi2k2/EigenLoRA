# Adapting RoBERTa using EigenLoRA

## Adapting to the GLUE Benchmark
Our experiments on the GLUE benchmark are run on 4 NVIDIA A5000 GPU cards. The results may vary due to different GPU models, drivers, CUDA SDK versions, floating-point precisions, and random seeds. 


<h1 align="center"> 
    <image src="../imgs/glue.png"/>
</h1>

## Steps to reproduce GLUE results in low-resource settings
### NLU
```console
cd NLU/
```

### Obtain the initial eigenloras
```console
python get_eigenlora.py
```

### Start the experiments for the obtained eigenloras
```console
sh ./scripts/glue_mrpc.sh
sh ./scripts/glue_stsb.sh
```


### Evaluate the trained EigenLoRAs
```console
python run_glue.py \
  --model_name_or_path roberta-base \
  --task_name cola \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --seed 0 \
  --learning_rate 4e-3 \
  --lr_scheduler_type 'reduce_lr_on_plateau' \
  --weight_decay 0.1 \
  --warmup_ratio 0.06 \
  --num_train_epochs 80 \
  --evaluation_strategy epoch \
  --apply_eigenlora True\
  --eigenlora_r 8 \
  --eigenlora_num_components 32 \
  --eigenlora_adapter_name cola \
  --eigenlora_load_path ankit-vaidya19/cola_eigenlora_r_8_c_32 \
  --eigenlora_save_path ./cola_eigenlora_trained \
  --output_dir ./cola \
  --overwrite_output_dir \
  --logging_dir ./cola \
  --logging_steps 10 \
  --report_to wandb \
  --run_name roberta_cola
```