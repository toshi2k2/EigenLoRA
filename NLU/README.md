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
