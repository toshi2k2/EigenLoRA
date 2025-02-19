# SDXL inference using EigenLoRA

## Adapting for SDXL
Our experiments are run on 1 NVIDIA A5000 GPU card. The results may vary due to different GPU models, drivers, CUDA SDK versions, floating-point precisions, and random seeds. 


<h1 align="center"> 
    <image src="../imgs/diffusion.png"/>
</h1>

## Steps to reproduce our results
### Diffusion
```console
cd Diffusion/
```

### Obtain the initial reconstructed LoRAs and EigenLoRAs
```console
python get_eigenlora.py
```

### Generate using the obtained reconstructed LoRAs and EigenLoRAs
```console
python sdxl_inference.py
```
