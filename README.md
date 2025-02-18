<h1 align="center">
    <p> EigenLoRAx: Recycling Adapters to Find Principal Subspaces for Resource-Efficient Adaptation and Inference</p>
</h1>
 
<h1 align="center"> 
    <image src="./imgs/elorax.png"/>
</h1>

The Official PyTorch implementation of [**EigenLoRAx: Recycling Adapters to Find Principal Subspaces for Resource-Efficient Adaptation and Inference**](https://arxiv.org/abs/2502.04700)

We are still updating the code and its instructions in the coming weeks. Please watch (and leave a star if you like our work) this space for continued update.

### Setup
```console
conda env create -f environment.yml
conda activate eigenlora
```


## Usage

In order to find the EigenLoRAs Principal Components, start with a few pretrained LoRA adapters for the same base model. 

## Citation
If you find EigenLoRAx useful, please consider giving a star and citation:
```bibtex
@misc{kaushik2025eigenloraxrecyclingadaptersprincipal,
      title={EigenLoRAx: Recycling Adapters to Find Principal Subspaces for Resource-Efficient Adaptation and Inference}, 
      author={Prakhar Kaushik and Ankit Vaidya and Shravan Chaudhari and Alan Yuille},
      year={2025},
      eprint={2502.04700},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.04700}, 
}
```
