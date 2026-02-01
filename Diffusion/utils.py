"""
Utility functions for SDXL EigenLoRA computation.

This module provides functions for:
- Loading and consolidating SDXL LoRA adapters
- Computing eigenvectors from LoRA weight matrices
- Calculating EigenLoRA components and loadings
- Reconstructing LoRAs from principal components
"""

import re
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
from tqdm import tqdm


def replace_key(text: str, substring: str, replacement: str) -> str:
    """
    Replace a substring and everything after it with a replacement string.

    Args:
        text: The original string.
        substring: The substring to find and replace from.
        replacement: The string to replace with.

    Returns:
        The modified string with the replacement applied.
    """
    pattern = re.compile(re.escape(substring) + r".*", re.DOTALL)
    return re.sub(pattern, replacement, text)


def consolidate_loras_sdxl(
    pipe: Any,
    lora_dict: Dict[str, Dict[str, torch.Tensor]],
    lora_name: str,
    key_name: str,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load and consolidate an SDXL LoRA adapter into the combined dictionary.

    This function loads a LoRA adapter using the diffusers pipeline and adds
    its weights to the combined lora_dict, organized by layer key.

    Args:
        pipe: The StableDiffusionXLPipeline instance.
        lora_dict: Existing dictionary of combined LoRAs, organized by layer.
        lora_name: HuggingFace model ID or path to the LoRA adapter.
        key_name: Name identifier for this LoRA adapter.

    Returns:
        Updated lora_dict with the new adapter's weights added.

    Example:
        >>> lora_dict = {}
        >>> lora_dict = consolidate_loras_sdxl(pipe, lora_dict, "CiroN2022/toy-face", "toy_face")
    """
    state_dict, alphas = pipe.lora_state_dict(lora_name, unet_config=pipe.unet.config)
    for key, value in state_dict.items():
        try:
            lora_dict[key].update({key_name: value.squeeze()})
        except KeyError:
            lora_dict[key] = {key_name: value.squeeze()}
    return lora_dict


def get_eigenvectors(
    lora_dict: Dict[str, Dict[str, torch.Tensor]],
    unwind_tensor: bool,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute eigenvectors from combined LoRA weight matrices.

    This function performs PCA on the combined LoRA weights across multiple
    adapters to find the principal directions of variation.

    Args:
        lora_dict: Dictionary of combined LoRA weights organized by layer,
            where each layer maps adapter names to weight tensors.
        unwind_tensor: If True, flatten weight matrices to vectors before PCA.
            If False, compute eigenvectors along the larger dimension.

    Returns:
        Dictionary mapping layer keys to eigenvector dictionaries containing:
        - 'eigenvectors': Principal components sorted by eigenvalue (descending)
        - 'eigenvalues': Corresponding eigenvalues

    Note:
        unwind_tensor=True produces eigenvectors of size (m*n, 1) for m×n matrices.
        unwind_tensor=False produces eigenvectors of size (max(m,n), 1).
    """
    eigen_dict = {}
    for layer_key in tqdm(lora_dict.keys(), desc="Computing eigenvectors"):
        tensor_list = []
        for lora_key in lora_dict[layer_key].keys():
            tensor = lora_dict[layer_key][lora_key]
            if unwind_tensor:
                tensor = tensor.reshape((tensor.shape[0] * tensor.shape[1], 1))
            if tensor.shape[0] < tensor.shape[1]:
                tensor = tensor.t()
            tensor_list.append(tensor)
        concat_tensors = torch.cat(tensor_list, dim=1).to(torch.float32)
        eig = eigendecomposition(concat_tensors)
        eigen_dict.update({layer_key: eig})
    return eigen_dict


def eigendecomposition(matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Perform eigendecomposition on a centered covariance matrix.

    This function centers the input matrix, computes its covariance,
    and returns eigenvectors sorted by eigenvalue in descending order.

    Args:
        matrix: Input matrix of shape (features, samples).

    Returns:
        Dictionary containing:
        - 'eigenvalues': Sorted eigenvalues (descending order), as bfloat16
        - 'eigenvectors': Corresponding eigenvectors as columns, as bfloat16
    """
    mean = matrix.mean(dim=1, keepdim=True)
    matrix = matrix - mean
    cov = torch.mm(matrix.t(), matrix)
    eigenvals, eigenvecs = torch.linalg.eig(cov)
    eigenvals = eigenvals.to(torch.float32)
    eigenvecs = eigenvecs.to(torch.float32)
    eigenvecs = torch.mm(matrix, eigenvecs)
    eigenvecs = torch.nn.functional.normalize(eigenvecs, p=2, dim=0)
    eigenvals, indices = eigenvals.sort(descending=True)
    eigenvecs = eigenvecs[:, indices]
    return {
        "eigenvalues": eigenvals.to(torch.bfloat16),
        "eigenvectors": eigenvecs.to(torch.bfloat16),
    }


def calculate_reconstructed_loras(
    pipe: Any,
    lora_name: str,
    eigenvectors: Dict[str, Dict[str, torch.Tensor]],
    num_components: int,
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct LoRA weights using principal components.

    This function projects the original LoRA weights onto the eigenvector
    subspace and reconstructs them, effectively performing dimensionality
    reduction and reconstruction.

    Args:
        pipe: The StableDiffusionXLPipeline instance.
        lora_name: HuggingFace model ID or path to the LoRA adapter.
        eigenvectors: Dictionary containing eigenvectors for each layer.
        num_components: Number of principal components to use for reconstruction.

    Returns:
        State dictionary containing reconstructed LoRA weights.

    Note:
        This is useful for evaluating how well the eigenvector basis
        captures the original LoRA's style.
    """
    recons_lora_sd = {}
    lora_sd, alphas = pipe.lora_state_dict(lora_name, unet_config=pipe.unet.config)

    for k in lora_sd.keys():
        if ".up." in k:
            components = nn.Parameter(
                eigenvectors[k]["eigenvectors"][:, :num_components]
            ).contiguous()
            loadings = nn.Parameter(torch.mm(components.t(), lora_sd[k]).squeeze(dim=1))
            recons = (
                torch.sum(
                    components.unsqueeze(0) * loadings.t().unsqueeze(1),
                    dim=-1,
                )
                .t()
                .contiguous()
            )
            recons_lora_sd.update({k: recons})
        elif ".down." in k:
            components = nn.Parameter(
                eigenvectors[k]["eigenvectors"][:, :num_components]
            ).contiguous()
            loadings = nn.Parameter(
                torch.mm(components.t(), lora_sd[k].t()).squeeze(dim=1)
            )
            recons = torch.sum(
                components.unsqueeze(0) * loadings.t().unsqueeze(1),
                dim=-1,
            ).contiguous()
            recons_lora_sd.update({k: recons})

    return recons_lora_sd


def calculate_eigenloras(
    pipe: Any,
    lora_name: str,
    eigenvectors: Dict[str, Dict[str, torch.Tensor]],
    num_components: int,
) -> Dict[str, torch.Tensor]:
    """
    Calculate EigenLoRA components and loadings from eigenvectors.

    This function projects LoRA weights onto the principal eigenvector subspace
    to create EigenLoRA components. It computes both the component matrices
    (fixed basis vectors) and the loadings (trainable coefficients).

    Args:
        pipe: The StableDiffusionXLPipeline instance.
        lora_name: HuggingFace model ID or path to the source LoRA adapter.
        eigenvectors: Dictionary containing eigenvectors for each layer.
        num_components: Number of principal components to retain.

    Returns:
        State dictionary containing EigenLoRA components and loadings,
        with keys in the format expected by the EigenLoRA diffusers integration.

    Note:
        The returned state dict contains both '.components' and '.loadings'
        keys for each layer, enabling parameter-efficient fine-tuning.
    """
    eigenlora_sd = {}
    lora_sd, alphas = pipe.lora_state_dict(lora_name, unet_config=pipe.unet.config)

    for k in lora_sd.keys():
        if ".up." in k:
            components = nn.Parameter(
                eigenvectors[k]["eigenvectors"][:, :num_components]
            ).contiguous()
            loadings = nn.Parameter(torch.mm(components.t(), lora_sd[k]).squeeze(dim=1))
            if ".lora_linear_layer." in k:
                new_key_c = replace_key(
                    k, "lora_linear_layer.up", "eigenlora_linear_layer.up.components"
                )
                new_key_l = replace_key(
                    k, "lora_linear_layer.up", "eigenlora_linear_layer.up.loadings"
                )
            else:
                new_key_c = replace_key(k, "lora.up", "eigenlora.up.components")
                new_key_l = replace_key(k, "lora.up", "eigenlora.up.loadings")
            eigenlora_sd.update({new_key_c: components})
            eigenlora_sd.update({new_key_l: loadings})
        elif ".down." in k:
            components = nn.Parameter(
                eigenvectors[k]["eigenvectors"][:, :num_components]
            ).contiguous()
            loadings = nn.Parameter(
                torch.mm(components.t(), lora_sd[k].t()).squeeze(dim=1)
            )
            if ".lora_linear_layer." in k:
                new_key_c = replace_key(
                    k,
                    "lora_linear_layer.down",
                    "eigenlora_linear_layer.down.components",
                )
                new_key_l = replace_key(
                    k, "lora_linear_layer.down", "eigenlora_linear_layer.down.loadings"
                )
            else:
                new_key_c = replace_key(k, "lora.down", "eigenlora.down.components")
                new_key_l = replace_key(k, "lora.down", "eigenlora.down.loadings")
            eigenlora_sd.update({new_key_c: components})
            eigenlora_sd.update({new_key_l: loadings})

    return eigenlora_sd
