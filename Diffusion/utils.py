import torch
import torch.nn as nn
from tqdm import tqdm


def consolidate_loras_sdxl(pipe, lora_dict, lora_name, key_name):
    state_dict, alphas = pipe.lora_state_dict(lora_name, unet_config=pipe.unet.config)
    for key, value in state_dict.items():
        try:
            lora_dict[key].update({key_name: value.squeeze()})
        except KeyError:
            lora_dict[key] = {key_name: value.squeeze()}
    return lora_dict


def get_eigenvectors(lora_dict, unwind_tensor):
    """
    unwind_tensor = True --> Eigenvectors of size m*n,1
    unwind_tensor = False --> Eigenvectors of size m,1
    """
    eigen_dict = {}
    for layer_key in tqdm(lora_dict.keys()):
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


def eigendecomposition(matrix):
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


def calculate_reconstructed_loras(pipe, lora_name, eigenvectors, num_components):
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
