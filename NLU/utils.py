import torch
import torch.nn as nn
import re


def replace_key(text, substring, replacement):
    pattern = re.compile(re.escape(substring) + r".*", re.DOTALL)
    return re.sub(pattern, replacement, text)


def combine_loras(lora_dict, state_dict, key_name, noise=1.00):
    for key, value in state_dict.items():
        if "classifier" in key:
            continue
        try:
            lora_dict[key].update({key_name: noise * value})
        except KeyError:
            lora_dict[key] = {key_name: noise * value}
    return lora_dict


def calculate_eigenloras(eigenvectors, lora_sd, num_components, loadings=True):
    """
    If non-random loadings then lora_sd should be of same task as EigenLoRA task
    """
    eigenlora_sd = {}
    for k in lora_sd.keys():
        if "lora_A" in k:
            components = nn.Parameter(
                eigenvectors[k]["eigenvectors"][:, :num_components]
            ).contiguous()
            new_key_c = replace_key(k, "lora_A", "eigenlora_A.components")
            eigenlora_sd.update({new_key_c: components})
            if loadings:
                loadings = nn.Parameter(
                    torch.mm(components.t(), lora_sd[k].t()).squeeze(dim=1)
                )
                new_key_l = replace_key(k, "lora_A", "eigenlora_A.loadings")
                eigenlora_sd.update({new_key_l: loadings})
        elif "lora_B" in k:
            components = nn.Parameter(
                eigenvectors[k]["eigenvectors"][:, :num_components]
            ).contiguous()
            new_key_c = replace_key(k, "lora_B", "eigenlora_B.components")
            eigenlora_sd.update({new_key_c: components})
            if loadings:
                loadings = nn.Parameter(
                    torch.mm(components.t(), lora_sd[k]).squeeze(dim=1)
                )
                new_key_l = replace_key(k, "lora_B", "eigenlora_B.loadings")
                eigenlora_sd.update({new_key_l: loadings})
    return eigenlora_sd


def add_classifier(eigenlora_dict, lora_dict):
    for key, value in lora_dict.items():
        if "classifier" in key:
            eigenlora_dict.update({key: value})
    return eigenlora_dict


def get_eigenvectors(lora_dict, unwind_tensor):
    """
    unwind_tensor = True --> Eigenvectors of size m*n,1
    unwind_tensor = False --> Eigenvectors of size m,1
    """
    eigen_dict = {}
    for layer_key in lora_dict.keys():
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
    mean = matrix.mean(axis=1, keepdim=True)
    matrix = matrix - mean
    cov = torch.mm(matrix, matrix.t())
    eigenvals, eigenvecs = torch.linalg.eig(cov)
    eigenvals = eigenvals.to(torch.float32)
    eigenvecs = eigenvecs.to(torch.float32)
    eigenvals, indices = eigenvals.sort(descending=True)
    eigenvecs = eigenvecs[:, indices]
    return {"eigenvalues": eigenvals, "eigenvectors": eigenvecs}


def gram_schmidt_normalization(matrix, vec, eps=1e-8):
    vec = vec.reshape(-1)
    for col in matrix.t():
        proj = (vec @ col) * col
        vec = vec - proj
    norm = torch.norm(vec)
    if norm < eps:
        raise ValueError("Vector is linearly dependent with existing basis")
    vec = vec / norm
    return vec


def add_gs_vectors(state_dict, num_rand_vecs):
    to_return = {}
    for k in state_dict.keys():
        if "components" in k:
            mat = state_dict[k].cpu()
            for i in range(num_rand_vecs):
                rand_vec = torch.rand(max(mat.shape))
                new_vector = gram_schmidt_normalization(mat, rand_vec).unsqueeze(1)
                mat = torch.cat([mat, new_vector], dim=1)
            to_return[k] = mat
    print(f"Number of components --> {min(mat.shape)}")
    return to_return
