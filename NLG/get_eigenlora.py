"""
Script to compute EigenLoRA reconstruction from pre-trained Mistral LoRA adapters.
This script computes eigenvectors from source LoRA adapters and generates
EigenLoRA initialization for the Lots-of-LoRAs benchmark.
"""

import os
import sys
import json
import argparse
from copy import deepcopy
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm
from peft import load_peft_weights, LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file, load_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute EigenLoRA reconstruction from pre-trained Mistral LoRA adapters"
    )

    # Source LoRA configuration
    parser.add_argument(
        "--source_lora_config",
        type=str,
        default="train.json",
        help="JSON file containing source LoRA names for eigenvector computation (default: train.json)",
    )

    # Target LoRA configuration
    parser.add_argument(
        "--target_lora_config",
        type=str,
        default=None,
        help="JSON file containing target LoRA names for reconstruction. If not provided, uses source config.",
    )
    parser.add_argument(
        "--target_lora_names",
        type=str,
        nargs="+",
        default=None,
        help="Specific target LoRA names to process (overrides target_lora_config)",
    )

    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model name or path (default: mistralai/Mistral-7B-Instruct-v0.2)",
    )

    # EigenLoRA configuration
    parser.add_argument(
        "--num_components",
        type=int,
        default=256,
        help="Number of eigenvector components to use (default: 256)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )

    # Component computation
    parser.add_argument(
        "--component_path",
        type=str,
        default=None,
        help="Path to pre-computed component dict. If not provided, will compute from source LoRAs.",
    )
    parser.add_argument(
        "--save_components",
        action="store_true",
        help="Save computed components to disk",
    )
    parser.add_argument(
        "--component_output_path",
        type=str,
        default="./lola_components.pt",
        help="Path to save computed components (default: ./lola_components.pt)",
    )

    # Output configuration
    parser.add_argument(
        "--output_type",
        type=str,
        choices=["reconstruction", "eigenlora"],
        default="reconstruction",
        help="Output type: 'reconstruction' for reconstructed LoRA weights, 'eigenlora' for EigenLoRA components+loadings (default: reconstruction)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for adapters (default: ./output)",
    )

    args = parser.parse_args()
    return args


def get_lora_names(json_file: str) -> List[str]:
    """
    Load LoRA names from a JSON configuration file.

    Args:
        json_file: Path to JSON file containing LoRA names.

    Returns:
        List of LoRA model paths/names.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    names = list(data.values())
    return names


def consolidate_loras(lora_batch: List[str]) -> Dict[str, torch.Tensor]:
    """
    Load and concatenate all LoRAs in a batch by layer.

    Args:
        lora_batch: List of LoRA model paths/names.

    Returns:
        Dictionary mapping layer keys to concatenated tensors.
    """
    lora_dict = {}
    condensed_dict = {}

    for lora_name in tqdm(lora_batch, desc="Loading LoRAs"):
        state_dict = load_peft_weights(lora_name)
        # Extract task name for identification
        name = lora_name.replace("Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-", "")
        for key, value in state_dict.items():
            try:
                lora_dict[key].update({name: value})
            except KeyError:
                lora_dict[key] = {name: value}

    for layer_key in tqdm(lora_dict.keys(), desc="Concatenating layers"):
        tensor_list = []
        for lora_key in lora_dict[layer_key].keys():
            tensor = lora_dict[layer_key][lora_key].cpu()
            if tensor.shape[0] < tensor.shape[1]:
                tensor = tensor.t()
            tensor_list.append(tensor)
        concat_tensors = torch.cat(tensor_list, dim=1)
        condensed_dict.update({layer_key: concat_tensors})

    return condensed_dict


def get_components(
    state_dict: Dict[str, torch.Tensor], num_components: int
) -> Dict[str, torch.Tensor]:
    """
    Compute principal components from concatenated LoRA matrices.

    Args:
        state_dict: Dictionary of concatenated LoRA tensors by layer.
        num_components: Number of principal components to compute.

    Returns:
        Dictionary mapping layer keys to component matrices (U from SVD).
    """
    to_return = {}
    for k in tqdm(state_dict.keys(), desc="Computing components"):
        matrix = state_dict[k].cuda()
        mean = matrix.mean(axis=1, keepdim=True)
        matrix = matrix - mean
        U, S, V = torch.svd_lowrank(matrix, q=num_components, niter=40)
        to_return.update({k: U.contiguous().cpu()})
    return to_return


def get_lora_reconstruction(
    lora_dict: Dict[str, torch.Tensor],
    component_dict: Dict[str, torch.Tensor],
    num_components: int,
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct LoRA weights using principal components.

    Args:
        lora_dict: Original LoRA state dictionary.
        component_dict: Dictionary of principal components.
        num_components: Number of components to use for reconstruction.

    Returns:
        Reconstructed LoRA state dictionary.
    """
    to_return = {}
    for k in tqdm(lora_dict.keys(), desc="Reconstructing"):
        if "lora_A" in k:
            components = component_dict[k][:, :num_components].cuda()
            lora = lora_dict[k].t()
            loadings = torch.mm(components.t(), lora).squeeze(dim=1)
            recons = torch.sum(
                components.unsqueeze(0) * loadings.t().unsqueeze(1),
                dim=-1,
            )
            to_return.update({k: recons.contiguous().cpu()})
        elif "lora_B" in k:
            components = component_dict[k][:, :num_components].cuda()
            lora = lora_dict[k]
            loadings = torch.mm(components.t(), lora).squeeze(dim=1)
            recons = torch.sum(
                components.unsqueeze(0) * loadings.t().unsqueeze(1),
                dim=-1,
            ).t()
            to_return.update({k: recons.contiguous().cpu()})
    return to_return


def get_eigenlora_components(
    lora_dict: Dict[str, torch.Tensor],
    component_dict: Dict[str, torch.Tensor],
    num_components: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute EigenLoRA components and loadings for a LoRA.

    Unlike reconstruction, this saves the components and loadings separately,
    enabling parameter-efficient fine-tuning.

    Args:
        lora_dict: Original LoRA state dictionary.
        component_dict: Dictionary of principal components.
        num_components: Number of components to use.

    Returns:
        EigenLoRA state dictionary with components and loadings.
    """
    to_return = {}
    for k in tqdm(lora_dict.keys(), desc="Computing EigenLoRA"):
        if "lora_A" in k:
            components = component_dict[k][:, :num_components].cuda()
            lora = lora_dict[k].t()
            loadings = torch.mm(components.t(), lora).squeeze(dim=1)
            # Create component and loading keys
            comp_key = k.replace("lora_A", "eigenlora_A.components")
            load_key = k.replace("lora_A", "eigenlora_A.loadings")
            to_return.update({comp_key: components.contiguous().cpu()})
            to_return.update({load_key: loadings.contiguous().cpu()})
        elif "lora_B" in k:
            components = component_dict[k][:, :num_components].cuda()
            lora = lora_dict[k]
            loadings = torch.mm(components.t(), lora).squeeze(dim=1)
            # Create component and loading keys
            comp_key = k.replace("lora_B", "eigenlora_B.components")
            load_key = k.replace("lora_B", "eigenlora_B.loadings")
            to_return.update({comp_key: components.contiguous().cpu()})
            to_return.update({load_key: loadings.contiguous().cpu()})
    return to_return


def main():
    args = parse_args()

    print("=" * 60)
    print("EigenLoRA Generation for Mistral (Lots-of-LoRAs)")
    print("=" * 60)
    print(f"Source config: {args.source_lora_config}")
    print(f"Components: {args.num_components}")
    print(f"Output type: {args.output_type}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Load or compute components
    if args.component_path and os.path.exists(args.component_path):
        print(f"Loading pre-computed components from {args.component_path}...")
        component_dict = torch.load(args.component_path, weights_only=False)
    else:
        # Load source LoRAs
        source_loras = get_lora_names(args.source_lora_config)
        print(f"Source LoRAs: {len(source_loras)} adapters")

        # Consolidate and compute components
        condensed_dict = consolidate_loras(source_loras)
        component_dict = get_components(condensed_dict, args.num_components)

        if args.save_components:
            torch.save(component_dict, args.component_output_path)
            print(f"Saved components to {args.component_output_path}")

    # Determine target LoRAs
    if args.target_lora_names:
        target_loras = args.target_lora_names
    elif args.target_lora_config:
        target_loras = get_lora_names(args.target_lora_config)
    else:
        target_loras = get_lora_names(args.source_lora_config)

    print(f"Target LoRAs: {len(target_loras)} adapters")

    # Load base model
    print(f"Loading base model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    lora_config = LoraConfig(
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    # Process each target LoRA
    os.makedirs(args.output_dir, exist_ok=True)

    for lora_name in tqdm(target_loras, desc="Processing targets"):
        lora_dict = load_peft_weights(lora_name)

        # Extract task name
        task_name = lora_name.replace(
            "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-", ""
        )

        # Determine output based on type
        if args.output_type == "reconstruction":
            output_weights = get_lora_reconstruction(
                lora_dict, component_dict, args.num_components
            )
            suffix = "_recons"
        else:  # eigenlora
            output_weights = get_eigenlora_components(
                lora_dict, component_dict, args.num_components
            )
            suffix = "_eigenlora"

        # Save adapter with appropriate suffix
        output_path = os.path.join(args.output_dir, f"{task_name}{suffix}")
        os.makedirs(output_path, exist_ok=True)

        # Create and save PEFT model config
        model_with_lora = get_peft_model(model, lora_config)
        model_with_lora.save_pretrained(output_path)

        # Save weights
        save_file(
            output_weights, os.path.join(output_path, "adapter_model.safetensors")
        )
        print(f"Saved {task_name}{suffix} to {output_path}")

    print("=" * 60)
    print(f"EigenLoRA {args.output_type} complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
