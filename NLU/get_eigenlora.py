"""
Script to compute EigenLoRA components from pre-trained LoRA adapters.
This script computes eigenvectors from source LoRA adapters and generates
EigenLoRA initialization for target tasks.
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peft import EigenLoRAConfig, get_peft_model, TaskType, load_peft_weights
from transformers import AutoModelForSequenceClassification, AutoConfig
from safetensors.torch import save_file
from utils import (
    combine_loras,
    get_eigenvectors,
    calculate_eigenloras,
    add_gs_vectors,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute EigenLoRA components from pre-trained LoRA adapters"
    )

    # Source LoRA adapters
    parser.add_argument(
        "--source_lora_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths or HuggingFace model IDs for source LoRA adapters",
    )
    parser.add_argument(
        "--source_lora_names",
        type=str,
        nargs="+",
        required=True,
        help="Names for each source LoRA adapter (must match number of paths)",
    )

    # Target task configuration
    parser.add_argument(
        "--target_task_name",
        type=str,
        required=True,
        help="Name of the target task for EigenLoRA initialization",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Number of labels for the target task (default: 2)",
    )

    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="roberta-base",
        help="Base model name or path (default: roberta-base)",
    )

    # EigenLoRA configuration
    parser.add_argument(
        "--eigenlora_r",
        type=int,
        default=8,
        help="LoRA rank for EigenLoRA (default: 8)",
    )
    parser.add_argument(
        "--num_eigenvector_components",
        type=int,
        default=32,
        help="Number of eigenvector components to use (default: 32)",
    )
    parser.add_argument(
        "--num_gram_schmidt_components",
        type=int,
        default=32,
        help="Number of random Gram-Schmidt components to add (default: 32)",
    )

    # Processing options
    parser.add_argument(
        "--unwind_tensor",
        action="store_true",
        help="Unwind tensors when computing eigenvectors",
    )
    parser.add_argument(
        "--loading_source_index",
        type=int,
        default=0,
        help="Index of source LoRA to use for computing loadings. Set to -1 for random loadings (default: 0)",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for EigenLoRA adapter",
    )
    parser.add_argument(
        "--save_eigenlora_components",
        action="store_true",
        default=True,
        help="Save EigenLoRA components (default: True)",
    )
    parser.add_argument(
        "--save_eigenlora_loadings",
        action="store_true",
        default=True,
        help="Save EigenLoRA loadings (default: True)",
    )

    args = parser.parse_args()

    # Validate arguments
    if len(args.source_lora_paths) != len(args.source_lora_names):
        parser.error(
            "Number of source_lora_paths must match number of source_lora_names"
        )

    # Infer total components
    args.total_components = (
        args.num_eigenvector_components + args.num_gram_schmidt_components
    )
    print(
        f"Total components inferred: {args.total_components} (Eigenvector: {args.num_eigenvector_components} + Gram-Schmidt: {args.num_gram_schmidt_components})"
    )

    return args


def load_source_loras(paths, names):
    """Load and combine source LoRA adapters."""
    lora_dict = {}
    lora_state_dicts = []

    for path, name in zip(paths, names):
        print(f"Loading LoRA adapter: {name} from {path}")
        state_dict = load_peft_weights(path)
        lora_state_dicts.append(state_dict)
        lora_dict = combine_loras(lora_dict, state_dict, name)

    return lora_dict, lora_state_dicts


def compute_eigenlora(args, lora_dict, loading_source_sd):
    """Compute EigenLoRA components and loadings."""
    print("Computing eigenvectors...")
    eig_dict = get_eigenvectors(lora_dict, args.unwind_tensor)

    # If loading_source_index is -1, keep loadings random
    compute_loadings = args.loading_source_index >= 0

    print(f"Calculating EigenLoRA with {args.num_eigenvector_components} components...")
    if compute_loadings:
        print(f"Computing loadings from source LoRA index {args.loading_source_index}")
    else:
        print("Keeping loadings random (loading_source_index=-1)")

    eigenlora_sd = calculate_eigenloras(
        eig_dict,
        loading_source_sd,
        args.num_eigenvector_components,
        compute_loadings,
    )

    print(
        f"Adding {args.num_gram_schmidt_components} Gram-Schmidt orthogonal vectors..."
    )
    eigenlora_sd = add_gs_vectors(eigenlora_sd, args.num_gram_schmidt_components)

    return eigenlora_sd


def save_eigenlora_adapter(args, eigenlora_sd):
    """Save EigenLoRA adapter to disk."""
    # Create EigenLoRA config
    eigenlora_config = EigenLoRAConfig(
        r=args.eigenlora_r,
        num_components=args.total_components,
        task_type=TaskType.SEQ_CLS,
    )

    # Load base model with target task config
    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=model_config,
    )

    # Apply EigenLoRA
    model = get_peft_model(model, eigenlora_config)

    # Save adapter
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving EigenLoRA adapter to {args.output_dir}")

    model.save_pretrained(
        args.output_dir,
        save_eigenlora_components=args.save_eigenlora_components,
        save_eigenlora_loadings=args.save_eigenlora_loadings,
    )

    # Save computed EigenLoRA state dict
    adapter_path = os.path.join(args.output_dir, "adapter_model.safetensors")
    save_file(eigenlora_sd, adapter_path)
    print(f"Saved adapter weights to {adapter_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("EigenLoRA Component Computation")
    print("=" * 60)
    print(f"Target task: {args.target_task_name}")
    print(f"Source LoRAs: {args.source_lora_names}")
    print(f"Base model: {args.model_name_or_path}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Load source LoRA adapters
    lora_dict, lora_state_dicts = load_source_loras(
        args.source_lora_paths, args.source_lora_names
    )

    # Get loading source state dict
    loading_source_sd = lora_state_dicts[args.loading_source_index]

    # Compute EigenLoRA
    eigenlora_sd = compute_eigenlora(args, lora_dict, loading_source_sd)

    # Save EigenLoRA adapter
    save_eigenlora_adapter(args, eigenlora_sd)

    print("=" * 60)
    print("EigenLoRA computation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
