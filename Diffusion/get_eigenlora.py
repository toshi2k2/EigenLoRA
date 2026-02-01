"""
Script to compute EigenLoRA components from pre-trained SDXL LoRA adapters.
This script computes eigenvectors from source LoRA adapters and generates
EigenLoRA initialization for target styles in Stable Diffusion XL.
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import save_file
from utils import (
    consolidate_loras_sdxl,
    get_eigenvectors,
    calculate_reconstructed_loras,
    calculate_eigenloras,
)


# Default source LoRAs for training eigenvector basis
DEFAULT_SOURCE_LORAS = [
    ("CiroN2022/toy-face", "toy_face"),
    ("CiroN2022/ascii-art", "ascii"),
    ("CiroN2022/cyber-aesthetic", "cyber"),
    ("CiroN2022/digital-human", "digital_human"),
    ("CiroN2022/awesome-toys", "awesome_toys"),
    ("CiroN2022/street-art", "street_art"),
    ("CiroN2022/road-sign", "road_sign"),
    ("CiroN2022/cube-craft", "cube_craft"),
    ("CiroN2022/mind-warp", "mind_warp"),
    ("CiroN2022/sigil", "sigil"),
    ("CiroN2022/anipunks", "anipunks"),
    ("CiroN2022/alchemy", "alchemy"),
    ("CiroN2022/fauna-portrait", "fauna_portrait"),
    ("CiroN2022/breakcore-style", "breakcore"),
    ("CiroN2022/skull-graphics", "skull_graphics"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute EigenLoRA components from pre-trained SDXL LoRA adapters"
    )

    # Source LoRA adapters
    parser.add_argument(
        "--source_lora_paths",
        type=str,
        nargs="+",
        default=None,
        help="Paths or HuggingFace model IDs for source LoRA adapters. If not provided, uses default CiroN2022 LoRAs.",
    )
    parser.add_argument(
        "--source_lora_names",
        type=str,
        nargs="+",
        default=None,
        help="Names for each source LoRA adapter (must match number of paths)",
    )

    # Target LoRA configuration
    parser.add_argument(
        "--target_lora_path",
        type=str,
        required=True,
        help="Path or HuggingFace model ID for target LoRA to compute EigenLoRA for",
    )
    parser.add_argument(
        "--target_lora_name",
        type=str,
        required=True,
        help="Name for the target LoRA adapter",
    )

    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model name or path (default: stabilityai/stable-diffusion-xl-base-1.0)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Torch dtype for model loading (default: float16)",
    )

    # EigenLoRA configuration
    parser.add_argument(
        "--num_eigenvector_components",
        type=int,
        default=32,
        help="Number of eigenvector components to use (default: 32)",
    )

    # Processing options
    parser.add_argument(
        "--unwind_tensor",
        action="store_true",
        help="Unwind tensors when computing eigenvectors",
    )
    parser.add_argument(
        "--compute_reconstruction",
        action="store_true",
        help="Also compute reconstructed LoRA in addition to EigenLoRA",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for EigenLoRA adapter",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="weights_sdxl.safetensors",
        help="Output filename for weights (default: weights_sdxl.safetensors)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.source_lora_paths is not None and args.source_lora_names is not None:
        if len(args.source_lora_paths) != len(args.source_lora_names):
            parser.error(
                "Number of source_lora_paths must match number of source_lora_names"
            )

    # Set torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    args.torch_dtype = dtype_map[args.torch_dtype]

    return args


def load_pipeline(model_name_or_path, torch_dtype):
    """Load the SDXL pipeline."""
    print(f"Loading SDXL pipeline from {model_name_or_path}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16" if torch_dtype == torch.float16 else None,
    )
    return pipe


def load_source_loras(pipe, source_loras):
    """Load and combine source LoRA adapters."""
    lora_dict = {}

    for path, name in source_loras:
        print(f"Loading LoRA adapter: {name} from {path}")
        lora_dict = consolidate_loras_sdxl(pipe, lora_dict, path, name)

    return lora_dict


def compute_and_save_eigenlora(args, pipe, eig_dict):
    """Compute EigenLoRA and save to disk."""
    print(f"Computing EigenLoRA for {args.target_lora_name}...")
    eigenlora_sd = calculate_eigenloras(
        pipe, args.target_lora_path, eig_dict, args.num_eigenvector_components
    )

    # Create output directory
    eigenlora_dir = os.path.join(args.output_dir, f"{args.target_lora_name}_eigenlora")
    os.makedirs(eigenlora_dir, exist_ok=True)

    # Save EigenLoRA weights
    output_path = os.path.join(eigenlora_dir, args.output_filename)
    save_file(eigenlora_sd, output_path)
    print(f"Saved EigenLoRA weights to {output_path}")


def compute_and_save_reconstruction(args, pipe, eig_dict):
    """Compute reconstructed LoRA and save to disk."""
    print(f"Computing reconstructed LoRA for {args.target_lora_name}...")
    recons_sd = calculate_reconstructed_loras(
        pipe, args.target_lora_path, eig_dict, args.num_eigenvector_components
    )

    # Create output directory
    recons_dir = os.path.join(args.output_dir, f"{args.target_lora_name}_recons")
    os.makedirs(recons_dir, exist_ok=True)

    # Save reconstructed LoRA weights
    output_path = os.path.join(recons_dir, args.output_filename)
    save_file(recons_sd, output_path)
    print(f"Saved reconstructed LoRA weights to {output_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("EigenLoRA Component Computation for SDXL")
    print("=" * 60)
    print(f"Target LoRA: {args.target_lora_name} ({args.target_lora_path})")
    print(f"Base model: {args.model_name_or_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Components: {args.num_eigenvector_components}")
    print("=" * 60)

    # Load pipeline
    pipe = load_pipeline(args.model_name_or_path, args.torch_dtype)

    # Determine source LoRAs
    if args.source_lora_paths is not None and args.source_lora_names is not None:
        source_loras = list(zip(args.source_lora_paths, args.source_lora_names))
    else:
        print("Using default source LoRAs...")
        source_loras = DEFAULT_SOURCE_LORAS

    print(f"Source LoRAs: {[name for _, name in source_loras]}")

    # Load source LoRAs and compute eigenvectors
    lora_dict = load_source_loras(pipe, source_loras)

    print("Computing eigenvectors...")
    eig_dict = get_eigenvectors(lora_dict, args.unwind_tensor)

    # Compute and save EigenLoRA
    compute_and_save_eigenlora(args, pipe, eig_dict)

    # Optionally compute and save reconstruction
    if args.compute_reconstruction:
        compute_and_save_reconstruction(args, pipe, eig_dict)

    print("=" * 60)
    print("EigenLoRA computation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
