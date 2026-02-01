"""
Script for SDXL inference using LoRA or EigenLoRA adapters.
This script loads a Stable Diffusion XL model and generates images
using specified LoRA weights.
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers import DiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images using SDXL with LoRA or EigenLoRA adapters"
    )

    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model name or path (default: stabilityai/stable-diffusion-xl-base-1.0)",
    )

    # LoRA configuration
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path or HuggingFace model ID for LoRA weights",
    )
    parser.add_argument(
        "--weight_name",
        type=str,
        default=None,
        help="Weight filename if loading from a directory (e.g., 'weights_sdxl.safetensors')",
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default="default",
        help="Name for the loaded adapter (default: default)",
    )
    parser.add_argument(
        "--use_eigenlora",
        action="store_true",
        help="Whether to use EigenLoRA format for loading",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="Scale factor for LoRA weights (default: 1.0)",
    )

    # Generation configuration
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt for image generation",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps (default: 30)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (default: 7.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)",
    )

    # Output configuration
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.png",
        help="Path to save the generated image (default: output.png)",
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Torch dtype for model loading (default: float16)",
    )

    args = parser.parse_args()

    # Set torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    args.torch_dtype = dtype_map[args.torch_dtype]

    return args


def load_pipeline_with_lora(args):
    """Load the SDXL pipeline with LoRA weights."""
    print(f"Loading SDXL pipeline from {args.model_name_or_path}...")
    pipe = DiffusionPipeline.from_pretrained(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype,
    ).to(args.device)

    print(f"Loading LoRA weights from {args.lora_path}...")
    load_kwargs = {
        "adapter_name": args.adapter_name,
        "use_eigenlora": args.use_eigenlora,
    }
    if args.weight_name:
        load_kwargs["weight_name"] = args.weight_name

    pipe.load_lora_weights(args.lora_path, **load_kwargs)

    return pipe


def generate_image(pipe, args):
    """Generate an image using the pipeline."""
    print(f"Generating image with prompt: '{args.prompt}'")

    generator = torch.manual_seed(args.seed)

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        cross_attention_kwargs={"scale": args.lora_scale},
        generator=generator,
    ).images[0]

    return image


def main():
    args = parse_args()

    print("=" * 60)
    print("SDXL Inference with LoRA/EigenLoRA")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"LoRA: {args.lora_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output_path}")
    print("=" * 60)

    # Load pipeline with LoRA
    pipe = load_pipeline_with_lora(args)

    # Generate image
    image = generate_image(pipe, args)

    # Save image
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    image.save(args.output_path)
    print(f"Saved image to {args.output_path}")

    print("=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
