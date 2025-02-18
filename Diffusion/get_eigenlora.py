import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers import StableDiffusionXLPipeline
import torch
from safetensors.torch import save_file
from utils import (
    consolidate_loras_sdxl,
    get_eigenvectors,
    calculate_reconstructed_loras,
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

lora_dict_train = {}


lora_dict_train = consolidate_loras_sdxl(
    pipe, lora_dict_train, "CiroN2022/toy-face", "toy_face"
)
lora_dict_train = consolidate_loras_sdxl(
    pipe, lora_dict_train, "CiroN2022/ascii-art", "ascii"
)
lora_dict_train = consolidate_loras_sdxl(
    pipe, lora_dict_train, "CiroN2022/cyber-aesthetic", "cyber"
)
lora_dict_train = consolidate_loras_sdxl(
    pipe, lora_dict_train, "CiroN2022/digital-human", "digital_human"
)
lora_dict_train = consolidate_loras_sdxl(
    pipe, lora_dict_train, "CiroN2022/awesome-toys", "awesome_toys"
)
lora_dict_train = consolidate_loras_sdxl(
    pipe, lora_dict_train, "CiroN2022/street-art", "street_art"
)
lora_dict_train = consolidate_loras_sdxl(
    pipe, lora_dict_train, "CiroN2022/road-sign", "road_sign"
)
lora_dict_train = consolidate_loras_sdxl(
    pipe, lora_dict_train, "CiroN2022/cube-craft", "cube_craft"
)
lora_dict_train = consolidate_loras_sdxl(
    pipe, lora_dict_train, "CiroN2022/mind-warp", "mind_warp"
)
lora_dict_train = consolidate_loras_sdxl(
    pipe, lora_dict_train, "CiroN2022/sigil", "sigil"
)
lora_dict_train = consolidate_loras_sdxl(
    pipe, lora_dict_train, "CiroN2022/anipunks", "anipunks"
)
lora_dict_train = consolidate_loras_sdxl(
    pipe, lora_dict_train, "CiroN2022/alchemy", "alchemy"
)
lora_dict_train = consolidate_loras_sdxl(
    pipe, lora_dict_train, "CiroN2022/fauna-portrait", "fauna_portrait"
)
lora_dict_train = consolidate_loras_sdxl(
    pipe, lora_dict_train, "CiroN2022/breakcore-style", "breakcore"
)
lora_dict_train = consolidate_loras_sdxl(
    pipe, lora_dict_train, "CiroN2022/skull-graphics", "skull_graphics"
)

eig_dict = get_eigenvectors(lora_dict_train, False)
recons_lora = calculate_reconstructed_loras(pipe, "CiroN2022/toy-face", eig_dict, 32)
os.mkdir("toy_face_recons")
save_file(recons_lora, "toy_face_recons/weights_sdxl.safetensors")


recons_lora = calculate_reconstructed_loras(pipe, "CiroN2022/mind-warp", eig_dict, 32)
os.mkdir("mind_warp_recons")
save_file(recons_lora, "mind_warp/weights_sdxl.safetensors")
