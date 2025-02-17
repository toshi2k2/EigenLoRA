import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers import DiffusionPipeline
import torch


pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")

pipe.load_lora_weights(
    "ankit-vaidya19/toy_face_lora_recons",
    weight_name="toy_face_recons.safetensors",
    adapter_name="toy",
    use_eigenlora=False,
)

prompt = "toy_face of a red headed man with a beard and blue eyes."

lora_scale = 1
image = pipe(
    prompt,
    num_inference_steps=30,
    cross_attention_kwargs={"scale": lora_scale},
    generator=torch.manual_seed(0),
).images[0]
image.save("image.png")
