from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)
pipe.to("cpu")

def generate_image(prompt: str):
    image = pipe(prompt, num_inference_steps=10).images[0]
    return image
