# !pip install transformers accelerate
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DDIMScheduler, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import torch
import cv2

init_image = load_image("/workspace/knguyen/gaussian-splatting/output/4/train/ours_30000/depth/00099.png")
size = init_image.size
init_image = init_image.resize((1024, 1024))

generator = torch.Generator(device="cpu").manual_seed(0)

mask_image = load_image("/workspace/data/spinnerf-dataset/4/seg/20220819_105840.png")
mask_image = mask_image.resize((1024, 1024))
# Convert the mask to grayscale and a NumPy array
mask_image = mask_image.convert("L")
mask_array = np.array(mask_image)

# Define a dilation kernel (structuring element)
kernel_size = 150  # Adjust as needed
kernel = np.ones((kernel_size, kernel_size), np.uint8)

# Perform dilation using cv2.dilate
dilated_mask = cv2.dilate(mask_array, kernel, iterations=1)

# Scale the result if needed (e.g., binary mask to 0-255 range)
scaled_mask = (dilated_mask > 0).astype(np.uint8) * 255

# Convert back to PIL Image and save
scaled_mask_image = Image.fromarray(scaled_mask)

control_image = load_image("/workspace/knguyen/sdxl/lama_depth.png")
control_image = control_image.resize((1024, 1024))

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")

pipe.enable_model_cpu_offload()

# generate image
image = pipe(
    "Inpaint to fit surrounding background color",
    num_inference_steps=20,
    generator=generator,
    eta=1.0,
    image=init_image,
    mask_image=scaled_mask_image,
    control_image=control_image,
).images[0]

image = image.resize(size)

image.save("controlnet_output.png")