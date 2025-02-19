from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from PIL import Image, ImageFilter
import torch
import numpy as np
import cv2

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

# img_url = "/workspace/knguyen/gaussian-splatting/output/4/train/ours_30000/depth/00000.png"
img_url = "/home/meomeo/workspace/data/spinnerf-dataset/book/images_4/20220811_113225.png"
mask_url = "/root/workspace/gaussian-splatting/output/9/train/ours_30000/depth/20220819_112915.jpg"


image = load_image(img_url)
shape = image.size

image = image.resize((1024, 1024))
# Load and resize the mask image
mask_image = load_image(mask_url).resize((1024, 1024))

# Convert the mask to grayscale and a NumPy array
mask_image = mask_image.convert("L")
mask_array = np.array(mask_image)

# Define a dilation kernel (structuring element)
kernel_size = 50  # Adjust as needed
kernel = np.ones((kernel_size, kernel_size), np.uint8)

# Perform dilation using cv2.dilate
dilated_mask = cv2.dilate(mask_array, kernel, iterations=1)

# Scale the result if needed (e.g., binary mask to 0-255 range)
scaled_mask = (dilated_mask > 0).astype(np.uint8) * 255

# Convert back to PIL Image and save
scaled_mask_image = Image.fromarray(scaled_mask)

prompt = "clear the wall and the table"
# prompt = """Inpaint the masked region of this depth image. The image shows a 3D scene where some parts are occluded or missing data (masked region). Fill the missing parts by reconstructing the depth values based on surrounding pixels. The inpainting should preserve the smooth transitions and consistent depth information in the scene, while seamlessly blending with the surrounding depth data. Ensure that the changes in the inpainted region align with the overall structure and depth information of the rest of the image."""
# prompt = """Restore the structure and consistency of the image.\n
#                         Fill in the areas with natural, seamless transitions,\n
#                         maintaining the original details and proportions. Ensure that the perspective\n
#                         is accurate and that the texture and color transitions are smooth. Repair any missing\n
#                         or misaligned elements, ensuring the scene looks cohesive and realistic, as if it was never\n
#                         altered."""
generator = torch.Generator(device="cuda").manual_seed(55)


image = pipe(
  prompt=prompt,
  negative_prompt="object, human",
  image=image,
  mask_image=scaled_mask_image,
  guidance_scale=8.0,
  num_inference_steps=20,  # steps between 15 and 30 work well for us
  strength=0.999,  # make sure to use `strength` below 1.0
  generator=generator,
).images[0]

image = image.resize(shape)

image.save("output_chair.jpg")