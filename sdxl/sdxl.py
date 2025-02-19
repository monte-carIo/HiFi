from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import os
from PIL import Image
from visualize.cam_extract import cam_distances
import utils as utils
import argparse
from datetime import datetime
import numpy as np
import cv2
from simple_lama_inpainting import SimpleLama

def parse_args():
    parser = argparse.ArgumentParser(description="Inpainting with Stable Diffusion XL")
    
    parser.add_argument("--cross_att", action="store_true", help="Enable cross attention")
    
    parser.add_argument("--output_folder", type=str, default="./output", help="Output folder")
    
    parser.add_argument("--viewmapped_path", type=str, help="Output folder")
    
    parser.add_argument("--prompt", type=str, default="""Restore the structure and consistency of the image.\n
                        Fill in the broken or distorted areas with natural, seamless transitions,\n
                        maintaining the original details and proportions. Ensure that the perspective\n
                        is accurate and that the texture and color transitions are smooth. Repair any missing\n
                        or misaligned elements, ensuring the scene looks cohesive and realistic, as if it was never\n
                        altered. clearly edged.""", help="Prompt for inpainting")
    
    parser.add_argument("--data_folder", type=str, default="/workspace/data", help="Data folder")
    
    parser.add_argument("--anchor", type=str, default="IMG_2707.jpg", help="Anchor image for cross attention")
    
    parser.add_argument("--strength", type=float, default=0.9, help="Strength of the inpainting")
    
    parser.add_argument("--cross_eff", type=float, default=0.8, help="Cross attention coefficent")
    
    parser.add_argument("--kernel_size", type=int, default=60, help="Kernel size for dilation")
    
    parser.add_argument("--mask", action="store_true", help="Using the whole mask for inpainting.\
                                                            use this when the error of view mapper is to large\
                                                            and need auto correct")
    
    parser.add_argument("--lama", action="store_true", help="Using the lama model for inpainting.")
    
    parser.add_argument("--num_infer", type=int, default=20, help="Num inference steps")
    
    return parser.parse_args()


def prepare_pipe(args):
    # Initialize the pipeline
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    if args.cross_att:
        pipe.unet.set_attn_processor(
                            processor=utils.CrossViewAttnProcessor(self_attn_coeff=args.cross_eff,
                            unet_chunk_size=2, num_ref=2))
    return pipe

def ensure_binary_mask(mask_image):
    # Ensure the mask is binary (0 and 255) after resizing
    mask_image = mask_image.convert("L")
    mask_array = np.array(mask_image)
    mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(mask_array)

def dialate_mask(mask_image, args):
    mask_image = mask_image.convert("L")
    mask_array = np.array(mask_image)
    if args:
        kernel_size = args.kernel_size  # Adjust as needed
    else:
        kernel_size = 30
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated_mask = cv2.dilate(mask_array, kernel, iterations=1)

    scaled_mask = ((dilated_mask > 0) * 255)
    
    mask = scaled_mask.astype(np.uint8)

    mask_image = Image.fromarray(mask)
    return mask_image
    

def generate(images_folder, masks_folder, output_folder, prompt, pipe, args):
    generator = torch.Generator(device="cuda").manual_seed(0)
    
    if args.cross_att:
        original_image = load_image(args.anchor)
        image = original_image.resize((1024, 1024))  # Resize to 1024x1024 for processing
        mask_image = load_image(os.path.join(masks_folder, os.path.basename(args.anchor).replace("jpg", "png")))
        mask_image = mask_image.resize((1024, 1024))
        ref_image = image
        ref_mask = mask_image

    # Process each image-mask pair
    for file_name in os.listdir(masks_folder):
        if file_name.endswith('png'):
            image_path = os.path.join(images_folder, file_name)
            mask_path = os.path.join(masks_folder, file_name)

            # Ensure the corresponding mask exists
            if not os.path.exists(mask_path):
                print(f"Mask for {file_name} not found, skipping...")
                continue

            # Load image and mask
            original_image = load_image(image_path)  # Load original image for size reference
            original_size = original_image.size  # (width, height)
            image = original_image.resize((1024, 1024))  # Resize to 1024x1024 for processing
            mask_image = load_image(mask_path)
            
            if not args.mask:
                mask_array = np.array(original_image.convert("L"))
                mask = np.where(mask_array < 30, 255, 0).astype(np.uint8)
                if mask.shape != np.array(mask_image.convert("L")).shape:
                    mask = np.array(Image.fromarray(mask).resize(mask_image.size))
                mask_image = dialate_mask(mask_image, None)
                mask = np.bitwise_and(mask, np.array(mask_image.convert("L")) * 255)
                mask_image = Image.fromarray(mask)
                
            mask_image = mask_image.resize((1024, 1024))
            mask_image = ensure_binary_mask(mask_image)
            if args.kernel_size != -1:
                mask_image = dialate_mask(mask_image, args)
            
            # Perform inpainting
            print(f"Processing: {image_path}")
            if args.lama:
                simple_lama = SimpleLama()
                result = simple_lama(image, mask_image)
            elif args.cross_att:
                result = pipe(
                    prompt=[prompt]*2,
                    image=[image, ref_image],
                    mask_image=[mask_image, ref_mask],
                    guidance_scale=2.0,
                    num_inference_steps=args.num_infer,
                    strength=args.strength,
                    generator=generator,
                ).images[0]
            else:
                result = pipe(
                    prompt=prompt,
                    image=image,
                    mask_image=mask_image,
                    guidance_scale=8.0,
                    num_inference_steps=args.num_infer,
                    strength=args.strength,
                    generator=generator,
                ).images[0]

            # Resize the result back to the original size
            result = result.resize(original_size)

            # Save the result
            output_path = os.path.join(output_folder, 'images', file_name.replace('png', 'jpg'))
            result.save(output_path)
            print(f"Saved: {output_path}")
            
def main(args):
    pipe = prepare_pipe(args)
    # Define input directories and output directory
    data_folder = args.data_folder
    images_folder = args.viewmapped_path
    masks_folder = os.path.join(data_folder, "images_4", "label")
    output_folder = args.output_folder
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_folder, current_time)
    prompt = args.prompt
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    
    # Save the prompt and args in the output folder as a text file
    prompt_path = os.path.join(output_folder, "config.txt")
    with open(prompt_path, "w") as f:
        f.write("Arguments:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    generate(images_folder, masks_folder, output_folder, prompt, pipe, args)
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
