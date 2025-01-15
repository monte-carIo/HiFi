import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# Load the image and mask
image_path = "/workspace/data/spinnerf-dataset/12/images_4/20220823_093810.png"  # Replace with your image path
mask_path = "/workspace/data/spinnerf-dataset/12/images_4/label/20220823_093810.png"    # Replace with your mask path

# Open the image and mask
image = np.array(Image.open(image_path).convert("RGB"))  # Ensure the image is in RGB format
mask = np.array(Image.open(mask_path).convert("L"))  # Ensure the mask is in grayscale

# Dilate the mask
kernel = np.ones((5, 5), np.uint8)  # Define the kernel size for dilation
dilated_mask = cv2.dilate(mask, kernel, iterations=1)
mask = dilated_mask

# Normalize the mask (optional, depends on the range of your mask)
# mask = mask / 255.0  # Scale to range [0, 1] if it's not already

# Overlay mask on the image (with transparency)
overlay_color = np.array([0, 0, 255])  # Red color for the mask overlay
alpha = 0.3  # Transparency level

# Create the overlay by blending the mask and the image
overlay = image.copy()
for c in range(3):  # Apply mask to each channel (R, G, B)
    overlay[..., c] = image[..., c] * (1-mask) + alpha * overlay_color[c] * mask + (1 - alpha) * image[..., c] * mask
overlay = overlay.astype(np.uint8)
overlay = Image.fromarray(overlay)
overlay.save("20220823_093810_overlay.jpg")
# # Plot the image, mask, and overlay
# plt.figure(figsize=(12, 6))

# # Original image
# plt.subplot(1, 3, 1)
# plt.imshow(image)
# plt.title("Original Image")
# plt.axis("off")

# # Mask
# plt.subplot(1, 3, 2)
# plt.imshow(mask, cmap="gray")
# plt.title("Mask")
# plt.axis("off")

# # Image with overlay
# plt.subplot(1, 3, 3)
# plt.imshow(overlay.astype(np.uint8))
# plt.title("Image with Mask Overlay")
# plt.axis("off")

# plt.tight_layout()
