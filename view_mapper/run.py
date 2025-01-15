import os
import numpy as np
import cv2
from os import makedirs
from get_c2w import save_scene_and_depth
import math

def project_points(image, K, R, T, depth_map):
    """
    Project 2D image onto a new view using intrinsic and extrinsic parameters.

    Parameters:
    image (ndarray): Input 2D image (H x W x C).
    K (ndarray): Intrinsic matrix (3 x 3).
    R (ndarray): Rotation matrix (3 x 3).
    T (ndarray): Translation vector (3 x 1).
    depth_map (ndarray): Depth map (H x W).

    Returns:
    projected_image (ndarray): Image rendered from the new view.
    """
    epsilon = 1e-6  # A small positive value
    depth_map = np.where(depth_map == 0, epsilon, depth_map) * 116.2376937866211

    
    height, width = image.shape[:2]
    
    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert pixel coordinates to normalized image coordinates in source view
    x_normalized = (u - K[0, 2]) / K[0, 0]
    y_normalized = (v - K[1, 2]) / K[1, 1]

    # Construct the 3D point cloud from 2D image and depth
    X_3D = np.stack((x_normalized * depth_map, y_normalized * depth_map, depth_map), axis=-1)  # (H, W, 3)

    # Reshape to (N, 3) for matrix operations
    X_3D_flat = X_3D.reshape(-1, 3)
    ()

    # Apply the extrinsic transformation
    X_3D_transformed = (R @ X_3D_flat.T + T.reshape(3, 1)).T
    ()

    # Project the points onto the image plane
    X_projected = K @ X_3D_transformed.T
    ()

    # Normalize by the third coordinate (depth)
    points_2D = X_projected[:2, :] / X_projected[2, :]
    ()

    # Reshape back to image dimensions
    points_2D = points_2D.T
    ()
    
    # Create an empty image for the projected view
    projected_image = np.zeros_like(image)

    # Map colors from the original image to the projected points
    for i in range(points_2D.shape[0]):
        x, y = int(points_2D[i, 0]), int(points_2D[i, 1])
        if 0 <= x < width and 0 <= y < height:
            projected_image[y, x] = image.flat[i * 3:(i * 3) + 3]  # Assign color

    return projected_image

def compute_R_ts_T_ts(c2w_s, c2w_t):
    # Extract rotation (3x3) and translation (3x1) from source and target c2w matrices
    R_s = c2w_s[:3, :3]
    T_s = c2w_s[:3, 3]
    
    R_t = c2w_t[:3, :3]
    T_t = c2w_t[:3, 3]

    # Compute relative rotation and translation
    R_ts = np.dot(R_t.T, R_s)
    T_ts = np.dot(R_t.T, (T_s - T_t))

    return R_ts, T_ts

def dilate_mask(mask, kernel_size):
    """Dilates the input mask with the specified kernel size and converts it to a binary mask."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Dilate the mask
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Convert to binary mask: set to 255 (white) or 0 (black)
    _, binary_mask = cv2.threshold(dilated_mask, 128, 255, cv2.THRESH_BINARY)
    
    return binary_mask

def process_images(input_folder, output_path, output_folder, anchor_image, anchor_path, depth_path):
    
    
    mask_folder = os.path.join(input_folder, "seg")
    image_folder = os.path.join(input_folder, "images_4")
    sparse_folder = os.path.join(input_folder, "sparse","1")
    
    
    
    K_anchor = np.load(os.path.join(output_path, "intri", anchor_image + ".npy"))
    c2w_anchor= np.load(os.path.join(output_path, "c2w", anchor_image + ".npy"))
    depth_anchor = cv2.imread(depth_path)
    image = cv2.imread(anchor_path)
    
    temp = os.listdir(image_folder)[0]
    temp_path = os.path.join(image_folder, temp)
    temp_image = cv2.imread(temp_path)
    temp_shape = temp_image.shape[:2]
    image = cv2.resize(image, (temp_shape[1], temp_shape[0]))
    depth_anchor = cv2.resize(depth_anchor, (temp_shape[1], temp_shape[0]))
    depth_anchor = cv2.cvtColor(depth_anchor, cv2.COLOR_BGR2GRAY)    
    
    # Loop over all images in the image folder
    for image_name in sorted(os.listdir(mask_folder)): 
        print(image_name)
        if image_name.endswith('.png'):  # Ensure it's an image file
            if image_name.split('.')[0] == anchor_image:
                continue
            image_path = os.path.join(image_folder, image_name)
            mask_path = os.path.join(mask_folder, image_name)  # Assuming masks are .png files


            # Load image, mask, depth, intrinsics, and c2w
            original_image = cv2.imread(image_path)
            c2w = np.load(os.path.join(output_path, "c2w", image_name.split('.')[0] + ".npy"))

            # Compute transformations for projection
            R_ts, T_ts = compute_R_ts_T_ts(c2w_anchor, c2w)
            projected_image = project_points(image, K_anchor, R_ts, T_ts, depth_anchor)
            
            # Check if mask exists and apply accordingly
            if mask_folder is not None:
            # if False:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) * 255
                mask = cv2.resize(mask, (temp_shape[1], temp_shape[0]))  # Resize if necessary
                mask = dilate_mask(mask, kernel_size=10)
                
                # Create inverse mask and apply blending
                mask_inv = cv2.bitwise_not(mask)
                masked_projected_image = cv2.bitwise_and(projected_image, projected_image, mask=mask)
                original_unmasked = cv2.bitwise_and(original_image, original_image, mask=mask_inv)
                final_image = cv2.add(original_unmasked, masked_projected_image)
            else:
                print(f"Mask not found for {image_name}, saving the whole mapped image.")
                final_image = projected_image

            # Save the final blended image
            output_path_1 = os.path.join(output_folder, image_name)
            os.makedirs(output_folder, exist_ok=True)
            cv2.imwrite(output_path_1, final_image)

process_images(
    input_folder='/workspace/data/spinnerf-dataset/4',
    output_path = '/workspace/knguyen/gaussian-splatting/output/4/train/ours_30000',
    output_folder='output/4',
    anchor_image='20220819_105840',
    anchor_path='/workspace/knguyen/sdxl/lama.png',
    depth_path='/workspace/knguyen/sdxl/controlnet_output.png',
)
