import cv2
import numpy as np
import os

def calculate_cdf(hist):
    """Calculate the cumulative distribution function from the histogram."""
    cdf = hist.cumsum()
    return cdf / cdf[-1]  # Normalize to [0, 1]

def histogram_matching(source_image, reference_image, mask=None):
    """
    Perform histogram matching of source image to the reference image, 
    considering only the non-masked region for histogram calculation.

    Parameters:
    - source_image (ndarray): Source image to be matched.
    - reference_image (ndarray): Reference image to match the histogram.
    - mask (ndarray): Binary mask (same size as images), 0 for excluded regions and 255 for included regions.

    Returns:
    - matched_image (ndarray): Histogram-matched image.
    """
    # Convert images to float32
    source_image = source_image.astype(np.float32)
    reference_image = reference_image.astype(np.float32)

    # If no mask is provided, consider the entire image
    if mask is None:
        mask = np.ones_like(source_image[:, :, 0], dtype=np.uint8) * 255

    # Calculate the histograms and CDFs for each channel
    hist_size = 256
    matched_image = source_image.copy()
    for i in range(3):  # Loop through B, G, R channels
        # Calculate histograms for the non-masked regions
        hist_source, _ = np.histogram(
            source_image[:, :, i][mask > 0], bins=hist_size, range=(0, 256)
        )
        hist_reference, _ = np.histogram(
            reference_image[:, :, i][mask > 0], bins=hist_size, range=(0, 256)
        )
        
        # Calculate CDFs
        cdf_source = calculate_cdf(hist_source)
        cdf_reference = calculate_cdf(hist_reference)

        # Create mapping from source to reference
        mapping = np.zeros(hist_size, dtype=np.uint8)
        for j in range(hist_size):
            mapping[j] = np.searchsorted(cdf_reference, cdf_source[j])

        # Apply the mapping to the entire image
        matched_image[:, :, i] = mapping[source_image[:, :, i].astype(np.uint8)]

    return matched_image.astype(np.uint8)

def process_folder(source_folder, reference_path, output_folder, mask_folder=None):
    """
    Perform histogram matching for all images in a folder against a reference image.

    Parameters:
    - source_folder (str): Path to the folder containing source images.
    - reference_path (str): Path to the reference image.
    - output_folder (str): Path to save the output images.
    - mask_folder (str): Path to the folder containing masks (optional).
    """
    # Load the reference image
    reference_image = cv2.imread(reference_path)
    if reference_image is None:
        print(f"Error: Unable to load reference image from {reference_path}")
        return

    # Convert the reference image to RGB
    reference_image = reference_image.astype(np.float32)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the source folder
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        if not os.path.isfile(source_path):
            continue

        # Load source image
        source_image = cv2.imread(source_path)
        if source_image.shape == (2268, 4032, 3):
            source_image = cv2.resize(source_image, 
                               (source_image.shape[1] // 4, source_image.shape[0] // 4))
        if source_image is None:
            print(f"Skipping {filename}: Unable to read image.")
            continue

        # Load mask if provided
        mask = None
        if mask_folder:
            mask_path = os.path.join(mask_folder, filename)
            if os.path.isfile(mask_path):
                mask = (1 - cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)) * 255

        # Perform histogram matching
        matched_image = histogram_matching(source_image, reference_image, mask)

        # Save the matched image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, matched_image)
        print(f"Processed and saved: {output_path}")

# Example usage
process_folder(
    source_folder='/home/meomeo/workspace/knguyen/sdxl/output/20250114_175454/images', 
    reference_path='/home/meomeo/workspace/knguyen/view_mapper/output/output_inpainted/7/20220819_111635.jpg', 
    output_folder='/home/meomeo/workspace/knguyen/sdxl/output/20250114_175454/images_balanced',
    mask_folder='/home/meomeo/workspace/data/spinnerf-dataset/7/images_4/label'
)