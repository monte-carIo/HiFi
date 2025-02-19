import os

def write_unique_images_to_txt(folder1, folder2, output_txt_path):
    """
    Writes image names that are in folder1 but not in folder2 to a text file.

    Parameters:
    - folder1 (str): Path to the first folder.
    - folder2 (str): Path to the second folder.
    - output_txt_path (str): Path to the output text file.

    Returns:
    - None
    """
    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Get all image names in both folders
    images_folder1 = {
        f for f in os.listdir(folder1)
        if os.path.isfile(os.path.join(folder1, f)) and os.path.splitext(f)[1].lower() in image_extensions
    }

    images_folder2 = {
        f for f in os.listdir(folder2)
        if os.path.isfile(os.path.join(folder2, f)) and os.path.splitext(f)[1].lower() in image_extensions
    }

    # Find images in folder1 but not in folder2
    unique_images = images_folder1 - images_folder2

    # Write unique image names to the text file
    with open(output_txt_path, "w") as txt_file:
        for image_name in unique_images:
            txt_file.write(image_name + "\n")

    print(f"Successfully written {len(unique_images)} unique image names to {output_txt_path}")

# Example usage
folder1 = "/root/workspace/data/spinnerf-dataset/10/images_4"  # Replace with the path to the first folder
folder2 = "/root/workspace/data/spinnerf-dataset/10/images_4/label"  # Replace with the path to the second folder
output_txt_path = "/root/workspace/data/spinnerf-dataset/10/sparse/0/test.txt"  # Replace with the desired output text file path



write_unique_images_to_txt(folder1, folder2, output_txt_path)
