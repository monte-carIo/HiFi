import os
import argparse

def get_image_files(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    return sorted([f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions])

def write_selected_files(folder_path, percentage, output_file):
    image_files = get_image_files(folder_path)
    num_files = max(1, int(len(image_files) * (percentage / 100)))  # Ensure at least 1 file is selected
    selected_files = image_files[:num_files]
    
    with open(output_file, 'w') as f:
        for file in selected_files:
            f.write(file + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and save a subset to a text file.")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing images.")
    parser.add_argument("--percentage", type=float, default=40, help="Percentage of files to include in the output.")
    args = parser.parse_args()
    output_file = args.folder_path + '/../sparse/0/test.txt'
    write_selected_files(args.folder_path, args.percentage, output_file)
    print(f"Selected {args.percentage}% of images. Output saved to {output_file}")
