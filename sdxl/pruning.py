import os
import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn.functional import cosine_similarity

def load_dino_model():
    """Load the pretrained DINOv2 model."""
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_lc')  # Load DINOv2 model
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
    model.eval()  # Set to evaluation mode
    return model


def preprocess_image(image_path):
    """Preprocess image for DINOv2 encoding."""
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)  # No batch dimension added yet


def preprocess_mask(mask_path):
    """Preprocess mask: Resize and convert to binary mask."""
    mask = Image.open(mask_path).convert("L")  # Convert to grayscale
    mask = mask.resize((224, 224))  # Resize to match image
    mask = np.array(mask) > 0  # Convert to binary mask (True/False)
    return torch.tensor(mask, dtype=torch.float32)  # Convert to tensor


def extract_features(model, image_tensor, mask_tensor):
    """Extract DINOv2 features from only the masked region."""
    # breakpoint()
    image_tensor = image_tensor * mask_tensor
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image_tensor).squeeze()  # Extract features
    
    # Apply mask: Zero out features outside the mask
    mask_tensor = mask_tensor.flatten()  # Flatten mask
    return features


def compute_cosine_similarity(anchor_features, image_features):
    """Compute cosine similarity between two masked feature vectors."""
    return cosine_similarity(anchor_features.unsqueeze(0), image_features.unsqueeze(0)).item()


def analyze_images(images_folder, masks_folder, anchor_image):
    """Compare images using DINOv2 and visualize cosine similarity."""
    model = load_dino_model()
    
    # Extract anchor image features
    anchor_image_path = os.path.join(images_folder, anchor_image)
    anchor_tensor = preprocess_image(anchor_image_path)
    anchor_mask_path = os.path.join(masks_folder, anchor_image.replace('jpg', 'png'))
    anchor_mask = preprocess_mask(anchor_mask_path)
    anchor_features = extract_features(model, anchor_tensor, anchor_mask)

    similarities = {}

    for image_name in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_name)
        mask_path = os.path.join(masks_folder, image_name.replace('jpg', 'png'))  # Mask should have the same name

        if os.path.exists(mask_path):
            image_tensor = preprocess_image(image_path)
            mask_tensor = preprocess_mask(mask_path)
            image_features = extract_features(model, image_tensor, mask_tensor)
            similarity = compute_cosine_similarity(anchor_features, image_features)

            similarities[image_name] = similarity

    # Sort by similarity
    sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))

    # Plot similarity chart
    plt.figure(figsize=(12, len(sorted_similarities) * 0.5))
    plt.barh(list(sorted_similarities.keys()), list(sorted_similarities.values()), color='blue')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Image")
    plt.title("DINOv2 Cosine Similarity (Masked Region) with Anchor Image")
    plt.gca().invert_yaxis()  # Invert to show highest similarity at the top
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    
    # Auto-adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig('chart2.png')
    

if __name__ == "__main__":
    analyze_images("/root/workspace/HiFi/sdxl/output/20250218_092905/images",
                   "/root/workspace/data/spinnerf-dataset/10/images_4/label",
                   "20220823_095135.jpg")