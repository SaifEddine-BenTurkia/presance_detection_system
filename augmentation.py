import os
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from tqdm import tqdm



def augment_image(image_path, num_augmentations=2):
    """
    Create augmented versions of a single image and save them to the same folder
    
    Args:
        image_path: Path to the original image
        num_augmentations: Number of augmented versions to create
    
    Returns:
        List of paths to the newly created augmented images
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return []
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    augmented_paths = []
    folder_path = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    # Define augmentation functions - simple transformations that preserve identity
    augmentations = [
        # Rotation (small angles)
        ("rotate_5", lambda img: rotate_image(img, angle=5)),
        ("rotate_neg5", lambda img: rotate_image(img, angle=-5)),
        
        # Horizontal flip
        ("flip_h", lambda img: cv2.flip(img, 1)),
        
        # Brightness changes
        ("bright_down", lambda img: adjust_brightness(img, factor=0.8)),
        ("bright_up", lambda img: adjust_brightness(img, factor=1.2)),
        
        # Contrast changes
        ("contrast_down", lambda img: adjust_contrast(img, factor=0.9)),
        ("contrast_up", lambda img: adjust_contrast(img, factor=1.1)),
        
        # Small crops
        ("crop_95", lambda img: crop_slight(img, ratio=0.95)),
        
        # Slight blur
        ("blur", lambda img: cv2.GaussianBlur(img, (3, 3), 0))
    ]
    
    # Randomly select augmentations
    selected_augmentations = random.sample(augmentations, min(num_augmentations, len(augmentations)))
    
    for aug_name, aug_func in selected_augmentations:
        # Apply augmentation
        aug_img = aug_func(img.copy())
        
        # Create output filename
        out_path = os.path.join(folder_path, f"{name}_{aug_name}{ext}")
        
        # Convert back to BGR for OpenCV
        save_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, save_img)
        augmented_paths.append(out_path)
        # print(f"Created augmented image: {out_path}")
    
    return augmented_paths

def rotate_image(image, angle):
    """Rotate image by angle degrees"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def adjust_brightness(image, factor):
    """Adjust brightness using PIL"""
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(factor)
    return np.array(pil_img)

def adjust_contrast(image, factor):
    """Adjust contrast using PIL"""
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(factor)
    return np.array(pil_img)

def crop_slight(image, ratio=0.95):
    """Crop the image slightly but keep most of the face"""
    h, w = image.shape[:2]
    crop_h, crop_w = int(h * ratio), int(w * ratio)
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    cropped = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
    return cv2.resize(cropped, (w, h))

def augment_single_image_people(dataset_dir, single_image_people_list, num_augmentations=2):
    """
    Augment images for people with only one image in the dataset
    
    Args:
        dataset_dir: Root directory of the dataset
        single_image_people_list: List of people names who have only one image
        num_augmentations: Number of augmented versions to create per image
    
    Returns:
        dict: Statistics about the augmentation process
    """
    stats = {
        "total_single_image_people": len(single_image_people_list),
        "processed": 0,
        "skipped": 0,
        "augmented_images_created": 0,
        "errors": 0
    }
    
    # Process each person in the list
    for person in tqdm(single_image_people_list, desc="Augmenting"):
        person_dir = os.path.join(dataset_dir, person)
        
        # Check if directory exists
        if not os.path.isdir(person_dir):
            print(f"Warning: Directory not found for person: {person}")
            stats["skipped"] += 1
            continue
        
        # Get all image files in the person's directory
        image_files = [f for f in os.listdir(person_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Verify this person indeed has only one image
        if len(image_files) != 1:
            # print(f"Warning: {person} has {len(image_files)} images, not just one. Skipping.")
            stats["skipped"] += 1
            continue
        
        # Get the single image path
        image_path = os.path.join(person_dir, image_files[0])
        
        try:
            # Augment the image and save to the same directory
            augmented_paths = augment_image(image_path, num_augmentations)
            stats["augmented_images_created"] += len(augmented_paths)
            stats["processed"] += 1
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            stats["errors"] += 1
    
    # Print summary
    print("\nAugmentation Summary:")
    print(f"Total single-image people: {stats['total_single_image_people']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Total augmented images created: {stats['augmented_images_created']}")
    print(f"Errors encountered: {stats['errors']}")
    
    return stats


# Example usage:
if __name__ == "__main__":
    # Replace with your actual dataset directory
    DATASET_DIR = "lfw-deepfunneled\lfw-deepfunneled"
    
    df = pd.read_csv('db_infos\lfw_allnames.csv')
    single_image_people = df[df['images'] == 1]['name'].tolist()
    
    # If your list is in a file, you can load it like this:
    # with open("single_image_people.txt", "r") as f:
    #     single_image_people = [line.strip() for line in f if line.strip()]
    
    # Run the augmentation
    augment_single_image_people(DATASET_DIR, single_image_people, num_augmentations=2)