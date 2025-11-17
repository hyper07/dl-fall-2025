#!/usr/bin/env python3
"""
Generate augmented images and their feature vectors for training and similarity search.

This script creates augmented versions of training images with rotations and mirrors,
saves them to disk, and stores their feature vectors in the database.
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Any
import logging
import shutil

# Add core to path
sys.path.append(str(Path(__file__).parent))

from core.model_utils import CNNTrainer
from core.database import get_vector_store

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def apply_augmentations(image: Image.Image) -> List[Tuple[Image.Image, str]]:
    """Apply rotations and mirrors to an image.

    Returns:
        List of (augmented_image, augmentation_name) tuples
    """
    augmentations = []

    # Original
    augmentations.append((image, "original"))

    # Rotations
    augmentations.append((image.rotate(90), "rot90"))    # 90 degrees
    augmentations.append((image.rotate(180), "rot180"))  # 180 degrees
    augmentations.append((image.rotate(270), "rot270"))  # 270 degrees

    # Mirrors (flips)
    augmentations.append((image.transpose(Image.FLIP_LEFT_RIGHT), "mirror_h"))  # Horizontal mirror
    augmentations.append((image.transpose(Image.FLIP_TOP_BOTTOM), "mirror_v"))  # Vertical mirror

    # Rotations + mirrors
    augmentations.append((image.rotate(90).transpose(Image.FLIP_LEFT_RIGHT), "rot90_mirror_h"))  # 90° + horizontal mirror
    augmentations.append((image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT), "rot180_mirror_h")) # 180° + horizontal mirror
    augmentations.append((image.rotate(270).transpose(Image.FLIP_LEFT_RIGHT), "rot270_mirror_h")) # 270° + horizontal mirror

    return augmentations


def save_augmented_images(original_path: str, class_name: str, output_base_dir: str) -> List[str]:
    """Generate and save augmented versions of an image.

    Returns:
        List of paths to saved augmented images
    """
    # Open original image
    image = Image.open(original_path)
    original_filename = Path(original_path).stem
    original_ext = Path(original_path).suffix

    # Create output directory for this class
    class_output_dir = Path(output_base_dir) / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    # Apply augmentations
    augmentations = apply_augmentations(image)

    for augmented_image, aug_name in augmentations:
        # Create new filename
        if aug_name == "original":
            new_filename = f"{original_filename}{original_ext}"
        else:
            new_filename = f"{original_filename}_{aug_name}{original_ext}"

        output_path = class_output_dir / new_filename

        # Save augmented image
        augmented_image.save(output_path)
        saved_paths.append(str(output_path))

    return saved_paths


def extract_and_store_features(trainer: CNNTrainer, image_paths: List[str],
                              class_name: str, vector_store, table_name: str) -> int:
    """Extract features from images and store in database.

    Returns:
        Number of vectors stored
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    
    vectors_data = []

    for image_path in image_paths:
        try:
            # Load and preprocess image
            img = load_img(image_path, target_size=(trainer.img_height, trainer.img_width))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Extract features
            features = trainer.extract_features_from_array(img_array)

            # Normalize the feature vector
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

            # Create content identifier
            path_obj = Path(image_path)
            filename = path_obj.name

            # Determine augmentation type from filename
            if "_rot90" in filename:
                aug_type = "rot90"
            elif "_rot180" in filename:
                aug_type = "rot180"
            elif "_rot270" in filename:
                aug_type = "rot270"
            elif "_mirror_h" in filename:
                aug_type = "mirror_h"
            elif "_mirror_v" in filename:
                aug_type = "mirror_v"
            elif "_rot90_mirror_h" in filename:
                aug_type = "rot90_mirror_h"
            elif "_rot180_mirror_h" in filename:
                aug_type = "rot180_mirror_h"
            elif "_rot270_mirror_h" in filename:
                aug_type = "rot270_mirror_h"
            else:
                aug_type = "original"

            # Extract original filename (remove augmentation suffix)
            if aug_type != "original":
                # Remove the augmentation suffix to get original filename
                original_filename = filename.replace(f"_{aug_type}", "")
            else:
                original_filename = filename

            content = f"ResNet50_{class_name}_{aug_type}_{original_filename}"
            model_name = 'resnet50'
            label = class_name
            augmentation = aug_type

            vectors_data.append((content, model_name, label, augmentation, original_filename, features))

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue

    # Store vectors in batches
    if vectors_data:
        batch_size = 100
        stored_count = 0

        for i in range(0, len(vectors_data), batch_size):
            batch = vectors_data[i:i+batch_size]
            vector_store.insert_vectors(table_name, batch)
            stored_count += len(batch)
            print(f"Stored batch of {len(batch)} vectors for {class_name}")

        return stored_count

    return 0


def main():
    """Main function to generate augmented images and features."""

    # Set database config for host
    os.environ['DB_HOST'] = 'localhost'
    os.environ['DB_PORT'] = '45432'
    os.environ['DB_USER'] = 'admin'
    os.environ['DB_PASSWORD'] = 'PassW0rd'
    os.environ['DB_NAME'] = 'db'

    print("Generating augmented images and feature vectors...")
    print("Each original image will be augmented into 8 versions:")
    print("- Original, 90°, 180°, 270° rotations")
    print("- Horizontal mirror, vertical mirror")
    print("- Rotations + horizontal mirror")

    # Directories
    input_dir = 'static/train_dataset'
    output_dir = 'static/train_dataset_augmented'

    # Clean and create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Load the trained model
    trainer = CNNTrainer()
    trainer.load_model('models/resnet50/wound_classifier.pkl')

    # Initialize vector store
    vector_store = get_vector_store()
    table_name = 'images_features'

    # Get all image files from dataset
    dataset_dir = Path(input_dir)
    total_original_images = 0
    total_augmented_images = 0
    total_vectors_stored = 0

    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            print(f"\nProcessing class: {class_name}")

            # Get all images for this class
            class_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                class_images.extend(list(class_dir.glob(ext)))

            if not class_images:
                continue

            print(f"Found {len(class_images)} original images")

            # Process each image in the class
            class_augmented_paths = []
            for img_path in class_images:
                # Generate and save augmented versions
                augmented_paths = save_augmented_images(str(img_path), class_name, output_dir)
                class_augmented_paths.extend(augmented_paths)

            print(f"Generated {len(class_augmented_paths)} augmented images for {class_name}")

            # Extract and store features for all augmented images
            vectors_stored = extract_and_store_features(
                trainer, class_augmented_paths, class_name, vector_store, table_name
            )

            total_original_images += len(class_images)
            total_augmented_images += len(class_augmented_paths)
            total_vectors_stored += vectors_stored

    # Get final database count
    final_db_count = vector_store.get_vector_count(table_name)

    print("\n🎉 Augmentation complete!")
    print("📊 Summary:")
    print(f"   Original images: {total_original_images}")
    print(f"   Augmented images created: {total_augmented_images}")
    print(f"   Augmentation factor: {total_augmented_images // max(total_original_images, 1)}x")
    print(f"   Vectors stored in database: {total_vectors_stored}")
    print(f"   Total vectors in database: {final_db_count}")
    print(f"   Augmented images saved to: {output_dir}")

    # Save summary
    summary = {
        'original_images': total_original_images,
        'augmented_images': total_augmented_images,
        'augmentation_factor': total_augmented_images // max(total_original_images, 1),
        'vectors_stored': total_vectors_stored,
        'total_vectors_in_db': final_db_count,
        'classes_processed': [d.name for d in dataset_dir.iterdir() if d.is_dir()],
        'table_name': table_name,
        'vector_dimension': 1536,
        'augmentation_types': [
            'original', 'rot90', 'rot180', 'rot270',
            'mirror_h', 'mirror_v',
            'rot90_mirror_h', 'rot180_mirror_h', 'rot270_mirror_h'
        ],
        'input_directory': input_dir,
        'output_directory': output_dir,
        'generation_date': '2025-11-16',
        'note': 'Generated augmented images and vectors for improved training and similarity search'
    }

    with open('augmentation_complete_summary.json', 'w') as f:
        import json
        json.dump(summary, f, indent=2)

    print("📄 Summary saved to augmentation_complete_summary.json")

    print("\n💡 Next steps:")
    print("1. Use the augmented images in training for better model generalization")
    print("2. The similarity search now has much more diverse representations")
    print("3. Consider training a new model with the augmented dataset")


if __name__ == "__main__":
    main()