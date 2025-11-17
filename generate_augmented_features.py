#!/usr/bin/env python3
"""
Generate and store augmented image features for similarity search.

This script creates 8 augmented versions of each training image and stores
their feature vectors in the database for improved similarity search.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

# Add core to path
sys.path.append(str(Path(__file__).parent))

from core.model_utils import CNNTrainer
from core.database import get_vector_store

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def apply_exact_transformations(image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    """Apply exact 90, 180, 270 rotations and flips to an image.

    Returns:
        List of (transformed_image, transformation_name) tuples
    """
    transformations = []

    # Original
    transformations.append((image, "original"))

    # Rotations
    transformations.append((np.rot90(image, k=1, axes=(0, 1)), "rot90"))  # 90 degrees
    transformations.append((np.rot90(image, k=2, axes=(0, 1)), "rot180"))  # 180 degrees
    transformations.append((np.rot90(image, k=3, axes=(0, 1)), "rot270"))  # 270 degrees

    # Horizontal flips of rotations
    transformations.append((np.fliplr(np.rot90(image, k=1, axes=(0, 1))), "rot90_flip_h"))  # 90 + horizontal flip
    transformations.append((np.fliplr(np.rot90(image, k=2, axes=(0, 1))), "rot180_flip_h"))  # 180 + horizontal flip
    transformations.append((np.fliplr(np.rot90(image, k=3, axes=(0, 1))), "rot270_flip_h"))  # 270 + horizontal flip

    # Vertical flip of original
    transformations.append((np.flipud(image), "flip_v"))  # Vertical flip

    return transformations


def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load and preprocess an image for feature extraction."""
    try:
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
    except ImportError:
        raise ImportError("TensorFlow not available")

    # Load image
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]

    return img_array


def generate_augmented_features(trainer: CNNTrainer, image_path: str, class_name: str) -> List[Dict[str, Any]]:
    """Generate feature vectors for all 8 augmented versions of an image.

    Args:
        trainer: Trained CNNTrainer instance
        image_path: Path to the original image
        class_name: Class name of the image

    Returns:
        List of dictionaries containing feature data for each augmentation
    """
    # Load original image
    original_image = load_and_preprocess_image(image_path)

    # Apply all transformations
    transformations = apply_exact_transformations(original_image)

    augmented_data = []

    for transformed_image, transformation_name in transformations:
        try:
            # Extract features from the transformed image
            # We need to expand dims for the model input
            img_array = np.expand_dims(transformed_image, axis=0)

            # Get feature vector (this will be truncated to 1024 dimensions by extract_features)
            feature_vector = trainer.extract_features_from_array(img_array)

            # Normalize the feature vector
            norm = np.linalg.norm(feature_vector)
            if norm > 0:
                feature_vector = feature_vector / norm

            # Create content identifier
            original_filename = Path(image_path).name
            content = f"ResNet50_{class_name}_{transformation_name}_{original_filename}"

            # Prepare data for database insertion
            augmented_data.append({
                'content': content,
                'model_name': 'resnet50',
                'label': class_name,
                'augmentation': transformation_name,
                'original_image': str(original_filename),
                'features': feature_vector
            })

        except Exception as e:
            logger.error(f"Error processing {transformation_name} for {image_path}: {e}")
            continue

    return augmented_data


def main():
    """Main function to generate and store augmented features."""

    # Set database config for host
    os.environ['DB_HOST'] = 'localhost'
    os.environ['DB_PORT'] = '45432'
    os.environ['DB_USER'] = 'admin'
    os.environ['DB_PASSWORD'] = 'PassW0rd'
    os.environ['DB_NAME'] = 'db'

    print("Generating augmented features for similarity search...")
    print("Each original image will be augmented into 8 versions (including original)")

    # Load the trained model
    trainer = CNNTrainer()
    trainer.load_model('models/resnet50/wound_classifier.pkl')

    # Initialize vector store
    vector_store = get_vector_store()
    table_name = 'images_features'

    # Get all image files from dataset
    dataset_dir = Path('static/train_dataset')
    image_paths = []
    class_labels = []

    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in class_dir.glob(ext):
                    image_paths.append(str(img_path))
                    class_labels.append(class_name)

    print(f"Found {len(image_paths)} original images across {len(set(class_labels))} classes")
    print(f"This will generate {len(image_paths) * 8} total augmented images")

    # Generate and store augmented features
    total_augmented = 0
    batch_size = 50  # Process in smaller batches to avoid memory issues
    all_augmented_data = []

    for i, (img_path, class_name) in enumerate(zip(image_paths, class_labels)):
        try:
            # Generate all 8 augmented versions for this image
            augmented_data = generate_augmented_features(trainer, img_path, class_name)
            all_augmented_data.extend(augmented_data)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images ({len(all_augmented_data)} augmented versions)")

            # Store in batches to avoid memory issues
            if len(all_augmented_data) >= batch_size:
                # Convert to format expected by insert_vectors
                batch_data = []
                for data in all_augmented_data:
                    batch_data.append((
                        data['content'],
                        data['model_name'],
                        data['label'],
                        data['augmentation'],
                        data['original_image'],
                        data['features']
                    ))

                vector_store.insert_vectors(table_name, batch_data)
                total_augmented += len(batch_data)
                all_augmented_data = []  # Clear batch

                print(f"Stored batch of {len(batch_data)} augmented vectors. Total stored: {total_augmented}")

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue

    # Store any remaining augmented data
    if all_augmented_data:
        batch_data = []
        for data in all_augmented_data:
            batch_data.append((
                data['content'],
                data['model_name'],
                data['label'],
                data['augmentation'],
                data['original_image'],
                data['features']
            ))

        vector_store.insert_vectors(table_name, batch_data)
        total_augmented += len(batch_data)
        print(f"Stored final batch of {len(batch_data)} augmented vectors. Total stored: {total_augmented}")

    # Get final count
    final_count = vector_store.get_vector_count(table_name)
    print(f"\nAugmentation complete!")
    print(f"Original images: {len(image_paths)}")
    print(f"Augmented versions per image: 8")
    print(f"Total vectors in database: {final_count}")
    print(f"Expected total: {len(image_paths) * 8 + len(image_paths)} (including original vectors)")

    # Save summary
    summary = {
        'original_images': len(image_paths),
        'augmentations_per_image': 8,
        'total_augmented_vectors': total_augmented,
        'total_vectors_in_db': final_count,
        'classes_processed': list(set(class_labels)),
        'table_name': table_name,
        'vector_dimension': 1024,
        'augmentation_types': [
            'original', 'rot90', 'rot180', 'rot270',
            'rot90_flip_h', 'rot180_flip_h', 'rot270_flip_h', 'flip_v'
        ],
        'generation_date': '2025-11-16',
        'note': 'Added augmented images for improved similarity search diversity'
    }

    with open('augmented_features_summary.json', 'w') as f:
        import json
        json.dump(summary, f, indent=2)

    print("Summary saved to augmented_features_summary.json")


if __name__ == "__main__":
    main()