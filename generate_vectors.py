#!/usr/bin/env python3
"""
Generate 1536 feature vectors from CNN model with optional augmentations and store in database.
Supports ResNet50, VGG16, and EfficientNet architectures.

Usage:
    python generate_vectors.py                    # Use ResNet50 (default)
    python generate_vectors.py --architecture vgg16
    python generate_vectors.py --architecture efficientnet
    python generate_vectors.py --skip-augmentation  # Skip augmentations, only original images
    python generate_vectors.py --replace-architecture --architecture vgg16  # Replace VGG16 vectors
    python generate_vectors.py --append  # Append to existing table
"""

import os
import sys
import math
import random
import numpy as np
import logging
import argparse
from pathlib import Path
from PIL import Image
import json
from typing import List, Tuple, Dict, Any

# Exact augmentation types should mirror EnhancedDataGenerator in core.model_utils
EXACT_AUGMENTATIONS = [
    'original',
    'rotate_90',
    'rotate_180',
    'rotate_270',
    'rotate_90_flip',
    'rotate_180_flip',
    'rotate_270_flip',
    'flip_vertical',
]

# Add core to path
sys.path.append(str(Path(__file__).parent / 'core'))

from core.model_utils import CNNTrainer
from core.database import get_vector_store

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vector_generation.log')
    ]
)
logger = logging.getLogger(__name__)


def get_image_paths_by_class(dataset_dir: str) -> Dict[str, List[str]]:
    """Get all image paths organized by class."""
    dataset_path = Path(dataset_dir)
    class_images = {}

    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_paths.extend(class_dir.glob(ext))

            class_images[class_name] = [str(p) for p in sorted(image_paths)]
            logger.info(f"Found {len(image_paths)} images in class '{class_name}'")

    return class_images


def augment_image(image_path: str, augmentation_type: str) -> np.ndarray:
    """Apply augmentation to image and return as numpy array."""
    img = Image.open(image_path)

    if augmentation_type == 'original':
        pass
    elif augmentation_type == 'rotate_90':
        img = img.transpose(Image.ROTATE_90)
    elif augmentation_type == 'rotate_180':
        img = img.transpose(Image.ROTATE_180)
    elif augmentation_type == 'rotate_270':
        img = img.transpose(Image.ROTATE_270)
    elif augmentation_type == 'rotate_90_flip':
        img = img.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT)
    elif augmentation_type == 'rotate_180_flip':
        img = img.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT)
    elif augmentation_type == 'rotate_270_flip':
        img = img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
    elif augmentation_type == 'flip_vertical':
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize to 224x224
    img = img.resize((224, 224))

    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0

    return img_array


def extract_features_from_array(trainer: CNNTrainer, img_array: np.ndarray) -> np.ndarray:
    """Extract features from a normalized image array using trainer's direct method."""
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)
    return trainer.extract_features_from_array(img_array)


def select_image_subset(image_paths: List[str], target_vectors: int,
                        augmentations_count: int) -> List[str]:
    """Optionally limit number of original images per class to balance vector counts."""
    if target_vectors is None or target_vectors <= 0:
        return image_paths

    augmentations_count = max(1, augmentations_count)
    max_images = max(1, math.ceil(target_vectors / augmentations_count))

    if len(image_paths) <= max_images:
        return image_paths

    rng = random.Random(42)
    selected = rng.sample(image_paths, max_images)
    logger.info(f"Limiting class to {max_images} original images to target ~{target_vectors} vectors")
    return sorted(selected)


def generate_vectors_for_class(
    trainer: CNNTrainer,
    class_name: str,
    image_paths: List[str],
    architecture: str,
    skip_augmentation: bool = False,
    target_vectors_per_class: int = 0
) -> Tuple[List[Tuple[str, str, str, str, str, np.ndarray]], int]:
    """Generate vectors for all images in class with optional augmentations."""
    vectors_data = []
    if skip_augmentation:
        augmentations = ['original']
    else:
        augmentations = EXACT_AUGMENTATIONS

    # Apply optional per-class sampling to keep the DB balanced
    image_paths = select_image_subset(image_paths, target_vectors_per_class, len(augmentations))
    processed_images = len(image_paths)

    # Generate augmentations for ALL images
    for image_path in image_paths:
        for aug_type in augmentations:
            try:
                if aug_type == 'original':
                    features = trainer.extract_features(image_path)
                else:
                    img_array = augment_image(image_path, aug_type)
                    features = extract_features_from_array(trainer, img_array)

                # Normalize the feature vector for proper cosine similarity
                norm = np.linalg.norm(features)
                if norm > 0:
                    features = features / norm
                else:
                    logger.warning(f"Zero norm feature vector for {image_path}, using zeros")
                    features = np.zeros_like(features)

                content = f"{architecture}_{class_name}_{aug_type}_{Path(image_path).name}"
                model_name = architecture
                label = class_name
                augmentation = aug_type
                original_image = str(Path(image_path).name)
                vectors_data.append((content, model_name, label, augmentation, original_image, features))
            except Exception as e:
                logger.error(f"Error processing {image_path} ({aug_type}): {e}")
                continue

    logger.info(f"Generated {len(vectors_data)} vectors for class '{class_name}' ({processed_images} images × {len(augmentations)} augmentations)")
    return vectors_data, processed_images


def main():
    """Main function to generate and store vectors."""
    parser = argparse.ArgumentParser(description='Generate feature vectors from CNN model')
    parser.add_argument('--architecture', choices=['resnet50', 'vgg16', 'efficientnet'], 
                       default='resnet50', help='CNN architecture to use (default: resnet50)')
    parser.add_argument('--skip-augmentation', action='store_true', 
                       help='Skip data augmentation and only process original images')
    parser.add_argument('--model-path', help='Path to the trained model file (overrides architecture-based path)')
    parser.add_argument('--dataset-dir', default='./files/train_dataset',
                       help='Path to the dataset directory')
    parser.add_argument('--table-name', default='images_features',
                       help='Name of the database table to store vectors')
    parser.add_argument('--target-vectors-per-class', type=int, default=0,
                       help='Approximate number of vectors to keep per class (0 = use all available images)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Number of vectors to insert per database batch')
    parser.add_argument('--replace-architecture', action='store_true',
                       help='Delete existing vectors for the specified architecture before inserting new ones')
    parser.add_argument('--append', action='store_true',
                       help='Append to existing table instead of dropping and recreating it')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.replace_architecture and args.append:
        parser.error("--replace-architecture and --append cannot be used together")
    
    # Configuration
    architecture = args.architecture
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = f'./models/{architecture}/wound_classifier_best.keras'
    
    dataset_dir = args.dataset_dir
    table_name = args.table_name
    skip_augmentation = args.skip_augmentation
    target_vectors_per_class = args.target_vectors_per_class
    batch_size = max(1, args.batch_size)

    logger.info(f"Starting vector generation for {architecture} model")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"Skip augmentation: {skip_augmentation}")
    logger.info(f"Target vectors per class: {target_vectors_per_class}")

    try:
        # Load trained model
        logger.info(f"Loading trained {architecture} model...")
        trainer = CNNTrainer(architecture=architecture, model_name='wound_classifier')
        trainer.load_model(model_path)
        # Mark as trained since we're loading a pre-trained model
        trainer.is_trained = True
        logger.info("Model loaded and marked as trained")

        # Get image paths by class
        logger.info("Scanning dataset...")
        class_images = get_image_paths_by_class(dataset_dir)

        # Initialize vector store
        logger.info(f"Initializing vector store with table '{table_name}'...")
        vector_store = get_vector_store()
        
        # Handle table operations based on mode
        if not args.append and not args.replace_architecture:
            # Default behavior: drop table
            try:
                with vector_store.db as conn:
                    cur = conn.cursor()
                    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                    conn.commit()
                    logger.info(f"Dropped existing table '{table_name}' before regeneration")
            except Exception as e:
                logger.warning(f"Could not drop table {table_name}: {e}")
        elif args.replace_architecture:
            # Delete existing vectors for this architecture
            try:
                with vector_store.db as conn:
                    cur = conn.cursor()
                    cur.execute(f"DELETE FROM {table_name} WHERE model_name = %s", (architecture,))
                    deleted_count = cur.rowcount
                    conn.commit()
                    logger.info(f"Deleted {deleted_count} existing vectors for architecture '{architecture}'")
            except Exception as e:
                logger.warning(f"Could not delete vectors for architecture {architecture}: {e}")
        else:
            # Append mode: preserve all existing data
            logger.info("Append mode enabled - existing vectors will be preserved")

        vector_store.create_vector_table(table_name, vector_dim=1536)

        # Generate vectors for each class and stream inserts
        total_vectors_written = 0
        processed_images_by_class = {}
        for class_name, image_paths in class_images.items():
            logger.info(f"Processing class '{class_name}' with {len(image_paths)} available images...")
            class_vectors, processed_image_count = generate_vectors_for_class(
                trainer,
                class_name,
                image_paths,
                architecture,
                skip_augmentation=skip_augmentation,
                target_vectors_per_class=target_vectors_per_class
            )
            processed_images_by_class[class_name] = processed_image_count

            if not class_vectors:
                logger.warning(f"No vectors generated for class '{class_name}'")
                continue

            for i in range(0, len(class_vectors), batch_size):
                batch = class_vectors[i:i+batch_size]
                vector_store.insert_vectors(table_name, batch)
                total_vectors_written += len(batch)
                logger.info(
                    f"Stored batch {i // batch_size + 1} for class '{class_name}' "
                    f"({len(batch)} vectors, total written: {total_vectors_written})"
                )

        # Verify total count
        final_count = vector_store.get_vector_count(table_name)
        logger.info(f"Vector generation completed! Total vectors in database: {final_count}")

        # Save summary
        total_original_images = sum(processed_images_by_class.values())
        if skip_augmentation:
            augmentations_per_image = 1
            augmentations_list = ['original']
        else:
            augmentations_per_image = len(EXACT_AUGMENTATIONS)
            augmentations_list = EXACT_AUGMENTATIONS
        
        summary = {
            'model_type': architecture,
            'total_vectors': final_count,
            'total_original_images': total_original_images,
            'augmentations_per_image': augmentations_per_image,
            'expected_total_vectors': total_original_images * augmentations_per_image,
            'classes_processed': list(class_images.keys()),
            'table_name': table_name,
            'augmentations': augmentations_list,
            'skip_augmentation': skip_augmentation,
            'target_vectors_per_class': target_vectors_per_class,
            'batch_size': batch_size,
            'append_mode': args.append,
            'replace_architecture_mode': args.replace_architecture,
            'vectors_inserted_this_run': total_vectors_written
        }

        with open('vector_generation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("Summary saved to vector_generation_summary.json")

    except Exception as e:
        logger.error(f"Vector generation failed: {e}")
        raise


if __name__ == "__main__":
    main()