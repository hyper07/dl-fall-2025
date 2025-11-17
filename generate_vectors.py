#!/usr/bin/env python3
"""
Generate 1024 feature vectors from ResNet50 model including augmentations and store in database.
Note: Using 1024-dimensional reduced features from GAP layer.
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
from PIL import Image
import json
from typing import List, Tuple, Dict, Any

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
        img = img.rotate(90)
    elif augmentation_type == 'rotate_180':
        img = img.rotate(180)
    elif augmentation_type == 'rotate_270':
        img = img.rotate(270)
    elif augmentation_type == 'flip_horizontal':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
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
    """Extract features from preprocessed image array using the trainer's extract_features method."""
    # Save the array as a temporary image file and use extract_features
    import tempfile
    import os
    from PIL import Image

    # Convert array back to PIL Image and save temporarily
    img_array = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        img.save(tmp_file.name)
        tmp_path = tmp_file.name

    try:
        # Use the trainer's extract_features method
        features = trainer.extract_features(tmp_path)
        return features
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


def generate_vectors_for_class(trainer: CNNTrainer, class_name: str, image_paths: List[str]) -> List[Tuple[str, str, str, str, str, np.ndarray]]:
    """Generate vectors for all images in class with all augmentations."""
    vectors_data = []
    augmentations = ['original', 'rotate_90', 'rotate_180', 'rotate_270', 'flip_horizontal', 'flip_vertical']

    # Generate ALL augmentations for ALL images
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

                content = f"ResNet50_{class_name}_{aug_type}_{Path(image_path).name}"
                model_name = 'resnet50'
                label = class_name
                augmentation = aug_type
                original_image = str(Path(image_path).name)
                vectors_data.append((content, model_name, label, augmentation, original_image, features))
            except Exception as e:
                logger.error(f"Error processing {image_path} ({aug_type}): {e}")
                continue

    logger.info(f"Generated {len(vectors_data)} vectors for class '{class_name}' ({len(image_paths)} images × {len(augmentations)} augmentations)")
    return vectors_data


def main():
    """Main function to generate and store vectors."""
    # Configuration
    model_path = './wound_classifier_best.keras'
    dataset_dir = './files/train_dataset'
    table_name = 'images_features'
    target_vectors_per_class = 200
    total_target = 1024

    logger.info("Starting vector generation for ResNet50 model")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"Target vectors per class: {target_vectors_per_class}")
    logger.info(f"Total target vectors: {total_target}")

    try:
        # Load trained model
        logger.info("Loading trained ResNet50 model...")
        trainer = CNNTrainer(architecture='resnet50', model_name='wound_classifier')
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
        
        # Drop table if it exists (to recreate with new schema)
        try:
            with vector_store.db as conn:
                cur = conn.cursor()
                cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.commit()
                logger.info(f"Dropped existing table '{table_name}'")
        except Exception as e:
            logger.warning(f"Could not drop table {table_name}: {e}")
        
        vector_store.create_vector_table(table_name, vector_dim=1024)

        # Generate vectors for each class - ALL images with ALL augmentations
        all_vectors_data = []
        for class_name, image_paths in class_images.items():
            logger.info(f"Processing class '{class_name}' with {len(image_paths)} images...")
            class_vectors = generate_vectors_for_class(trainer, class_name, image_paths)
            all_vectors_data.extend(class_vectors)

        # Store all vectors in database
        logger.info(f"Storing {len(all_vectors_data)} vectors in database...")
        batch_size = 100
        for i in range(0, len(all_vectors_data), batch_size):
            batch = all_vectors_data[i:i+batch_size]
            vector_store.insert_vectors(table_name, batch)
            logger.info(f"Stored batch {i//batch_size + 1}/{(len(all_vectors_data) + batch_size - 1)//batch_size}")

        # Verify total count
        final_count = vector_store.get_vector_count(table_name)
        logger.info(f"Vector generation completed! Total vectors in database: {final_count}")

        # Save summary
        total_original_images = sum(len(paths) for paths in class_images.values())
        summary = {
            'model_type': 'resnet50',
            'total_vectors': final_count,
            'total_original_images': total_original_images,
            'augmentations_per_image': 6,
            'expected_total_vectors': total_original_images * 6,
            'classes_processed': list(class_images.keys()),
            'table_name': table_name,
            'augmentations': ['original', 'rotate_90', 'rotate_180', 'rotate_270', 'flip_horizontal', 'flip_vertical']
        }

        with open('vector_generation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("Summary saved to vector_generation_summary.json")

    except Exception as e:
        logger.error(f"Vector generation failed: {e}")
        raise


if __name__ == "__main__":
    main()