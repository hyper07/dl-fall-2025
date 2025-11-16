"""
Image similarity search using PostgreSQL with pgvector extension.
Supports 2000-dimensional feature vectors for wound image classification.
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging
from pathlib import Path
import json

# Conditional imports for PostgreSQL dependencies
try:
    from .database import get_vector_store, DatabaseConnection
    from .model_utils import CNNTrainer
    POSTGRESQL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PostgreSQL dependencies not available: {e}")
    POSTGRESQL_AVAILABLE = False
    # Create dummy classes/functions for when dependencies are missing
    class DatabaseConnection:
        pass
    def get_vector_store():
        raise ImportError("PostgreSQL dependencies not available")
    class CNNTrainer:
        pass

logger = logging.getLogger(__name__)


class ImageSimilaritySearch:
    """Image similarity search using feature vectors stored in PostgreSQL."""

    def __init__(self, table_name: str = "images_features"):
        if not POSTGRESQL_AVAILABLE:
            raise ImportError("PostgreSQL dependencies not available. This module requires psycopg2 and pgvector to be installed.")
        self.table_name = table_name
        self.vector_store = get_vector_store()
        self.vector_store.create_vector_table(table_name)
        self.trainer = None

    def load_trainer(self, model_path: str, config_path: str) -> None:
        """Load trained CNN model for feature extraction."""
        self.trainer = CNNTrainer()
        self.trainer.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")

    def extract_features_batch(self, image_paths: List[str]) -> np.ndarray:
        """Extract 2000-dimensional features from a batch of images."""
        if self.trainer is None:
            raise ValueError("Model not loaded. Call load_trainer() first.")

        features = []
        for image_path in image_paths:
            try:
                # Load and preprocess image
                img = self.trainer.load_and_preprocess_image(image_path)
                # Extract features (model outputs 2000-dim vector)
                feature_vector = self.trainer.model.predict(img, verbose=0)[0]
                features.append(feature_vector)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue

        return np.array(features)

    def store_image_features(self, image_paths: List[str], class_labels: List[str],
                           batch_size: int = 32) -> None:
        """Extract features from images and store in vector database."""
        if self.trainer is None:
            raise ValueError("Model not loaded. Call load_trainer() first.")

        total_images = len(image_paths)
        logger.info(f"Processing {total_images} images for feature extraction...")

        # Process in batches
        for i in range(0, total_images, batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_labels = class_labels[i:i+batch_size]

            # Extract features for batch
            features = self.extract_features_batch(batch_paths)

            # Prepare data for vector store
            vectors_data = []
            for j, (path, label, feature) in enumerate(zip(batch_paths, batch_labels, features)):
                content = f"Image: {Path(path).name} | Class: {label}"
                model_name = 'resnet50'  # Default model
                augmentation = 'original'  # Default augmentation
                original_image = Path(path).name
                vectors_data.append((content, model_name, label, augmentation, original_image, feature))

            # Store in database
            self.vector_store.insert_vectors(self.table_name, vectors_data)

            logger.info(f"Processed batch {i//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")

        logger.info(f"Successfully stored features for {len(image_paths)} images")

    def find_similar_images(self, query_image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find top-k most similar images to the query image.

        Returns:
            List of dicts with keys: 'image_path', 'class', 'similarity_score', 'filename'
        """
        if self.trainer is None:
            raise ValueError("Model not loaded. Call load_trainer() first.")

        # Extract features from query image
        query_features = self.extract_features_batch([query_image_path])[0]

        # Search for similar vectors
        results = self.vector_store.search_similar(self.table_name, query_features, limit=top_k)

        # Format results
        similar_images = []
        for result in results:
            db_id, content, similarity_score, model_name, label, augmentation, original_image = result
            similar_images.append({
                'image_path': f"files/train_dataset/{label}/{original_image}",  # Reconstruct path
                'class': label,
                'similarity_score': float(similarity_score),
                'filename': original_image,
                'model_name': model_name,
                'augmentation': augmentation
            })

        return similar_images

    def calculate_average_class_similarity(self, similar_images: List[Dict[str, Any]],
                                        query_class: str) -> Dict[str, float]:
        """Calculate average similarity scores grouped by class.

        Args:
            similar_images: Results from find_similar_images
            query_class: The class of the query image

        Returns:
            Dict with class names as keys and average similarity as values
        """
        class_similarities = {}
        class_counts = {}

        for img in similar_images:
            img_class = img['class']
            score = img['similarity_score']

            if img_class not in class_similarities:
                class_similarities[img_class] = 0.0
                class_counts[img_class] = 0

            class_similarities[img_class] += score
            class_counts[img_class] += 1

        # Calculate averages
        avg_similarities = {}
        for class_name in class_similarities:
            avg_similarities[class_name] = class_similarities[class_name] / class_counts[class_name]

        # Sort by average similarity (highest first)
        sorted_avg = dict(sorted(avg_similarities.items(), key=lambda x: x[1], reverse=True))

        return sorted_avg

    def get_similar_images_with_class_analysis(self, query_image_path: str, query_class: str,
                                             top_k: int = 5) -> Dict[str, Any]:
        """Find similar images and calculate class-based similarity analysis.

        Returns:
            Dict with 'similar_images' list and 'class_similarities' dict
        """
        similar_images = self.find_similar_images(query_image_path, top_k=top_k)
        class_similarities = self.calculate_average_class_similarity(similar_images, query_class)

        return {
            'similar_images': similar_images,
            'class_similarities': class_similarities,
            'query_class': query_class,
            'query_image': query_image_path
        }

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        try:
            total_vectors = self.vector_store.get_vector_count(self.table_name)

            # Get class distribution
            with self.vector_store.db as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    SELECT metadata->>'class' as class_name, COUNT(*) as count
                    FROM {self.table_name}
                    GROUP BY metadata->>'class'
                    ORDER BY count DESC
                """)
                class_distribution = dict(cur.fetchall())

            return {
                'total_images': total_vectors,
                'class_distribution': class_distribution,
                'vector_dimension': 2000
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}


def create_similarity_search(model_path: str, config_path: str,
                           table_name: str = "image_features") -> ImageSimilaritySearch:
    """Factory function to create and initialize ImageSimilaritySearch."""
    if not POSTGRESQL_AVAILABLE:
        raise ImportError("PostgreSQL dependencies not available. This function requires psycopg2 and pgvector to be installed.")
    search = ImageSimilaritySearch(table_name=table_name)
    search.load_trainer(model_path, config_path)
    return search