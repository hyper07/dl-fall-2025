#!/usr/bin/env python3
"""
Test script to check feature vector values and identify the zero issue.
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
from tensorflow.keras.models import Model

# Add core to path
sys.path.append(str(Path(__file__).parent / 'core'))

from core.model_utils import CNNTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_feature_extraction():
    """Test feature extraction and check for zero values."""
    # Load trained model
    logger.info("Loading trained ResNet50 model...")
    trainer = CNNTrainer(architecture='resnet50', model_name='wound_classifier')
    trainer.load_model('./models/resnet50/wound_classifier.pkl')
    logger.info(f"Model is_trained: {trainer.is_trained}")
    logger.info("Model loaded successfully")

    # Find a sample image
    dataset_dir = './files/train_dataset'
    sample_image = None
    for class_dir in Path(dataset_dir).iterdir():
        if class_dir.is_dir():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                images = list(class_dir.glob(ext))
                if images:
                    sample_image = str(images[0])
                    break
            if sample_image:
                break

    if not sample_image:
        logger.error("No sample image found")
        return

    logger.info(f"Testing with sample image: {sample_image}")

    # Load and preprocess image
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    img = load_img(sample_image, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    logger.info(f"Input image shape: {img_array.shape}")
    logger.info(f"Input image range: {img_array.min()} to {img_array.max()}")

    # Test base model output
    try:
        base_output = trainer.model.layers[1].predict(img_array, verbose=0)  # ResNet50 base
        logger.info(f"Base model output shape: {base_output.shape}")
        logger.info(f"Base model output range: {base_output.min()} to {base_output.max()}")
        logger.info(f"Base model non-zero: {np.count_nonzero(base_output)}")
    except Exception as e:
        logger.error(f"Error testing base model: {e}")

    # Test GlobalAveragePooling output
    try:
        gap_input = trainer.model.layers[1](img_array, training=False)
        gap_output = trainer.model.layers[2](gap_input)  # GlobalAveragePooling2D
        logger.info(f"GAP output shape: {gap_output.shape}")
        logger.info(f"GAP output range: {gap_output.min()} to {gap_output.max()}")
        logger.info(f"GAP non-zero: {np.count_nonzero(gap_output)}")
    except Exception as e:
        logger.error(f"Error testing GAP: {e}")

    # Check model layers
    logger.info("Model layers:")
    for i, layer in enumerate(trainer.model.layers):
        logger.info(f"  {i}: {layer.name} - {layer.__class__.__name__}")

    # Check if feature_layer exists
    try:
        feature_layer = trainer.model.get_layer('feature_layer')
        logger.info(f"Feature layer found: {feature_layer}")
        logger.info(f"Feature layer output shape: {feature_layer.output.shape}")

        # Check feature layer weights
        weights = feature_layer.get_weights()
        logger.info(f"Feature layer has {len(weights)} weight arrays")
        if len(weights) >= 2:
            kernel_weights = weights[0]
            bias_weights = weights[1]
            logger.info(f"Kernel weights shape: {kernel_weights.shape}")
            logger.info(f"Bias weights shape: {bias_weights.shape}")
            logger.info(f"Kernel weights range: {kernel_weights.min():.6f} to {kernel_weights.max():.6f}")
            logger.info(f"Bias weights range: {bias_weights.min():.6f} to {bias_weights.max():.6f}")
            logger.info(f"Non-zero kernel weights: {np.count_nonzero(kernel_weights)}")
            logger.info(f"Non-zero bias weights: {np.count_nonzero(bias_weights)}")
    except ValueError as e:
        logger.error(f"Feature layer not found: {e}")

    # Test GAP output (input to feature layer)
    try:
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        gap_layer = trainer.model.get_layer('global_average_pooling2d')
        gap_model = Model(inputs=trainer.model.input, outputs=gap_layer.output)

        # Load and preprocess image
        img = load_img(sample_image, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        gap_output = gap_model.predict(img_array)
        logger.info(f"GAP output shape: {gap_output.shape}")
        logger.info(f"GAP output range: {gap_output.min()} to {gap_output.max()}")
        logger.info(f"GAP non-zero values: {np.count_nonzero(gap_output)}")
        logger.info(f"GAP mean: {gap_output.mean()}")
        logger.info(f"GAP std: {gap_output.std()}")
    except Exception as e:
        logger.error(f"Error testing GAP: {e}")

    # Test feature layer pre-activation (without ReLU)
    try:
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.models import Model

        # Load and preprocess image
        img = load_img(sample_image, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Get GAP output
        gap_layer = trainer.model.get_layer('global_average_pooling2d')
        gap_model = Model(inputs=trainer.model.input, outputs=gap_layer.output)
        gap_output = gap_model.predict(img_array)

        # Manually compute Dense layer output without activation
        feature_layer = trainer.model.get_layer('feature_layer')
        kernel, bias = feature_layer.get_weights()
        pre_activation = np.dot(gap_output, kernel) + bias

        logger.info(f"Pre-activation shape: {pre_activation.shape}")
        logger.info(f"Pre-activation range: {pre_activation.min()} to {pre_activation.max()}")
        logger.info(f"Pre-activation mean: {pre_activation.mean()}")
        logger.info(f"Pre-activation std: {pre_activation.std()}")
        logger.info(f"Pre-activation negative values: {np.sum(pre_activation < 0)}")
        logger.info(f"Pre-activation positive values: {np.sum(pre_activation > 0)}")
        logger.info(f"Pre-activation zero values: {np.sum(pre_activation == 0)}")

    except Exception as e:
        logger.error(f"Error testing pre-activation: {e}")

    # Test classification output
    try:
        from tensorflow.keras.preprocessing.image import load_img, img_to_array

        # Load and preprocess image
        img = load_img(sample_image, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Get classification predictions
        class_predictions = trainer.model.predict(img_array)
        logger.info(f"Classification predictions shape: {class_predictions.shape}")
        logger.info(f"Classification predictions: {class_predictions}")
        logger.info(f"Predicted class: {np.argmax(class_predictions)}")
        logger.info(f"Prediction confidence: {np.max(class_predictions)}")

    except Exception as e:
        logger.error(f"Error testing classification: {e}")

    # Extract features
    features = trainer.extract_features(sample_image)
    logger.info(f"Feature vector shape: {features.shape}")
    logger.info(f"Feature vector dtype: {features.dtype}")

    # Analyze the features
    non_zero_count = np.count_nonzero(features)
    zero_count = len(features) - non_zero_count
    zero_percentage = (zero_count / len(features)) * 100

    logger.info(f"Non-zero values: {non_zero_count}")
    logger.info(f"Zero values: {zero_count}")
    logger.info(f"Zero percentage: {zero_percentage:.2f}%")

    logger.info(f"Min value: {features.min()}")
    logger.info(f"Max value: {features.max()}")
    logger.info(f"Mean value: {features.mean()}")
    logger.info(f"Std value: {features.std()}")

    logger.info(f"First 10 values: {features[:10]}")
    logger.info(f"Last 10 values: {features[-10:]}")

    # Check if values are all positive (ReLU effect)
    negative_count = np.sum(features < 0)
    logger.info(f"Negative values: {negative_count}")

    return features


if __name__ == "__main__":
    test_feature_extraction()