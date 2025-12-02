#!/usr/bin/env python3
"""
Debug script to test EfficientNet model predictions
"""

import sys
import numpy as np
from pathlib import Path

# Add core to path
sys.path.append(str(Path(__file__).parent))

from core.model_utils import CNNTrainer

def main():
    # Load the model
    trainer = CNNTrainer()
    trainer.load_model("models/efficientnet/wound_classifier_best.keras")
    
    print(f"Model architecture: {trainer.architecture}")
    print(f"Model name: {trainer.model_name}")
    print(f"Is trained: {trainer.is_trained}")
    print(f"Model summary:")
    trainer.model.summary()
    
    # Get class indices
    # Since we don't have the generator, let's assume alphabetical order
    class_names = ['Abrasions', 'Bruises', 'Burns', 'Cut', 'Diabetic Wounds', 'Laseration', 'Normal', 'Pressure Wounds', 'Surgical Wounds', 'Venous Wounds']
    print(f"Class names: {class_names}")

    # Get some test images
    data_dir = Path("files/train_dataset")
    test_images = []

    # Get one image from each class
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            if images:
                test_images.append((str(images[0]), class_dir.name))

    print(f"Testing on {len(test_images)} images...")

    # Predict on each
    for img_path, true_class in test_images:
        try:
            result = trainer.predict_dual(img_path)
            class_probs = result['class'][0]
            predicted_idx = np.argmax(class_probs)
            predicted_class = class_names[predicted_idx]
            confidence = class_probs[predicted_idx]

            print(f"Image: {Path(img_path).name} | True: {true_class} | Predicted: {predicted_class} | Confidence: {confidence:.3f}")
            print(f"  All probs: {class_probs}")

        except Exception as e:
            print(f"Error predicting {img_path}: {e}")

if __name__ == "__main__":
    main()