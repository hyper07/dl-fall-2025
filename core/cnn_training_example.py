"""
Example usage of the CNN training functionality from core.model_utils
"""

import os
import sys
sys.path.append('/home/jovyan/core')  # Add core to path for Jupyter

from core.model_utils import CNNTrainer, train_cnn_model
from core.config import config

def train_wound_classifier():
    """Example: Train a wound classification model using the core CNN trainer."""

    # Dataset path (adjust based on your data location)
    dataset_path = "/home/jovyan/files/train_dataset"  # Adjust path as needed

    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} not found. Please adjust the path.")
        return

    # Get number of classes from dataset
    class_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    num_classes = len(class_names)

    print(f"Found {num_classes} classes: {class_names}")

    # Method 1: Step-by-step training with more control
    print("\n=== Method 1: Step-by-step training ===")

    trainer = CNNTrainer(architecture="resnet50", model_name="wound_classifier")
    trainer.build_model(num_classes=num_classes, feature_dim=config.model.vector_dimension)

    train_gen, val_gen = trainer.create_data_generators(
        train_dir=dataset_path,
        validation_split=0.2,
        batch_size=config.model.batch_size,
        augment=True
    )

    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")

    # Train the model
    history = trainer.train(
        train_gen,
        val_gen,
        epochs=config.model.epochs
    )

    # Save the trained model
    trainer.save_model("wound_classifier_model.pkl")

    # Test feature extraction
    print("\n=== Testing feature extraction ===")
    # Find a sample image
    sample_image = None
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample_image = os.path.join(class_dir, img_file)
                break
        if sample_image:
            break

    if sample_image:
        print(f"Extracting features from: {sample_image}")
        features = trainer.extract_features(sample_image)
        print(f"Feature vector shape: {features.shape}")
        print(f"Feature vector (first 10 values): {features[:10]}")

        # Test prediction
        predicted_class, feature_vector = trainer.predict_image(sample_image)
        print(f"Predicted class index: {predicted_class}")
        print(f"Predicted class name: {class_names[predicted_class]}")

    # Method 2: Convenience function for quick training
    print("\n=== Method 2: Convenience function ===")

    quick_model = train_cnn_model(
        train_dir=dataset_path,
        num_classes=num_classes,
        architecture="efficientnet",
        epochs=5,  # Quick training for demo
        batch_size=16,
        model_name="quick_wound_classifier"
    )

    print("Quick training completed!")

    return trainer, quick_model

def fine_tune_example(trainer, dataset_path):
    """Example of fine-tuning a pre-trained model."""

    print("\n=== Fine-tuning example ===")

    # Create new data generators
    train_gen, val_gen = trainer.create_data_generators(
        train_dir=dataset_path,
        validation_split=0.2,
        batch_size=16,
        augment=False  # Less augmentation for fine-tuning
    )

    # Fine-tune the model
    trainer.fine_tune(
        train_gen,
        val_gen,
        epochs=3,
        unfreeze_layers=20,
        learning_rate=1e-5
    )

    print("Fine-tuning completed!")

if __name__ == "__main__":
    print("CNN Training Example")
    print("=" * 50)

    # Train models
    model1, model2 = train_wound_classifier()

    # Example fine-tuning (uncomment to run)
    # fine_tune_example(model1, "/path/to/dataset")

    print("\nTraining examples completed!")
    print("Models saved and ready for inference or vector search.")