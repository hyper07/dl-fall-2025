#!/usr/bin/env python3
"""
Manual CNN Training Script for Wound Classification

This script provides a standalone way to train CNN models for wound classification
using the core utilities. It supports multiple architectures and training configurations.

Usage:
    python train_model.py --architecture resnet50 --epochs 20 --batch_size 32
    python train_model.py --config config.json
    python train_model.py --help
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

# Add core to path
sys.path.append(str(Path(__file__).parent))

# Keywords that typically appear in augmented filenames (rotations, mirrors, etc.)
AUGMENTED_NAME_MARKERS = (
    "_rot",
    "_mirror"
)

from core.model_utils import CNNTrainer, train_cnn_model
from core.config import config, initialize_app
from core.data_processing import DataValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def str2bool(value):
    """Parse flexible boolean CLI arguments."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    value_str = str(value).strip().lower()
    if value_str in {"true", "t", "1", "yes", "y"}:
        return True
    if value_str in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'. Use true/false.")


def add_boolean_arg(parser, name, default=False, help_text=None):
    """Add a --flag [true|false] style argument with sensible defaults."""
    parser.add_argument(
        f'--{name}',
        type=str2bool,
        nargs='?',
        const=True,
        default=default,
        metavar='{true,false}',
        help=(help_text or f"Set {name.replace('_', ' ')} (true/false). "
              f"Defaults to {str(default).lower()}.")
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CNN model for wound classification')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./files/train_dataset_augmented',
                       help='Path to training dataset directory')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Directory to save trained models')

    # Model arguments
    parser.add_argument('--architecture', type=str, default='resnet50',
                       choices=['resnet50', 'vgg16', 'efficientnet'],
                       help='CNN architecture to use')
    parser.add_argument('--model_name', type=str, default='wound_classifier',
                       help='Name for the trained model')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--test_split', type=float, default=0.1,
                       help='Fraction of data to use for testing (stratified by class)')
    parser.add_argument('--feature_dim', type=int, default=1536,
                       help='Dimension of feature vectors for vector search')

    # Training options
    add_boolean_arg(parser, 'augment', default=False,
                    help_text='Use data augmentation (true/false)')
    add_boolean_arg(parser, 'exact_rotations', default=True,
                    help_text='Use exact 90/180/270 degree rotations and flips (true/false)')
    parser.add_argument('--augmentation_factor', type=int, default=None,
                       help='Augmentation factor for enhanced data generation '
                            '(defaults to 8 when exact_rotations is true)')
    add_boolean_arg(parser, 'progress_bar', default=True,
                    help_text='Show TQDM progress bars during training (true/false)')
    add_boolean_arg(parser, 'quiet', default=False,
                    help_text='Suppress TensorFlow warnings and reduce output verbosity (true/false)')
    add_boolean_arg(parser, 'fine_tune', default=False,
                    help_text='Perform fine-tuning after initial training (true/false)')
    parser.add_argument('--fine_tune_epochs', type=int, default=5,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--unfreeze_layers', type=int, default=20,
                       help='Number of layers to unfreeze for fine-tuning')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'gpu', 'mps'],
                       help='Device to use for training (auto detects MPS on macOS)')

    # Configuration
    parser.add_argument('--config', type=str,
                       help='Path to JSON config file')
    parser.add_argument('--save_config', action='store_true', default=False,
                       help='Save current configuration to file')

    return parser.parse_args()


def load_config_from_file(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Update global config
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)

    logger.info(f"Loaded configuration from {config_path}")


def validate_dataset(data_dir, include_augmented=True):
    """Validate the training dataset."""
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory {data_dir} does not exist")

    # Get class directories
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class directories found in {data_dir}")

    logger.info(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

    # Count images per class
    class_counts = {}
    total_images = 0

    for class_dir in class_dirs:
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))

        if not include_augmented:
            filtered_images = [
                img for img in image_files
                if not any(marker in img.name.lower() for marker in AUGMENTED_NAME_MARKERS)
            ]

            if not filtered_images:
                logger.warning(
                    "All images for class %s were flagged as augmented; falling back to full set.",
                    class_dir.name
                )
            else:
                image_files = filtered_images

        class_counts[class_dir.name] = len(image_files)
        total_images += len(image_files)

    logger.info(f"Total images: {total_images}")
    for class_name, count in class_counts.items():
        logger.info(f"  {class_name}: {count} images")

    # Check for minimum samples
    min_samples = min(class_counts.values())
    if min_samples < 10:
        logger.warning(f"Some classes have very few samples (min: {min_samples}). Consider adding more data.")

    return len(class_dirs), class_counts


def create_output_directory(output_dir, architecture):
    """Create output directory with timestamp."""
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = output_dir / f"{architecture}"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created output directory: {model_dir}")
    return model_dir


def save_training_summary(model_dir, trainer, class_counts, training_history, args, eval_results=None):
    """Save training summary and configuration."""
    summary = {
        'training_date': datetime.now().isoformat(),
        'model_name': trainer.model_name,
        'architecture': trainer.architecture,
        'num_classes': len(class_counts),
        'class_counts': class_counts,
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'test_split': args.test_split,
            'feature_dim': args.feature_dim,
            'augmentation': args.augment,
            'exact_rotations': args.exact_rotations,
            'augmentation_factor': args.augmentation_factor,
            'fine_tuning': args.fine_tune
        },
        'final_metrics': {
            'train_accuracy': training_history.get('class_output_accuracy', [-1])[-1],
            'val_accuracy': training_history.get('val_class_output_accuracy', [-1])[-1],
            'train_loss': training_history.get('class_output_loss', [-1])[-1],
            'val_loss': training_history.get('val_class_output_loss', [-1])[-1]
        } if training_history else None,
        'evaluation_metrics': eval_results.get('metrics') if eval_results else None,
        'model_paths': {
            'model_file': str(model_dir / f"{trainer.model_name}.pkl"),
            'config_file': str(model_dir / 'training_config.json')
        }
    }

    # Save summary
    with open(model_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save training arguments
    with open(model_dir / 'training_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    logger.info(f"Training summary saved to {model_dir / 'training_summary.json'}")


def main():
    """Main training function."""
    args = parse_arguments()

    # Derive augmentation factor if not explicitly provided
    if args.augmentation_factor is None:
        args.augmentation_factor = 8 if args.exact_rotations else 1
        logger.info(f"Auto-selected augmentation_factor={args.augmentation_factor} "
                    f"(exact_rotations={'enabled' if args.exact_rotations else 'disabled'})")

    # Configure TensorFlow logging based on quiet flag
    if args.quiet:
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings and info
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

    # Initialize app
    initialize_app(args.config)

    # Load config from file if specified
    if args.config:
        load_config_from_file(args.config)

    # Override config with command line arguments
    config.model.epochs = args.epochs
    config.model.batch_size = args.batch_size
    config.model.learning_rate = args.learning_rate
    config.model.vector_dimension = args.feature_dim

    logger.info("Starting manual CNN training")
    logger.info(f"Architecture: {args.architecture}")
    logger.info(f"Dataset: {args.data_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")

    try:
        # Validate dataset
        num_classes, class_counts = validate_dataset(
            args.data_dir,
            include_augmented=args.augment
        )

        # Create output directory
        model_dir = create_output_directory(args.output_dir, args.architecture)

        # Save config if requested
        if args.save_config:
            config.save_to_file(model_dir / 'config.json')

        # Method 1: Step-by-step training for more control
        logger.info("Building CNN model...")
        trainer = CNNTrainer(
            architecture=args.architecture,
            model_name=args.model_name
        )

        trainer.build_model(
            num_classes=num_classes,
            feature_dim=args.feature_dim,
            freeze_base=True,
            device=args.device
        )

        logger.info("Creating data generators...")
        if args.exact_rotations and args.augment:
            logger.info("Using enhanced augmentation: exact 90/180/270° rotations + horizontal/vertical flips")
        elif args.augment:
            logger.info("Using standard augmentation: random rotations + shifts + flips")
        else:
            logger.info("No data augmentation enabled")

        train_gen, val_gen = trainer.create_data_generators(
            train_dir=args.data_dir,
            test_split=args.test_split,
            batch_size=args.batch_size,
            augment=args.augment,
            exact_rotations=args.exact_rotations,
            augmentation_factor=args.augmentation_factor,
            include_augmented_files=args.augment
        )

        logger.info(f"Training samples: {train_gen.samples}")
        logger.info(f"Validation samples: {val_gen.samples}")
        if hasattr(train_gen, 'augmentation_factor') and train_gen.augmentation_factor > 1:
            logger.info(f"Effective training samples with augmentation: {train_gen.samples} "
                       f"(factor: {train_gen.augmentation_factor}x)")

        # Train the model
        logger.info("Starting training...")
        if args.progress_bar:
            logger.info("TQDM progress bars enabled - training progress will be displayed")
        else:
            logger.info("Progress bars disabled - using standard Keras output")
        
        best_model_path = model_dir / f"{trainer.model_name}_best.keras"

        training_history = trainer.train(
            train_gen,
            val_gen,
            epochs=args.epochs,
            progress_bar=args.progress_bar,
            best_model_path=best_model_path
        )

        # Fine-tune if requested
        if args.fine_tune:
            logger.info("Starting fine-tuning...")
            trainer.fine_tune(
                train_gen,
                val_gen,
                epochs=args.fine_tune_epochs,
                unfreeze_layers=args.unfreeze_layers,
                learning_rate=args.learning_rate / 10,  # Lower LR for fine-tuning
                best_model_path=best_model_path
            )

        # Evaluate model on test set
        logger.info("Evaluating model on test set...")
        eval_results = trainer.evaluate(val_gen, save_path=model_dir)

        # Save the trained model
        model_path = model_dir / f"{trainer.model_name}.pkl"
        trainer.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

        # Save training summary
        save_training_summary(model_dir, trainer, class_counts, training_history.history, args, eval_results)

        # Test feature extraction
        logger.info("Testing feature extraction...")
        test_image = find_sample_image(args.data_dir, class_counts)
        if test_image:
            features = trainer.extract_features(str(test_image))
            logger.info(f"Feature extraction successful. Feature shape: {features.shape}")
            logger.info(f"Sample features (first 5): {features[:5]}")

        logger.info("Training completed successfully!")
        logger.info(f"Model saved in: {model_dir}")
        logger.info(f"Training logs saved in: training.log")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def find_sample_image(data_dir, class_counts):
    """Find a sample image for testing."""
    data_dir = Path(data_dir)

    for class_name in class_counts.keys():
        class_dir = data_dir / class_name
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images = list(class_dir.glob(ext))
            if images:
                return images[0]

    return None


if __name__ == "__main__":
    main()