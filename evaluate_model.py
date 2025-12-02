#!/usr/bin/env python3
"""
Model Evaluation Script for Wound Classification

This script loads a trained CNN model and evaluates it on test images,
generating confusion matrices and detailed metrics.

Usage:
    python evaluate_model.py --model_path models/resnet50/wound_classifier.pkl --data_dir files/train_dataset
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

from core.model_utils import CNNTrainer
from core.config import initialize_app

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained CNN model for wound classification')

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file (.pkl)')
    parser.add_argument('--data_dir', type=str, default='./files/train_dataset',
                       help='Path to training dataset directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save evaluation results (default: same as model directory)')
    parser.add_argument('--test_split', type=float, default=0.1,
                       help='Fraction of data used for testing (must match training split, default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_arguments()

    # Initialize app
    initialize_app()

    logger.info("Starting model evaluation")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Dataset: {args.data_dir}")
    logger.info(f"Test split: {args.test_split}")

    try:
        # Load the trained model
        logger.info("Loading trained model...")
        trainer = CNNTrainer()
        trainer.load_model(args.model_path)

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Use same directory as model
            output_dir = Path(args.model_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create test data generator
        logger.info("Creating test data generator...")
        train_gen, test_gen = trainer.create_data_generators(
            train_dir=args.data_dir,
            test_split=args.test_split,
            batch_size=args.batch_size,
            augment=False,  # No augmentation for evaluation
            exact_rotations=False,
            augmentation_factor=1
        )

        logger.info(f"Test samples: {test_gen.samples}")

        # Evaluate model
        logger.info("Evaluating model on test set...")
        eval_results = trainer.evaluate(test_gen, save_path=output_dir)

        # Print summary
        metrics = eval_results['metrics']
        logger.info("Evaluation Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")

        # Save evaluation summary
        summary = {
            'evaluation_date': datetime.now().isoformat(),
            'model_path': args.model_path,
            'dataset': args.data_dir,
            'test_split': args.test_split,
            'test_samples': test_gen.samples,
            'metrics': metrics,
            'output_files': {
                'confusion_matrix_png': str(output_dir / 'confusion_matrix.png'),
                'confusion_matrix_csv': str(output_dir / 'confusion_matrix.csv'),
                'classification_report_csv': str(output_dir / 'classification_report.csv'),
                'evaluation_metrics_json': str(output_dir / 'evaluation_metrics.json')
            }
        }

        with open(output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved in: {output_dir}")
        logger.info(f"Confusion matrix: {output_dir}/confusion_matrix.png")
        logger.info(f"Detailed metrics: {output_dir}/evaluation_summary.json")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()