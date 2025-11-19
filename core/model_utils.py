"""
Machine learning model utilities for training, evaluation, and deployment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import joblib
from pathlib import Path
import json
from datetime import datetime
import warnings
import sys
import math

logger = logging.getLogger(__name__)

# Global TensorFlow configuration to prevent mutex issues
try:
    import tensorflow as tf
    # Configure TensorFlow globally to prevent threading issues
    tf.get_logger().setLevel('ERROR')
    # Disable eager execution warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # Set memory growth for GPUs if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    # Configure MPS for macOS if available
    mps_devices = tf.config.list_physical_devices('MPS')
    if mps_devices:
        tf.config.set_visible_devices(mps_devices, 'MPS')
        logger.info("MPS devices configured globally for macOS")
    _tensorflow_configured = True
except ImportError:
    tf = None
    _tensorflow_configured = False

# Provide safe bases so downstream classes import cleanly with or without TensorFlow
if _tensorflow_configured:
    from tensorflow.keras.callbacks import Callback as _KerasCallback
    try:
        from tensorflow.keras.utils import Sequence as _KerasSequence
    except ImportError:
        _KerasSequence = tf.keras.utils.Sequence  # type: ignore[attr-defined]
    CallbackBase = _KerasCallback
    SequenceBase = _KerasSequence
else:
    class CallbackBase:  # type: ignore[too-few-public-methods]
        """Fallback callback base when TensorFlow is unavailable."""

        def __init__(self, *args, **kwargs):
            pass

    class SequenceBase:  # type: ignore[too-few-public-methods]
        """Fallback sequence base when TensorFlow is unavailable."""

        def __len__(self):
            return 0

        def __getitem__(self, index):
            raise IndexError(index)


class ModelTrainer:
    """Base class for model training with common functionality."""

    def __init__(self, model_name: str = "base_model"):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_history = []

    def train(self, X_train: Union[np.ndarray, pd.DataFrame],
              y_train: Union[np.ndarray, pd.Series],
              X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
              y_val: Optional[Union[np.ndarray, pd.Series]] = None,
              **kwargs) -> Dict[str, Any]:
        """Train the model. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement train method")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make probability predictions if supported."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("Model does not support probability predictions")

    def save_model(self, filepath: Union[str, Path]):
        """Save model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'saved_at': datetime.now().isoformat()
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Union[str, Path]):
        """Load model from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data.get('training_history', [])
        logger.info(f"Model loaded from {filepath}")


class ModelEvaluator:
    """Model evaluation utilities."""

    @staticmethod
    def calculate_metrics(y_true: Union[np.ndarray, pd.Series],
                         y_pred: Union[np.ndarray, pd.Series],
                         task_type: str = 'classification') -> Dict[str, float]:
        """Calculate common evaluation metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_squared_error, mean_absolute_error, r2_score
        )

        metrics = {}

        if task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        elif task_type == 'regression':
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2_score'] = r2_score(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])

        return metrics

    @staticmethod
    def plot_confusion_matrix(y_true: Union[np.ndarray, pd.Series],
                            y_pred: Union[np.ndarray, pd.Series],
                            class_names: Optional[List[str]] = None):
        """Plot confusion matrix for classification tasks."""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    @staticmethod
    def plot_feature_importance(model, feature_names: List[str], top_n: int = 20):
        """Plot feature importance if available."""
        import matplotlib.pyplot as plt

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            logger.warning("Model does not have feature importance information")
            return

        # Sort features by importance
        indices = np.argsort(importance)[::-1][:top_n]
        names = [feature_names[i] for i in indices]
        values = importance[indices]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(names)), values[::-1])
        plt.yticks(range(len(names)), names[::-1])
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.show()


class EnsembleTrainer:
    """Ensemble model training utilities."""

    @staticmethod
    def create_ensemble(models: List[ModelTrainer],
                       method: str = 'voting') -> 'EnsembleModel':
        """Create ensemble from multiple trained models."""
        return EnsembleModel(models, method)

    @staticmethod
    def train_multiple_models(X_train: Union[np.ndarray, pd.DataFrame],
                            y_train: Union[np.ndarray, pd.Series],
                            model_configs: List[Dict[str, Any]]) -> List[ModelTrainer]:
        """Train multiple models with different configurations."""
        trained_models = []

        for config in model_configs:
            model_class = config['model_class']
            model_params = config.get('params', {})
            model_name = config.get('name', f"{model_class.__name__}_model")

            model = model_class(model_name)
            model.train(X_train, y_train, **model_params)
            trained_models.append(model)

            logger.info(f"Trained {model_name}")

        return trained_models


class EnsembleModel:
    """Ensemble model for combining predictions."""

    def __init__(self, models: List[ModelTrainer], method: str = 'voting'):
        self.models = models
        self.method = method

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        if self.method == 'voting':
            # For classification: majority vote
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions.astype(int))
        elif self.method == 'averaging':
            # For regression: average predictions
            return np.mean(predictions, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")


class TQDMProgressBar(CallbackBase):
    """TQDM progress bar callback for Keras training."""

    def __init__(self):
        super().__init__()
        self.epoch_bar = None
        self.current_epoch = 0
        self.tqdm_available = False

    def set_model(self, model):
        """Set the model for the callback. Required by Keras."""
        self.model = model

    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of each training batch."""
        pass

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of each training batch."""
        pass

    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of each test batch."""
        pass

    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of each test batch."""
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of each prediction batch."""
        pass

    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of each prediction batch."""
        pass

    def on_test_begin(self, logs=None):
        """Called at the beginning of testing/evaluation."""
        pass

    def on_test_end(self, logs=None):
        """Called at the end of testing/evaluation."""
        pass

    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction."""
        pass

    def on_predict_end(self, logs=None):
        """Called at the end of prediction."""
        pass

    def set_params(self, params):
        """Set training parameters."""
        self.params = params

    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        try:
            from tqdm import tqdm
            self.tqdm_available = True
            logger.info("TQDM progress bars initialized")
        except ImportError:
            logger.warning("TQDM not available. Install with: pip install tqdm")
            self.tqdm_available = False

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        if not self.tqdm_available:
            return

        from tqdm import tqdm
        self.current_epoch = epoch

        steps_total = self.params.get('steps') if hasattr(self, 'params') else None
        if steps_total in (None, 0):
            steps_total = self.params.get('steps_per_epoch', 0) if hasattr(self, 'params') else 0
        if not steps_total:
            logger.debug("TQDMProgressBar: no step information available; skipping bar creation")
            return

        # Close previous bar if exists
        if self.epoch_bar:
            self.epoch_bar.close()

        # Use more compatible TQDM settings
        self.epoch_bar = tqdm(
            total=steps_total,
            desc=f'Epoch {epoch+1}/{self.params["epochs"]}',
            unit='batch',
            leave=True,
            position=0,
            ncols=100,
            disable=False,  # Ensure not disabled
            file=sys.stdout  # Explicitly set output
        )
        logger.debug(f"Progress bar created for epoch {epoch+1}")

    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch."""
        if not self.tqdm_available or self.epoch_bar is None:
            return

        # Update progress bar with metrics
        metrics_str = ""
        if logs:
            metrics = []
            for key, value in logs.items():
                if key in ['loss', 'val_loss'] and isinstance(value, (int, float)):
                    metrics.append(f"loss: {value:.4f}")
            if metrics:
                metrics_str = f" | {' | '.join(metrics)}"

        self.epoch_bar.set_postfix_str(metrics_str)
        self.epoch_bar.update(1)

        # Debug logging for first few batches
        if batch < 2:
            logger.debug(f"Batch {batch} completed with metrics: {metrics_str}")

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        if self.epoch_bar:
            self.epoch_bar.close()
            self.epoch_bar = None

        # Log validation metrics if available
        if logs and self.tqdm_available:
            val_metrics = []
            for key, value in logs.items():
                if key.startswith('val_') and isinstance(value, (int, float)):
                    clean_key = key.replace('val_class_output_', 'val_')
                    if 'accuracy' in clean_key:
                        val_metrics.append(f"{clean_key}: {value:.3f}")
                    else:
                        val_metrics.append(f"{clean_key}: {value:.4f}")
            if val_metrics:
                logger.info(f"Validation | {' | '.join(val_metrics)}")

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        if self.epoch_bar:
            self.epoch_bar.close()
            self.epoch_bar = None


class EnhancedDataGenerator(SequenceBase):
    """Enhanced data generator that applies exact rotations and flips to augment data."""

    def __init__(self, base_generator, exact_rotations: bool = True, augmentation_factor: int = 8):
        if hasattr(super(), "__init__"):
            try:
                super().__init__()
            except TypeError:
                super().__init__()
        self.base_generator = base_generator
        self.exact_rotations = exact_rotations

        # Use provided augmentation factor or default based on exact_rotations
        if augmentation_factor is not None:
            self._augmentation_factor = augmentation_factor
        else:
            self._augmentation_factor = 8 if exact_rotations else 1
        
        # Limit to maximum available transformations (8)
        self._augmentation_factor = min(self._augmentation_factor, 8)
        
        self.batch_size = base_generator.batch_size

        # Store original samples count
        self._original_samples = base_generator.samples
        self._original_batch_size = base_generator.batch_size

        # Define transformation functions
        import numpy as np
        self.transformations = [
            lambda x: x,  # original
            lambda x: np.rot90(x, k=1, axes=(0, 1)),  # 90 degrees
            lambda x: np.rot90(x, k=2, axes=(0, 1)),  # 180 degrees
            lambda x: np.rot90(x, k=3, axes=(0, 1)),  # 270 degrees
            lambda x: np.fliplr(np.rot90(x, k=1, axes=(0, 1))),  # 90 + horizontal flip
            lambda x: np.fliplr(np.rot90(x, k=2, axes=(0, 1))),  # 180 + horizontal flip
            lambda x: np.fliplr(np.rot90(x, k=3, axes=(0, 1))),  # 270 + horizontal flip
            lambda x: np.flipud(x),  # vertical flip
        ]

    def _apply_exact_transformations(self, image):
        """Apply exact 90, 180, 270 rotations and flips to an image."""
        import numpy as np

        transformations = []

        # Original
        transformations.append(image)

        # Rotations
        transformations.append(np.rot90(image, k=1, axes=(0, 1)))  # 90 degrees
        transformations.append(np.rot90(image, k=2, axes=(0, 1)))  # 180 degrees
        transformations.append(np.rot90(image, k=3, axes=(0, 1)))  # 270 degrees

        # Horizontal flips of rotations
        transformations.append(np.fliplr(np.rot90(image, k=1, axes=(0, 1))))  # 90 + horizontal flip
        transformations.append(np.fliplr(np.rot90(image, k=2, axes=(0, 1))))  # 180 + horizontal flip
        transformations.append(np.fliplr(np.rot90(image, k=3, axes=(0, 1))))  # 270 + horizontal flip

        # Vertical flip of original
        transformations.append(np.flipud(image))  # Vertical flip

        return transformations

    def __len__(self):
        """Return the number of batches per epoch."""
        if self.exact_rotations:
            # Each original image becomes 8 augmented images
            total_augmented_samples = self._original_samples * self._augmentation_factor
            return max(1, math.ceil(total_augmented_samples / float(self.batch_size)))
        else:
            return max(1, math.ceil(self._original_samples / float(self.batch_size)))

    def __getitem__(self, index):
        """Generate one batch of data."""
        import numpy as np
        
        if self.exact_rotations:
            # Calculate which original batch and transformations to use
            augmented_batch_size = self.batch_size
            images_needed = augmented_batch_size

            batch_x_list = []
            batch_y_list = []

            # Keep track of current position in the augmented sequence
            current_augmented_index = index * augmented_batch_size

            while len(batch_x_list) < augmented_batch_size:
                # Map augmented index back to original batch index
                orig_batch_index = (current_augmented_index // self._augmentation_factor) // self._original_batch_size
                orig_image_index = (current_augmented_index // self._augmentation_factor) % self._original_batch_size
                trans_index = current_augmented_index % self._augmentation_factor

                # Get the original batch
                try:
                    orig_batch_x, orig_batch_y = self.base_generator[orig_batch_index % len(self.base_generator)]
                except IndexError:
                    # If we run out of data, reset and continue
                    self.base_generator.reset()
                    orig_batch_x, orig_batch_y = self.base_generator[0]

                if orig_image_index < len(orig_batch_x):
                    image = orig_batch_x[orig_image_index]
                    label = orig_batch_y[orig_image_index]

                    # Apply specific transformation
                    transformed_image = self.transformations[trans_index](image)

                    batch_x_list.append(transformed_image)
                    batch_y_list.append(label)

                current_augmented_index += 1

            # Convert to numpy arrays
            import numpy as np
            batch_x = np.array(batch_x_list[:augmented_batch_size])
            batch_y = np.array(batch_y_list[:augmented_batch_size])

            return batch_x, batch_y
        else:
            # Return original batch without augmentation
            return self.base_generator[index]

    @property
    def samples(self):
        """Return total number of samples (augmented)."""
        return int(self._original_samples * self._augmentation_factor)

    @property
    def num_classes(self):
        """Return number of classes."""
        return self.base_generator.num_classes

    @property
    def augmentation_factor(self):
        """Return the augmentation factor."""
        return self._augmentation_factor


class CNNTrainer(ModelTrainer):
    """CNN model trainer with support for transfer learning and feature extraction."""

    def __init__(self, model_name: str = "cnn_model", architecture: str = "resnet50"):
        super().__init__(model_name)
        self.architecture = architecture.lower()
        self.img_height = 224
        self.img_width = 224
        self.num_classes = None
        self.feature_dim = 1536

    def build_model(self, num_classes: int, feature_dim: int = 1536,
                   freeze_base: bool = True, device: str = 'auto') -> 'CNNTrainer':
        """Build CNN model with specified architecture."""
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device

        try:
            import tensorflow as tf
            # Disable TensorFlow warnings and set up session properly
            tf.get_logger().setLevel('ERROR')
            # Note: Removed GPU reset to allow MPS detection

            from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
            from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
            from tensorflow.keras.models import Model
            from tensorflow.keras.optimizers import Adam

            # Configure device based on user preference
            import platform
            system = platform.system()

            if self.device == 'cpu':
                tf.config.set_visible_devices([], 'GPU')
                tf.config.set_visible_devices([], 'MPS')  # Also disable MPS
                logger.info("Forcing CPU usage")
            elif self.device == 'mps' and system == 'Darwin':
                mps_devices = tf.config.list_physical_devices('MPS')
                gpu_devices = tf.config.list_physical_devices('GPU')
                if mps_devices:
                    tf.config.set_visible_devices(mps_devices, 'MPS')
                    logger.info("MPS acceleration enabled for macOS")
                elif gpu_devices:
                    tf.config.set_visible_devices(gpu_devices, 'GPU')
                    for gpu in gpu_devices:
                        try:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        except Exception:  # Memory growth unsupported on some Metal builds
                            logger.debug("Memory growth not supported for device %s", getattr(gpu, 'name', 'unknown'))
                    logger.info("Using Apple GPU via Metal backend (GPU device type)")
                else:
                    logger.warning("MPS requested but not available, falling back to CPU")
                    tf.config.set_visible_devices([], 'GPU')
            elif self.device == 'gpu':
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Using GPU: {gpus}")
                else:
                    logger.warning("GPU requested but not available, using CPU")
            else:  # auto
                if system == 'Darwin':
                    mps_devices = tf.config.list_physical_devices('MPS')
                    gpu_devices = tf.config.list_physical_devices('GPU')
                    if mps_devices:
                        tf.config.set_visible_devices(mps_devices, 'MPS')
                        logger.info("Auto-detected MPS on macOS")
                    elif gpu_devices:
                        tf.config.set_visible_devices(gpu_devices, 'GPU')
                        for gpu in gpu_devices:
                            try:
                                tf.config.experimental.set_memory_growth(gpu, True)
                            except Exception:
                                logger.debug("Memory growth not supported for device %s", getattr(gpu, 'name', 'unknown'))
                        logger.info("Auto-detected Apple GPU via Metal backend")
                    else:
                        logger.info("MPS/GPU not detected on macOS, using CPU")
                else:
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    if gpus:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"Auto-detected GPU: {gpus}")
                    else:
                        logger.info("Auto-detected CPU usage")

        except ImportError:
            raise ImportError("TensorFlow/Keras not available. Install with: pip install tensorflow")

        # Select base model
        if self.architecture == "resnet50":
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
        elif self.architecture == "vgg16":
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
        elif self.architecture == "efficientnet":
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        # Freeze base model layers
        if freeze_base:
            base_model.trainable = False

        # Build model with DUAL outputs (like Flask app)
        inputs = base_model.input
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)

        # Feature layer - dimensional vectors for similarity search
        feature_output = Dense(feature_dim, activation=None, name="feature_output")(x)

        # Classification output
        class_output = Dense(num_classes, activation="softmax", name="class_output")(feature_output)

        # Model with DUAL outputs: [class_output, feature_output]
        self.model = Model(inputs=inputs, outputs=[class_output, feature_output])

        # Store feature model for extraction (outputs features only)
        self.feature_model = Model(inputs=inputs, outputs=feature_output)

        # Compile model for dual-output training
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                "class_output": "categorical_crossentropy",
                "feature_output": "mean_squared_error"
            },
            metrics={"class_output": "accuracy"}
        )

        logger.info(f"Built {self.architecture} model with {num_classes} classes and {feature_dim}-dimensional feature vectors")
        return self

    def create_data_generators(self, train_dir: str, test_split: float = 0.1,
                              batch_size: int = 32, augment: bool = True, exact_rotations: bool = True,
                              augmentation_factor: int = 8):
        """Create training and test data generators with stratified splitting (90% train, 10% test per class)."""
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            import os
            import numpy as np
            from pathlib import Path
            import pandas as pd
        except ImportError:
            raise ImportError("TensorFlow not available")

        train_dir = Path(train_dir)

        # Collect all image files by class for stratified splitting
        class_files = {}
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                image_files = []
                for ext in supported_extensions:
                    image_files.extend(class_dir.glob(f'*{ext}'))
                    image_files.extend(class_dir.glob(f'*{ext.upper()}'))
                class_files[class_name] = sorted([str(f) for f in image_files])

        if not class_files:
            raise ValueError(f"No image files found in {train_dir}")

        # Stratified split: 90% train, 10% test per class
        train_files = []
        test_files = []

        for class_name, files in class_files.items():
            n_files = len(files)
            n_test = max(1, int(n_files * test_split))  # At least 1 test sample per class

            # Shuffle files for random split
            np.random.seed(42)  # For reproducibility
            shuffled_files = np.random.permutation(files)

            # Split files
            test_files.extend([(f, class_name) for f in shuffled_files[:n_test]])
            train_files.extend([(f, class_name) for f in shuffled_files[n_test:]])

        logger.info(f"Stratified split: {len(train_files)} training samples, {len(test_files)} test samples")
        logger.info(f"Classes: {list(class_files.keys())}")

        # Create data generators
        if augment:
            # Enhanced augmentation for training
            if self.architecture == "efficientnet":
                # EfficientNet uses its own preprocessing
                from tensorflow.keras.applications.efficientnet import preprocess_input
                train_datagen = ImageDataGenerator(
                    preprocessing_function=preprocess_input,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode="nearest"
                )
                test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
            else:
                # Standard preprocessing for other architectures
                train_datagen = ImageDataGenerator(
                    rescale=1.0/255,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode="nearest"
                )
                test_datagen = ImageDataGenerator(rescale=1.0/255)
        else:
            if self.architecture == "efficientnet":
                from tensorflow.keras.applications.efficientnet import preprocess_input
                train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
                test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
            else:
                train_datagen = ImageDataGenerator(rescale=1.0/255)
                test_datagen = ImageDataGenerator(rescale=1.0/255)

        # Create generators from file lists
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame(train_files, columns=['filename', 'class']),
            x_col='filename',
            y_col='class',
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True
        )

        test_generator = test_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame(test_files, columns=['filename', 'class']),
            x_col='filename',
            y_col='class',
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False
        )

        # If augmentation is enabled, create enhanced generators with exact transformations
        if augment:
            train_generator = self.create_enhanced_generator(train_generator, exact_rotations=exact_rotations, augmentation_factor=augmentation_factor)
            # Note: Test generator keeps original images for consistency

        return train_generator, test_generator

    def create_enhanced_generator(self, base_generator, exact_rotations: bool = True, augmentation_factor: int = 8):
        """Create enhanced generator with exact rotations and flips."""
        return EnhancedDataGenerator(base_generator, exact_rotations=exact_rotations, augmentation_factor=augmentation_factor)

    def custom_generator(self, generator):
        """Convert generator to work with dual-output model (like Flask app)."""
        for batch_x, batch_y in generator:
            # Create dummy labels for feature output (same shape as feature vectors)
            dummy_feature_labels = np.zeros((batch_y.shape[0], self.feature_dim))
            yield batch_x, {"class_output": batch_y, "feature_output": dummy_feature_labels}

    def train(self, train_generator, val_generator=None, epochs: int = 20,
              callbacks: List = None, progress_bar: bool = True,
              best_model_path: Optional[Union[str, Path]] = None, **kwargs):
        """Train the feature extraction model."""
        if self.model is None:
            raise ValueError("Model must be built before training. Call build_model() first.")

        # Default callbacks - simplified for feature extraction
        if callbacks is None:
            try:
                from tensorflow.keras.callbacks import ModelCheckpoint
                checkpoint_path = Path(best_model_path) if best_model_path else Path(f'{self.model_name}_best.keras')
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                callbacks = []
                if progress_bar:
                    tqdm_callback = TQDMProgressBar()
                    callbacks.append(tqdm_callback)
                    logger.info("TQDM progress bar callback added")
                else:
                    logger.info("Progress bars disabled")
                callbacks.extend([
                    ModelCheckpoint(str(checkpoint_path), save_best_only=True)
                ])
                logger.info(f"Total callbacks configured: {len(callbacks)}")
            except ImportError:
                callbacks = []

        # Calculate steps
        steps_per_epoch = train_generator.samples // train_generator.batch_size
        validation_steps = val_generator.samples // val_generator.batch_size if val_generator else None

        # Use custom generator for dual outputs (like Flask app)
        train_gen = self.custom_generator(train_generator)
        val_gen = self.custom_generator(val_generator) if val_generator else None

        # Train model epoch by epoch to handle generator resets properly
        full_history = None
        for epoch in range(epochs):
            logger.info(f"Training epoch {epoch + 1}/{epochs}")
            
            # Recreate generators for each epoch to ensure proper reset
            train_gen = self.custom_generator(train_generator)
            val_gen = self.custom_generator(val_generator) if val_generator else None
            
            history = self.model.fit(
                train_gen,
                epochs=epoch + 1,
                initial_epoch=epoch,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=callbacks,
                **kwargs
            )
            
            # Accumulate history
            if full_history is None:
                full_history = history
            else:
                # Merge histories
                for key in history.history:
                    if key in full_history.history:
                        full_history.history[key].extend(history.history[key])
                    else:
                        full_history.history[key] = history.history[key]

        self.is_trained = True
        self.training_history = full_history.history if full_history else {}

        logger.info(f"Feature extraction training completed. Model ready for vector similarity search.")
        return full_history

    def extract_features_from_array(self, img_array: np.ndarray) -> np.ndarray:
        """Extract feature vector from a preprocessed image array."""
        if not self.is_trained:
            raise ValueError("Model must be trained before feature extraction")

        if self.feature_model is None:
            raise ValueError("Feature model not available. Model may not be properly loaded.")

        logger.info(f"Extracting features from image array with shape: {img_array.shape}")
        # Extract features using the feature model (outputs feature vector)
        feature_vector = self.feature_model.predict(img_array, verbose=0)[0]

        # Truncate to 1536 dimensions to match database schema
        if len(feature_vector) > 1536:
            feature_vector = feature_vector[:1536]
            logger.info(f"Truncated feature vector from {len(feature_vector)} to 1536 dimensions")

        # Check for NaN values and handle them
        if np.any(np.isnan(feature_vector)):
            logger.warning(f"Feature vector contains NaN values, replacing with zeros")
            feature_vector = np.nan_to_num(feature_vector, nan=0.0)

        logger.info(f"Feature extraction completed. Feature vector shape: {feature_vector.shape}")
        return feature_vector

    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract feature vector from an image file."""
        try:
            from tensorflow.keras.preprocessing.image import load_img, img_to_array
        except ImportError:
            raise ImportError("TensorFlow not available")

        if not self.is_trained:
            raise ValueError("Model must be trained before feature extraction")

        # Load image
        img = load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Apply architecture-specific preprocessing
        if self.architecture == "efficientnet":
            # For custom trained EfficientNet models, use standard normalization
            # (not the ImageNet EfficientNet preprocessing, since model was trained with standard normalization)
            img_array = img_array / 255.0
        else:
            # ResNet50, VGG16, and other architectures use standard normalization
            img_array = img_array / 255.0

        # Extract features using the feature model
        return self.extract_features_from_array(img_array)

    def predict_image(self, image_path: str) -> Tuple[int, np.ndarray]:
        """Predict class and extract features from an image."""
        try:
            from tensorflow.keras.preprocessing.image import load_img, img_to_array
        except ImportError:
            raise ImportError("TensorFlow not available")

        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Load image
        img = load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Apply architecture-specific preprocessing
        if self.architecture == "efficientnet":
            # For custom trained EfficientNet models, use standard normalization
            img_array = img_array / 255.0
        else:
            img_array = img_array / 255.0

        # Get predictions
        predictions = self.model.predict(img_array)
        class_probs = predictions[0]
        feature_vector = self.feature_model.predict(img_array, verbose=0)[0]

        predicted_class = np.argmax(class_probs)

        return predicted_class, feature_vector

    def predict_dual(self, image_path: str) -> Dict[str, np.ndarray]:
        """Predict both class and features from an image (like Flask app)."""
        try:
            from tensorflow.keras.preprocessing.image import load_img, img_to_array
        except ImportError:
            raise ImportError("TensorFlow not available")

        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Load image
        img = load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Apply architecture-specific preprocessing
        if self.architecture == "efficientnet":
            # For custom trained EfficientNet models, use standard normalization
            img_array = img_array / 255.0
        else:
            img_array = img_array / 255.0

        # Get predictions - handle both single and dual output models
        predictions = self.model.predict(img_array, verbose=0)

        if len(predictions) == 2:
            # Dual output model (new architecture)
            class_predictions = predictions[0]  # Class probabilities
            feature_predictions = predictions[1]  # Feature vector
        else:
            # Single output model (old architecture) - get features separately
            class_predictions = predictions[0]  # Class probabilities
            feature_predictions = self.feature_model.predict(img_array, verbose=0)  # Feature vector

        return {
            "class": class_predictions,  # Class probabilities for all classes
            "feature": feature_predictions.squeeze()  # Feature vector (remove batch dimension)
        }

    def fine_tune(self, train_generator, val_generator=None, epochs: int = 10,
                  unfreeze_layers: int = 10, learning_rate: float = 1e-5,
                  best_model_path: Optional[Union[str, Path]] = None, callbacks: List = None):
        """Fine-tune the model by unfreezing some base layers."""
        if not self.is_trained:
            raise ValueError("Model must be trained before fine-tuning")

        try:
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            raise ImportError("TensorFlow not available")

        # Unfreeze base model layers for fine-tuning
        self.model.layers[1].trainable = True  # Base model is typically at index 1

        # Unfreeze the last N layers
        base_model = self.model.layers[1]
        for layer in base_model.layers[-unfreeze_layers:]:
            layer.trainable = True

        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                "class_output": "categorical_crossentropy",
                "feature_output": "mean_squared_error"
            },
            metrics={"class_output": "accuracy"}
        )

        logger.info(f"Fine-tuning with {unfreeze_layers} unfrozen layers and lr={learning_rate}")

        # Train with fine-tuning
        return self.train(
            train_generator,
            val_generator,
            epochs=epochs,
            best_model_path=best_model_path,
            callbacks=callbacks
        )

    def load_model(self, filepath: Union[str, Path]):
        """Load model from disk and setup feature model."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Try loading as joblib first (for models saved with save_model)
        try:
            import joblib
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.is_trained = model_data['is_trained']
            self.training_history = model_data.get('training_history', [])
            logger.info(f"Model loaded from joblib format: {filepath}")
        except Exception as e:
            logger.info(f"Joblib loading failed ({e}), trying Keras format...")
            # Fallback to Keras load_model for .keras/.h5 files
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(filepath)
                self.is_trained = True
                logger.info(f"Keras model loaded from {filepath}")
            except Exception as keras_error:
                raise ValueError(f"Could not load model from {filepath}. Tried both joblib and Keras formats. Joblib error: {e}, Keras error: {keras_error}")

        # Recreate feature model for feature extraction
        try:
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Dense
            # For dual-output model, feature output is the second output
            if len(self.model.outputs) >= 2:
                # Use the feature output directly from the dual-output model
                self.feature_model = Model(inputs=self.model.input, outputs=self.model.outputs[1])
                self.feature_dim = self.model.outputs[1].shape[-1]
                logger.info(f"Feature model recreated from dual-output model with {self.feature_dim}-dim features")
            else:
                # Fallback: recreate from GAP layer (for backward compatibility)
                gap_layer = self.model.get_layer('global_average_pooling2d')
                reduction_layer = Dense(1536, activation='relu', name='feature_reduction')(gap_layer.output)
                self.feature_model = Model(inputs=self.model.input, outputs=reduction_layer)
                self.feature_dim = 1536
                logger.info("Feature model recreated using GAP layer with 1536-dim reduction")
        except Exception as e:
            logger.warning(f"Could not recreate feature model: {e}")
            self.feature_model = None


def create_cnn_trainer(architecture: str = "resnet50", model_name: str = "cnn_model") -> CNNTrainer:
    """Factory function for CNN trainer."""
    return CNNTrainer(model_name=model_name, architecture=architecture)


def train_cnn_model(train_dir: str, num_classes: int, architecture: str = "resnet50",
                   epochs: int = 20, batch_size: int = 32, feature_dim: int = 1536,
                   test_split: float = 0.1, model_name: str = "cnn_model") -> CNNTrainer:
    """Convenience function to train a CNN model end-to-end."""

    # Create trainer
    trainer = CNNTrainer(model_name=model_name, architecture=architecture)

    # Build model
    trainer.build_model(num_classes=num_classes, feature_dim=feature_dim)

    # Create data generators
    train_gen, val_gen = trainer.create_data_generators(
        train_dir=train_dir,
        test_split=test_split,
        batch_size=batch_size
    )

    # Train model
    history = trainer.train(train_gen, val_gen, epochs=epochs)

    return trainer


def cross_validate_model(model_class: Callable,
                        X: Union[np.ndarray, pd.DataFrame],
                        y: Union[np.ndarray, pd.Series],
                        cv_folds: int = 5,
                        **model_params) -> Dict[str, List[float]]:
    """Perform cross-validation on a model."""
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import make_scorer

    model = model_class(**model_params)

    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted'
    }

    try:
        scores = cross_validate(model, X, y, cv=cv_folds, scoring=scoring)
        return {metric: scores[f'test_{metric}'].tolist() for metric in scoring.keys()}
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}")
        return {}


def save_model_metrics(metrics: Dict[str, Any], filepath: Union[str, Path]):
    """Save model metrics to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Metrics saved to {filepath}")


def load_model_metrics(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load model metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)