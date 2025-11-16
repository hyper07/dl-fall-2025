# Core Utilities Package

This package contains shared functionality for the DL Fall 2025 project, providing utilities for database operations, data processing, model training, file processing, and configuration management.

## Modules

### `database.py`
Database utilities for PostgreSQL with vector search support.

**Key Classes:**
- `DatabaseConnection`: Context manager for database connections
- `VectorStore`: Vector operations for similarity search

**Features:**
- Automatic vector extension registration
- Vector table creation and management
- Similarity search with cosine distance
- Batch vector insertion

**Usage:**
```python
from core.database import get_db_connection, get_vector_store

# Connect to database
db = get_db_connection()

# Create vector store
store = get_vector_store(db)

# Create table for 2000-dimensional vectors
store.create_vector_table("documents", vector_dim=2000)

# Insert vectors
vectors = [("doc1", embedding1, {"title": "Doc 1"}), ...]
store.insert_vectors("documents", vectors)

# Search similar vectors
results = store.search_similar("documents", query_embedding, limit=5)
```

### `data_processing.py`
Data loading, cleaning, and preprocessing utilities.

**Key Classes:**
- `DataLoader`: Load data from various file formats
- `DataPreprocessor`: Handle missing values, normalization, encoding
- `DataValidator`: Data quality checks and schema validation

**Features:**
- CSV, Excel, JSON file loading
- Missing value imputation (drop, mean, median, mode)
- Numeric normalization (standard, min-max)
- Categorical encoding (label, one-hot)
- Data quality reporting
- Train/validation/test splitting

**Usage:**
```python
from core.data_processing import DataLoader, DataPreprocessor, split_data

# Load data
loader = DataLoader()
df = loader.load_csv("data.csv")

# Preprocess
preprocessor = DataPreprocessor()
df = preprocessor.handle_missing_values(df, strategy="mean")
df, scalers = preprocessor.normalize_numeric_columns(df)

# Split data
train_df, val_df, test_df, _ = split_data(df, "target_column")
```

### `model_utils.py`
Machine learning model training and evaluation utilities.

**Key Classes:**
- `ModelTrainer`: Base class for model training
- `ModelEvaluator`: Model evaluation metrics and plotting
- `EnsembleTrainer`: Ensemble model utilities
- `CNNTrainer`: CNN model training with transfer learning and feature extraction

**Features:**
- Unified training interface
- Comprehensive evaluation metrics
- Model serialization and cross-validation
- Ensemble model creation and feature importance plotting
- **CNN Training**: Support for ResNet50, VGG16, EfficientNet architectures
- Transfer learning with feature extraction
- Fine-tuning capabilities
- Image data generators with augmentation

**CNN Training Usage:**
```python
from core.model_utils import CNNTrainer, train_cnn_model

# Method 1: Step-by-step training
trainer = CNNTrainer(architecture="resnet50", model_name="wound_classifier")
trainer.build_model(num_classes=10, feature_dim=2000)
train_gen, val_gen = trainer.create_data_generators("data/train", batch_size=32)
history = trainer.train(train_gen, val_gen, epochs=20)

# Extract features for vector search
features = trainer.extract_features("image.jpg")

# Method 2: Convenience function
model = train_cnn_model(
    train_dir="data/train",
    num_classes=10,
    architecture="efficientnet",
    epochs=15
)

# Fine-tune the model
model.fine_tune(train_gen, val_gen, epochs=5, unfreeze_layers=20)
```

**Traditional ML Usage:**
```python
from core.model_utils import ModelEvaluator, cross_validate_model
from sklearn.ensemble import RandomForestClassifier

# Evaluate model
metrics = ModelEvaluator.calculate_metrics(y_true, y_pred, task_type="classification")

# Cross-validation
scores = cross_validate_model(RandomForestClassifier, X, y, cv_folds=5)

# Plot confusion matrix
ModelEvaluator.plot_confusion_matrix(y_true, y_pred)
```

### `file_processing.py`
File processing utilities for unstructured data.

**Key Classes:**
- `PDFProcessor`: PDF text and metadata extraction
- `ImageProcessor`: Image metadata and basic feature extraction
- `DocumentProcessor`: General document processing
- `FileManager`: File upload and management

**Features:**
- PDF text extraction (PyPDF + pdfminer)
- Image metadata extraction (OpenCV + PIL)
- Document element extraction (unstructured library)
- File upload management
- Batch processing with threading

**Usage:**
```python
from core.file_processing import process_file, FileManager

# Process a single file
result = process_file("document.pdf")
print(result['text'])

# Manage file uploads
file_manager = FileManager("/tmp/uploads")
filepath = file_manager.save_uploaded_file(uploaded_file, "document.pdf")
```

### `config.py`
Configuration management for the application.

**Key Classes:**
- `AppConfig`: Main application configuration
- `DatabaseConfig`: Database-specific settings
- `ModelConfig`: Model training parameters
- `FileConfig`: File processing settings
- `ConfigManager`: Configuration singleton manager

**Features:**
- Environment variable integration
- JSON configuration files
- Nested configuration structures
- Type-safe configuration with dataclasses
- Logging setup

**Usage:**
```python
from core.config import config, initialize_app

# Initialize app with config
initialize_app("config.json")

# Access configuration
db_host = config.database.host
vector_dim = config.model.vector_dimension
upload_dir = config.file.upload_dir
```

## Installation Requirements

The core package requires the following libraries (already included in the Docker containers):

- Database: `psycopg2-binary`, `pgvector`
- Data Processing: `pandas`, `numpy`, `scikit-learn`
- File Processing: `pypdf`, `pdfminer`, `opencv-python`, `unstructured`, `Pillow`
- Model Utils: `joblib`, `matplotlib`, `seaborn`, `tensorflow`
- Config: Standard library only

## Examples

See `cnn_training_example.py` for a complete example of training CNN models for image classification with feature extraction for vector search.

## Development

To extend the core package:

1. Add new utility functions to appropriate modules
2. Update type hints and docstrings
3. Add unit tests in a `tests/` directory
4. Update this README with new functionality

## Error Handling

All modules include comprehensive error handling and logging. Check the logs for debugging information when issues occur.