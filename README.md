# Wound Detection Platform

AI-Powered Wound Classification & Medical Image Analysis System

*Columbia University SPS × Advanced Medical Imaging - Deep Learning Project Fall 2025*

## 🎯 Overview

This project implements an intelligent wound detection and classification system using deep learning techniques. The platform provides healthcare professionals with automated wound analysis capabilities, supporting the identification of various wound types through computer vision and machine learning.

### Key Features

- **Multi-Class Wound Classification**: Identifies 10 different wound types including abrasions, bruises, burns, cuts, diabetic wounds, lacerations, pressure wounds, surgical wounds, venous wounds, and normal skin
- **Apple Silicon MPS Support**: Optimized for macOS with Metal Performance Shaders (MPS) acceleration for faster training on Apple Silicon Macs
- **Web-Based Interface**: User-friendly Streamlit application for image upload and analysis
- **Vector Search**: PostgreSQL with pgvector for efficient similarity search and image retrieval
- **Model Training Pipeline**: Automated training scripts with support for multiple CNN architectures
- **Docker Deployment**: Containerized environment for easy deployment and scaling
- **Real-time Analysis**: Instant wound classification and confidence scoring

## 🏗️ Architecture

The system consists of multiple interconnected services:

- **Streamlit Web App**: Frontend interface for wound analysis
- **PostgreSQL + pgvector**: Vector database for image embeddings and metadata
- **Jupyter Environment**: Interactive notebooks for model development and experimentation
- **Training Pipeline**: Standalone scripts for model training and evaluation

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Git
- Python 3.12+ (recommended for local development)
- At least 8GB RAM recommended for model training

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/hyper07/dl-fall-2025.git
   cd dl-fall-2025
   ```

2. **Environment Setup**
   ```bash
   # Copy environment file (if needed)
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Launch the Platform**
   ```bash
   # Start all services
   docker-compose up --build

   # Or start individual services
   docker-compose up dl-streamlit-app dl-postgres
   ```

4. **Access the Application**
   - **Streamlit App**: http://localhost:48501
   - **Jupyter Notebook**: http://localhost:48888 (token: empty)
   - **PostgreSQL**: localhost:45432

## 📊 Dataset

The system uses a comprehensive wound image dataset with the following categories:

- Abrasions
- Bruises
- Burns
- Cut
- Diabetic Wounds
- Laceration
- Normal (healthy skin)
- Pressure Wounds
- Surgical Wounds
- Venous Wounds

**Dataset Location**: `files/train_dataset/`

## 🧠 Model Training

### Advanced Training Options

```bash
# Full enhanced training with MPS acceleration and progress bars
python train_model.py \
  --architecture efficientnet \
  --exact_rotations \
  --device mps \
  --progress_bar \
  --epochs 50 \
  --data_dir ./files/train_dataset

# Quiet training (suppress TensorFlow warnings)
python train_model.py \
  --quiet \
  --device auto

# Minimal training without progress bars
python train_model.py \
  --augment \
  --exact_rotations=false \
  --progress_bar=false \
  --device auto

# Fine-tuning after initial training
python train_model.py --fine_tune --fine_tune_epochs 10 --unfreeze_layers 30
```

### Data Augmentation

The training pipeline supports comprehensive data augmentation:

- **Standard Augmentation**: Random rotations (±40°), shifts, and flips
- **Enhanced Augmentation** (`--exact_rotations`): Exact 90°, 180°, 270° rotations + horizontal/vertical flips
- **Effective Dataset Size**: Up to 8x multiplication with exact transformations

**Augmentation Types:**
- Rotations: 0°, 90°, 180°, 270°
- Flips: Horizontal, vertical, and combined transformations
- Random augmentations: Small rotations, shifts, and scaling

### Training Progress

The training script provides detailed progress monitoring:

- **TQDM Progress Bars**: Real-time progress bars for epochs and batches
- **Live Metrics**: Training loss, accuracy, and validation metrics displayed in progress bars
- **Detailed Logging**: Comprehensive logging to console and training.log file
- **Model Checkpoints**: Automatic saving of best models during training

### Supported Architectures

- **ResNet50**: Default choice, good balance of accuracy and speed
- **VGG16**: Deeper architecture for complex feature extraction
- **EfficientNet**: Optimized for computational efficiency

### Configuration

Training parameters can be customized via `training_config.json`:

```json
{
  "model": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 20,
    "validation_split": 0.2
  }
}
```

## � Similarity Search

The platform includes advanced image similarity search capabilities powered by PostgreSQL with pgvector extension, providing MongoDB Atlas Search-like functionality.

### Features

- **1024-Dimensional Feature Vectors**: Optimized feature extraction from ResNet50 GAP layer with Dense reduction
- **Cosine Similarity Search**: Efficient vector similarity using pgvector exact search
- **Class-Based Analysis**: Average similarity scores grouped by wound class
- **Real-time Search**: Instant similarity search with numerical scores (no "nan" values)
- **Database Integration**: Seamless integration with PostgreSQL vector database
- **Dual Output Model**: Single forward pass returns both class predictions and feature vectors (like Flask app)

### How It Works

1. **Dual Output Prediction**: Single model forward pass extracts both class probabilities and 1024-dimensional feature vectors
2. **Feature Extraction**: ResNet50 backbone with GAP pooling and Dense reduction to 1024 dimensions
3. **Vector Storage**: Features stored in PostgreSQL with pgvector extension for efficient similarity search
4. **Similarity Search**: Cosine similarity comparison between query image and database vectors
5. **Class Analysis**: Results grouped by wound type with average similarity scores

### Usage

```python
# Dual output prediction (like Flask app)
from core.model_utils import CNNTrainer

trainer = CNNTrainer()
trainer.load_model('wound_classifier_best.keras')

# Single forward pass returns both class and features
result = trainer.predict_dual('wound_image.jpg')
print(f"Class probabilities: {result['class']}")  # Shape: (10,)
print(f"Feature vector: {result['feature']}")     # Shape: (1024,)

# Similarity search with automatic class prediction
from core.image_similarity import create_similarity_search

search = create_similarity_search(
    model_path='models/resnet50/wound_classifier_best.keras',
    config_path='training_config.json',
    table_name='images_features'
)

results = search.get_similar_images_with_class_analysis(
    query_image_path='query_wound.jpg',
    top_k=5
)
```

### Database Schema

```sql
CREATE TABLE images_features (
    id SERIAL PRIMARY KEY,
    content TEXT,
    model_name VARCHAR(50),
    label VARCHAR(100),
    augmentation VARCHAR(50),
    original_image VARCHAR(255),
    embedding VECTOR(1024)
);
```

### Performance

- **Vector Dimension**: 1024 (optimized for pgvector HNSW compatibility)
- **Database Size**: 9,186+ vectors with augmentations
- **Search Speed**: Sub-second similarity search
- **Accuracy**: Numerical similarity scores with proper L2 normalization

## �🔧 Development

### Project Structure

```
dl-fall-2025/
├── app-streamlit/          # Streamlit web application
│   ├── pages/             # App pages (Training, Analysis, etc.)
│   ├── functions/         # Utility functions
│   ├── components/        # Reusable UI components
│   └── styles/            # CSS styling
├── core/                  # Shared utilities
│   ├── database.py        # Database operations
│   ├── data_processing.py # Data loading & preprocessing
│   ├── model_utils.py     # Model utilities
│   └── config.py          # Configuration management
├── jupyter/               # Jupyter environment
├── files/                 # Dataset and static files
├── models/                # Trained model artifacts
├── logs/                  # Application logs
├── train_model.py         # Training script
├── training_config.json   # Training configuration
├── docker-compose.yml     # Multi-service orchestration
└── requirements.txt       # Python dependencies
```

### Adding New Features

1. **Model Architectures**: Extend `core/model_utils.py`
2. **Data Processing**: Modify `core/data_processing.py`
3. **UI Components**: Add to `app-streamlit/components/`

### Testing

```bash
# Run unit tests
python -m pytest

# Test specific module
python -m pytest core/tests/
```

## 📈 Performance & Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance metrics
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis
- **ROC Curves**: Multi-class classification performance

## � Troubleshooting

### TensorFlow Mutex Lock Warnings

If you see messages like `[mutex.cc : 452] RAW: Lock blocking 0x...`, this is normal TensorFlow behavior on macOS with MPS. These warnings indicate TensorFlow is properly initializing GPU acceleration.

**To suppress these warnings:**
```bash
python train_model.py --quiet
```

**What these warnings mean:**
- TensorFlow is acquiring internal locks for thread-safe operations
- MPS (Metal Performance Shaders) initialization on macOS
- Safe to ignore - they don't affect training performance

### Common Issues

**MPS/GPU Acceleration Not Working:**
- Ensure you have TensorFlow 2.13+ installed
- On macOS, MPS is automatically enabled
- Use `--device cpu` to force CPU-only training

**Memory Issues:**
- Reduce batch size: `--batch_size 8`
- Use CPU training: `--device cpu`
- Close other GPU-intensive applications

**Import Errors:**
- Install dependencies: `pip install -r requirements.txt`
- For Docker: `docker-compose build --no-cache`

### Python Version

This project requires **Python 3.12+** for optimal compatibility with:
- TensorFlow 2.15+ (latest MPS support for macOS)
- TQDM progress bars
- Modern scikit-learn and data science libraries

**For local development:**
```bash
python --version  # Should show Python 3.12.x
pip install -r requirements.txt
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Ensure Docker compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Columbia University School of Professional Studies
- Advanced Medical Imaging partners
- Open-source deep learning community
- Contributors and maintainers

## 📞 Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/hyper07/dl-fall-2025/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hyper07/dl-fall-2025/discussions)
- **Email**: [Project Maintainers]

---

*Built with ❤️ for advancing healthcare through AI*
