# Manual CNN Training Script

This script provides a standalone way to train CNN models for wound classification using the core utilities.

## Features

- **Multiple Architectures**: ResNet50, VGG16, EfficientNet
- **Transfer Learning**: Pre-trained models with fine-tuning options
- **Feature Extraction**: Dual-output models for classification + vector search
- **Data Validation**: Automatic dataset validation and statistics
- **Progress Monitoring**: Detailed logging and training summaries
- **Model Saving**: Automatic model and configuration saving

## Quick Start

### Basic Training
```bash
# Train with default settings (ResNet50, 20 epochs)
python train_model.py

# Train with custom architecture
python train_model.py --architecture efficientnet --epochs 30

# Train with custom batch size and learning rate
python train_model.py --batch_size 16 --learning_rate 0.0001
```

### Using Configuration File
```bash
# Use a custom configuration file
python train_model.py --config my_config.json

# Save current configuration for future use
python train_model.py --save_config
```

### Fine-tuning
```bash
# Train and then fine-tune
python train_model.py --fine_tune --fine_tune_epochs 10 --unfreeze_layers 30
```

## Command Line Arguments

### Data Options
- `--data_dir`: Path to training dataset (default: `./static/train_dataset`)
- `--output_dir`: Directory to save models (default: `./models`)

### Model Options
- `--architecture`: CNN architecture (`resnet50`, `vgg16`, `efficientnet`)
- `--model_name`: Name for the trained model
- `--feature_dim`: Dimension of feature vectors (default: 2000)

### Training Options
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--validation_split`: Validation data fraction (default: 0.2)
- `--augment`: Enable data augmentation (default: True)

### Fine-tuning Options
- `--fine_tune`: Enable fine-tuning after initial training
- `--fine_tune_epochs`: Number of fine-tuning epochs (default: 5)
- `--unfreeze_layers`: Number of layers to unfreeze (default: 20)

### Configuration
- `--config`: Path to JSON configuration file
- `--save_config`: Save current configuration to output directory

## Dataset Structure

Your training data should be organized as:

```
data/train_dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

## Output Files

The script creates a timestamped output directory containing:

- `{model_name}.pkl`: Trained model (joblib format)
- `training_summary.json`: Training statistics and configuration
- `training_args.json`: Command line arguments used
- `config.json`: Full configuration (if `--save_config` used)

## Examples

### Example 1: Basic Training
```bash
python train_model.py \
    --data_dir ./data/wounds \
    --architecture resnet50 \
    --epochs 25 \
    --batch_size 32 \
    --model_name wound_classifier_v1
```

### Example 2: Advanced Training with Fine-tuning
```bash
python train_model.py \
    --architecture efficientnet \
    --epochs 15 \
    --fine_tune \
    --fine_tune_epochs 8 \
    --unfreeze_layers 25 \
    --learning_rate 0.0005 \
    --save_config
```

### Example 3: Using Configuration File
```bash
# First, create/edit training_config.json
# Then run:
python train_model.py --config training_config.json
```

## Configuration File Format

See `training_config.json` for an example configuration file. You can modify:

- Database settings (for vector search integration)
- Model hyperparameters
- File processing settings
- Logging levels

## Monitoring Training

Training progress is logged to both console and `training.log`. The script validates your dataset and provides statistics before training begins.

## Troubleshooting

- **Dataset not found**: Check `--data_dir` path
- **CUDA out of memory**: Reduce `--batch_size`
- **Poor validation accuracy**: Try `--fine_tune` or increase `--epochs`
- **Import errors**: Ensure you're running in the Docker container with all dependencies

## GPU Support

The script automatically uses GPU if available through TensorFlow. No additional configuration needed.