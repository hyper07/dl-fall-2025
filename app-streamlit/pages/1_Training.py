import streamlit as st
import pandas as pd
import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime

# Conditional imports for PostgreSQL functionality
try:
    from core.image_similarity import ImageSimilaritySearch, create_similarity_search, POSTGRESQL_AVAILABLE
except ImportError:
    POSTGRESQL_AVAILABLE = False

from core.model_utils import CNNTrainer

# Page config
st.set_page_config(
    page_title="CNN Training",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 CNN Model Training")
st.markdown("Train CNN models for wound classification with real-time progress monitoring")

# Initialize session state
if 'training_status' not in st.session_state:
    st.session_state.training_status = 'idle'
if 'training_logs' not in st.session_state:
    st.session_state.training_logs = []
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = {
        'epoch': 0,
        'total_epochs': 0,
        'train_acc': 0.0,
        'val_acc': 0.0,
        'message': ''
    }
if 'training_thread' not in st.session_state:
    st.session_state.training_thread = None
if 'stop_training' not in st.session_state:
    st.session_state.stop_training = False

# Configuration section
st.header("Training Configuration")

col1, col2 = st.columns(2)

with col1:
    architecture = st.selectbox(
        "Model Architecture",
        ["resnet50", "vgg16", "efficientnet"],
        index=0,
        help="Choose the CNN architecture for training"
    )

    epochs = st.slider(
        "Training Epochs",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="Number of training epochs"
    )

    batch_size = st.selectbox(
        "Batch Size",
        [16, 32, 64],
        index=1,
        help="Batch size for training"
    )

with col2:
    learning_rate = st.selectbox(
        "Learning Rate",
        [0.001, 0.0001, 0.00001],
        index=0,
        help="Initial learning rate"
    )

    use_augmentation = st.checkbox(
        "Data Augmentation",
        value=True,
        help="Use data augmentation during training"
    )

    fine_tune = st.checkbox(
        "Fine-tuning",
        value=False,
        help="Perform fine-tuning after initial training"
    )

# Dataset validation
st.header("Dataset Overview")

data_dir = Path("./files/train_dataset")

if data_dir.exists():
    # Validate dataset manually
    try:
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if class_dirs:
            class_counts = {}
            total_images = 0

            for class_dir in class_dirs:
                image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
                class_counts[class_dir.name] = len(image_files)
                total_images += len(image_files)

            st.success(f"✅ Found {len(class_counts)} classes with {total_images} total images")

            # Display class distribution
            df_classes = pd.DataFrame({
                'Class': list(class_counts.keys()),
                'Images': list(class_counts.values())
            })

            st.dataframe(df_classes, use_container_width=True)

            # Visualization
            st.bar_chart(df_classes.set_index('Class'))
        else:
            st.error("❌ No class directories found")
    except Exception as e:
        st.error(f"❌ Error validating dataset: {e}")
else:
    st.error(f"❌ Dataset directory not found: {data_dir}")

# Training control
st.header("Training Control")

col_start, col_stop, col_clear = st.columns(3)

def start_training():
    if st.session_state.training_status == 'running':
        st.error("Training is already running")
        return

    st.session_state.training_status = 'running'
    st.session_state.training_logs = []
    st.session_state.training_progress = {
        'epoch': 0,
        'total_epochs': epochs,
        'train_acc': 0.0,
        'val_acc': 0.0,
        'message': 'Initializing training...'
    }
    st.session_state.stop_training = False

    # Start training in background thread
    st.session_state.training_thread = threading.Thread(
        target=train_model_background,
        args=(architecture, epochs, batch_size, learning_rate, use_augmentation, fine_tune, data_dir)
    )
    st.session_state.training_thread.daemon = True
    st.session_state.training_thread.start()

def stop_training():
    st.session_state.stop_training = True
    st.session_state.training_progress['message'] = 'Stopping training...'

def clear_logs():
    st.session_state.training_logs = []
    st.session_state.training_progress = {
        'epoch': 0,
        'total_epochs': 0,
        'train_acc': 0.0,
        'val_acc': 0.0,
        'message': ''
    }

with col_start:
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        start_training()

with col_stop:
    if st.button("⏹️ Stop Training", use_container_width=True):
        stop_training()

with col_clear:
    if st.button("🧹 Clear Logs", use_container_width=True):
        clear_logs()

# Progress display
st.header("Training Progress")

# Status indicators
col_status, col_epoch, col_acc = st.columns(3)

with col_status:
    status_color = {
        'idle': 'gray',
        'running': 'blue',
        'completed': 'green',
        'stopped': 'orange',
        'error': 'red'
    }.get(st.session_state.training_status, 'gray')

    st.metric(
        "Status",
        st.session_state.training_status.upper(),
        delta=None
    )

with col_epoch:
    progress_bar = st.progress(
        min(st.session_state.training_progress['epoch'] / max(st.session_state.training_progress['total_epochs'], 1), 1.0)
    )
    st.metric(
        "Epoch",
        f"{st.session_state.training_progress['epoch']} / {st.session_state.training_progress['total_epochs']}"
    )

with col_acc:
    col_acc1, col_acc2 = st.columns(2)
    with col_acc1:
        st.metric("Train Acc", ".3f")
    with col_acc2:
        st.metric("Val Acc", ".3f")

# Current message
if st.session_state.training_progress['message']:
    if st.session_state.training_status == 'error':
        st.error(st.session_state.training_progress['message'])
    else:
        st.info(st.session_state.training_progress['message'])

# Logs display
st.header("Training Logs")

# Create a placeholder for logs
log_placeholder = st.empty()

# Display logs
if st.session_state.training_logs:
    log_text = "".join(st.session_state.training_logs[-100:])  # Show last 100 log entries
    log_placeholder.code(log_text, language="text")
else:
    log_placeholder.info("No training logs yet. Click 'Start Training' to begin.")

# Auto-refresh
if st.session_state.training_status == 'running':
    time.sleep(1)
    st.rerun()

# Training summary (when completed)
if st.session_state.training_status in ['completed', 'stopped']:
    st.header("Training Summary")

    # Try to load the training summary
    try:
        # Find the latest training directory
        models_dir = Path("./models")
        if models_dir.exists():
            training_dirs = sorted(models_dir.glob("*/"), key=lambda x: x.stat().st_mtime, reverse=True)
            if training_dirs:
                latest_dir = training_dirs[0]
                summary_file = latest_dir / "training_summary.json"

                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Model Information")
                        st.write(f"**Architecture:** {summary.get('architecture', 'N/A')}")
                        st.write(f"**Classes:** {summary.get('num_classes', 'N/A')}")
                        st.write(f"**Training Date:** {summary.get('training_date', 'N/A')}")

                    with col2:
                        st.subheader("Final Metrics")
                        metrics = summary.get('final_metrics', {})
                        st.write(f"**Train Accuracy:** {metrics.get('train_accuracy', 'N/A')}")
                        st.write(f"**Validation Accuracy:** {metrics.get('val_accuracy', 'N/A')}")
                        st.write(f"**Train Loss:** {metrics.get('train_loss', 'N/A')}")
                        st.write(f"**Validation Loss:** {metrics.get('val_loss', 'N/A')}")

                    # Download button for summary
                    st.download_button(
                        label="📥 Download Summary",
                        data=json.dumps(summary, indent=2),
                        file_name="training_summary.json",
                        mime="application/json"
                    )
                else:
                    st.warning("Training summary file not found")
            else:
                st.warning("No training directories found")
        else:
            st.warning("Models directory not found")
    except Exception as e:
        st.error(f"Error loading training summary: {e}")

# Database Initialization for Similarity Search
st.header("Database Setup for Similarity Search")

st.markdown("""
After training, initialize the PostgreSQL vector database to enable similarity search functionality.
This will extract 1024-dimensional features from all training images and store them for fast similarity search.
""")

col_init, col_status = st.columns(2)

with col_init:
    if st.button("🚀 Initialize Vector Database", type="secondary", use_container_width=True):
        if not POSTGRESQL_AVAILABLE:
            st.error("❌ PostgreSQL dependencies not available. Database initialization requires psycopg2 and pgvector.")
            st.info("💡 This feature is available in the Docker environment with PostgreSQL.")
        elif st.session_state.training_status == 'completed':
            with st.spinner("Initializing vector database..."):
                try:
                    # Find the latest trained model
                    models_dir = Path("./models")
                    if models_dir.exists():
                        training_dirs = sorted(models_dir.glob("*/"), key=lambda x: x.stat().st_mtime, reverse=True)
                        if training_dirs:
                            latest_dir = training_dirs[0]
                            model_files = list(latest_dir.glob("*.pkl"))
                            config_files = list(latest_dir.glob("*.json"))

                            if model_files:
                                model_path = model_files[0]
                                config_path = config_files[0] if config_files else Path("./training_config.json")

                                # Initialize similarity search
                                similarity_search = create_similarity_search(
                                    model_path=str(model_path),
                                    config_path=str(config_path),
                                    table_name="image_features"
                                )

                                # Collect all training images
                                image_paths = []
                                class_labels = []

                                for class_dir in data_dir.iterdir():
                                    if class_dir.is_dir():
                                        for img_file in class_dir.glob('*.jpg'):
                                            image_paths.append(str(img_file))
                                            class_labels.append(class_dir.name)
                                        for img_file in class_dir.glob('*.jpeg'):
                                            image_paths.append(str(img_file))
                                            class_labels.append(class_dir.name)
                                        for img_file in class_dir.glob('*.png'):
                                            image_paths.append(str(img_file))
                                            class_labels.append(class_dir.name)

                                if image_paths:
                                    # Store features in database
                                    similarity_search.store_image_features(
                                        image_paths=image_paths,
                                        class_labels=class_labels,
                                        batch_size=16  # Smaller batch for database operations
                                    )

                                    st.success(f"✅ Database initialized with {len(image_paths)} images!")
                                    st.info("You can now use the Similarity Search page to find similar wound images.")
                                else:
                                    st.error("No training images found to populate database.")
                            else:
                                st.error("No trained model file found.")
                        else:
                            st.error("No training directories found.")
                    else:
                        st.error("Models directory not found.")
                except Exception as e:
                    st.error(f"Database initialization failed: {e}")
        else:
            st.warning("Please complete model training first.")

with col_status:
    st.subheader("Database Status")
    try:
        # Try to get database stats
        models_dir = Path("./models")
        if models_dir.exists():
            training_dirs = sorted(models_dir.glob("*/"), key=lambda x: x.stat().st_mtime, reverse=True)
            if training_dirs:
                latest_dir = training_dirs[0]
                model_files = list(latest_dir.glob("*.pkl"))
                config_files = list(latest_dir.glob("*.json"))

                if model_files:
                    model_path = model_files[0]
                    config_path = config_files[0] if config_files else Path("./training_config.json")

                    similarity_search = create_similarity_search(
                        model_path=str(model_path),
                        config_path=str(config_path),
                        table_name="image_features"
                    )

                    db_stats = similarity_search.get_database_stats()
                    if 'error' in db_stats:
                        st.error(f"Database error: {db_stats['error']}")
                    else:
                        st.metric("Images in Database", db_stats.get('total_images', 0))
                        st.metric("Vector Dimension", db_stats.get('vector_dimension', 1024))
                        if db_stats.get('class_distribution'):
                            st.write(f"Classes: {len(db_stats['class_distribution'])}")
                else:
                    st.info("No model available for database check.")
            else:
                st.info("No trained models found.")
        else:
            st.info("Models directory not found.")
    except Exception as e:
        st.error(f"Error checking database status: {e}")

def train_model_background(architecture, epochs, batch_size, learning_rate, augment, fine_tune, data_dir):
    """Background training function"""
    try:
        # Update progress
        st.session_state.training_progress['message'] = f"Starting training with {architecture}..."

        # Log to file and session state
        log_file = Path("./logs/training.log")
        log_file.parent.mkdir(exist_ok=True)

        def log_message(message):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            st.session_state.training_logs.append(log_entry)

            # Also write to file
            with open(log_file, 'a') as f:
                f.write(log_entry)

        log_message("Initializing CNN Trainer...")

        # Create trainer
        trainer = CNNTrainer(
            architecture=architecture,
            model_name=f"wound_classifier_{architecture}"
        )

        # Get class counts for num_classes
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        class_counts = {}
        for class_dir in class_dirs:
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            class_counts[class_dir.name] = len(image_files)
        num_classes = len(class_counts)

        # Build model
        trainer.build_model(
            num_classes=num_classes,
            feature_dim=1024,
            freeze_base=True
        )

        log_message(f"Model built with {num_classes} classes")

        # Create data generators
        train_gen, val_gen = trainer.create_data_generators(
            train_dir=str(data_dir),
            validation_split=0.2,
            batch_size=batch_size,
            augment=augment
        )

        log_message(f"Data generators created. Train: {train_gen.samples}, Val: {val_gen.samples}")

        # Custom callback to update progress
        class StreamlitCallback:
            def __init__(self):
                self.epoch = 0

            def on_epoch_end(self, epoch, logs=None):
                self.epoch = epoch + 1
                st.session_state.training_progress['epoch'] = self.epoch
                st.session_state.training_progress['train_acc'] = float(logs.get('class_output_accuracy', 0.0))
                st.session_state.training_progress['val_acc'] = float(logs.get('val_class_output_accuracy', 0.0))

                log_message(f"Epoch {self.epoch}/{epochs} - Train Acc: {logs.get('class_output_accuracy', 0.0):.4f}, Val Acc: {logs.get('val_class_output_accuracy', 0.0):.4f}")

                if st.session_state.stop_training:
                    log_message("Training stopped by user")
                    raise KeyboardInterrupt("Training stopped")

        callback = StreamlitCallback()

        # Train the model
        st.session_state.training_progress['message'] = "Training in progress..."
        training_history = trainer.train(
            train_gen,
            val_gen,
            epochs=epochs,
            callbacks=[callback]
        )

        # Fine-tune if requested
        if fine_tune and not st.session_state.stop_training:
            log_message("Starting fine-tuning...")
            st.session_state.training_progress['message'] = "Fine-tuning in progress..."
            trainer.fine_tune(
                train_gen,
                val_gen,
                epochs=5,
                unfreeze_layers=20,
                learning_rate=learning_rate / 10
            )

        # Save the model
        model_dir = Path("./models")
        model_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = model_dir / f"{timestamp}_{trainer.model_name}"
        save_dir.mkdir(exist_ok=True)

        model_path = save_dir / f"{trainer.model_name}.pkl"
        trainer.save_model(str(model_path))

        log_message(f"Model saved to {model_path}")

        # Save training summary
        summary = {
            'training_date': datetime.now().isoformat(),
            'model_name': trainer.model_name,
            'architecture': architecture,
            'num_classes': num_classes,
            'class_counts': class_counts,
            'training_config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'augmentation': augment,
                'fine_tuning': fine_tune
            },
            'final_metrics': {
                'train_accuracy': float(training_history.history.get('class_output_accuracy', [-1])[-1]),
                'val_accuracy': float(training_history.history.get('val_class_output_accuracy', [-1])[-1]),
                'train_loss': float(training_history.history.get('class_output_loss', [-1])[-1]),
                'val_loss': float(training_history.history.get('val_class_output_loss', [-1])[-1])
            } if training_history else None,
            'model_paths': {
                'model_file': str(model_path),
                'log_file': str(log_file)
            }
        }

        summary_file = save_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        log_message(f"Training summary saved to {summary_file}")

        st.session_state.training_progress['message'] = "Training completed successfully!"
        st.session_state.training_status = 'completed'

    except KeyboardInterrupt:
        st.session_state.training_progress['message'] = "Training stopped by user"
        st.session_state.training_status = 'stopped'
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        st.session_state.training_progress['message'] = error_msg
        st.session_state.training_status = 'error'
        log_message(error_msg)