import streamlit as st
import pandas as pd
import json
import time
import threading
import html
import os
from pathlib import Path
from datetime import datetime

# Conditional imports for PostgreSQL functionality
try:
    from core.image_similarity import create_similarity_search, POSTGRESQL_AVAILABLE
except ImportError:
    create_similarity_search = None
    POSTGRESQL_AVAILABLE = False

from core.model_utils import CNNTrainer

# Page config
st.set_page_config(
    page_title="CNN Training",
    page_icon="",
    layout="wide"
)

st.title("CNN Model Training")

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

status = st.session_state.training_status
progress_state = st.session_state.training_progress
total_epochs = progress_state.get('total_epochs', 0) or 0
current_epoch = progress_state.get('epoch', 0) or 0
progress_pct = min(current_epoch / total_epochs, 1.0) if total_epochs else 0.0
train_acc = progress_state.get('train_acc', 0.0) or 0.0
val_acc = progress_state.get('val_acc', 0.0) or 0.0
status_descriptions = {
        'idle': 'Configure a run and launch the training pipeline when you are ready.',
        'running': 'Training in progress. Metrics and logs update live as epochs finish.',
        'completed': 'Training completed. Review metrics and publish to the vector database.',
        'error': 'An error interrupted training. Inspect the log console for details.',
        'stopped': 'Training stopped gracefully. You can adjust parameters and resume.',
}.get(status, 'Orchestrate and observe the entire model lifecycle from one cockpit.')

with st.container(border=True):

    st.markdown("Configure architectures, monitor convergence, and sync embeddings to your vector database without leaving this dashboard.")

    # Progress bar for epochs
    st.progress(progress_pct, text=f"Epoch Progress: {progress_pct * 100:.0f}%")
        
    col1, col2 = st.columns([2,1])
    with col1:
        status_color = {
            'idle': '🟡',
            'running': '🟢',
            'completed': '✅',
            'error': '❌',
            'stopped': '⏹️'
        }.get(status, '🟡')
        st.metric("Status", f"{status_color} {status.upper()}")
        
    with col2:
        # Status with color coding

        # Accuracy with better labeling
        st.metric("Accuracy (Train/Val)", f"{train_acc:.3f} / {val_acc:.3f}")

# col1, col2, col3 = st.columns(3)
# with col1:
#     st.metric("📋 Training Status", status.capitalize(), help=status_descriptions)
# with col2:
#     st.metric("🔢 Epochs", f"{current_epoch} / {total_epochs if total_epochs else '—'}", f"{progress_pct * 100:.0f}% complete")
# with col3:
#     st.metric("💬 Current Message", progress_state.get('message', 'Awaiting updates...')[:20] + ('...' if len(progress_state.get('message', '')) > 20 else ''), help=progress_state.get('message', 'Awaiting updates...'))

with st.expander("Configuration", expanded=True):

    col1, col2 = st.columns(2)

    with col1:
        data_dir_input = st.text_input(
            "Dataset Directory",
            value="./files/train_dataset",
            help="Path to the training dataset directory"
        )

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
            value=False,
            help="Use data augmentation during training"
        )

        fine_tune = st.checkbox(
            "Fine-tuning",
            value=False,
            help="Perform fine-tuning after initial training"
        )

        if fine_tune:
            col1, col2 = st.columns(2)
            with col1:
                fine_tune_epochs = st.slider(
                    "Fine-tuning Epochs",
                    min_value=1,
                    max_value=20,
                    value=5,
                    step=1,
                    help="Number of fine-tuning epochs"
                )
                unfreeze_layers = st.slider(
                    "Unfreeze Layers",
                    min_value=5,
                    max_value=100,
                    value=20,
                    step=5,
                    help="Number of layers to unfreeze for fine-tuning"
                )
            with col2:
                fine_tune_lr = st.selectbox(
                    "Fine-tuning Learning Rate",
                    [0.0001, 0.00001, 0.000001],
                    index=1,
                    help="Learning rate for fine-tuning (typically lower than initial)"
                )
        else:
            fine_tune_epochs = 5
            unfreeze_layers = 20
            fine_tune_lr = learning_rate / 10


with st.expander("Training Controls", expanded=True):
    col_start, col_stop, col_clear = st.columns(3)

    def start_training():
        if st.session_state.training_status == 'running':
            st.error("Training is already running")
            return

        st.session_state.training_status = 'running'
        st.session_state.training_logs = []
        total_training_epochs = epochs + (fine_tune_epochs if fine_tune else 0)
        st.session_state.training_progress = {
            'epoch': 0,
            'total_epochs': total_training_epochs,
            'train_acc': 0.0,
            'val_acc': 0.0,
            'message': 'Initializing training...'
        }
        st.session_state.stop_training = False

        # Start training in background thread
        st.session_state.training_thread = threading.Thread(
            target=train_model_background,
            args=(architecture, epochs, batch_size, learning_rate, use_augmentation, fine_tune, data_dir_input, fine_tune_epochs, unfreeze_layers, fine_tune_lr)
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
        if st.button("Start Training", type="primary", use_container_width=True):
            start_training()

    with col_stop:
        if st.button("Stop Training", type="secondary", use_container_width=True):
            stop_training()

    with col_clear:
        if st.button("Clear Logs", use_container_width=True):
            clear_logs()

with st.expander("Progress", expanded=True):

    st.progress(progress_pct)
    progress_message = st.session_state.training_progress.get('message', '')
    if progress_message:
        if st.session_state.training_status == 'error':
            st.error(progress_message)
        else:
            st.info(progress_message)
            
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Train accuracy", f"{train_acc:.3f}", "Last epoch")
    with col2:
        st.metric("Validation accuracy", f"{val_acc:.3f}", "Last epoch")

with st.expander("Logs", expanded=True):

    log_placeholder = st.empty()

    if st.session_state.training_logs:
        log_text = "".join(st.session_state.training_logs[-100:])
        escaped_logs = html.escape(log_text)
        log_placeholder.markdown(
            f'<div class="log-console"><pre>{escaped_logs}</pre></div>',
            unsafe_allow_html=True
        )
    else:
        log_placeholder.info("No training logs yet. Click 'Start Training' to begin.")

# Auto-refresh
if st.session_state.training_status == 'running':
    time.sleep(1)
    st.rerun()

# Training summary (when completed)
if st.session_state.training_status in ['completed', 'stopped']:
    with st.expander("Training Summary", expanded=True):
        st.markdown(
            """
            <div class="section-heading">
              <div>
                <h2>Latest Training Snapshot</h2>
                <p class="subtitle">Review key metrics from the most recent run and export the report.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        try:
            models_dir = Path("./models")
            if models_dir.exists():
                model_dirs = sorted(
                    [d for d in models_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime,
                    reverse=True
                )
                if model_dirs:
                    latest_model_dir = model_dirs[0]
                    summary_file = latest_model_dir / "training_summary.json"
                    if summary_file.exists():
                        with open(summary_file, 'r') as f:
                            summary_data = json.load(f)

                        st.metric("Model", summary_data.get('model_name', 'N/A'))
                        
                        final_metrics = summary_data.get('final_metrics', {})
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Final Train Accuracy", f"{final_metrics.get('train_accuracy', 0):.3f}")
                            st.metric("Final Train Loss", f"{final_metrics.get('train_loss', 0):.3f}")
                        with col2:
                            st.metric("Final Validation Accuracy", f"{final_metrics.get('val_accuracy', 0):.3f}")
                            st.metric("Final Validation Loss", f"{final_metrics.get('val_loss', 0):.3f}")

                        if 'class_counts' in summary_data:
                            st.subheader("Class Distribution")
                            st.bar_chart(summary_data['class_counts'])

                        st.download_button(
                            label="Download Training Summary",
                            data=json.dumps(summary_data, indent=2),
                            file_name=f"{latest_model_dir.name}_summary.json",
                            mime="application/json"
                        )
                    else:
                        st.info("No training summary found for the latest model.")
                else:
                    st.info("No trained models found in the './models' directory.")
            else:
                st.warning("Models directory not found.")
        except Exception as exc:
            st.error(f"Error loading training summary: {exc}")


data_dir = Path("./files/train_dataset")
class_counts = {}
total_images = 0
dataset_error = None


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

def train_model_background(architecture, epochs, batch_size, learning_rate, augment, fine_tune, data_dir_str, fine_tune_epochs=5, unfreeze_layers=20, fine_tune_lr=0.00001):
    """Background training function"""
    try:
        data_dir = Path(data_dir_str)
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
            feature_dim=1536,
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

            # Create callback for fine-tuning
            class FineTuneCallback:
                def __init__(self, initial_epochs):
                    self.initial_epochs = initial_epochs

                def on_epoch_end(self, epoch, logs=None):
                    self.epoch = self.initial_epochs + epoch + 1
                    st.session_state.training_progress['epoch'] = self.epoch
                    st.session_state.training_progress['train_acc'] = float(logs.get('class_output_accuracy', 0.0))
                    st.session_state.training_progress['val_acc'] = float(logs.get('val_class_output_accuracy', 0.0))

                    log_message(f"Fine-tuning Epoch {self.epoch}/{epochs + fine_tune_epochs} - Train Acc: {logs.get('class_output_accuracy', 0.0):.4f}, Val Acc: {logs.get('val_class_output_accuracy', 0.0):.4f}")

            fine_tune_callback = FineTuneCallback(epochs)

            trainer.fine_tune(
                train_gen,
                val_gen,
                epochs=fine_tune_epochs,
                unfreeze_layers=unfreeze_layers,
                learning_rate=fine_tune_lr,
                callbacks=[fine_tune_callback]
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
                'fine_tuning': fine_tune,
                'fine_tune_epochs': fine_tune_epochs if fine_tune else 0,
                'unfreeze_layers': unfreeze_layers if fine_tune else 0,
                'fine_tune_learning_rate': fine_tune_lr if fine_tune else 0
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