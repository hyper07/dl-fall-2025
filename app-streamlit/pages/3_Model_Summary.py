import streamlit as st
import json
from pathlib import Path
from datetime import datetime
import sys

# Add core to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core.image_similarity import create_similarity_search
except Exception:
    create_similarity_search = None

st.set_page_config(page_title="Model Summary", page_icon="📚", layout="wide")

st.title("Model Summary")

if 'similarity_search' not in st.session_state:
    st.session_state.similarity_search = None

models_dir = Path("./models")
available_models = []

if models_dir.exists():
    for model_dir in models_dir.glob("*/"):
        if model_dir.is_dir():
            model_files = list(model_dir.glob("*.keras")) or list(model_dir.glob("*.h5")) or list(model_dir.glob("*.pkl"))
            config_file = model_dir / "training_args.json"
            training_summary_file = model_dir / "training_summary.json"
            evaluation_summary_file = model_dir / "evaluation_summary.json"
            if model_files:
                model_info = {
                    'name': model_dir.name,
                    'model_file': model_files[0],
                    'config_file': config_file if config_file.exists() else None,
                    'training_summary': None,
                    'evaluation_summary': None,
                    'updated_at': model_dir.stat().st_mtime
                }
                if training_summary_file.exists():
                    try:
                        with open(training_summary_file, 'r') as f:
                            model_info['training_summary'] = json.load(f)
                    except Exception:
                        model_info['training_summary'] = None

                if evaluation_summary_file.exists():
                    try:
                        with open(evaluation_summary_file, 'r') as f:
                            model_info['evaluation_summary'] = json.load(f)
                    except Exception:
                        model_info['evaluation_summary'] = None

                available_models.append(model_info)

total_models = len(available_models)

if not available_models:
        st.warning("No trained models found in ./models. Train a model or mount the models directory.")
        st.stop()

models_with_summary = sum(1 for m in available_models if m['training_summary'])
models_with_config = sum(1 for m in available_models if m['config_file'])
models_with_evaluation = sum(1 for m in available_models if m['evaluation_summary'])
latest_info = max(available_models, key=lambda m: m['updated_at'])
latest_timestamp = datetime.fromtimestamp(latest_info['updated_at']).strftime("%Y-%m-%d %H:%M")

with st.container(border=True):
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        st.metric("Models tracked", total_models)
    with col2:
        st.metric("Configs ready", models_with_config)
    with col3:
        st.metric("Summaries curated", models_with_summary)
    with col4:
        st.metric("Evaluations completed", models_with_evaluation)

model_names = [m['name'] for m in available_models]

with st.expander("Model Catalogue", expanded=True):
    selected = st.selectbox("Select model to inspect", model_names, key="model_selector")
    selected_info = next((m for m in available_models if m['name'] == selected), None)

    if selected_info:
        has_summary = bool(selected_info['training_summary'])
        has_config = bool(selected_info['config_file'])
        has_evaluation = bool(selected_info['evaluation_summary'])
        summary_metrics = (selected_info['training_summary'] or {}).get('final_metrics', {})
        evaluation_metrics = (selected_info['evaluation_summary'] or {}).get('metrics', {})
        class_counts = (selected_info['training_summary'] or {}).get('class_counts', {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", selected, help=f"Primary artefact: {selected_info['model_file'].name}")
        with col2:
            st.metric("Summary", "Available" if has_summary else "Missing", help="Final metrics captured" if has_summary else "unavailable")
        with col3:
            st.metric("Evaluation", "Available" if has_evaluation else "Missing", help="Test set evaluation completed" if has_evaluation else "Run evaluation to see test performance")

        if has_summary:
            col1, col2, col3 = st.columns(3)
            with col1:
                train_acc = summary_metrics.get('train_accuracy', 'N/A')
                if isinstance(train_acc, (int, float)):
                    train_acc = f"{train_acc:.4f}"
                st.metric("Train accuracy", train_acc, "Final epoch")
            with col2:
                val_acc = summary_metrics.get('val_accuracy', 'N/A')
                if isinstance(val_acc, (int, float)):
                    val_acc = f"{val_acc:.4f}"
                st.metric("Validation accuracy", val_acc, "Generalisation snapshot")
            with col3:
                train_loss = summary_metrics.get('train_loss', 'N/A')
                val_loss = summary_metrics.get('val_loss', 'N/A')
                if isinstance(train_loss, (int, float)):
                    train_loss = f"{train_loss:.4f}"
                if isinstance(val_loss, (int, float)):
                    val_loss = f"{val_loss:.4f}"
                st.metric("Loss profile", f"{train_loss} / {val_loss}", "Train / validation")

            # Display evaluation metrics if available
with st.expander("Evaluation Metrics", expanded=False):

    if has_evaluation:
        # Create metrics table with metrics as columns
        import pandas as pd
        metrics_data = {
            'Accuracy': [f"{evaluation_metrics.get('accuracy', 0):.4f}"],
            'Precision': [f"{evaluation_metrics.get('precision', 0):.4f}"],
            'Recall': [f"{evaluation_metrics.get('recall', 0):.4f}"],
            'F1-Score': [f"{evaluation_metrics.get('f1_score', 0):.4f}"]
        }


        metrics_df = pd.DataFrame(metrics_data)

        # Center the table and make it look better
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            # Use HTML table for better control over styling
            table_html = f"""
            <table style="width: 100%; border-collapse: collapse; margin: 0 auto; font-size: 16px;">
                <thead>
                    <tr style="background-color: #f0f2f6;">
                        <th style="padding: 12px; text-align: center; border: 1px solid #ddd; font-weight: bold;">Accuracy</th>
                        <th style="padding: 12px; text-align: center; border: 1px solid #ddd; font-weight: bold;">Precision</th>
                        <th style="padding: 12px; text-align: center; border: 1px solid #ddd; font-weight: bold;">Recall</th>
                        <th style="padding: 12px; text-align: center; border: 1px solid #ddd; font-weight: bold;">F1-Score</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 12px; text-align: center; border: 1px solid #ddd; font-weight: bold;">{evaluation_metrics.get('accuracy', 0):.4f}</td>
                        <td style="padding: 12px; text-align: center; border: 1px solid #ddd; font-weight: bold;">{evaluation_metrics.get('precision', 0):.4f}</td>
                        <td style="padding: 12px; text-align: center; border: 1px solid #ddd; font-weight: bold;">{evaluation_metrics.get('recall', 0):.4f}</td>
                        <td style="padding: 12px; text-align: center; border: 1px solid #ddd; font-weight: bold;">{evaluation_metrics.get('f1_score', 0):.4f}</td>
                    </tr>
                </tbody>
            </table>
            <br />
            """

            st.markdown(table_html, unsafe_allow_html=True)

with st.expander("Confusion Matrix", expanded=False):
    # Display confusion matrix if available
    confusion_matrix_path = Path("./models") / selected / "confusion_matrix.png"
    if confusion_matrix_path.exists():
        st.image(str(confusion_matrix_path), caption=f"Confusion Matrix - {selected} Model", use_container_width=True)

    else:
        st.info("💡 No evaluation results available. Run evaluation to see test performance metrics.")

# Display evaluation details
with st.expander("Evaluation Details", expanded=False):
    if selected_info['evaluation_summary']:
        st.json(selected_info['evaluation_summary'])




# Import Vector Embeddings button
if st.session_state.similarity_search:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if st.button("Import Vector Embeddings", key="import_vectors", type="secondary", use_container_width=True):
        with st.spinner("Generating and inserting vector embeddings for the dataset..."):
            try:
                # Scan dataset directory
                dataset_dir = Path("./files/train_dataset")
                image_paths = []
                class_labels = []
                
                if dataset_dir.exists():
                    for class_dir in dataset_dir.iterdir():
                        if class_dir.is_dir():
                            class_name = class_dir.name
                            for ext in ['*.jpg', '*.jpeg', '*.png']:
                                for img_path in class_dir.glob(ext):
                                    image_paths.append(str(img_path))
                                    class_labels.append(class_name)
                    
                    if image_paths:
                        st.info(f"Found {len(image_paths)} images across {len(set(class_labels))} classes. Processing...")
                        # Generate and store features
                        st.session_state.similarity_search.store_image_features(image_paths, class_labels)
                        st.success(f"✅ Successfully imported vector embeddings for {len(image_paths)} images!")
                        st.info("🔍 Vector embeddings are now stored in the database and ready for similarity search.")
                    else:
                        st.warning("No images found in ./files/train_dataset. Ensure the dataset is mounted.")
                else:
                    st.error("Dataset directory ./files/train_dataset not found.")
            except Exception as e:
                st.error(f"❌ Failed to import vector embeddings: {e}")
                st.info("💡 Check database connectivity and ensure the model is compatible.")
elif has_config and create_similarity_search:
    st.caption("💡 Activate similarity search first to enable vector embedding import.")
