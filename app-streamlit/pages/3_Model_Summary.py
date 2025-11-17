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
            if model_files:
                model_info = {
                    'name': model_dir.name,
                    'model_file': model_files[0],
                    'config_file': config_file if config_file.exists() else None,
                    'training_summary': None,
                    'updated_at': model_dir.stat().st_mtime
                }
                if training_summary_file.exists():
                    try:
                        with open(training_summary_file, 'r') as f:
                            model_info['training_summary'] = json.load(f)
                    except Exception:
                        model_info['training_summary'] = None

                available_models.append(model_info)

total_models = len(available_models)

if not available_models:
        st.warning("No trained models found in ./models. Train a model or mount the models directory.")
        st.stop()

models_with_summary = sum(1 for m in available_models if m['training_summary'])
models_with_config = sum(1 for m in available_models if m['config_file'])
latest_info = max(available_models, key=lambda m: m['updated_at'])
latest_timestamp = datetime.fromtimestamp(latest_info['updated_at']).strftime("%Y-%m-%d %H:%M")

with st.container(border=True):
    col1, col2, col3= st.columns([1,1,1])
    with col1:
        st.metric("Models tracked", total_models)
    with col2:
        st.metric("Configs ready", models_with_config)
    with col3:
        st.metric("Summaries curated", models_with_summary)

model_names = [m['name'] for m in available_models]

with st.expander("Model Catalogue", expanded=True):
    selected = st.selectbox("Select model to inspect", model_names, key="model_selector")
    selected_info = next((m for m in available_models if m['name'] == selected), None)

    if selected_info:
        has_summary = bool(selected_info['training_summary'])
        has_config = bool(selected_info['config_file'])
        summary_metrics = (selected_info['training_summary'] or {}).get('final_metrics', {})
        class_counts = (selected_info['training_summary'] or {}).get('class_counts', {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", selected, help=f"Primary artefact: {selected_info['model_file'].name}")
        with col2:
            st.metric("Summary", "Available" if has_summary else "Missing", help="Final metrics captured" if has_summary else "unavailable")
        with col3:
            st.metric("Config", "Ready" if has_config else "Missing", help="Required for vector extraction.")

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

            # if class_counts:
            #     top_classes = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)[:6]
            #     class_markup = ''.join([
            #         f'<span class="chip">{label}: {count}</span>' for label, count in top_classes
            #     ])
            #     st.markdown(
            #         f"<div class=\"chip-row\" style=\"margin-bottom:1rem;\">{class_markup}</div>",
            #         unsafe_allow_html=True
            #     )

            with st.expander("Training summary"):
                st.json(selected_info['training_summary'])
        else:
            st.info("No training summary available for this model.")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        init_col1, init_col2 = st.columns(2)
      
        button_disabled = not has_config or not create_similarity_search
        if st.button("Activate Similarity Search", key="init_similarity", disabled=button_disabled, type="primary", use_container_width=True):
            if not has_config:
                st.error("❌ Model does not have a config file (training_args.json). Cannot initialize similarity search.")
                st.info("💡 Ensure training_args.json is saved alongside the model checkpoint.")
            elif not create_similarity_search:
                st.error("❌ Core similarity utilities not available: ensure dependencies are installed.")
                st.info("💡 Install pgvector and PostgreSQL dependencies to enable this feature.")
            else:
                with st.spinner("Activating similarity search engine..."):
                    try:
                        st.info("⚙️ Loading model checkpoint and initializing feature encoder...")
                        
                        similarity_search = create_similarity_search(
                            model_path=str(selected_info['model_file']),
                            config_path=str(selected_info['config_file']),
                            table_name="images_features"
                        )
                        st.session_state.similarity_search = similarity_search
                        
                        st.metric("Status", "Active", "Encoder ready for queries")
                        st.metric("Model", selected, "Loaded checkpoint")
                        
                        st.success("✅ Similarity search activated successfully!")
                        st.info("🔍 Navigate to the Similarity Search page to query wound images.")
                    except Exception as e:
                        st.error(f"❌ Activation failed: {e}")
                        st.info("💡 Verify database connectivity and model compatibility.")
        
        if button_disabled and create_similarity_search:
            st.caption("💡 Provide a training_args.json next to the model to enable activation.")
        elif button_disabled:
            st.caption("💡 Install pgvector dependencies to enable this action.")

     
        st.info("How it works: We load the classifier, rebuild preprocessing from the config, extract embeddings for every training image, and persist them into the configured PostgreSQL table.")

