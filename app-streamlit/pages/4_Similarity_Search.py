import streamlit as st
import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
from PIL import Image
import time

# Ensure core package is importable
sys.path.append(str(Path(__file__).parent.parent))

# Try to import create_similarity_search helper
try:
    from core.image_similarity import create_similarity_search
    CORE_AVAILABLE = True
except Exception:
    create_similarity_search = None
    CORE_AVAILABLE = False

# Try to import Ollama client
try:
    from core.ollama_client import OllamaClient
    OLLAMA_AVAILABLE = True
except Exception:
    OllamaClient = None
    OLLAMA_AVAILABLE = False

st.set_page_config(page_title="Image Similarity Search", page_icon="🔍", layout="wide")

st.title("Image Similarity Search")

if 'similarity_search' not in st.session_state:
    st.session_state.similarity_search = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# Model selection
st.header("Model Selection")
models_dir = Path(__file__).parent.parent / "models"
available_models = []
if models_dir.exists():
    for model_dir in models_dir.glob("*/"):
        if model_dir.is_dir():
            model_files = list(model_dir.glob("*.keras")) + list(model_dir.glob("*.h5")) + list(model_dir.glob("*.pkl"))
            config_file = model_dir / "training_args.json"
            summary_file = model_dir / "training_summary.json"
            if model_files:
                info = {
                    'name': model_dir.name,
                    'model_files': model_files,
                    'config_file': config_file if config_file.exists() else None,
                    'summary': None
                }
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            info['summary'] = json.load(f)
                    except Exception:
                        info['summary'] = None
                available_models.append(info)

model_count = len(available_models)
active_search = st.session_state.similarity_search
db_stats = {}
if active_search is not None:
        try:
                stats = active_search.get_database_stats()
                if isinstance(stats, dict):
                        db_stats = stats
        except Exception:
                db_stats = {}

vector_count = db_stats.get('total_images', 0)
vector_dimension = db_stats.get('vector_dimension', 1536)
table_name = db_stats.get('table', db_stats.get('table_name', 'images_features'))
active_model_name = getattr(getattr(active_search, 'trainer', None), 'model_name', 'Not initialized') if active_search else 'Not initialized'

with st.expander("Encoder Setup", expanded=True):

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active session", "Ready" if active_search else "Not initialised", help="Streaming similarity search" if active_search else "Select a model and activate to begin")
    with col2:
        st.metric("Indexed vectors", vector_count, help="Rows currently stored in the vector table.")
    with col3:
        st.metric("Vector dimension", vector_dimension, help="Embedding length used for similarity.")

    if not available_models:
        st.info("No trained models found in ./models. Initialise a model in the Model Summary page or train a new one.")
    else:
        model_names = [m['name'] for m in available_models]
        selected_model_name = model_names[0] if len(model_names) == 1 else st.selectbox("Select trained model", model_names)
        selected_model_info = next((m for m in available_models if m['name'] == selected_model_name), None)

        if selected_model_info:
            model_files = selected_model_info.get('model_files', [])
            if len(model_files) == 1:
                chosen_model_file = model_files[0]
            else:
                model_file_names = [p.name for p in model_files]
                chosen_name = st.selectbox("Choose model file", model_file_names)
                chosen_model_file = next(p for p in model_files if p.name == chosen_name)

            summary = selected_model_info.get('summary') or {}
            has_config = selected_model_info.get('config_file') is not None

            st.metric("Model artefact", chosen_model_file.name, "Selected checkpoint for activation.")

            if summary:
                with st.expander("Training summary"):
                    st.json(summary)

        
            disabled = not (CORE_AVAILABLE and has_config)
            activate_label = "Activate Encoder" if CORE_AVAILABLE else "Activate Encoder (unavailable)"
            if st.button(activate_label, disabled=disabled, type="primary", use_container_width=True):
                if not CORE_AVAILABLE:
                    st.error("❌ Core similarity utilities not available in this environment.")
                    st.info("💡 Initialise on the Model Summary page or ensure dependencies are installed.")
                elif not has_config:
                    st.error("❌ Config file missing. Provide training_args.json to initialise the encoder.")
                    st.info("💡 The config file should be located next to the model checkpoint.")
                else:
                    with st.spinner("Activating similarity search engine..."):
                        try:
                            st.info(f"⚙️ Loading {selected_model_name} and connecting to vector database...")
                            
                            similarity_search = create_similarity_search(
                                model_path=str(chosen_model_file),
                                config_path=str(selected_model_info['config_file']),
                                table_name="images_features"
                            )
                            st.session_state.similarity_search = similarity_search
                            st.success("✅ Similarity search activated successfully!")
                            st.rerun()
                            
                        except Exception as err:
                            st.error(f"❌ Activation failed: {err}")

                if disabled and CORE_AVAILABLE:
                    st.caption("💡 Attach training_args.json next to the model file to enable activation.")
                elif disabled:
                    st.caption("💡 Install pgvector dependencies to enable activation in this environment.")

            
            st.info("Need data? Ensure the vector database already contains embeddings. Use the Training page to populate it after a successful run.")

with st.expander("Upload & Analyse", expanded=True):
    st.markdown(
        """
        <div class="section-heading">
          <div>
            <h2>Upload & Analyse</h2>
            <p class="subtitle">Submit a wound image, compute embeddings, and explore reinforced matches.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Choose an image to find similar wounds", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", width=320)

        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)
        temp_image_path = temp_dir / f"search_{int(time.time())}_{uploaded_file.name}"

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(temp_image_path, 'JPEG', quality=95)
        st.session_state['last_query_image'] = str(temp_image_path)

        if st.button("🔍 Find Similar Images", key="search_button", type="primary", use_container_width=True):
            if st.session_state.similarity_search is None:
                st.error("❌ Similarity search not initialized. Activate a model first.")
                st.info("💡 Use the 'Activate Encoder' button above to initialize the search engine.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.info("🧠 Extracting features from query image...")
                    progress_bar.progress(0.3)
                    
                    status_text.info("🔍 Computing similarity scores across vector database...")
                    progress_bar.progress(0.6)
                    
                    results = st.session_state.similarity_search.get_similar_images_with_class_analysis(
                        query_image_path=str(temp_image_path), top_k=10
                    )
                    st.session_state.search_results = results
                    
                    progress_bar.progress(1.0)
                    status_text.empty()
                    progress_bar.empty()
                    
                    similar_count = len(results.get('similar_images', []))
                    st.metric("Search complete", f"{similar_count} similar images retrieved")
                    
                    st.success("✅ Similarity search completed successfully!")
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Search failed: {e}")
                    st.info("💡 Ensure the database contains indexed vectors and the connection is active.")
    else:
        st.caption("Upload a wound image to trigger the similarity pipeline.")

if st.session_state.search_results:
    results = st.session_state.search_results
    query_image_path = st.session_state.get('last_query_image')

    with st.expander("Similarity Insights", expanded=True):
        st.markdown(
            """
            <div class="section-heading">
              <div>
                <h2>Similarity Insights</h2>
                <p class="subtitle">Review the closest matches, class probabilities, and per-class similarity aggregates.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        top_match = None
        similar_images = results.get('similar_images', [])
        if similar_images:
            top_match = similar_images[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Top match class", top_match.get('class', 'N/A'), "Rank 1 result")
            with col2:
                st.metric("Top similarity", f"{top_match.get('similarity_score', 0):.4f}", "Cosine similarity")
            with col3:
                st.metric("Matches returned", len(similar_images), "Top-k results")

        if query_image_path and st.session_state.similarity_search is not None:
            try:
                dual_result = st.session_state.similarity_search.trainer.predict_dual(query_image_path)
                class_probs = dual_result['class'][0] if dual_result['class'].ndim > 1 else dual_result['class']
                predicted_idx = int(np.argmax(class_probs))
                st.info(f"Predicted class index: {predicted_idx} (confidence: {class_probs[predicted_idx]:.3f})")
                with st.expander("📊 Class probabilities", expanded=False):
                    class_probs_df = pd.DataFrame(
                        {'Class': list(range(len(class_probs))), 'Probability': class_probs}
                    ).sort_values('Probability', ascending=False)
                    st.dataframe(class_probs_df, use_container_width=True)
                    st.bar_chart(class_probs_df.set_index('Class'))
            except Exception:
                pass

        if similar_images:
            df = pd.DataFrame({
                'Rank': range(1, len(similar_images) + 1),
                'Class': [img.get('class') for img in similar_images],
                'Filename': [img.get('filename') for img in similar_images],
                'Similarity Score': [f"{img.get('similarity_score', 0):.4f}" for img in similar_images]
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

        if results.get('class_similarities'):
            class_sim_df = pd.DataFrame({
                'Class': list(results['class_similarities'].keys()),
                'Average Similarity': list(results['class_similarities'].values())
            }).sort_values('Average Similarity', ascending=False)
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.subheader("Class similarity analysis")
            st.dataframe(class_sim_df, use_container_width=True)
            st.bar_chart(class_sim_df.set_index('Class'))

with st.expander("AI Medical Analysis", expanded=True):
    st.markdown(
        """
        <div class="section-heading">
          <div>
            <h2>AI Medical Analysis</h2>
            <p class="subtitle">Get AI-powered medical insights and treatment recommendations based on similar cases.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not OLLAMA_AVAILABLE:
        st.warning("⚠️ Ollama client not available. Install dependencies or ensure Ollama service is running.")
    else:
        # Create summary string from results
        summary_lines = []
        for i, img in enumerate(similar_images, 1):
            summary_lines.append(f"{i}\t{img.get('class', 'Unknown')}\t{img.get('filename', 'N/A')}\t{img.get('similarity_score', 0):.4f}")
        summary_text = "\n".join(summary_lines)

        # Create class similarities summary
        class_similarities = results.get('class_similarities', {})
        class_summary = ", ".join([f"{cls} ({score*100:.2f}%)" for cls, score in sorted(class_similarities.items(), key=lambda x: x[1], reverse=True)])

        # st.markdown("**Similarity Summary:**")
        # st.code(summary_text, language="text")
        st.markdown(f"**Class Similarities:** {class_summary}")

        # Model selection (hidden - using default)
        selected_model = "gpt-oss:20b"
        api_url = "http://hyper07.ddns.net:11434"
        
        if st.button("Get AI Medical Opinion", type="primary", use_container_width=True):
            with st.spinner("Consulting AI medical assistant..."):
                try:
                    ollama_client = OllamaClient(api_url=api_url)
                    
                    prompt = f"""As a medical AI assistant specializing in wound care, analyze these similar wound images and their class similarities to provide treatment recommendations:

                        Based on the similarity search results showing:
                        {class_summary}

                        What treatment do you recommend for the most prevalent wound types identified?

                        Please provide evidence-based treatment recommendations based on standard wound care protocols."""

                    response = ollama_client.generate_text_stream(
                        prompt=prompt,
                        model=selected_model,
                        options={"temperature": 0.3, "num_predict": 2000}
                    )
                    
                    st.success("✅ AI analysis completed!")
                    st.markdown("### AI Medical Opinion")
                    st.write_stream(response)
                    
                except Exception as e:
                    st.error(f"❌ Failed to get AI analysis: {e}")
                    if "404" in str(e):
                        st.info("💡 **404 Error**: The selected model may not be available. Try 'Check Available Models' first, or install the model with: `ollama pull <model_name>`")
                    elif "Connection refused" in str(e):
                        st.info("💡 **Connection Error**: Ollama service may not be running or accessible. Check your Ollama configuration.")
                    else:
                        st.info("💡 **Tip**: Try a different model or ensure Ollama is properly configured.")

