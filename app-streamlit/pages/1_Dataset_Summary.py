import streamlit as st
import pandas as pd
from pathlib import Path

# Conditional imports for PostgreSQL functionality
try:
    from core.image_similarity import create_similarity_search, POSTGRESQL_AVAILABLE
except ImportError:
    create_similarity_search = None
    POSTGRESQL_AVAILABLE = False

st.set_page_config(
    page_title="Dataset Summary",
    page_icon="📊",
    layout="wide"
)

st.title("Dataset Summary")


data_dir = Path("./files/train_dataset")
class_counts = {}
total_images = 0
dataset_error = None

if data_dir.exists():
    try:
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if class_dirs:
            for class_dir in class_dirs:
                class_name = class_dir.name
                num_images = len(list(class_dir.glob('*')))
                class_counts[class_name] = num_images
                total_images += num_images
        else:
            dataset_error = "No class subdirectories found in the dataset folder."
    except Exception as exc:
        dataset_error = f"Error validating dataset: {exc}"
else:
    dataset_error = f"Dataset directory not found: {data_dir}"

with st.expander("Training Dataset Summary", expanded=True):
    if class_counts:
        avg_images = total_images / len(class_counts) if class_counts else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Classes detected", len(class_counts), help="Unique wound categories available for training.")
        with col2:
            st.metric("Total images", total_images, help="Aggregated samples across all folders.")
        with col3:
            st.metric("Avg per class", f"{avg_images:.1f}", help="Helps gauge class balance.")

        df_classes = pd.DataFrame({
            'Class': list(class_counts.keys()),
            'Images': list(class_counts.values())
        })

        # st.dataframe(df_classes, use_container_width=True)
        st.bar_chart(df_classes.set_index('Class'))
    elif dataset_error:
        st.error(f"❌ {dataset_error}")
    else:
        st.info("Upload wound imagery into subfolders under ./files/train_dataset to begin training.")



with st.expander("Database Initialization", expanded=True):
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <h2>Similarity Database</h2>
                <p class="subtitle">Persist embeddings into PostgreSQL + pgvector to unlock nearest-neighbour search.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info("Initialize once per trained model. Feature extraction runs on every image in the training corpus and stores 1536-D vectors along with class labels for rapid similarity lookup.")

    # Main action area
    st.markdown("### Database Setup")
    col_init, col_space, col_status = st.columns([2, 0.5, 2])

    with col_init:
        if 'similarity_search' in st.session_state and st.session_state.similarity_search is not None:
            db_stats = st.session_state.similarity_search.get_database_stats()
            st.success("✅ Database Ready")
        if st.button("Initialize Now", type="primary", use_container_width=True):
            if not POSTGRESQL_AVAILABLE or not create_similarity_search:
                st.error("PostgreSQL dependencies are not installed. Please install them to use this feature.")
            elif st.session_state.training_status == 'completed':
                with st.spinner("Initializing vector database... This may take a few minutes."):
                    try:
                        # Find the latest trained model
                        models_dir = Path("./models")
                        model_dirs = sorted(
                            [d for d in models_dir.iterdir() if d.is_dir()],
                            key=lambda d: d.stat().st_mtime,
                            reverse=True
                        )
                        if not model_dirs:
                            st.error("No trained models found to initialize the database.")
                        else:
                            latest_model_dir = model_dirs[0]
                            model_file = next(latest_model_dir.glob("*.pkl"), None)
                            config_file = latest_model_dir / "training_args.json"

                            if model_file and config_file.exists():
                                st.session_state.similarity_search = create_similarity_search(
                                    model_path=str(model_file),
                                    config_path=str(config_file),
                                    table_name="images_features"
                                )
                                st.success("Vector database initialized successfully!")
                                st.rerun()  # Refresh to update status
                            else:
                                st.error("Latest model is missing a .pkl file or training_args.json.")
                    except Exception as e:
                        st.error(f"Error initializing vector database: {e}")
            else:
                st.warning("Training must be completed before initializing the vector database.")


    with col_space:
        st.empty()  # Spacer

    with col_status:
        st.markdown("**Database Status**")
        if not create_similarity_search:
            st.info("Vector utilities unavailable in this environment.")
        else:
            try:
                if 'similarity_search' in st.session_state and st.session_state.similarity_search is not None:
                    db_stats = st.session_state.similarity_search.get_database_stats()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Indexed Images", db_stats.get('total_images', 'N/A'))
                    with col2:
                        st.metric("Vector Dimension", db_stats.get('vector_dimension', 'N/A'))
                else:
                    st.info("❌ Database not initialized.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Indexed Images", 0)
                    with col2:
                        st.metric("Vector Dimension", "N/A")
            except Exception as e:
                st.error(f"Could not retrieve database status: {e}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Indexed Images", "Error")
                with col2:
                    st.metric("Vector Dimension", "Error")
