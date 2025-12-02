import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add core to path
sys.path.append(str(Path(__file__).parent.parent))

# Conditional imports for PostgreSQL functionality
try:
    from core.image_similarity import create_similarity_search, POSTGRESQL_AVAILABLE
    from core.database import get_vector_store
    DB_AVAILABLE = True
except ImportError:
    create_similarity_search = None
    get_vector_store = None
    POSTGRESQL_AVAILABLE = False
    DB_AVAILABLE = False

# Add functions path for database queries
sys.path.append(str(Path(__file__).parent.parent.parent / "app-streamlit"))
try:
    from functions.database import execute_query, test_connection
    QUERY_AVAILABLE = True
except ImportError:
    execute_query = None
    test_connection = None
    QUERY_AVAILABLE = False

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

    st.info("**Initialize once per trained model.** Feature extraction runs on every image in the training corpus and stores 1536-D vectors along with class labels for rapid similarity lookup.")
    
    # Check database status directly
    db_vector_count = 0
    db_vector_dim = None
    db_initialized = False
    db_error = None
    
    if DB_AVAILABLE:
        try:
            vector_store = get_vector_store()
            db_vector_count = vector_store.get_vector_count("images_features")
            db_vector_dim = vector_store.get_table_vector_dim("images_features")
            db_initialized = db_vector_count > 0
        except Exception as e:
            db_error = str(e)

    # Display status in prominent metrics
    st.markdown("### Database Status")
    status_col1, status_col2 = st.columns(2)

    
    with status_col2:
        if db_error and "does not exist" not in db_error:
            st.error(f"⚠️ Database connection error")
        elif not DB_AVAILABLE:
            st.info("ℹ️ Database dependencies not available")
    
    # Metrics
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(
            "Indexed Images", 
            db_vector_count if db_initialized else 0,
            help="Total number of image embeddings stored in the vector database"
        )
    with metric_col2:
        st.metric(
            "Vector Dimension", 
            db_vector_dim if db_vector_dim else "N/A",
            help="Dimensionality of feature vectors (1536 for standard embeddings)"
        )

    # Display sample embeddings if database is initialized
    if db_initialized and QUERY_AVAILABLE:
        st.markdown("### Sample Embeddings")
        st.markdown("View actual embedding vectors stored in the database:")
        
        try:
            query = "SELECT id, label, model_name, embedding FROM images_features WHERE embedding IS NOT NULL ORDER BY ID DESC LIMIT 10"
            result = execute_query(query, return_df=True)
            if result is not None and len(result) > 0:
                data = []
                for idx, row in result.iterrows():
                    embedding = np.array(row['embedding'])
                    try:
                        if embedding.ndim > 0 and len(embedding) > 0:
                            embedding_preview = f"[{', '.join(f'{x:.4f}' for x in embedding[:5])}, ..., {', '.join(f'{x:.4f}' for x in embedding[-5:])}]"
                        else:
                            embedding_preview = str(row['embedding'])
                    except Exception as e:
                        embedding_preview = f"Error: {e}"
                    
                    data.append({
                        'ID': row['id'],
                        'Label': row['label'],
                        'Model Name': row['model_name'],
                        'Embedding Preview': embedding_preview
                    })
                
                df = pd.DataFrame(data)
                st.markdown('<div style="overflow-x: auto; width: 100%;">', unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No embeddings found in database")
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")

    st.markdown("---")
    
    # Training requirement check
    models_dir = Path("./models")
    available_models = []
    
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                model_file = next(model_dir.glob("*.pkl"), None) or next(model_dir.glob("*.keras"), None)
                config_file = model_dir / "training_args.json"
                if model_file and config_file.exists():
                    available_models.append({
                        'name': model_dir.name,
                        'model_file': model_file,
                        'config_file': config_file
                    })
    
    if not available_models:
        st.warning("⚠️ **Training must be completed before initializing the vector database.**")
        st.info("💡 Train a model in the **Training** page first, then return here to initialize the database.")
    else:
        # Model selection
        st.markdown("### Initialize Vectors")
        
        if len(available_models) == 1:
            selected_model = available_models[0]
            st.info(f"📦 Found trained model: **{selected_model['name']}**")
        else:
            model_names = [m['name'] for m in available_models]
            selected_name = st.selectbox(
                "Select model to initialize",
                model_names,
                help="Choose which trained model to use for feature extraction"
            )
            selected_model = next(m for m in available_models if m['name'] == selected_name)
        
        # Action buttons
        col_action1, col_action2 = st.columns(2)
        
        with col_action1:
            if st.button("Regenerate Vectors", type="primary", use_container_width=True, disabled=not DB_AVAILABLE):
                if not DB_AVAILABLE:
                    st.error("❌ Database dependencies not available. Install psycopg2 and pgvector.")
                else:
                    with st.spinner("🔄 Generating vectors from training dataset... This may take several minutes."):
                        try:
                            st.info(f"⚙️ Using model: **{selected_model['name']}**")
                            st.info("📊 Extracting features from all training images...")
                            
                            # Initialize similarity search which will create the connection
                            similarity_search = create_similarity_search(
                                model_path=str(selected_model['model_file']),
                                config_path=str(selected_model['config_file']),
                                table_name="images_features"
                            )
                            
                            # Get all image paths from training dataset
                            from core.data_processing import get_image_paths_by_class
                            class_images = get_image_paths_by_class(str(data_dir))
                            
                            # Flatten to lists
                            all_image_paths = []
                            all_labels = []
                            for class_name, paths in class_images.items():
                                all_image_paths.extend(paths)
                                all_labels.extend([class_name] * len(paths))
                            
                            st.info(f"📸 Processing {len(all_image_paths)} images...")
                            
                            # Store features in database
                            similarity_search.store_image_features(all_image_paths, all_labels, batch_size=32)
                            
                            st.success(f"✅ Successfully generated {len(all_image_paths)} vectors!")
                            st.balloons()
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Vector generation failed: {e}")
                            st.exception(e)
        
        with col_action2:
            if st.button("🗑️ Clear Database", type="secondary", use_container_width=True, disabled=not db_initialized):
                if st.warning("⚠️ This will delete all vectors from the database. Are you sure?"):
                    try:
                        vector_store = get_vector_store()
                        vector_store.clear_table("images_features")
                        st.success("✅ Database cleared successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to clear database: {e}")
        
        # Information box
        if db_initialized:
            st.success("✅ **Database is ready!** You can now use the Similarity Search page to find similar wound images.")
        else:
            st.info("💡 **Next step:** Click 'Regenerate Vectors' to populate the database with image embeddings.")
