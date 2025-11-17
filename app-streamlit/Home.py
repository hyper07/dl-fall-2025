try:
    import streamlit as st
except Exception:
    # Minimal streamlit stub for environments without streamlit (tests/tools).
    class _DummyColumn:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def markdown(self, *args, **kwargs):
            return None
        def button(self, *args, **kwargs):
            return False

    class _StubSpinner:
        def __init__(self, *_a, **_k):
            pass
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc, tb):
            return False

    class _StreamlitStub:
        def set_page_config(self, *a, **k):
            return None
        def error(self, *a, **k):
            print('STREAMLIT ERROR:', *a)
        def warning(self, *a, **k):
            print('STREAMLIT WARNING:', *a)
        def info(self, *a, **k):
            print('STREAMLIT INFO:', *a)
        def markdown(self, *a, **k):
            return None
        def button(self, *a, **k):
            return False
        def spinner(self, *a, **k):
            return _StubSpinner()
        def success(self, *a, **k):
            print('STREAMLIT SUCCESS:', *a)
        def switch_page(self, *a, **k):
            return None
        def columns(self, n):
            return [_DummyColumn() for _ in range(n)]

    st = _StreamlitStub()
import sys
import os
from pathlib import Path

st.set_page_config(
    layout="wide",
    page_title="Wound Detection Platform",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# Ensure local app-streamlit directory and project root are first on sys.path
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from utils import initialize_workspace
except Exception as e:
    # If local utils cannot be imported, provide a lightweight fallback
    initialize_workspace = lambda: None
    # Provide a clearer message for common shared-lib issues (libGL)
    err_str = str(e)
    if 'libGL' in err_str or 'libglib' in err_str or 'GLIBC' in err_str:
        st.error(
            "Could not import local utils.initialize_workspace due to a system graphics library error:\n"
            f"{err_str}\n\n"
            "If you're running in a Docker/Linux environment, install the system package providing libGL (for Debian/Ubuntu):\n"
            "  apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev\n\n"
            "Alternatively, use the headless OpenCV Python wheel by installing `opencv-python-headless` instead of `opencv-python` to avoid GUI deps."
        )
    else:
        st.warning(f"Could not import local utils.initialize_workspace: {e}")

# Import functions.database defensively (may rely on DB libs not present in all environments)
try:
    from functions.database import SELECTED_COLUMNS
    import functions.database as fs_db  # for DB helpers
except Exception as e:
    SELECTED_COLUMNS = []
    fs_db = None
    st.warning(f"Could not import functions.database: {e}")

# Initialize workspace path and imports
initialize_workspace()

# Ensure project root is on sys.path so `core` can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Check for vector table and offer restore if backups exist
try:
    tables_df = fs_db.get_all_tables(use_host=True)
    table_names = [str(t).lower() for t in tables_df['table_name'].tolist()] if tables_df is not None and not tables_df.empty else []
except Exception as e:
    table_names = []
    st.warning(f"Database connection check failed: {e}")

images_table_missing = 'images_features' not in table_names
backup_dir = Path(project_root) / 'backup'

try:
    # Attempt to import restore utilities from core (fallback to local file check if import fails)
    from core.database import restore_database, list_backups
    core_restore_available = True
except Exception:
    restore_database = None
    list_backups = None
    core_restore_available = False

available_backups = []
if core_restore_available:
    try:
        # Pass the resolved backup directory to the core helper so it looks in the
        # same place this module will use for the filesystem fallback.
        available_backups = list_backups(str(backup_dir)) or []
    except Exception:
        available_backups = []

# Fallback: check filesystem backup folder(s). Prefer the repository backup dir,
# but also look at the container-common path '/backup' in case the compose mapping
# uses that inside the container. Keep newest-first ordering to match
# core.database.list_backups.
def _collect_backups_from_path(path: Path):
    if not path.exists():
        return []
    files = [str(p) for p in path.glob('*.backup')]
    return sorted(files, key=lambda x: Path(x).stat().st_mtime, reverse=True)

if not available_backups:
    # First try the project-local backup directory
    available_backups = _collect_backups_from_path(backup_dir)

    # If none found, try the conventional `/backup` mount inside containers
    if not available_backups:
        available_backups = _collect_backups_from_path(Path('/backup'))

if images_table_missing and available_backups:
    st.warning("Similarity vector table `images_features` not present in database. Detected backup files in ./backup.")
    st.info(f"Found {len(available_backups)} backup(s). Latest: {Path(available_backups[0]).name}")

    # AUTO_RESTORE_FROM_BACKUP env var controls automatic restore behavior.
    # Default: enabled for environments that want automatic recovery from existing backups.
    auto_restore = os.getenv("AUTO_RESTORE_FROM_BACKUP", "true").lower() in ("1", "true", "yes")

    latest_backup = available_backups[0]
    if auto_restore:
        # Perform automatic restore immediately (non-interactive). Keep same restore flow.
        with st.spinner(f"Automatically restoring database from {Path(latest_backup).name}..."):
            try:
                if core_restore_available and restore_database is not None:
                    restore_database(latest_backup)
                else:
                    # Try to import core.database dynamically if it wasn't available before
                    try:
                        from core.database import restore_database as _restore
                        _restore(latest_backup)
                    except Exception as err:
                        raise RuntimeError(f"Could not restore: {err}")

                st.success("Database restore completed automatically. You may need to reload the app to refresh DB status.")
            except Exception as e:
                st.error(f"Automatic database restore failed: {e}")
    else:
        # Fallback: keep manual restore button available when auto-restore is disabled
        if st.button("Restore database from latest backup"):
            with st.spinner(f"Restoring database from {Path(latest_backup).name}..."):
                try:
                    if core_restore_available and restore_database is not None:
                        restore_database(latest_backup)
                    else:
                        # Try to import core.database dynamically if it wasn't available before
                        try:
                            from core.database import restore_database as _restore
                            _restore(latest_backup)
                        except Exception as err:
                            raise RuntimeError(f"Could not restore: {err}")

                    st.success("Database restore completed. You may need to reload the app to refresh DB status.")
                except Exception as e:
                    st.error(f"Database restore failed: {e}")

# Header & footer are rendered by the router (streamlit_app.py) when using st.navigation
# CSS is loaded globally from styles/app.css in the router
# Fallback: load CSS here if page is run directly
try:
    css_path = os.path.join(os.path.dirname(__file__), "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass
# Resolve a robust project root by walking up and looking for repository markers
def find_project_root(start_path: Path = None):
    """Walk up from start_path (default: this file) and look for common repo markers.

    Markers checked (any present): 'files' directory, 'models' directory, '.git', 'README.md'.
    Returns the first path that contains any marker, otherwise falls back to the provided
    `project_root` variable or the current working directory.
    """
    # start from the directory containing the file (not the file path itself)
    start = (Path(start_path) if start_path else Path(__file__)).resolve()
    if start.is_file():
        start = start.parent
    git_candidate = None
    files_models_candidate = None
    readme_candidate = None
    # iterate from the start up to filesystem root
    for parent in [start] + list(start.parents):
        try:
            if (parent / '.git').exists():
                # immediate winner if this is a git repo root
                return parent
        except Exception:
            pass
        try:
            if (parent / 'files').exists() and (parent / 'models').exists():
                # strong signal for the project root; keep this but continue in case .git exists higher
                if files_models_candidate is None:
                    files_models_candidate = parent
        except Exception:
            pass
        try:
            if (parent / 'README.md').exists() and readme_candidate is None:
                readme_candidate = parent
        except Exception:
            pass

    # Prefer .git (already handled), then files+models, then README, then fallbacks
    if files_models_candidate:
        return files_models_candidate
    if readme_candidate:
        return readme_candidate
    try:
        return Path(project_root)
    except Exception:
        return Path.cwd()

# Choose a reliable base root for path calculations
base_root = find_project_root()

# Count classes and images in the canonical training dataset. Use a
# case-insensitive suffix check and be resilient to common image formats.
dataset_dir = base_root / "files" / "train_dataset"
class_count = 0
image_count = 0
image_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
if dataset_dir.exists():
    for class_dir in sorted(dataset_dir.iterdir()):
        if class_dir.is_dir():
            class_count += 1
            image_count += sum(1 for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in image_suffixes)

# Also report augmented dataset counts (if present) so the UI can show both
aug_dataset_dir = base_root / "files" / "train_dataset_augmented"
aug_class_count = 0
aug_image_count = 0
if aug_dataset_dir.exists():
    for class_dir in sorted(aug_dataset_dir.iterdir()):
        if class_dir.is_dir():
            aug_class_count += 1
            aug_image_count += sum(1 for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in image_suffixes)

# Detect trained model runs. Prefer directories that contain a model file
# (common extensions: .keras, .h5, .pt, .pth). Use model-file mtime to pick
# the latest model rather than directory mtime where possible.
models_dir = base_root / "models"
trained_models_count = 0
latest_model_name = "No runs yet"
if models_dir.exists():
    model_dirs = [d for d in sorted(models_dir.iterdir()) if d.is_dir()]
    # Directories that contain at least one model artifact
    model_dirs_with_artifact = []
    model_file_suffixes = {'.keras', '.h5', '.pt', '.pth'}
    for d in model_dirs:
        if any(f.is_file() and f.suffix.lower() in model_file_suffixes for f in d.iterdir()):
            model_dirs_with_artifact.append(d)

    # If we found dirs with artifacts, count those; otherwise fall back to any subdirs
    if model_dirs_with_artifact:
        trained_models_count = len(model_dirs_with_artifact)
    else:
        trained_models_count = len(model_dirs)

    # Find the latest model directory by newest model-file mtime first, then by dir mtime
    latest_dir = None
    latest_mtime = 0
    for d in (model_dirs_with_artifact or model_dirs):
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in model_file_suffixes:
                mtime = f.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_dir = d
        # if we didn't find a model file inside this dir, consider dir mtime as fallback
        if latest_dir is None:
            # use directory modification time as a last resort
            try:
                m = d.stat().st_mtime
                if m > latest_mtime:
                    latest_mtime = m
                    latest_dir = d
            except Exception:
                pass

    if latest_dir:
        latest_model_name = latest_dir.name

backup_count = len(available_backups)
db_status_label = "Ready" if not images_table_missing else "Missing"
db_status_state = "completed" if not images_table_missing else "idle"

hero_html = f"""
<div class=\"page-hero\">
    <div class=\"page-hero__content\">
        <div>
            <span class=\"eyebrow\">Wound Intelligence</span>
            <h1>Unified training & discovery cockpit</h1>
            <p>Drive end-to-end wound classification workflows with curated dashboards, model orchestration, and pgvector-powered search.</p>
            <div class=\"chip-row\" style=\"margin-top:1.35rem;\">
                <span class=\"chip\">Streamlit experience</span>
                <span class=\"chip\">CNN automation</span>
                <span class=\"chip\">Vector analytics</span>
            </div>
        </div>
        <div class=\"page-hero__metrics\">
            <div class=\"hero-metric\">
                <span class=\"label\">Dataset classes</span>
                <span class=\"value\">{class_count}</span>
                    <span class="subtext">{image_count} images (base); {aug_image_count} images (aug)</span>
            </div>
            <div class=\"hero-metric\">
                <span class=\"label\">Trained models</span>
                <span class=\"value\">{trained_models_count}</span>
            </div>
            <div class=\"hero-metric\">
                <span class=\"label\">Vector table</span>
                <span class=\"value\">{db_status_label}</span>
            </div>
        </div>
    </div>
</div>
"""

st.markdown(hero_html, unsafe_allow_html=True)

status_html = f"""
<div class=\"status-grid\">
    <div class=\"status-chip\" data-status=\"{'completed' if class_count else 'idle'}\">
        <span class=\"label\">Training corpus</span>
        <span class=\"value\">{class_count} classes</span>
        <span class=\"subtext\">{image_count} images discovered in files/train_dataset.</span>
    </div>
        <div class=\"status-chip\" data-status=\"{'completed' if aug_class_count else 'idle'}\">
            <span class=\"label\">Augmented corpus</span>
            <span class=\"value\">{aug_class_count} classes</span>
            <span class=\"subtext\">{aug_image_count} images discovered in files/train_dataset_augmented.</span>
        </div>
    <div class=\"status-chip\">
        <span class=\"label\">Latest model</span>
        <span class=\"value\">{latest_model_name}</span>
        <span class=\"subtext\">Manage checkpoints from the Model Summary page.</span>
    </div>
    <div class=\"status-chip\" data-status=\"{db_status_state}\">
        <span class=\"label\">pgvector backups</span>
        <span class=\"value\">{backup_count}</span>
        <span class=\"subtext\">Restore options detected in /backup.</span>
    </div>
</div>
"""

st.markdown(status_html, unsafe_allow_html=True)


nav_items = [
        {
                "title": "CNN Training",
                "description": "Configure architectures, monitor learning curves, and publish embeddings.",
                "button": "Open Training",
                "page": "pages/1_Training.py",
                "key": "home_nav_training"
        },
        {
                "title": "Model Registry",
                "description": "Review checkpoints, summaries, and activate similarity search.",
                "button": "Open Model Summary",
                "page": "pages/2_Model_Summary.py",
                "key": "home_nav_models"
        },
        {
                "title": "Similarity Search",
                "description": "Query the wound atlas with vector search and class analytics.",
                "button": "Open Similarity Search",
                "page": "pages/3_Similarity_Search.py",
                "key": "home_nav_similarity"
        }
]

nav_cols = st.columns(len(nav_items))
for col, item in zip(nav_cols, nav_items):
        with col:
                st.markdown(
                        f"""
                        <div class=\"nav-card\">
                            <h4>{item['title']}</h4>
                            <p>{item['description']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                )
                if st.button(item['button'], key=item['key'], use_container_width=True):
                        st.switch_page(item['page'])

st.markdown('</div>', unsafe_allow_html=True)

