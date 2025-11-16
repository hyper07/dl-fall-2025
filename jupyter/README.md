# Capstone AML Jupyter Lab Environment

This is the Jupyter Lab environment for the Capstone AML Project (Columbia University SPS × Société Générale).

## Overview

The Jupyter service provides an interactive development environment for:
- Data analysis and exploration
- Machine learning model development
- AML feature engineering and modeling
- Notebook-based experimentation

## Structure

```
jupyter/
├── workspace/               # Jupyter workspace (mounted from host)
├── Documents/               # Shared documents
├── Downloads/               # Downloads folder
├── Dockerfile               # Jupyter container
└── requirements.txt         # Python dependencies
```

## Running the Environment

### Using Docker Compose (Recommended)

From the project root:

```bash
# Start all services
docker-compose up --build

# Or start only Jupyter
docker-compose up --build dl-jupyter
```

Access Jupyter Lab at: http://localhost:48888/lab/workspaces/auto-1

### Local Development

If you prefer to run Jupyter locally:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Jupyter Lab:
```bash
jupyter lab
```

## Features

- **Workspace Mounting**: The `workspace/` directory is mounted, allowing persistent storage of notebooks
- **Pre-installed Libraries**: Common data science and ML libraries are included
- **AML Integration**: Access to AML toolkit and datasets
- **Shared Environment**: Consistent environment across team members

## Notebooks

The workspace contains various AML analysis notebooks including:
- EDA notebooks for data exploration
- Feature engineering and modeling scripts
- Risk analysis and visualization notebooks

## Dependencies

See `requirements.txt` for Python dependencies.
