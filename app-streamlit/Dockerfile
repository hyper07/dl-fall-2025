FROM python:3.12-slim

WORKDIR /usr/src/app

# Build-time arguments for requirements paths - defaults for building from app-streamlit dir
ARG REQUIREMENTS_PATH=requirements.txt
ARG REQUIREMENTS_DEV_PATH=requirements-dev.txt

# dont write pyc files
# dont buffer to stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY ${REQUIREMENTS_PATH} /usr/src/app/requirements.txt
COPY ${REQUIREMENTS_DEV_PATH} /usr/src/app/requirements-dev.txt

# RUN apt-get update && apt-get install -y libgomp1 libglib2.0-0
RUN apt-get update && apt-get install -y libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgthread-2.0-0 poppler-utils tesseract-ocr libtesseract-dev libleptonica-dev build-essential

RUN apt-get update && apt-get -y install --no-install-recommends \
    curl sqlite3 libsqlite3-dev gcc 

# dependencies
RUN pip install --upgrade pip setuptools wheel \
    # && set -x HDF5_DIR "$(brew --prefix hdf5)" \
    && pip3 install --only-binary h5py h5py \
    && pip install -r requirements.txt \
    && pip install -r requirements-dev.txt \
    && rm -rf /root/.cache/pip

# COPY ./app /usr/src/app
