FROM quay.io/jupyter/minimal-notebook:python-3.12

# Build-time argument for requirements path - defaults for building from jupyter dir
ARG REQUIREMENTS_PATH=requirements.txt

USER root
RUN apt-get update && \
    apt-get install -y default-jdk ca-certificates-java libopenblas-dev \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgthread-2.0-0 \
    poppler-utils tesseract-ocr libtesseract-dev libleptonica-dev \
    build-essential && \
    apt-get clean && \
    update-ca-certificates -f

    # Install R and IRkernel
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends r-base \
#     r-base-dev\
#     libcurl4-openssl-dev \
#     libssl-dev \
#     libxml2-dev && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/* && \
#     R -e "install.packages('IRkernel', repos='http://cran.rstudio.com/')" && \
#     R -e "IRkernel::installspec(user = FALSE)"

# Set JAVA_HOME environment variable 
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
# Set PATH environment variable
ENV PATH=$JAVA_HOME/bin:$PATH

# Copy and install requirements as root first
COPY ${REQUIREMENTS_PATH} /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Switch to non-root user after installation
USER ${NB_UID}
