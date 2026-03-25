# Use a modern PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for segmentation and Jupyter
RUN pip install --no-cache-dir \
    segmentation-models-pytorch==0.3.3 \
    albumentations==1.4.0 \
    pandas \
    matplotlib \
    tqdm \
    pillow \
    scikit-image \
    scikit-learn \
    ipywidgets \
    jupyterlab

# Copy the notebook into the container
COPY leaf_segmentation.ipynb .
# COPY run_inference.py . # Keeping this for reference if needed

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose Jupyter port
EXPOSE 8888

# Default command: Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
