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

# Install Python dependencies
# Using a fixed version for consistency where possible
RUN pip install --no-cache-dir \
    segmentation-models-pytorch==0.3.3 \
    albumentations==1.4.0 \
    pandas \
    matplotlib \
    tqdm \
    pillow \
    scikit-image \
    scikit-learn

# Create directories for data and output
RUN mkdir -p /app/sample_images /app/output_masks

# Copy inference script and weights
# Note: Users should ideally mount their own images and weights
COPY run_inference.py .
# COPY best_model.pth . 

# Set environmental variables
ENV PYTHONUNBUFFERED=1

# Default command: run inference
# Expects images in /app/sample_images and weights at /app/best_model.pth
CMD ["python", "run_inference.py"]
