# TTS API Server Docker Image
# KugelAudio 7B - Official Implementation
# Optimized for NVIDIA GPU (RTX 5060 Ti)

# Gebruik een nieuwere CUDA base image
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_HOME=/app/.cache/torch
ENV HF_HOME=/app/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv (recommended by KugelAudio)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install PyTorch nightly with CUDA 12.8 support (needed for sm_120 / Blackwell)
RUN pip install --no-cache-dir --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install KugelAudio from GitHub (not yet on PyPI)
RUN pip install --no-cache-dir git+https://github.com/Kugelaudio/kugelaudio-open.git

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create cache directories
RUN mkdir -p /app/.cache/torch /app/.cache/huggingface

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the server
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
