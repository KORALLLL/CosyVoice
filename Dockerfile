# CosyVoice3 TTS Server Dockerfile
# Supports backends: native, vllm, trtllm, trtllm-serve

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libsndfile1 \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/CosyVoice

# Copy CosyVoice source code
COPY . /opt/CosyVoice/

# Install Python dependencies
# Note: tensorrt-cu12 versions in requirements.txt may need manual pinning.
# We install core deps first, then handle optional heavy deps separately.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    conformer==0.3.2 \
    fastapi==0.115.6 \
    fastapi-cli==0.0.4 \
    gdown==5.1.0 \
    grpcio==1.57.0 \
    grpcio-tools==1.57.0 \
    hydra-core==1.3.2 \
    HyperPyYAML==1.2.3 \
    inflect==7.3.1 \
    librosa==0.10.2 \
    lightning==2.2.4 \
    matplotlib==3.7.5 \
    modelscope==1.20.0 \
    networkx==3.1 \
    numpy==1.26.4 \
    omegaconf==2.3.0 \
    onnx==1.16.0 \
    onnxruntime-gpu==1.18.0 \
    openai-whisper==20231117 \
    protobuf==4.25 \
    pyarrow==18.1.0 \
    pydantic==2.7.0 \
    pyworld==0.3.4 \
    rich==13.7.1 \
    soundfile==0.12.1 \
    tensorboard==2.14.0 \
    torch==2.3.1 \
    torchaudio==2.3.1 \
    transformers==4.51.3 \
    x-transformers==2.11.24 \
    uvicorn==0.30.0 \
    wetext==0.0.4 \
    wget==3.2 \
    diffusers==0.29.0 \
    && rm -rf /root/.cache/pip

# Install Prometheus client for metrics (used by server)
RUN pip install --no-cache-dir prometheus-client

# Optional: Install vLLM (for vllm backend)
# Uncomment if you plan to use vllm backend
# RUN pip install --no-cache-dir vllm==0.5.0

# Optional: Install TensorRT-LLM (for trtllm / trtllm-serve backends)
# TensorRT-LLM installation is environment-specific; install via your preferred method.
# Example:
# RUN pip install --no-cache-dir tensorrt_llm==0.11.0

# Ensure fastapi runtime is in PYTHONPATH
ENV PYTHONPATH=/opt/CosyVoice/runtime/python/fastapi:/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}

# Expose server port
EXPOSE 8000

# Entrypoint
ENTRYPOINT ["/opt/CosyVoice/runtime/python/fastapi/start_server.sh"]
