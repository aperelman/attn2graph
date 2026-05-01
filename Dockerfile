FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace stack
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    datasets \
    huggingface-hub \
    sentencepiece \
    safetensors

# Models cache will be mounted from host — just set the env var
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

COPY . /app/

CMD ["python", "-u", "aga_script.py"]
