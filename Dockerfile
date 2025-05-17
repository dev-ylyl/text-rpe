ARG WORKER_CUDA_VERSION=12.6.2
FROM runpod/base:0.6.2-cuda${WORKER_CUDA_VERSION}

# 安装系统依赖和工具
RUN apt-get update -o Acquire::Retries=5 && \
    apt-get install -y --no-install-recommends git wget && \
    rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV HF_HOME=/runpod-volume \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# 安装 Python 依赖
COPY requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# ✅ 构建时预加载模型
COPY scripts/preload_models.py /app/scripts/preload_models.py
RUN python3.11 /app/scripts/preload_models.py

# 拷贝所有代码
COPY . /app
WORKDIR /app

# 启动 RunPod Worker
CMD ["python3.11", "-u", "handler.py"]