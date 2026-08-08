# NVIDIA CUDA 11.8 + cuDNN 8 — L4 GPU Runtime
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0

# ─── 1. System deps ───
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    ffmpeg libglib2.0-0 libgomp1 wget curl git unzip \
    libgl1-mesa-glx libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN python3.11 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# ─── 2. PyTorch CUDA 11.8 ───
RUN python3.11 -m pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# ─── 3. NumPy + ONNX (sabitlə) ───
RUN python3.11 -m pip install --no-cache-dir "numpy==1.26.4" && \
    python3.11 -m pip install --no-cache-dir onnxruntime-gpu==1.15.1

# ─── 4. App requirements ───
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# ─── 5. Project files ───
COPY . .

# ─── 6. Qovluqlar ───
RUN mkdir -p /app/models /app/uploads /app/outputs

# ─── 7. FaceFusion — SIZIN FORK (əvvəl varsa sil) ───
RUN rm -rf /app/facefusion && \
    git clone --depth 1 https://github.com/Elsen001/facefusion.git /app/facefusion && \
    cd /app/facefusion && \
    sed -i '/^onnxruntime\b/d' requirements.txt && \
    sed -i '/^onnxruntime-gpu\b/d' requirements.txt && \
    sed -i '/^numpy\b/d' requirements.txt && \
    python3.11 -m pip install --no-cache-dir -r requirements.txt

# ─── 8. ONNX GPU yenidən möhkəmləndir ───
RUN python3.11 -m pip install --no-cache-dir --force-reinstall onnxruntime-gpu==1.15.1 && \
    python3.11 -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# ─── 9. INSwapper modeli ───
RUN mkdir -p /app/facefusion/.assets/models && \
    wget -q --timeout=60 -O /app/facefusion/.assets/models/inswapper_128.onnx \
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx"

# ─── 10. GFPGAN modeli (optional, fail olsa da davam et) ───
RUN wget -q --timeout=60 -O /app/models/GFPGANv1.4.pth \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" || \
    echo "GFPGAN optional, skipped"

# ─── 11. GPU verify ───
RUN python3.11 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" && \
    python3.11 -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers())"

EXPOSE 7860
CMD ["python", "app.py"]
