FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    ffmpeg libglib2.0-0 libgomp1 wget git unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch GPU
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models uploads outputs models/buffalo_l

# INSwapper modeli
RUN wget -q -O models/inswapper_128.onnx \
    "https://github.com/facefusion/facefusion-assets/releases/download/models-2.2.0/inswapper_128.onnx" \
    && test $(stat -c%s models/inswapper_128.onnx) -gt 8000000

# Buffalo_L
RUN wget -q -O /tmp/buffalo_l.zip \
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" \
    && cd models/buffalo_l && unzip -o /tmp/buffalo_l.zip && rm /tmp/buffalo_l.zip

# GFPGAN
RUN wget -q -O models/GFPGANv1.4.pth \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.8/GFPGANv1.4.pth" \
    && test $(stat -c%s models/GFPGANv1.4.pth) -gt 8000000

EXPOSE 7860

CMD ["python", "app.py"]
