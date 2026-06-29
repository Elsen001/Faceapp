FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libglib2.0-0 libgomp1 wget git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Əvvəlcə PyTorch CPU (kiçik versiya)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Sonra digər paketlər (unzip də lazımdır)
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    flask insightface onnxruntime \
    opencv-python-headless numpy pillow \
    gfpgan basicsr facexlib

COPY . .

RUN mkdir -p models uploads outputs models/buffalo_l

# INSwapper modeli — uğursuz olarsa build dayanır (ölçü yoxlanılır)
RUN wget -q -O models/inswapper_128.onnx \
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx" \
    && test $(stat -c%s models/inswapper_128.onnx) -gt 8000000

# Buffalo_L üz aşkarlama modeli — runtime-da internet olmaya bilər,
# build zamanı əvvəlcədən endirib qoyuruq
RUN wget -q -O /tmp/buffalo_l.zip \
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" \
    && cd models/buffalo_l && unzip -o /tmp/buffalo_l.zip && rm /tmp/buffalo_l.zip

# GFPGAN modeli
RUN wget -q -O models/GFPGANv1.4.pth \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" \
    && test $(stat -c%s models/GFPGANv1.4.pth) -gt 8000000

EXPOSE 7860

CMD ["python", "app.py"]
