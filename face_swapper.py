"""
DeepFace / InsightFace tabanlı video yüz değiştirme motoru.
Model: inswapper_128.onnx  (FaceFusion GitHub Releases)
       inswapper_128_fp16.onnx  (alternativ)
"""

import os
import cv2
import numpy as np
import threading
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable

import insightface
from insightface.app import FaceAnalysis

_face_app: Optional[FaceAnalysis] = None
_swapper = None
_lock = threading.Lock()

MODEL_DIR = Path(__file__).parent / "models"

# FP16 variantını da qəbul et
def _find_swapper_path() -> Optional[str]:
    for name in ("inswapper_128.onnx", "inswapper_128_fp16.onnx"):
        p = MODEL_DIR / name
        if p.exists() and p.stat().st_size > 10_000_000:
            return str(p)
    return None


def get_face_app() -> FaceAnalysis:
    global _face_app
    if _face_app is None:
        with _lock:
            if _face_app is None:
                app = FaceAnalysis(
                    name="buffalo_l",
                    root=str(MODEL_DIR),
                    providers=["CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=(640, 640))
                _face_app = app
    return _face_app


def get_swapper():
    global _swapper
    if _swapper is None:
        with _lock:
            if _swapper is None:
                path = _find_swapper_path()
                if not path:
                    raise FileNotFoundError(
                        "Swap modeli tapılmadı!\n"
                        "Zəhmət olmasa aşağıdakı əmri çalışdırın:\n"
                        "  python download_models.py\n\n"
                        "Əl ilə yükləmək üçün:\n"
                        "https://github.com/facefusion/facefusion-assets"
                        "/releases/download/models-3.0.0/inswapper_128.onnx\n"
                        "→ models/inswapper_128.onnx"
                    )
                _swapper = insightface.model_zoo.get_model(
                    path,
                    providers=["CPUExecutionProvider"],
                )
    return _swapper


def detect_faces(img_bgr: np.ndarray):
    return get_face_app().get(img_bgr)


def swap_faces_in_frame(frame: np.ndarray, source_face, all_faces: bool = False) -> np.ndarray:
    swapper = get_swapper()
    faces = detect_faces(frame)
    if not faces:
        return frame

    result = frame.copy()
    if all_faces:
        for face in faces:
            result = swapper.get(result, face, source_face, paste_back=True)
    else:
        biggest = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]
        result = swapper.get(result, biggest, source_face, paste_back=True)
    return result


def extract_source_face(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Şəkil oxunmadı: {image_path}")
    faces = detect_faces(img)
    if not faces:
        raise ValueError("Şəkildə üz aşkarlanmadı. Aydın, tam üzlü şəkil seçin.")
    return sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]


def process_video(
    video_path: str,
    source_image_path: str,
    output_path: str,
    all_faces: bool = False,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> str:
    MODEL_DIR.mkdir(exist_ok=True)

    if progress_cb:
        progress_cb(2, "Mənbə üzü analiz edilir...")
    source_face = extract_source_face(source_image_path)

    if progress_cb:
        progress_cb(5, "Video açılır...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video açılmadı.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 9999
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = tempfile.mktemp(suffix="_noaudio.mp4")
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(swap_faces_in_frame(frame, source_face, all_faces))
        idx += 1
        if progress_cb and idx % 5 == 0:
            pct = 10 + int((idx / max(total, 1)) * 80)
            progress_cb(min(pct, 89), f"Frame: {idx}/{total}")

    cap.release()
    writer.release()

    if progress_cb:
        progress_cb(90, "Audio birləşdirilir...")
    _merge_audio(video_path, tmp, output_path)

    try:
        os.remove(tmp)
    except OSError:
        pass

    if progress_cb:
        progress_cb(100, "Tamamlandı!")
    return output_path


def _merge_audio(orig: str, processed: str, out: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", processed, "-i", orig,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest", out,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        subprocess.run([
            "ffmpeg", "-y", "-i", processed,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", out,
        ], capture_output=True, check=True)


def check_model_available() -> bool:
    return _find_swapper_path() is not None


def get_model_info() -> dict:
    path = _find_swapper_path()
    return {
        "swapper_path": path,
        "swapper_exists": path is not None,
        "model_dir": str(MODEL_DIR),
        "insightface_version": insightface.__version__,
        "download_cmd": "python download_models.py",
        "manual_url": (
            "https://github.com/facefusion/facefusion-assets"
            "/releases/download/models-3.0.0/inswapper_128.onnx"
        ),
    }
