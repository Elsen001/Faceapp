"""
DeepFace / InsightFace tabanlı video yüz değiştirme motoru.
En yüksek kalite için INSwapper (inswapper_128.onnx) kullanılır.
"""

import os
import cv2
import numpy as np
import threading
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable

# ── InsightFace ──────────────────────────────────────────────────────────────
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

# ── Globals ──────────────────────────────────────────────────────────────────
_face_app: Optional[FaceAnalysis] = None
_swapper = None
_lock = threading.Lock()
_init_done = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
SWAPPER_PATH = os.path.join(MODEL_DIR, "inswapper_128.onnx")


def get_face_app() -> FaceAnalysis:
    """Lazy-init FaceAnalysis (detection + recognition)."""
    global _face_app, _init_done
    if _face_app is None:
        with _lock:
            if _face_app is None:
                app = FaceAnalysis(
                    name="buffalo_l",
                    root=MODEL_DIR,
                    providers=["CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=(640, 640))
                _face_app = app
                _init_done = True
    return _face_app


def get_swapper():
    """Lazy-init INSwapper."""
    global _swapper
    if _swapper is None:
        with _lock:
            if _swapper is None:
                if not os.path.exists(SWAPPER_PATH):
                    raise FileNotFoundError(
                        f"inswapper_128.onnx tapılmadı: {SWAPPER_PATH}\n"
                        "Modeli https://huggingface.co/deepinsight/inswapper "
                        "ünvanından yükləyin."
                    )
                _swapper = insightface.model_zoo.get_model(
                    SWAPPER_PATH,
                    providers=["CPUExecutionProvider"],
                )
    return _swapper


def detect_faces(img_bgr: np.ndarray):
    """Görüntüdəki bütün üzləri tap."""
    app = get_face_app()
    return app.get(img_bgr)


def swap_faces_in_frame(
    frame: np.ndarray,
    source_face,
    target_face_index: int = 0,
    all_faces: bool = False,
) -> np.ndarray:
    """
    Bir frame içindəki üz(lər)i source_face ilə əvəz et.
    all_faces=True olduqda bütün üzlər dəyişdirilir.
    """
    swapper = get_swapper()
    faces = detect_faces(frame)

    if not faces:
        return frame

    result = frame.copy()

    if all_faces:
        for face in faces:
            result = swapper.get(result, face, source_face, paste_back=True)
    else:
        # Ən böyük üzü seç (ən yaxında olan)
        faces_sorted = sorted(faces, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        if target_face_index < len(faces_sorted):
            target = faces_sorted[target_face_index]
            result = swapper.get(result, target, source_face, paste_back=True)

    return result


def extract_source_face(image_path: str):
    """Mənbə şəkildən üzü çıxar."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Şəkil oxunmadı: {image_path}")

    faces = detect_faces(img)
    if not faces:
        raise ValueError("Şəkildə üz aşkarlanmadı. Zəhmət olmasa aydın üz şəkli seçin.")

    # Ən böyük / ən yaxşı üzü qaytar
    faces_sorted = sorted(faces, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
    return faces_sorted[0]


def process_video(
    video_path: str,
    source_image_path: str,
    output_path: str,
    all_faces: bool = False,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Videodakı üzü mənbə şəkildəki üzlə əvəz et.
    Orijinal audio qorunur.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Mənbə üzü çıxar
    if progress_cb:
        progress_cb(2, "Mənbə şəkildəki üz analiz edilir...")
    source_face = extract_source_face(source_image_path)

    # 2. Videoyu aç
    if progress_cb:
        progress_cb(5, "Video açılır...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video açılmadı.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        total_frames = 9999  # stream üçün fallback

    # 3. Müvəqqəti video faylı (audiysuz)
    tmp_video = tempfile.mktemp(suffix="_noaudio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (width, height))

    # 4. Frame-ləri işlə
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        swapped = swap_faces_in_frame(frame, source_face, all_faces=all_faces)
        writer.write(swapped)
        frame_idx += 1

        if progress_cb and frame_idx % 5 == 0:
            pct = 10 + int((frame_idx / max(total_frames, 1)) * 80)
            progress_cb(
                min(pct, 89),
                f"Frame işlənir: {frame_idx}/{total_frames}",
            )

    cap.release()
    writer.release()

    # 5. Orijinal audioyu geri qoş
    if progress_cb:
        progress_cb(90, "Audio birləşdirilir...")

    _merge_audio(video_path, tmp_video, output_path)

    # Təmizlə
    try:
        os.remove(tmp_video)
    except OSError:
        pass

    if progress_cb:
        progress_cb(100, "Tamamlandı!")

    return output_path


def _merge_audio(original_video: str, processed_video: str, output: str):
    """
    FFmpeg ilə orijinal audio + işlənmiş video birləşdir.
    Audio yoxdursa, yalnız videoyu kopyala.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", processed_video,
        "-i", original_video,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Audio yoxdur – yalnız videoyu çevir
        cmd_novid = [
            "ffmpeg", "-y",
            "-i", processed_video,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            output,
        ]
        subprocess.run(cmd_novid, capture_output=True, check=True)


def check_model_available() -> bool:
    return os.path.exists(SWAPPER_PATH)


def get_model_info() -> dict:
    return {
        "swapper_path": SWAPPER_PATH,
        "swapper_exists": os.path.exists(SWAPPER_PATH),
        "model_dir": MODEL_DIR,
        "insightface_version": insightface.__version__,
    }
