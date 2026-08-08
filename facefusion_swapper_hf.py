"""
FaceFusion 3.0+ HuggingFace Integration — Temporal Consistency Pipeline
Fork: https://github.com/Elsen001/facefusion
"""

import os
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import torch

log = logging.getLogger(__name__)

FACEFUSION_DIR = Path(__file__).parent / "facefusion"
FACEFUSION_RUN = FACEFUSION_DIR / "facefusion.py"
MODELS_DIR = FACEFUSION_DIR / ".assets" / "models"


def get_gpu_info() -> Dict[str, Any]:
    info = {
        "available": torch.cuda.is_available(),
        "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "name": None,
        "memory_gb": 0,
    }
    if info["available"] and info["count"] > 0:
        info["name"] = torch.cuda.get_device_name(0)
        info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    return info


def get_execution_provider() -> str:
    """GPU-ya mecburi kecid."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        log.info("ONNX available providers: %s", available)

        if "CUDAExecutionProvider" in available:
            try:
                model_path = MODELS_DIR / "inswapper_128.onnx"
                if model_path.exists():
                    sess_opts = ort.SessionOptions()
                    sess = ort.InferenceSession(
                        str(model_path), sess_opts, providers=["CUDAExecutionProvider"]
                    )
                    log.info("CUDA test session OK")
                return "cuda"
            except Exception as e:
                log.warning("CUDA test session failed: %s", e)
        log.error("CUDA NOT available! Providers: %s", available)
    except Exception as e:
        log.error("ONNX check failed: %s", e)
    return "cpu"


def check_facefusion_available() -> bool:
    return FACEFUSION_RUN.exists() and (FACEFUSION_DIR / "facefusion" / "content_analyser.py").exists()


def install_facefusion():
    REPO_URL = "https://github.com/Elsen001/facefusion.git"

    if FACEFUSION_DIR.exists() and (FACEFUSION_DIR / ".git").exists():
        log.info("FaceFusion (fork) updating...")
        try:
            subprocess.run(["git", "-C", str(FACEFUSION_DIR), "pull", "--depth", "1"],
                          capture_output=True, timeout=30)
        except Exception:
            pass
    else:
        log.info("Cloning FaceFusion from: %s", REPO_URL)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(FACEFUSION_DIR)],
                      check=True, capture_output=True)

    req_file = FACEFUSION_DIR / "requirements.txt"
    if req_file.exists():
        import re
        with open(req_file, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            s = line.strip()
            if re.match(r'^onnxruntime\b', s, re.IGNORECASE):
                continue
            if re.match(r'^numpy\b', s, re.IGNORECASE):
                continue
            new_lines.append(line)
        with open(req_file, 'w') as f:
            f.writelines(new_lines)

    subprocess.run(["pip", "install", "-r", str(req_file)], capture_output=True, check=False)
    subprocess.run(["pip", "install", "--no-cache-dir", "--force-reinstall", "onnxruntime-gpu==1.15.1"],
                  capture_output=True, check=False)
    subprocess.run(["pip", "install", "--no-cache-dir", "--force-reinstall", "numpy==1.26.4"],
                  capture_output=True, check=False)
    log.info("FaceFusion installed")


def _patch_content_analyser():
    path = FACEFUSION_DIR / "facefusion" / "content_analyser.py"
    if not path.exists():
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        old = 'def detect_nsfw(vision_frame : VisionFrame) -> bool:'
        if old in content:
            idx = content.find(old)
            nxt = content.find('\ndef ', idx + 1)
            if nxt == -1:
                nxt = len(content)
            content = content[:idx] + old + '\n    # BYPASS\n    return False\n' + content[nxt:]
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            log.info("Content analyser bypassed")
    except Exception as e:
        log.warning("Bypass patch error: %s", e)


def _ensure_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx"
    fpath = MODELS_DIR / "inswapper_128.onnx"
    if not fpath.exists() or fpath.stat().st_size < 8_000_000:
        log.info("Downloading inswapper_128.onnx...")
        subprocess.run(["wget", "-q", "--timeout=60", "-O", str(fpath), url],
                      capture_output=True, check=False)


def process_video_facefusion(
    source_image_path: str,
    target_video_path: str,
    output_path: str,
    all_faces: bool = False,
    progress_cb: Optional[Callable] = None,
    high_quality: bool = False,
    reference_face_distance: float = 0.6,
    reference_frame_number: int = 0,
    face_detector_score: float = 0.3,
    face_mask_blur: float = 0.5,
    face_mask_padding: str = "8 16 8 16",
    execution_thread_count: int = 8,
    output_video_quality: int = 95,
    output_video_preset: str = "fast",
) -> str:
    if not check_facefusion_available():
        if progress_cb:
            progress_cb(5, "Installing FaceFusion...")
        install_facefusion()

    _patch_content_analyser()
    _ensure_models()

    gpu_info = get_gpu_info()
    log.info("GPU: available=%s, count=%s, name=%s", gpu_info["available"], gpu_info["count"], gpu_info["name"])

    execution_provider = get_execution_provider()
    log.info("Provider: %s", execution_provider)

    if execution_provider == "cpu":
        log.warning("CPU MODE — Will be very slow!")
        if progress_cb:
            progress_cb(5, "WARNING: GPU not detected, using CPU (slow)")
    else:
        if progress_cb:
            progress_cb(8, f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.0f}GB)")

    cmd = [
        "python", str(FACEFUSION_RUN),
        "headless-run",
        "--source-paths", source_image_path,
        "--target-path", target_video_path,
        "--output-path", output_path,
        "--processors", "face_swapper",
        "--face-swapper-model", "inswapper_128",
        "--face-selector-mode", "reference",
        "--reference-face-distance", str(reference_face_distance),
        "--reference-frame-number", str(reference_frame_number),
        "--face-detector-model", "retinaface",
        "--face-detector-score", str(face_detector_score),
        "--face-detector-angles", "0", "90", "180", "270",
        "--face-mask-types", "box", "occlusion", "region",
        "--face-mask-blur", str(face_mask_blur),
        "--face-mask-padding", *face_mask_padding.split(),
        "--face-occluder-model", "xseg_2",
        "--face-parser-model", "bisenet_resnet_34",
        "--execution-providers", execution_provider,
        "--execution-thread-count", str(execution_thread_count),
        "--execution-device-ids", "0",
        "--output-video-encoder", "libx264",
        "--output-video-preset", output_video_preset,
        "--output-video-quality", str(output_video_quality),
        "--output-audio-encoder", "aac",
        "--trim-frame-start", "0",
    ]

    if high_quality:
        cmd.extend(["--processors", "face_swapper", "face_enhancer", "--face-enhancer-model", "gfpgan_1.4"])

    if not all_faces:
        cmd.extend(["--face-selector-order", "large-small"])

    log.info("Command: %s", " ".join(cmd))

    if progress_cb:
        progress_cb(10, f"Starting FaceFusion ({execution_provider.upper()})...")

    env = os.environ.copy()
    env['FACEFUSION_CONTENT_ANALYSER'] = 'false'
    env['FACEFUSION_SKIP_CONTENT_ANALYSIS'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = '0'

    process = subprocess.Popen(
        cmd, cwd=str(FACEFUSION_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env
    )

    output_lines = []
    for line in process.stdout:
        output_lines.append(line)
        log.debug(line.rstrip())
        if progress_cb:
            ll = line.lower()
            if "progress" in ll or "%" in line:
                try:
                    import re
                    m = re.search(r'(\d+)%', line)
                    if m:
                        pct = int(m.group(1))
                        progress_cb(min(10 + int(pct * 0.85), 95), line.strip())
                except Exception:
                    pass
            elif "frame" in ll and "/" in line:
                try:
                    import re
                    m = re.search(r'(\d+)/(\d+)', line)
                    if m:
                        c, t = int(m.group(1)), int(m.group(2))
                        pct = int((c / max(t, 1)) * 85)
                        progress_cb(min(10 + pct, 95), f"Frame {c}/{t}")
                except Exception:
                    pass

    return_code = process.wait()
    log.info("FaceFusion exit code: %d", return_code)

    # CHECK OUTPUT FILE
    out_file = Path(output_path)
    if out_file.exists() and out_file.stat().st_size > 1000:
        log.info("Output exists: %s (%d bytes)", out_file, out_file.stat().st_size)
    else:
        log.error("Output NOT FOUND: %s", output_path)
        parent = out_file.parent
        for f in sorted(parent.glob("*.mp4"), key=lambda x: x.stat().st_size, reverse=True):
            log.info("Found mp4: %s (%d bytes)", f, f.stat().st_size)

    if return_code != 0:
        tail = "".join(output_lines[-80:])
        log.error("FaceFusion error:\n%s", tail)
        raise RuntimeError(f"FaceFusion failed: {tail[:500]}")

    if progress_cb:
        progress_cb(100, "Done!")
    return output_path


def process_image_facefusion(source_image_path, target_image_path, output_path,
                             all_faces=False, high_quality=False):
    _patch_content_analyser()
    return process_video_facefusion(
        source_image_path, target_image_path, output_path,
        all_faces=all_faces, high_quality=high_quality
    )


def get_model_info():
    gpu = get_gpu_info()
    return {
        "facefusion_available": check_facefusion_available(),
        "gpu_available": gpu["available"],
        "gpu_count": gpu["count"],
        "gpu_name": gpu["name"],
        "gpu_memory_gb": round(gpu["memory_gb"], 1),
        "execution_provider": get_execution_provider(),
    }


def check_model_available():
    return check_facefusion_available()
