"""
HuggingFace Spaces — FaceFusion 3.0+ Video Face Swap
"""

import os
import sys
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from facefusion_swapper_hf import (
    process_video_facefusion,
    process_image_facefusion,
    get_model_info,
)

import gradio as gr

UPLOADS_DIR = Path("/tmp/facefusion_outputs")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

_status = {"progress": 0, "message": "Ready"}


def _progress_cb(pct: int, msg: str):
    _status["progress"] = pct
    _status["message"] = msg
    log.info("[%d%%] %s", pct, msg)


def _find_output(expected_path: str, work_dir: Path) -> str:
    """Find FaceFusion output file anywhere in work_dir."""
    expected = Path(expected_path)

    # Direct path
    if expected.exists() and expected.stat().st_size > 1000:
        return str(expected)

    # Same directory
    parent = expected.parent
    candidates = [f for f in parent.glob("*.mp4") if f.stat().st_size > 1000]
    if candidates:
        return str(max(candidates, key=lambda p: p.stat().st_size))

    # Recursive search in work_dir
    all_mp4 = [f for f in work_dir.rglob("*.mp4") if f.stat().st_size > 1000]
    if all_mp4:
        return str(max(all_mp4, key=lambda p: p.stat().st_size))

    raise FileNotFoundError(f"Output not found. Expected: {expected_path}")


def process_video_gradio(source_image, target_video, all_faces, high_quality,
                         ref_dist, det_score, mask_blur, preset):
    if not source_image or not target_video:
        return None, "Error: Source image and target video required!"

    _status["progress"] = 0

    try:
        sid = os.urandom(4).hex()
        work_dir = UPLOADS_DIR / sid
        work_dir.mkdir(parents=True, exist_ok=True)

        src = work_dir / "source.jpg"
        tgt = work_dir / "target.mp4"
        out = work_dir / "output.mp4"

        shutil.copy2(source_image, src)
        shutil.copy2(target_video, tgt)

        log.info("Session: %s | Source: %s bytes | Target: %s bytes",
                 sid, src.stat().st_size, tgt.stat().st_size)

        result = process_video_facefusion(
            source_image_path=str(src),
            target_video_path=str(tgt),
            output_path=str(out),
            all_faces=all_faces,
            high_quality=high_quality,
            progress_cb=_progress_cb,
            reference_face_distance=float(ref_dist),
            face_detector_score=float(det_score),
            face_mask_blur=float(mask_blur),
            output_video_preset=preset,
            execution_thread_count=8,
        )

        actual = _find_output(result, work_dir)
        size_mb = Path(actual).stat().st_size / (1024 * 1024)

        log.info("Result: %s (%.1f MB)", actual, size_mb)
        return actual, f"Done! {size_mb:.1f} MB | Provider check logs"

    except Exception as e:
        log.exception("Processing failed")
        return None, f"Error: {str(e)}"


def process_image_gradio(source_image, target_image, all_faces, high_quality):
    if not source_image or not target_image:
        return None, "Error: Both images required!"

    try:
        sid = os.urandom(4).hex()
        work_dir = UPLOADS_DIR / sid
        work_dir.mkdir(parents=True, exist_ok=True)

        src = work_dir / "source.jpg"
        tgt = work_dir / "target.jpg"
        out = work_dir / "output.jpg"

        shutil.copy2(source_image, src)
        shutil.copy2(target_image, tgt)

        result = process_image_facefusion(
            source_image_path=str(src),
            target_image_path=str(tgt),
            output_path=str(out),
            all_faces=all_faces,
            high_quality=high_quality,
        )

        actual = _find_output(result, work_dir)
        return actual, "Done!"

    except Exception as e:
        log.exception("Image processing failed")
        return None, f"Error: {str(e)}"


def get_system_info():
    info = get_model_info()
    return f"""GPU Available: {info.get('gpu_available', False)}
GPU Count: {info.get('gpu_count', 0)}
GPU Name: {info.get('gpu_name', 'N/A')}
GPU Memory: {info.get('gpu_memory_gb', 0)} GB
Provider: {info.get('execution_provider', 'N/A')}
FaceFusion Ready: {info.get('facefusion_available', False)}""".strip()


with gr.Blocks(title="FaceFusion 3.0+ Video Face Swap", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎭 FaceFusion 3.0+ Temporal Consistency
    **Professional video face swap — zero jitter & extreme pose support**
    """)

    with gr.Tab("🎬 Video Face Swap"):
        with gr.Row():
            with gr.Column(scale=1):
                source_img = gr.Image(label="Source Face Image", type="filepath", image_mode="RGB")
                target_vid = gr.Video(label="Target Video")

            with gr.Column(scale=1):
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    all_faces_chk = gr.Checkbox(label="Swap all faces", value=False)
                    hq_chk = gr.Checkbox(label="High quality (GFPGAN)", value=False)
                    ref_dist = gr.Slider(label="Reference Face Distance (0.3=strict, 0.6=loose)",
                                         minimum=0.2, maximum=0.8, step=0.05, value=0.6)
                    det_score = gr.Slider(label="Face Detector Score (0.3=profile, 0.5=frontal)",
                                          minimum=0.2, maximum=0.8, step=0.05, value=0.3)
                    mask_blur = gr.Slider(label="Mask Blur (edge smoothness)",
                                          minimum=0.0, maximum=1.0, step=0.05, value=0.5)
                    preset = gr.Dropdown(label="Output Preset",
                                         choices=["ultrafast", "superfast", "veryfast",
                                                  "faster", "fast", "medium", "slow", "slower"],
                                         value="fast")

                process_btn = gr.Button("🚀 Process", variant="primary", size="lg")
                output_vid = gr.Video(label="Result")
                status_txt = gr.Textbox(label="Status", interactive=False)

        process_btn.click(
            fn=process_video_gradio,
            inputs=[source_img, target_vid, all_faces_chk, hq_chk,
                    ref_dist, det_score, mask_blur, preset],
            outputs=[output_vid, status_txt],
        )

    with gr.Tab("🖼️ Image Face Swap"):
        with gr.Row():
            with gr.Column(scale=1):
                src_img = gr.Image(label="Source Face", type="filepath")
                tgt_img = gr.Image(label="Target Image", type="filepath")
            with gr.Column(scale=1):
                all_faces_img = gr.Checkbox(label="Swap all faces", value=False)
                hq_img = gr.Checkbox(label="High quality", value=False)
                process_img_btn = gr.Button("🚀 Process", variant="primary")
                output_img = gr.Image(label="Result")
                status_img = gr.Textbox(label="Status", interactive=False)

        process_img_btn.click(
            fn=process_image_gradio,
            inputs=[src_img, tgt_img, all_faces_img, hq_img],
            outputs=[output_img, status_img],
        )

    with gr.Tab("ℹ️ System Info"):
        info_btn = gr.Button("Show System Info")
        info_out = gr.Textbox(label="Info", lines=8, interactive=False)
        info_btn.click(fn=get_system_info, outputs=info_out)
        demo.load(fn=get_system_info, outputs=info_out)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, show_error=True)
