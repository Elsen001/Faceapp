
"""
DeepFace Video Yüz Dəyişdirmə – Flask Backend
"""

import os
import uuid
import json
import time
import threading
import logging
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    Response,
    stream_with_context,
)
from werkzeug.utils import secure_filename

# ── App setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv", "webm"}
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_VIDEO_MB = 500
MAX_IMAGE_MB = 20

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = (MAX_VIDEO_MB + MAX_IMAGE_MB) * 1024 * 1024
app.secret_key = os.urandom(32)

# ── Job store (in-memory) ────────────────────────────────────────────────────
# job_id -> {"status": "pending|running|done|error",
#             "progress": 0-100, "message": str,
#             "output": str|None, "error": str|None}
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def allowed(filename: str, kinds: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in kinds


def unique_path(directory: Path, suffix: str) -> Path:
    return directory / f"{uuid.uuid4().hex}{suffix}"


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model-status")
def model_status():
    from face_swapper import check_model_available, get_model_info
    info = get_model_info()
    return jsonify(info)


@app.route("/swap", methods=["POST"])
def swap():
    """
    multipart/form-data:
      - video  : video file
      - image  : face source image
      - all_faces: "true"/"false"
    """
    if "video" not in request.files or "image" not in request.files:
        return jsonify({"error": "video və image sahələri tələb olunur"}), 400

    vid_file = request.files["video"]
    img_file = request.files["image"]
    all_faces = request.form.get("all_faces", "false").lower() == "true"

    if not vid_file.filename or not allowed(vid_file.filename, ALLOWED_VIDEO):
        return jsonify({"error": "Dəstəklənən video formatları: mp4, avi, mov, mkv, webm"}), 400

    if not img_file.filename or not allowed(img_file.filename, ALLOWED_IMAGE):
        return jsonify({"error": "Dəstəklənən şəkil formatları: jpg, jpeg, png, bmp, webp"}), 400

    vid_path = unique_path(UPLOAD_DIR, Path(secure_filename(vid_file.filename)).suffix)
    img_path = unique_path(UPLOAD_DIR, Path(secure_filename(img_file.filename)).suffix)
    vid_file.save(str(vid_path))
    img_file.save(str(img_path))

    job_id = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"{job_id}_swapped.mp4"

    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "message": "Növbədə gözlənilir...",
            "output": None,
            "error": None,
        }

    # Run in background thread
    t = threading.Thread(
        target=_run_job,
        args=(job_id, str(vid_path), str(img_path), str(output_path), all_faces),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


def _run_job(job_id, video_path, image_path, output_path, all_faces):
    def progress(pct, msg):
        with jobs_lock:
            jobs[job_id]["progress"] = pct
            jobs[job_id]["message"] = msg
            jobs[job_id]["status"] = "running"

    try:
        from face_swapper import process_video, check_model_available

        if not check_model_available():
            raise FileNotFoundError(
                "inswapper_128.onnx modeli tapılmadı. "
                "Zəhmət olmasa modeli yükləyin (bax: /model-status)."
            )

        progress(1, "Başlanır...")
        process_video(
            video_path=video_path,
            source_image_path=image_path,
            output_path=output_path,
            all_faces=all_faces,
            progress_cb=progress,
        )
        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["output"] = Path(output_path).name
            jobs[job_id]["message"] = "Uğurla tamamlandı!"

    except Exception as e:
        log.exception("Job %s failed", job_id)
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["message"] = "Xəta baş verdi"
    finally:
        # Clean up uploads
        for p in (video_path, image_path):
            try:
                os.remove(p)
            except OSError:
                pass


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "İş tapılmadı"}), 404
    return jsonify(job)


@app.route("/stream/<job_id>")
def stream(job_id):
    """SSE endpoint for real-time progress."""
    def generate():
        while True:
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'tapılmadı'})}\n\n"
                break
            yield f"data: {json.dumps(job)}\n\n"
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.8)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/download/<filename>")
def download(filename):
    safe = secure_filename(filename)
    return send_from_directory(str(OUTPUT_DIR), safe, as_attachment=True)


@app.route("/preview/<filename>")
def preview(filename):
    safe = secure_filename(filename)
    return send_from_directory(str(OUTPUT_DIR), safe)


if __name__ == "__main__":
    print("=" * 60)
    print("  DeepFace Video Yüz Dəyişdirmə Serveri")
    print("  http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
