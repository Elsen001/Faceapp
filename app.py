import os, uuid, json, time, threading, logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2048 * 1024 * 1024

jobs = {}
jobs_lock = threading.Lock()

ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv", "webm"}
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "bmp", "webp"}


def allowed_video(f):
    return "." in f and f.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO


def allowed_image(f):
    return "." in f and f.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE


@app.after_request
def after_request(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return r


@app.route("/")
def index():
    # templates/ olmadan birbaşa index.html oxu
    html_paths = [
        BASE_DIR / "index.html",
        BASE_DIR / "templates" / "index.html",
    ]
    for p in html_paths:
        if p.exists():
            return p.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html"}
    return "<h2>index.html tapılmadı</h2>", 404


@app.route("/model-status")
def model_status():
    try:
        from face_swapper import get_model_info
        return jsonify(get_model_info())
    except Exception as e:
        return jsonify({"swapper_exists": False, "error": str(e)})


@app.route("/swap", methods=["POST", "OPTIONS"])
def swap():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        mode = request.form.get("mode", "video")

        if "source" not in request.files or "target" not in request.files:
            return jsonify({"error": "source ve target lazimdir"}), 400

        source_file = request.files["source"]
        target_file = request.files["target"]
        all_faces = request.form.get("all_faces", "false").lower() == "true"

        # Fayl adı boşdursa xəta
        if not source_file.filename:
            return jsonify({"error": "Source fayl seçilməyib"}), 400
        if not target_file.filename:
            return jsonify({"error": "Target fayl seçilməyib"}), 400

        if not allowed_image(source_file.filename):
            return jsonify({"error": "Üz şəkli: jpg, png, webp olmalıdır"}), 400

        if mode == "image":
            if not allowed_image(target_file.filename):
                return jsonify({"error": "Hədəf şəkil: jpg, png, webp olmalıdır"}), 400
        else:
            if not allowed_video(target_file.filename):
                return jsonify({"error": "Video: mp4, avi, mov, mkv olmalıdır"}), 400

        src_suffix = Path(secure_filename(source_file.filename)).suffix or ".jpg"
        tgt_suffix = Path(secure_filename(target_file.filename)).suffix or ".mp4"

        src_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{src_suffix}"
        tgt_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{tgt_suffix}"
        source_file.save(str(src_path))
        target_file.save(str(tgt_path))

        job_id = uuid.uuid4().hex
        out_ext = ".jpg" if mode == "image" else ".mp4"
        out_path = OUTPUT_DIR / f"{job_id}_swapped{out_ext}"

        with jobs_lock:
            jobs[job_id] = {
                "status": "pending",
                "progress": 0,
                "message": "Gözlənilir...",
                "output": None,
                "error": None,
                "mode": mode,
            }

        t = threading.Thread(
            target=_run_job,
            args=(job_id, str(src_path), str(tgt_path), str(out_path), all_faces, mode),
            daemon=True,
        )
        t.start()
        return jsonify({"job_id": job_id})

    except Exception as e:
        log.exception("swap error")
        return jsonify({"error": str(e)}), 500


def _run_job(job_id, src_path, tgt_path, out_path, all_faces, mode):
    def progress(pct, msg):
        with jobs_lock:
            jobs[job_id].update({"progress": pct, "message": msg, "status": "running"})

    try:
        from face_swapper import process_video, process_image, check_model_available

        if not check_model_available():
            raise FileNotFoundError(
                "inswapper_128.onnx modeli tapılmadı! models/ qovluğuna yükləyin."
            )

        progress(1, "Başlanır...")

        if mode == "image":
            # FIX: source birinci, target ikinci — ardıcıllıq düzgündür
            process_image(src_path, tgt_path, out_path, all_faces)
            progress(100, "Tamamlandı!")
        else:
            # FIX: process_video(video_path, source_image_path, ...) — sıra düzgündür
            process_video(tgt_path, src_path, out_path, all_faces, progress)

        # Output faylın yarandığını yoxla
        if not Path(out_path).exists():
            raise RuntimeError("Output fayl yaranmadı.")

        with jobs_lock:
            jobs[job_id].update({
                "status": "done",
                "progress": 100,
                "output": Path(out_path).name,
                "message": "Tamamlandı!",
            })

    except Exception as e:
        log.exception("job error")
        with jobs_lock:
            jobs[job_id].update({
                "status": "error",
                "error": str(e),
                "message": f"Xəta: {e}",
            })
    finally:
        for p in (src_path, tgt_path):
            try:
                os.remove(p)
            except Exception:
                pass


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "tapılmadı"}), 404
    return jsonify(job)


@app.route("/stream/<job_id>")
def stream(job_id):
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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(
        str(OUTPUT_DIR), secure_filename(filename), as_attachment=True
    )


@app.route("/preview/<filename>")
def preview(filename):
    return send_from_directory(str(OUTPUT_DIR), secure_filename(filename))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Server: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
