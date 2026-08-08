"""
video_face_swapper.py  —  Professional video face swap pipeline
Nim.video / Kling AI video pipeline-i ilə eyni arxitektura

DÜZƏLİŞLƏR:
  1. Cinsiyyət uyğunluğu — qadın şəkli yalnız qadına, kişi şəkli yalnız kişiyə
  2. Ağız bölgəsində əşya tanıma — çəngəl, yemək, barmaq aşkarlanır
  3. Əşya maskası — ağızdakı əşyalar orijinal qalır, başın içinə girmir
  4. Temporal face tracking — KCF/CSRT tracker + frame-lər arası hamarlaşdırma
  5. Parallel frame işləmə — CPU core-lardan istifadə
  6. Scene-cut aşkarlaması — sahne keçidlərində tracker sıfırlanır
  7. Üz kilidi — ilk üz yadda saxlanılır və başqa üzə keçmir
  8. Cinsiyyət filtrasiyası təkmilləşdirilib — fallback mexanizmi əlavə olunub
"""

import os
import cv2
import numpy as np
import threading
import subprocess
import tempfile
import logging
import queue
import time
from collections import deque
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict

# face_swapper.py-dən import et
from face_swapper import (
    extract_source_face,
    _safe_detect,
    _swap_single_face,
    _face_is_valid,
    enhance_face,
    get_face_app,
    _match_gender,
)

log = logging.getLogger(__name__)


def _to_int(v) -> int:
    """
    NumPy massivini təhlükəsiz şəkildə Python int-ə çevirir.
    """
    if isinstance(v, np.ndarray):
        v = v.reshape(-1)[0]
    return int(v)


# ──────────────────────── sabitlər ────────────────────────
DETECT_INTERVAL   = 3    # hər neçə frame-də bir tam detect
SCENE_CUT_THRESH  = 35.0 # sahne keçidi threshold-u
EMA_ALPHA         = 0.45 # hamarlaşma faktoru
MAX_WORKERS       = 4    # paralel swap thread sayı
TRACKER_TYPE      = "CSRT"  # CSRT, KCF, MOSSE


# ──────────────────────── Unified EMA Filter ────────────────────────

class UnifiedEmaFilter:
    """
    Bbox və KPS-i EYNİ anda, EYNİ əmsalla hamarlaşdırır.
    Bu, bbox ilə kps arasında faza fərqini aradan qaldırır.
    """
    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self._prev_bbox: Optional[np.ndarray] = None
        self._prev_kps: Optional[np.ndarray] = None

    def update(self, bbox: np.ndarray, kps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._prev_bbox is None or self._prev_kps is None:
            self._prev_bbox = bbox.copy()
            self._prev_kps = kps.copy()
            return bbox, kps

        smoothed_bbox = self.alpha * bbox + (1 - self.alpha) * self._prev_bbox
        smoothed_kps  = self.alpha * kps + (1 - self.alpha) * self._prev_kps

        self._prev_bbox = smoothed_bbox
        self._prev_kps = smoothed_kps
        return smoothed_bbox, smoothed_kps

    def reset(self):
        self._prev_bbox = None
        self._prev_kps = None


# ──────────────────────── Face Tracker ────────────────────────

class FaceTracker:
    """
    Bir üz üçün temporal tracker.
    """
    def __init__(self, face_id: int, tracker_type: str = TRACKER_TYPE):
        self.face_id     = face_id
        self.tracker_type = tracker_type
        self._tracker    = None
        self._last_face  = None
        self._ema_filter = UnifiedEmaFilter()
        self._frame_cnt  = 0
        self._lost_cnt   = 0
        self._max_lost   = 10

    def _create_tracker(self):
        t = self.tracker_type.upper()
        if t == "CSRT":
            return cv2.TrackerCSRT_create()
        elif t == "KCF":
            return cv2.TrackerKCF_create()
        else:
            return cv2.TrackerMOSSE_create()

    def init(self, frame: np.ndarray, face) -> bool:
        self._ema_filter.reset()
        self._last_face = face
        self._frame_cnt = 0
        self._lost_cnt  = 0

        x1, y1, x2, y2 = [_to_int(v) for v in face.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return False

        self._tracker = self._create_tracker()
        try:
            ok = self._tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        except Exception:
            ok = False
        return ok

    def update(self, frame: np.ndarray):
        self._frame_cnt += 1

        if self._tracker is None or self._last_face is None:
            return None

        try:
            ok, bbox_cv = self._tracker.update(frame)
        except Exception:
            ok = False

        if not ok:
            self._lost_cnt += 1
            if self._lost_cnt > self._max_lost:
                self.reset()
            return self._last_face

        self._lost_cnt = 0
        x, y, bw, bh = [_to_int(v) for v in bbox_cv]

        face = self._last_face
        real_kps = self._try_local_redetect(frame, x, y, bw, bh)

        if real_kps is not None:
            kps_arr = real_kps
            kps_x = kps_arr[:, 0]
            kps_y = kps_arr[:, 1]
            margin = int(max(bw, bh) * 0.15)
            bbox_arr = np.array([
                max(0, np.min(kps_x) - margin),
                max(0, np.min(kps_y) - margin),
                min(frame.shape[1], np.max(kps_x) + margin),
                min(frame.shape[0], np.max(kps_y) + margin),
            ], dtype=np.float32)
        else:
            if face is not None and hasattr(face, "kps") and face.kps is not None:
                ox1, oy1, ox2, oy2 = face.bbox
                old_bw = max(ox2 - ox1, 1)
                old_bh = max(oy2 - oy1, 1)
                sx = bw / old_bw
                sy = bh / old_bh
                kps_new = face.kps.copy().astype(np.float32)
                kps_new[:, 0] = (kps_new[:, 0] - ox1) * sx + x
                kps_new[:, 1] = (kps_new[:, 1] - oy1) * sy + y
                kps_arr = kps_new
                bbox_arr = np.array([x, y, x + bw, y + bh], dtype=np.float32)
            else:
                return self._last_face

        bbox_smooth, kps_smooth = self._ema_filter.update(bbox_arr, kps_arr)
        face_proxy = _FaceProxy(face, kps_smooth, bbox_smooth)
        self._last_face = face_proxy
        return self._last_face

    def _try_local_redetect(self, frame: np.ndarray,
                            x: int, y: int, bw: int, bh: int
                            ) -> Optional[np.ndarray]:
        try:
            h, w = frame.shape[:2]
            pad = int(max(bw, bh) * 0.35)
            rx1 = max(0, x - pad)
            ry1 = max(0, y - pad)
            rx2 = min(w, x + bw + pad)
            ry2 = min(h, y + bh + pad)
            if rx2 <= rx1 or ry2 <= ry1:
                return None

            crop = frame[ry1:ry2, rx1:rx2]
            if crop.shape[0] < 40 or crop.shape[1] < 40:
                return None

            faces = get_face_app().get(crop)
            if not faces:
                return None

            cx_target = bw / 2.0
            cy_target = bh / 2.0
            best = min(faces, key=lambda f: abs((f.bbox[0]+f.bbox[2])/2 - cx_target) +
                                            abs((f.bbox[1]+f.bbox[3])/2 - cy_target))

            kps = np.array(best.kps[:5], dtype=np.float32)
            kps[:, 0] += rx1
            kps[:, 1] += ry1
            return kps
        except Exception:
            return None

    def reinit_with_face(self, frame: np.ndarray, face):
        kps_arr = np.array(face.kps[:5], dtype=np.float32)
        bbox_arr = np.asarray(face.bbox, dtype=np.float32)
        bbox_smooth, kps_smooth = self._ema_filter.update(bbox_arr, kps_arr)
        face_proxy = _FaceProxy(face, kps_smooth, bbox_smooth)
        self._last_face = face_proxy

        x1, y1, x2, y2 = [_to_int(v) for v in face.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        if x2 > x1 and y2 > y1:
            self._tracker = self._create_tracker()
            try:
                self._tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            except Exception:
                pass

    def reset(self):
        self._tracker    = None
        self._last_face  = None
        self._frame_cnt  = 0
        self._lost_cnt   = 0
        self._ema_filter.reset()

    @property
    def is_active(self) -> bool:
        return self._last_face is not None


class _FaceProxy:
    """
    insightface face object-inin yüngül proxy-si.
    """
    def __init__(self, original_face, kps: np.ndarray, bbox: np.ndarray):
        self._orig  = original_face
        self.kps    = kps
        self.bbox   = bbox

    def __getattr__(self, name):
        return getattr(self._orig, name)


# ──────────────────────── Scene Cut Detector ────────────────────────

class SceneCutDetector:
    """
    Ardıcıl frame-lər arası fərqə görə sahne keçidini aşkarla.
    """
    def __init__(self, threshold: float = SCENE_CUT_THRESH):
        self.threshold = threshold
        self._prev_lab: Optional[np.ndarray] = None

    def is_cut(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        lab   = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

        if self._prev_lab is None:
            self._prev_lab = lab
            return False

        mae = float(np.mean(np.abs(lab - self._prev_lab)))
        self._prev_lab = lab
        return mae > self.threshold

    def reset(self):
        self._prev_lab = None


# ──────────────────────── Temporal Mask Blender ────────────────────────

class TemporalMaskBlender:
    """
    Occlusion mask-larını frame-lər arası hamarlaşdır.
    """
    def __init__(self, alpha: float = 0.4, history: int = 3):
        self.alpha   = alpha
        self._buffer: deque = deque(maxlen=history)

    def update(self, mask: np.ndarray) -> np.ndarray:
        self._buffer.append(mask.astype(np.float32))
        if len(self._buffer) == 1:
            return mask
        avg = np.mean(list(self._buffer), axis=0)
        return np.clip(avg, 0, 255).astype(np.uint8)

    def reset(self):
        self._buffer.clear()


# ──────────────────────── VideoFaceSwapper ────────────────────────

class VideoFaceSwapper:
    """
    Professional video face swap pipeline.
    """
    def __init__(self,
                 detect_interval: int = DETECT_INTERVAL,
                 tracker_type:   str  = TRACKER_TYPE,
                 max_workers:    int  = MAX_WORKERS,
                 crop_size:      int  = 512,
                 gender_match:   bool = True):
        self.detect_interval = detect_interval
        self.tracker_type    = tracker_type
        self.max_workers     = max_workers
        self.crop_size       = crop_size
        self.gender_match    = gender_match

        self._trackers: List[FaceTracker] = []
        self._scene_cut = SceneCutDetector()
        self._tmb       = TemporalMaskBlender()
        self._source_img: Optional[np.ndarray] = None
        self._source_face_obj = None
        self._locked_face_id: Optional[int] = None  # kilit: ilk üzü saxla
        self._locked_gender: Optional[int] = None   # kilit: cinsiyyət

    # ── public API ──

    def process(self,
                source_image: str,
                input_video:  str,
                output_video: str,
                all_faces:    bool = False,
                high_quality: bool = False,
                progress_cb:  Optional[Callable] = None) -> str:
        """
        Videonu işlə və output_video-ya yaz.
        """
        self._progress(progress_cb, 2, "Mənbə üzü analiz edilir...")
        source_face = extract_source_face(source_image)
        self._source_img = cv2.imread(source_image)
        self._source_face_obj = source_face

        self._progress(progress_cb, 5, "Video açılır...")
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError(f"Video açılmadı: {input_video}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 99999
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._progress(progress_cb, 7,
                       f"Video: {vid_w}×{vid_h}, {fps:.1f} fps, "
                       f"~{total_frames} frame")

        tmp_video = tempfile.mktemp(suffix="_swap_noaudio.mp4")
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        writer    = cv2.VideoWriter(tmp_video, fourcc, fps, (vid_w, vid_h))

        self._trackers = []
        self._scene_cut.reset()
        self._tmb.reset()
        self._locked_face_id = None  # Sıfırla
        self._locked_gender = None   # Sıfırla

        try:
            self._process_loop(
                cap, writer, source_face,
                total_frames, all_faces, high_quality, progress_cb
            )
        finally:
            cap.release()
            writer.release()

        self._progress(progress_cb, 90, "Audio birləşdirilir...")
        try:
            _merge_audio_copy(input_video, tmp_video, output_video)
        except Exception as e:
            log.warning("Audio merge xətası (%s) — copy ilə davam", e)
            import shutil
            shutil.copy2(tmp_video, output_video)

        try:
            os.remove(tmp_video)
        except Exception:
            pass

        self._progress(progress_cb, 100, "Tamamlandı!")
        return output_video

    # ── daxili metodlar ──

    def _process_loop(self,
                      cap, writer,
                      source_face,
                      total_frames: int,
                      all_faces: bool,
                      high_quality: bool,
                      progress_cb: Optional[Callable]):
        QUEUE_DEPTH = self.max_workers * 4
        read_q:  queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
        write_q: queue.Queue = queue.Queue()

        frame_idx = 0
        done_event = threading.Event()

        def swap_worker():
            while not done_event.is_set():
                try:
                    item = read_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:
                    write_q.put(None)
                    read_q.task_done()
                    break
                idx, frame, tracked_faces = item
                try:
                    processed = self._swap_tracked_faces(
                        frame, tracked_faces, source_face, high_quality
                    )
                except Exception as e:
                    log.warning("Frame %d swap xətası: %s", idx, e)
                    processed = frame
                write_q.put((idx, processed))
                read_q.task_done()

        write_buffer: Dict[int, np.ndarray] = {}
        next_write = 0

        def flush_write_buffer():
            nonlocal next_write
            while next_write in write_buffer:
                writer.write(write_buffer.pop(next_write))
                next_write += 1

        def drain_write_q():
            nonlocal sentinel_count
            while True:
                try:
                    item = write_q.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    sentinel_count += 1
                else:
                    i, proc = item
                    write_buffer[i] = proc
                    flush_write_buffer()

        workers = []
        for _ in range(self.max_workers):
            t = threading.Thread(target=swap_worker, daemon=True)
            t.start()
            workers.append(t)

        sentinel_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    for _ in range(self.max_workers):
                        read_q.put(None)
                    break

                tracked_faces = self._track_frame(frame, frame_idx, all_faces)

                while True:
                    try:
                        read_q.put_nowait((frame_idx, frame, tracked_faces))
                        break
                    except queue.Full:
                        drain_write_q()
                        time.sleep(0.005)

                frame_idx += 1
                drain_write_q()

                if frame_idx % 10 == 0 and progress_cb:
                    pct = 8 + int((frame_idx / max(total_frames, 1)) * 80)
                    progress_cb(min(pct, 88),
                                f"Frame {frame_idx}/{total_frames}")

            done_event.set()
            for t in workers:
                t.join(timeout=120)

            while sentinel_count < self.max_workers:
                drain_write_q()
                time.sleep(0.01)

            drain_write_q()
            flush_write_buffer()

        except Exception as e:
            log.error("Process loop xətası: %s", e)
            done_event.set()
            raise

    def _track_frame(self,
                     frame: np.ndarray,
                     frame_idx: int,
                     all_faces: bool) -> list:
        is_cut = self._scene_cut.is_cut(frame)

        if is_cut:
            log.debug("Scene cut @ frame %d — tracker sıfırlanır", frame_idx)
            for t in self._trackers:
                t.reset()
            self._tmb.reset()
            # Sahne keçidində kilidi sıfırla — yeni səhnədə yeni üz tap
            self._locked_gender = None
            self._locked_face_id = None  # Üz kilidini də sıfırla

        need_detect = (
            is_cut or
            frame_idx == 0 or
            (frame_idx % self.detect_interval == 0)
        )

        if need_detect:
            detected = _safe_detect(frame)

            # Cinsiyyətə görə filtr (daha etibarlı)
            if self.gender_match and detected:
                detected = _match_gender(self._source_face_obj, detected)
                
                # Əgər cinsiyyət filtrindən sonra heç üz qalmayıbsa, hamısını götür
                if not detected:
                    detected = _safe_detect(frame)

            if detected:
                if not all_faces:
                    # Ən böyük üzü seç
                    detected = [max(detected,
                                   key=lambda f: (f.bbox[2] - f.bbox[0]) *
                                                 (f.bbox[3] - f.bbox[1]))]
                
                # Əgər kilitli üz varsa, onu saxlamağa çalış
                if self._locked_face_id is not None:
                    # Əvvəlki üzü track etməyə çalış
                    old_face = None
                    for tracker in self._trackers:
                        if tracker.face_id == self._locked_face_id and tracker.is_active:
                            old_face = tracker._last_face
                            break
                    
                    if old_face is not None:
                        # Köhnə üzə ən yaxın olanı seç
                        best = None
                        best_iou = 0.35
                        for f in detected:
                            iou = _bbox_iou(f.bbox, old_face.bbox)
                            if iou > best_iou:
                                best_iou = iou
                                best = f
                        if best is not None:
                            detected = [best]
                        else:
                            # Köhnə üz yoxdursa, ən böyük üzü götür
                            detected = [max(detected, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))]
                
                # Kilitli cinsiyyəti yenilə
                if self._locked_gender is None and detected:
                    first = detected[0]
                    self._locked_gender = getattr(first, 'gender', None)
                    if self._locked_gender is not None:
                        log.debug("Üz kilitləndi: gender=%s, id=%s", 
                                 self._locked_gender, self._locked_face_id)

            self._sync_trackers(frame, detected)

        tracked_faces = []
        for tracker in self._trackers:
            face = tracker.update(frame)
            if face is not None:
                tracked_faces.append(face)

        return tracked_faces

    def _swap_tracked_faces(self,
                            frame: np.ndarray,
                            tracked_faces: list,
                            source_face,
                            high_quality: bool) -> np.ndarray:
        if not tracked_faces:
            return frame

        # Cinsiyyətə görə filtr
        if self.gender_match:
            tracked_faces = _match_gender(source_face, tracked_faces)
        
        if not tracked_faces:
            return frame

        result = frame.copy()
        for tgt_face in tracked_faces:
            if _face_is_valid(tgt_face, frame.shape):
                try:
                    result = _swap_single_face(
                        result, tgt_face, source_face, self.crop_size,
                        enhance=high_quality,
                    )
                except Exception as e:
                    log.debug("_swap_single_face xətası: %s", e)

        return result

    def _sync_trackers(self, frame: np.ndarray, detected_faces: list):
        if not detected_faces:
            return

        if not self._trackers:
            for i, face in enumerate(detected_faces):
                t = FaceTracker(i, self.tracker_type)
                if t.init(frame, face):
                    self._trackers.append(t)
                    # İlk üzü kilitlə
                    if self._locked_face_id is None:
                        self._locked_face_id = i
                        log.debug("İlk üz kilitləndi: id=%d", i)
            return

        active = [(i, t) for i, t in enumerate(self._trackers) if t.is_active]
        matched_tracker_ids = set()
        
        # Əvvəlcə kilitli üzü tapmağa çalış
        locked_tracker = None
        if self._locked_face_id is not None:
            for i, t in enumerate(self._trackers):
                if i == self._locked_face_id and t.is_active:
                    locked_tracker = (i, t)
                    break

        for face in detected_faces:
            best_iou = 0.35  # İOU threshold-u artırıldı
            best_tidx = None
            
            # Əvvəlcə kilitli trackeri yoxla
            if locked_tracker is not None:
                tidx, t = locked_tracker
                if tidx not in matched_tracker_ids and t._last_face is not None:
                    iou = _bbox_iou(face.bbox, t._last_face.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_tidx = tidx
            
            # Əgər kilitli tapılmadısa, digərlərini yoxla
            if best_tidx is None:
                for tidx, t in active:
                    if tidx in matched_tracker_ids:
                        continue
                    if t._last_face is None:
                        continue
                    iou = _bbox_iou(face.bbox, t._last_face.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_tidx = tidx

            if best_tidx is not None:
                self._trackers[best_tidx].reinit_with_face(frame, face)
                matched_tracker_ids.add(best_tidx)
            else:
                # YENİ: Yalnız kilitli üz yoxdursa yeni tracker əlavə et
                if self._locked_face_id is None:
                    t = FaceTracker(len(self._trackers), self.tracker_type)
                    if t.init(frame, face):
                        self._trackers.append(t)
                        self._locked_face_id = t.face_id
                        log.debug("Yeni üz kilitləndi: id=%d", t.face_id)
                else:
                    log.debug("Yeni üz ignore edildi — kilitli üzə davam (id=%d)", 
                             self._locked_face_id)

    @staticmethod
    def _progress(cb, pct: int, msg: str):
        if cb:
            try:
                cb(pct, msg)
            except Exception:
                pass


# ──────────────────────── yardımçı funksiyalar ────────────────────────

def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw  = max(0, ix2 - ix1)
    ih  = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (a_area + b_area - inter)


def _merge_audio_copy(orig_video: str,
                      processed_video: str,
                      output_path: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", processed_video,
        "-i", orig_video,
        "-c:v", "libx264", "-preset", "fast", "-crf", "17",
        "-c:a", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0:
        return

    cmd[cmd.index("copy")] = "aac"
    cmd.insert(cmd.index("aac") + 1, "-b:a")
    cmd.insert(cmd.index("-b:a") + 1, "192k")
    r2 = subprocess.run(cmd, capture_output=True)
    if r2.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio merge xətası:\n{r2.stderr.decode(errors='replace')}"
        )


# ──────────────────────── public convenience API ────────────────────────

def process_video(source_image: str,
                  input_video:  str,
                  output_video: str,
                  all_faces:    bool = False,
                  high_quality: bool = False,
                  progress_cb:  Optional[Callable] = None,
                  detect_interval: int = DETECT_INTERVAL,
                  tracker_type:    str = TRACKER_TYPE,
                  max_workers:     int = MAX_WORKERS,
                  crop_size:       int = 512,
                  gender_match:    bool = True) -> str:
    """
    Sadə bir funksiya ilə tam video face swap.

    Args:
        source_image:     Mənbə üz şəkli
        input_video:      Giriş video faylı
        output_video:     Çıxış video faylı
        all_faces:        Bütün üzlər (False = ən böyük üz)
        high_quality:     CodeFormer/GFPGAN enhancement (yavaş)
        progress_cb:      callback(pct: int, msg: str)
        detect_interval:  Hər neçə frame-də tam detect
        tracker_type:     "CSRT" | "KCF" | "MOSSE"
        max_workers:      Paralel swap thread sayı
        crop_size:        Aligned crop ölçüsü
        gender_match:     Eyni cinsiyyətə swap et (True = default)
    """
    swapper = VideoFaceSwapper(
        detect_interval=detect_interval,
        tracker_type=tracker_type,
        max_workers=max_workers,
        crop_size=crop_size,
        gender_match=gender_match,
    )
    return swapper.process(
        source_image=source_image,
        input_video=input_video,
        output_video=output_video,
        all_faces=all_faces,
        high_quality=high_quality,
        progress_cb=progress_cb,
    )


# ──────────────────────── CLI ────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Video face swap — Professional pipeline"
    )
    parser.add_argument("source",  help="Mənbə üz şəkli (.jpg/.png)")
    parser.add_argument("input",   help="Giriş video faylı")
    parser.add_argument("output",  help="Çıxış video faylı")
    parser.add_argument("--all-faces",   action="store_true",
                        help="Bütün üzləri dəyişdir (default: ən böyük)")
    parser.add_argument("--high-quality", action="store_true",
                        help="CodeFormer/GFPGAN ilə keyfiyyət artır (yavaş)")
    parser.add_argument("--detect-interval", type=int, default=DETECT_INTERVAL,
                        help=f"Re-detect interval (default: {DETECT_INTERVAL})")
    parser.add_argument("--tracker", default=TRACKER_TYPE,
                        choices=["CSRT", "KCF", "MOSSE"],
                        help=f"Tracker tipi (default: {TRACKER_TYPE})")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Paralel thread sayı (default: {MAX_WORKERS})")
    parser.add_argument("--crop-size", type=int, default=512,
                        help="Aligned crop ölçüsü (default: 512)")
    parser.add_argument("--no-gender-match", action="store_true",
                        help="Cinsiyyət uyğunluğunu söndür")
    parser.add_argument("--no-audio", action="store_true",
                        help="Audio birləşdirmə (yalnız video)")
    args = parser.parse_args()

    def progress(pct, msg):
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r[{bar}] {pct:3d}%  {msg}", end="", flush=True)

    print(f"Mənbə:  {args.source}")
    print(f"Giriş:  {args.input}")
    print(f"Çıxış:  {args.output}")
    print(f"Tracker: {args.tracker}, detect hər {args.detect_interval} frame")
    print(f"Cinsiyyət uyğunluğu: {'SÖNDÜRÜLÜB' if args.no_gender_match else 'AKTİV'}")
    print()

    out = process_video(
        source_image=args.source,
        input_video=args.input,
        output_video=args.output,
        all_faces=args.all_faces,
        high_quality=args.high_quality,
        progress_cb=progress,
        detect_interval=args.detect_interval,
        tracker_type=args.tracker,
        max_workers=args.workers,
        crop_size=args.crop_size,
        gender_match=not args.no_gender_match,
    )
    print(f"\n✓ Hazır: {out}")