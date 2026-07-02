"""
video_face_swapper.py  —  Professional video face swap pipeline
Nim.video / Kling AI video pipeline-i ilə eyni arxitektura

Şəkil pipeline-dən fərqlər (video üçün əlavələr):
  1. Temporal face tracking  — hər frame-də yenidən detect etmək əvəzinə
                               KCF/CSRT tracker + N frame-də bir re-detect
  2. Landmark smoothing      — üz keypoint-lərini frame-lər arası Kalman + EMA
                               ilə hamarla → titrəmə (flicker) yox olur
  3. Parallel frame işləmə  — ThreadPoolExecutor ilə CPU core-lardan istifadə
  4. Temporal mask blending  — ardıcıl frame-lərdə maska kənarlarını yumşat
  5. Scene-cut aşkarlaması   — sahne keçidlərini tap, tracker-i sıfırla
  6. Audio pass-through      — orijinal audio FFmpeg ilə copy (re-encode yox)
  7. Occlusion tracking      — əl/cisim occlusion-u frame-lər arası izlə,
                               ani keçidlər əvəzinə hamar geçiş et
  8. GPU dəstəyi             — CUDA mövcuddursa avtomatik aktivləşir

İstifadə:
  from video_face_swapper import VideoFaceSwapper

  swapper = VideoFaceSwapper()
  swapper.process(
      source_image="menbə.jpg",
      input_video="video.mp4",
      output_video="cixis.mp4",
      all_faces=False,
      progress_cb=lambda pct, msg: print(f"{pct}% — {msg}")
  )
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict

# face_swapper.py-dən bütün mövcud funksiyaları import et
from face_swapper import (
    extract_source_face,
    _safe_detect,
    _swap_single_face,
    _face_is_valid,
    enhance_face,
    get_face_app,
)

log = logging.getLogger(__name__)


def _to_int(v) -> int:
    """
    NumPy massivini (skalyar, (1,) və ya (1,1) formalı olsa belə) təhlükəsiz
    şəkildə Python int-ə çevirir.
    NumPy >= 1.25-də çox-ölçülü/yastı olmayan massiv üzərində birbaşa int(v)
    çağırmaq "only 0-dimensional arrays can be converted to Python scalars"
    xətası yaradır — bu funksiya bunun qarşısını alır.
    """
    if isinstance(v, np.ndarray):
        v = v.reshape(-1)[0]
    return int(v)

# ──────────────────────── eynək + saç transfer köməkçiləri ────────────────────────

def _extract_glasses_region(image: np.ndarray, face) -> Optional[np.ndarray]:
    """
    Şəkildəki üz landmark-larına görə eynək regionunu çıxar.
    Göz zonasını qaytarır (bbox-ın yuxarı ~35%-i).
    """
    try:
        x1, y1, x2, y2 = [_to_int(v) for v in face.bbox]
        face_h = y2 - y1
        face_w = x2 - x1
        # Eynək zonası: gözlər face hündürlüyünün ~25%-55%-i arasında
        gy1 = y1 + int(face_h * 0.22)
        gy2 = y1 + int(face_h * 0.55)
        gx1 = x1 + int(face_w * 0.04)
        gx2 = x2 - int(face_w * 0.04)
        gy1 = max(0, gy1); gy2 = min(image.shape[0], gy2)
        gx1 = max(0, gx1); gx2 = min(image.shape[1], gx2)
        if gy2 <= gy1 or gx2 <= gx1:
            return None
        return image[gy1:gy2, gx1:gx2].copy()
    except Exception:
        return None


def _extract_hair_region(image: np.ndarray, face) -> Optional[np.ndarray]:
    """
    Şəkildəki üzün üstündəki saç bölgəsini çıxar.
    Üz bbox-ının üstündəki ~50% hündürlük regionu.
    """
    try:
        x1, y1, x2, y2 = [_to_int(v) for v in face.bbox]
        face_h = y2 - y1
        face_w = x2 - x1
        # Saç regionu: face-in üstündə, face_h qədər yuxarıya qədər
        # Genişlik: bir qədər kənarları da al
        hair_expand_w = int(face_w * 0.15)
        hx1 = max(0, x1 - hair_expand_w)
        hx2 = min(image.shape[1], x2 + hair_expand_w)
        # Üstdən face_h * 0.6 qədər yuxarı
        hy2 = y1 + int(face_h * 0.12)   # alnın üstünə qədər
        hy1 = max(0, y1 - int(face_h * 0.7))
        if hy2 <= hy1 or hx2 <= hx1:
            return None
        return image[hy1:hy2, hx1:hx2].copy()
    except Exception:
        return None


def _blend_region_onto_frame(
    frame: np.ndarray,
    region_img: np.ndarray,
    face,
    region_type: str,  # "glasses" | "hair"
    alpha: float = 0.88,
) -> np.ndarray:
    """
    Çıxarılmış eynək/saç regionunu hədəf face-in üzərinə yapışdır.
    Seamless clone ilə kənarlar hamar birləşir.
    """
    try:
        x1, y1, x2, y2 = [_to_int(v) for v in face.bbox]
        face_h = y2 - y1
        face_w = x2 - x1

        if region_type == "glasses":
            gy1 = y1 + int(face_h * 0.22)
            gy2 = y1 + int(face_h * 0.55)
            gx1 = x1 + int(face_w * 0.04)
            gx2 = x2 - int(face_w * 0.04)
        else:  # hair
            hair_expand_w = int(face_w * 0.15)
            gx1 = max(0, x1 - hair_expand_w)
            gx2 = min(frame.shape[1], x2 + hair_expand_w)
            gy2 = y1 + int(face_h * 0.12)
            gy1 = max(0, y1 - int(face_h * 0.7))

        # Koordinatları çərçivə hüdudlarına uyğunlaşdır
        gy1 = max(0, gy1); gy2 = min(frame.shape[0], gy2)
        gx1 = max(0, gx1); gx2 = min(frame.shape[1], gx2)
        if gy2 <= gy1 or gx2 <= gx1:
            return frame

        target_h = gy2 - gy1
        target_w = gx2 - gx1

        # Region-u hədəf ölçüsünə yenidən ölçüləndir
        resized = cv2.resize(region_img, (target_w, target_h),
                             interpolation=cv2.INTER_LANCZOS4)

        result = frame.copy()

        if region_type == "glasses":
            # Eynək üçün: laplacian mask ilə yalnız eynək pikselləri götür
            # Eynək adətən tünd/metalik piksellərdən ibarətdir
            gray_r = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # Eynək pikselləri: tünd kənar xəttlər
            edges = cv2.Canny(gray_r, 30, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(edges, kernel, iterations=2)
            # Yalnız eynək kənar xəttlərini blend et
            mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
            roi = result[gy1:gy2, gx1:gx2].astype(np.float32)
            blended = roi * (1 - mask_3c * alpha) + resized.astype(np.float32) * (mask_3c * alpha)
            result[gy1:gy2, gx1:gx2] = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            # Saç üçün: color segmentation — arka planı xaric et
            # HSV-də saç rəngi aşkarla
            hsv_r = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            # Saç adətən düşük saturation + orta/tünd value (hər rəng saçı işləyir)
            # Göy arka fon vs saç: saturation fərqi
            # Daha universal: grab-cut yanaşması
            mask = np.zeros(resized.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            # Mərkəz saç bölgəsi foreground kimi işarələ
            rect = (2, 2, target_w - 4, target_h - 4)
            try:
                cv2.grabCut(resized, mask, rect, bgd_model, fgd_model, 3,
                            cv2.GC_INIT_WITH_RECT)
                hair_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
            except Exception:
                hair_mask = np.ones((target_h, target_w), dtype=np.uint8)

            # Mask kənarlarını yumşat
            hair_mask_blur = cv2.GaussianBlur(
                hair_mask.astype(np.float32) * 255, (15, 15), 5
            ) / 255.0
            mask_3c = np.stack([hair_mask_blur] * 3, axis=-1)
            roi = result[gy1:gy2, gx1:gx2].astype(np.float32)
            blended = roi * (1 - mask_3c * alpha) + resized.astype(np.float32) * (mask_3c * alpha)
            result[gy1:gy2, gx1:gx2] = np.clip(blended, 0, 255).astype(np.uint8)

        return result

    except Exception as e:
        log.debug("_blend_region_onto_frame xətası (%s): %s", region_type, e)
        return frame


# ──────────────────────── sabitlər ────────────────────────
# Mənbə şəkildəki eynək/saçı videoya köçürən kobud transfer (grabCut/Canny
# düzbucaq blend). Titrəmə, görünən qutu kənarları yaradır və çox yavaşdır —
# default söndürülüb. Lazım olsa True et.
ENABLE_GLASSES_HAIR_TRANSFER = False

DETECT_INTERVAL   = 2    # hər neçə frame-də bir yenidən tam detect (aşağı = az sürüşmə)
SCENE_CUT_THRESH  = 35.0 # MAE threshold sahne keçidi üçün (LAB space)
EMA_ALPHA         = 0.55 # keypoint EMA hamarlaşma faktoru (0=tam hamar, 1=ham)
MAX_WORKERS       = 4    # paralel swap thread sayı
TRACKER_TYPE      = "CSRT"  # KCF, CSRT, MOSSE — CSRT ən dəqiq

# ──────────────────────── KPS hamarlaşdırıcı ────────────────────────

class KpsEmaFilter:
    """
    Üz keypoint-lərini Exponential Moving Average ilə hamarlaşdır.
    Frame-lər arası titrəməni aradan qaldırır.
    """
    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self._prev: Optional[np.ndarray] = None

    def update(self, kps: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = kps.copy()
            return kps
        smoothed = self.alpha * kps + (1 - self.alpha) * self._prev
        self._prev = smoothed
        return smoothed

    def reset(self):
        self._prev = None


class BboxEmaFilter:
    """Üz bbox-ını hamarlaşdır."""
    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self._prev: Optional[np.ndarray] = None

    def update(self, bbox: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = bbox.copy()
            return bbox
        smoothed = self.alpha * bbox + (1 - self.alpha) * self._prev
        self._prev = smoothed
        return smoothed

    def reset(self):
        self._prev = None


class FaceTracker:
    """
    Bir üz üçün temporal tracker.
    - KCF/CSRT tracker ilə üzü izlə
    - N frame-də bir tam re-detect et
    - Scene cut-da sıfırla
    - Keypoint-ləri EMA ilə hamarlaşdır
    """

    def __init__(self, face_id: int, tracker_type: str = TRACKER_TYPE):
        self.face_id     = face_id
        self.tracker_type = tracker_type
        self._tracker    = None
        self._last_face  = None
        self._kps_ema    = KpsEmaFilter()
        self._bbox_ema   = BboxEmaFilter()
        self._frame_cnt  = 0
        self._lost_cnt   = 0
        self._max_lost   = 10  # bu qədər frame itirilsə tracker reset

    def _create_tracker(self):
        t = self.tracker_type.upper()
        if t == "CSRT":
            return cv2.TrackerCSRT_create()
        elif t == "KCF":
            return cv2.TrackerKCF_create()
        else:
            return cv2.TrackerMOSSE_create()

    def init(self, frame: np.ndarray, face) -> bool:
        """Tracker-i ilk üz ilə inisializasiya et."""
        self._kps_ema.reset()
        self._bbox_ema.reset()
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
        """
        Frame üçün tracker-i yenilə.
        Returns: hamarlanmış face (kps + bbox) və ya None

        DÜZƏLTMƏ — sürüşmə (sliding) problemi:
        Əvvəlki versiya kps-ləri YALNIZ xətti scale ilə hesablayırdı.
        Bu, üz döndükdə (yaw rotasiya) keypoint-lərin bbox daxilindəki
        qeyri-xətti yerdəyişməsini tuta bilmir → align matrisi səhv
        hesablanır → swap edilmiş üz sürüşür/əyilir.

        Həll: hər tracker update-də əlavə yüngül lokal üz detect
        (yalnız bbox ətrafı crop-da, sürətli) cəhd edilir. Tapılarsa
        REAL kps istifadə olunur (yalnız EMA ilə hamarlanır), tapılmazsa
        köhnə xətti scale fallback kimi qalır.
        """
        self._frame_cnt += 1

        if self._tracker is None or self._last_face is None:
            return None

        # OpenCV tracker ilə bbox izlə
        try:
            ok, bbox_cv = self._tracker.update(frame)
        except Exception:
            ok = False

        if not ok:
            self._lost_cnt += 1
            if self._lost_cnt > self._max_lost:
                self.reset()
            return self._last_face  # son uğurlu frame-i ver

        self._lost_cnt = 0
        x, y, bw, bh = [_to_int(v) for v in bbox_cv]

        face = self._last_face
        real_kps = self._try_local_redetect(frame, x, y, bw, bh)

        # DÜZƏLTMƏ — bbox da kps kimi EMA ilə hamarlanmalıdır, əks halda
        # kps hamar, bbox isə titrəyən qalır → paste-back region sürüşür.
        bbox_smooth = self._bbox_ema.update(
            np.array([x, y, x + bw, y + bh], dtype=np.float32)
        )

        if real_kps is not None:
            # ── Real detection tapıldı: bunu istifadə et (dəqiq, pose-aware) ──
            kps_smooth = self._kps_ema.update(real_kps)
            face_proxy = _FaceProxy(face, kps_smooth, bbox_smooth)
            self._last_face = face_proxy
        elif face is not None and hasattr(face, "kps") and face.kps is not None:
            # ── Fallback: xətti scale (real detect tapılmadıqda) ──
            ox1, oy1, ox2, oy2 = face.bbox
            old_bw = max(ox2 - ox1, 1)
            old_bh = max(oy2 - oy1, 1)
            sx = bw / old_bw
            sy = bh / old_bh
            kps_new = face.kps.copy().astype(np.float32)
            kps_new[:, 0] = (kps_new[:, 0] - ox1) * sx + x
            kps_new[:, 1] = (kps_new[:, 1] - oy1) * sy + y
            kps_smooth = self._kps_ema.update(kps_new)
            face_proxy = _FaceProxy(face, kps_smooth, bbox_smooth)
            self._last_face = face_proxy

        return self._last_face

    def _try_local_redetect(self, frame: np.ndarray,
                            x: int, y: int, bw: int, bh: int
                            ) -> Optional[np.ndarray]:
        """
        Tracker bbox-ı ətrafında genişləndirilmiş bölgədə sürətli üz detect
        cəhd et. Tapılarsa real (pose-doğru) kps qaytarır.

        Bu, sürüşmənin əsas həllidir: hər frame-də (və ya hər 2-3 frame-də)
        həqiqi landmark istifadə edir, sadəcə əvvəlki bbox-dan xətti
        scale etmir.
        """
        try:
            from face_swapper import get_face_app
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

            # Bbox mərkəzinə ən yaxın üzü seç
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
        """Re-detect sonrası tracker-i yeni üzlə yenilə."""
        kps_smooth = self._kps_ema.update(
            np.array(face.kps[:5], dtype=np.float32)
        )
        bbox_smooth = self._bbox_ema.update(
            np.asarray(face.bbox, dtype=np.float32)
        )
        face_proxy = _FaceProxy(face, kps_smooth, bbox_smooth)
        self._last_face = face_proxy

        # Tracker-i yenilə
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
        self._kps_ema.reset()
        self._bbox_ema.reset()

    @property
    def is_active(self) -> bool:
        return self._last_face is not None


class _FaceProxy:
    """
    insightface face object-inin yüngül proxy-si.
    Hamarlanmış kps + bbox saxlayır, digər atributlar original-dan gəlir.
    """
    def __init__(self, original_face, kps: np.ndarray, bbox: np.ndarray):
        self._orig  = original_face
        self.kps    = kps
        self.bbox   = bbox

    def __getattr__(self, name):
        return getattr(self._orig, name)


# ──────────────────────── sahne keçidi aşkarlaması ────────────────────────

class SceneCutDetector:
    """
    Ardıcıl frame-lər arası LAB color histogram fərqinə görə
    sahne keçidini aşkarla.
    """
    def __init__(self, threshold: float = SCENE_CUT_THRESH):
        self.threshold = threshold
        self._prev_lab: Optional[np.ndarray] = None

    def is_cut(self, frame: np.ndarray) -> bool:
        """True qaytarırsa sahne keçidi var."""
        # Kiçilt — sürət üçün
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


# ──────────────────────── temporal mask blender ────────────────────────

class TemporalMaskBlender:
    """
    Occlusion mask-larını frame-lər arası hamarlaşdır.
    Qəfil görünüb-itən əl/cisim-lər zamanı mask kəskin keçiş etmər.
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


# ──────────────────────── əsas video pipeline ────────────────────────

class VideoFaceSwapper:
    """
    Professional video face swap pipeline.

    Xüsusiyyətlər:
    - Temporal tracking: üzü frame-lər arası izlə, flicker yox
    - Keypoint EMA smoothing: titrəmə aradan qalxır
    - Scene cut aşkarlaması: keçidlərdə tracker sıfırlanır
    - Parallel frame processing: çox core istifadəsi
    - Occlusion tracking: əl/cisim frame-lər arası izlənir
    - Audio copy: orijinal audio keyfiyyətsiz qalmadan kopyalanır
    - GPU auto-detect: CUDA varsa aktiv olur
    """

    def __init__(self,
                 detect_interval: int = DETECT_INTERVAL,
                 tracker_type:   str  = TRACKER_TYPE,
                 max_workers:    int  = MAX_WORKERS,
                 crop_size:      int  = 512):
        self.detect_interval = detect_interval
        self.tracker_type    = tracker_type
        self.max_workers     = max_workers
        self.crop_size       = crop_size

        # Per-face tracker-lər
        self._trackers: List[FaceTracker]  = []
        self._scene_cut = SceneCutDetector()
        self._tmb       = TemporalMaskBlender()
        # Eynək + saç transfer üçün
        self._source_img:      Optional[np.ndarray] = None
        self._source_face_obj  = None

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

        Args:
            source_image:  Mənbə üz şəkli (.jpg/.png)
            input_video:   Hədəf video (.mp4/.avi/...)
            output_video:  Çıxış video yolu
            all_faces:     True → bütün üzlər; False → ən böyük üz
            high_quality:  True → hər frame-ə CodeFormer/GFPGAN tətbiq et
                           (yavaş, yalnız qısa videolar üçün)
            progress_cb:   İsteğe bağlı: callback(percent: int, message: str)
        """
        self._progress(progress_cb, 2, "Mənbə üzü analiz edilir...")
        source_face = extract_source_face(source_image)
        # DÜZƏLTMƏ 2 & 3: Eynək + saç transfer üçün mənbə şəklini saxla
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

        # Müvəqqəti fayl (audio olmadan)
        tmp_video = tempfile.mktemp(suffix="_swap_noaudio.mp4")
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        writer    = cv2.VideoWriter(tmp_video, fourcc, fps, (vid_w, vid_h))

        # State sıfırla
        self._trackers  = []
        self._scene_cut.reset()
        self._tmb.reset()
        self._source_img      = None
        self._source_face_obj = None

        frame_idx = 0
        try:
            # ── Frame oxuma + swap + yazma lopu ──
            # Pipeline: ox → track/detect → swap → yaz
            # Paralel versiya: I/O (ox+yaz) sequential, swap parallel
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
        """
        Frame-ləri sıraya götür, paralel swap et, sıralı yaz.

        Paralel strategiya:
          - Ana thread: oxuma + yazma (sequential — VideoCapture thread-safe deyil)
          - Worker thread-lər: swap (CPU-intensive, paralel)

        Sıra qorunması: frame_idx ilə ordered dict + yazma sırası

        Deadlock qaydası:
          - read_q.put() bloklamadan cəhd edir; dolubsa əvvəlcə write_q boşaldılır
          - write_q-nun ölçüsü limitsizdir → worker heç vaxt bloklanmır
          - swap_lock: tracker state-i qoruyur, amma write_q.put() lock xaricindədir
        """
        QUEUE_DEPTH = self.max_workers * 4
        read_q:  queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
        write_q: queue.Queue = queue.Queue()   # LİMİTSİZ — deadlock yox

        swap_lock  = threading.Lock()   # tracker state-i qorumaq üçün
        frame_idx  = 0
        done_event = threading.Event()

        # ── Worker: frame-ləri swap et ──
        def swap_worker():
            while not done_event.is_set():
                try:
                    item = read_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:
                    write_q.put(None)   # sentinel — lock xaricində
                    read_q.task_done()
                    break
                idx, frame = item
                try:
                    with swap_lock:     # tracker state race-ini önlə
                        processed = self._process_frame(
                            frame, source_face, all_faces, high_quality, idx
                        )
                except Exception as e:
                    log.warning("Frame %d swap xətası: %s", idx, e)
                    processed = frame
                write_q.put((idx, processed))   # lock xaricində — bloklanmır
                read_q.task_done()

        # ── Writer: çıxışa yaz (sıralı) ──
        write_buffer: Dict[int, np.ndarray] = {}
        next_write = 0

        def flush_write_buffer():
            nonlocal next_write
            while next_write in write_buffer:
                writer.write(write_buffer.pop(next_write))
                next_write += 1

        def drain_write_q():
            """write_q-dan bütün hazır frame-ləri al, buffer-a yaz."""
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

        # Worker-ləri başlat
        workers = []
        for _ in range(self.max_workers):
            t = threading.Thread(target=swap_worker, daemon=True)
            t.start()
            workers.append(t)

        sentinel_count = 0

        try:
            while True:
                # Oxu
                ret, frame = cap.read()
                if not ret:
                    for _ in range(self.max_workers):
                        read_q.put(None)
                    break

                # read_q dolubsa əvvəlcə write_q-nu boşalt, sonra yenidən cəhd et
                while True:
                    try:
                        read_q.put_nowait((frame_idx, frame))
                        break
                    except queue.Full:
                        drain_write_q()
                        time.sleep(0.005)

                frame_idx += 1

                # Yazma buffer-ını daim boşalt
                drain_write_q()

                # Mütəmadi progress
                if frame_idx % 10 == 0 and progress_cb:
                    pct = 8 + int((frame_idx / max(total_frames, 1)) * 80)
                    progress_cb(min(pct, 88),
                                f"Frame {frame_idx}/{total_frames}")

            # Worker-lər bitənə qədər gözlə, qalan frame-ləri yaz
            done_event.set()
            for t in workers:
                t.join(timeout=120)

            # Qalan write_q-nu boşalt
            while sentinel_count < self.max_workers:
                drain_write_q()
                time.sleep(0.01)

            # Son flush
            drain_write_q()
            flush_write_buffer()

        except Exception as e:
            log.error("Process loop xətası: %s", e)
            done_event.set()
            raise

    def _process_frame(self,
                       frame: np.ndarray,
                       source_face,
                       all_faces: bool,
                       high_quality: bool,
                       frame_idx: int) -> np.ndarray:
        """
        Bir frame-i işlə: track + detect + swap.
        Thread-safe: hər çağırış müstəqildir (tracker state lock-lanır).
        """
        # Sahne keçidini yoxla
        is_cut = self._scene_cut.is_cut(frame)

        if is_cut:
            log.debug("Scene cut @ frame %d — tracker sıfırlanır", frame_idx)
            for t in self._trackers:
                t.reset()
            self._tmb.reset()

        # Full detect: ilk frame, scene cut, və ya interval
        need_detect = (
            is_cut or
            frame_idx == 0 or
            (frame_idx % self.detect_interval == 0)
        )

        if need_detect:
            detected = _safe_detect(frame)
            if not all_faces and detected:
                # Ən böyük üzü seç
                detected = [max(detected,
                               key=lambda f: (f.bbox[2] - f.bbox[0]) *
                                             (f.bbox[3] - f.bbox[1]))]
            self._sync_trackers(frame, detected)

        # Tracker-lərdən cari face-ləri al
        tracked_faces = []
        for tracker in self._trackers:
            face = tracker.update(frame)
            if face is not None:
                tracked_faces.append(face)

        if not tracked_faces:
            return frame

        # Swap — enhancement yalnız üz crop-una tətbiq olunur (tam frame deyil)
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

        # Mənbə şəkildəki eynək/saçı videoya köçür (default söndürülüb — bax
        # ENABLE_GLASSES_HAIR_TRANSFER). Kobud transfer titrəmə/qutu kənarı yaradır.
        if (ENABLE_GLASSES_HAIR_TRANSFER and
                self._source_img is not None and
                self._source_face_obj is not None):
            src_glasses = _extract_glasses_region(
                self._source_img, self._source_face_obj
            )
            if src_glasses is not None:
                for tgt_face in tracked_faces:
                    if _face_is_valid(tgt_face, frame.shape):
                        result = _blend_region_onto_frame(
                            result, src_glasses, tgt_face, "glasses"
                        )

            src_hair = _extract_hair_region(
                self._source_img, self._source_face_obj
            )
            if src_hair is not None:
                for tgt_face in tracked_faces:
                    if _face_is_valid(tgt_face, frame.shape):
                        result = _blend_region_onto_frame(
                            result, src_hair, tgt_face, "hair"
                        )

        return result

    def _sync_trackers(self, frame: np.ndarray, detected_faces: list):
        """
        Yeni aşkarlanan üzləri mövcud tracker-lərə uyğunlaşdır.
        Hungarian matching: bbox IoU əsasında.
        """
        if not detected_faces:
            # Heç üz yoxdur — aktiv tracker-ləri sıfırlama, itmiş say
            return

        if not self._trackers:
            # İlk init
            for i, face in enumerate(detected_faces):
                t = FaceTracker(i, self.tracker_type)
                if t.init(frame, face):
                    self._trackers.append(t)
            return

        # Mövcud tracker-lərin bbox-ları
        active = [(i, t) for i, t in enumerate(self._trackers) if t.is_active]

        # Hər detected face üçün ən yaxın tracker-i tap (IoU)
        matched_tracker_ids = set()
        for face in detected_faces:
            best_iou  = 0.25  # minimum IoU threshold
            best_tidx = None
            for tidx, t in active:
                if tidx in matched_tracker_ids:
                    continue
                if t._last_face is None:
                    continue
                iou = _bbox_iou(face.bbox, t._last_face.bbox)
                if iou > best_iou:
                    best_iou  = iou
                    best_tidx = tidx

            if best_tidx is not None:
                # Mövcud tracker-i yenilə
                self._trackers[best_tidx].reinit_with_face(frame, face)
                matched_tracker_ids.add(best_tidx)
            else:
                # Yeni tracker yarat
                t = FaceTracker(len(self._trackers), self.tracker_type)
                if t.init(frame, face):
                    self._trackers.append(t)

    @staticmethod
    def _progress(cb, pct: int, msg: str):
        if cb:
            try:
                cb(pct, msg)
            except Exception:
                pass


# ──────────────────────── yardımçı funksiyalar ────────────────────────

def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """İki bbox arasında Intersection over Union hesabla."""
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
    """
    Orijinal audio-nu stream copy ilə yapışdır (re-encode yox → keyfiyyət itkisi sıfır).
    Fallback: AAC 192k encode.
    """
    # Əvvəl stream-copy cəhd et
    cmd = [
        "ffmpeg", "-y",
        "-i", processed_video,
        "-i", orig_video,
        "-c:v", "libx264", "-preset", "fast", "-crf", "17",
        "-c:a", "copy",           # audio re-encode yox, orijinal kimi
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0:
        return

    # Fallback: AAC encode
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
                  crop_size:       int = 512) -> str:
    """
    Sadə bir funksiya ilə tam video face swap.

    Args:
        source_image:     Mənbə üz şəkli
        input_video:      Giriş video faylı
        output_video:     Çıxış video faylı
        all_faces:        Bütün üzlər (False = ən böyük üz)
        high_quality:     CodeFormer/GFPGAN enhancement (yavaş)
        progress_cb:      callback(pct: int, msg: str)
        detect_interval:  Hər neçə frame-də tam detect (default: 8)
        tracker_type:     "CSRT" | "KCF" | "MOSSE"
        max_workers:      Paralel swap thread sayı
        crop_size:        Aligned crop ölçüsü (512 standard)

    Returns:
        output_video yolu
    """
    swapper = VideoFaceSwapper(
        detect_interval=detect_interval,
        tracker_type=tracker_type,
        max_workers=max_workers,
        crop_size=crop_size,
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
        description="Video face swap — nim.video / Kling AI pipeline"
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
    args = parser.parse_args()

    def progress(pct, msg):
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r[{bar}] {pct:3d}%  {msg}", end="", flush=True)

    print(f"Mənbə:  {args.source}")
    print(f"Giriş:  {args.input}")
    print(f"Çıxış:  {args.output}")
    print(f"Tracker: {args.tracker}, detect hər {args.detect_interval} frame")
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
    )
    print(f"\n✓ Hazır: {out}")
