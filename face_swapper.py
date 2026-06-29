"""
face_swapper.py  —  nim.video / Kling AI arxitekturası ilə eyni pipeline

Pipeline:
  1. SCRFD/RetinaFace (buffalo_l) → üz + 5 keypoint
  2. Similarity-Transform → 512×512 aligned crop  (insightface-nin daxili normu)
  3. INSwapper 128 → swap  (paste_back=False: raw crop alırıq)
  4. BiSeNet face-parser → piksel-dəqiq maska  (saç, qaş, göz, dəri, ağız)
  5. LAB color-transfer (yalnız mask bölgəsində)
  6. Occlusion-aware paste_back: əl/cisim aşkarlanır, o bölgələr orijinal saxlanılır
  7. Opsional: GFPGAN / CodeFormer enhancement

Üz ölçüsü düzəltməsi:
  • INSwapper paste_back=True işlədəndə hedefin transformation matrixi ilə
    yapışdırır — bu hedefin üz ölçüsünü saxlayır.
  • Həll: mənbənin kps bbox nisbətini hesabla, hedef frame-inə scale edərək
    mənbənin üz ölçüsünü qoru.

Occlusion (əl/ağız) düzəltməsi:
  • Hedef frame-indəki dəri rəngi olmayan bölgələri (əl daxil) aşkar et.
  • Həmin bölgələrdə üz maskasını sıfırla — orijinal şəkil qalsın.
"""

import os, cv2, numpy as np, threading, subprocess, tempfile, logging
from pathlib import Path
from typing import Optional, Callable, Tuple

import insightface
from insightface.app import FaceAnalysis

log = logging.getLogger(__name__)

# ─── GPU / CPU provider auto-select ──────────────────────────────────────────
def _get_providers() -> list:
    try:
        import torch
        if torch.cuda.is_available():
            log.info("CUDA GPU: %s", torch.cuda.get_device_name(0))
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]

_PROVIDERS = _get_providers()
# ─────────────────────────────────────────────────────────────────────────────


# ──────────────────────── global singletons ────────────────────────
_face_app  = None
_swapper   = None
_gfpgan    = None
_codeformer= None
_bisenet   = None          # face parser
_lock = threading.Lock()

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ──────────────────────── ArcFace 5-pt template (112×112) ────────────────────────
# sol göz, sağ göz, burun, sol ağız kənarı, sağ ağız kənarı
_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# ──────────────────────── model loaders ────────────────────────

def _find_swapper_path() -> Optional[str]:
    for name in ("inswapper_128.onnx", "inswapper_128_fp16.onnx"):
        p = MODEL_DIR / name
        if p.exists() and p.stat().st_size > 8_000_000:
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
                    providers=_PROVIDERS,
                )
                ctx = 0 if "CUDAExecutionProvider" in _PROVIDERS else -1
                app.prepare(ctx_id=ctx, det_size=(640, 640))
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
                        "inswapper_128.onnx tapılmadı! models/ qovluğuna yükləyin."
                    )
                _swapper = insightface.model_zoo.get_model(
                    path, providers=_PROVIDERS
                )
    return _swapper


def get_bisenet():
    """
    BiSeNet face parser — 19 sinif semantik seqmentasiya.
    Model mövcud deyilsə None qaytarır; pipeline landmark mask-a fallback edir.
    """
    global _bisenet
    if _bisenet is None:
        with _lock:
            if _bisenet is None:
                try:
                    import torch
                    from torchvision import transforms

                    model_path = MODEL_DIR / "bisenet_face_parsing.pth"
                    if not model_path.exists():
                        _bisenet = "unavailable"
                    else:
                        # facexlib-in BiSeNet-i
                        from facexlib.parsing import init_parsing_model
                        net = init_parsing_model(
                            model_name="bisenet",
                            half=False,
                            device="cpu",
                            model_rootpath=str(MODEL_DIR),
                        )
                        _bisenet = net
                except Exception as e:
                    log.debug("BiSeNet yüklənmədi: %s", e)
                    _bisenet = "unavailable"
    return None if _bisenet == "unavailable" else _bisenet


def get_gfpgan():
    global _gfpgan
    if _gfpgan is None:
        with _lock:
            if _gfpgan is None:
                try:
                    from gfpgan import GFPGANer
                    model_path = MODEL_DIR / "GFPGANv1.4.pth"
                    if model_path.exists():
                        _gfpgan = GFPGANer(
                            model_path=str(model_path),
                            upscale=1,
                            arch="clean",
                            channel_multiplier=2,
                            bg_upsampler=None,
                        )
                    else:
                        _gfpgan = "unavailable"
                except Exception:
                    _gfpgan = "unavailable"
    return None if _gfpgan == "unavailable" else _gfpgan


def get_codeformer():
    global _codeformer
    if _codeformer is None:
        with _lock:
            if _codeformer is None:
                try:
                    import torch
                    model_path = MODEL_DIR / "CodeFormer.pth"
                    if model_path.exists():
                        from basicsr.archs.codeformer_arch import CodeFormer as CF
                        net = CF(
                            dim_embd=512, codebook_size=1024,
                            n_head=8, n_layers=9,
                            connect_list=["32", "64", "128", "256"],
                        )
                        ckpt = torch.load(str(model_path), map_location="cpu")
                        net.load_state_dict(ckpt["params_ema"])
                        net.eval()
                        _codeformer = net
                    else:
                        _codeformer = "unavailable"
                except Exception:
                    _codeformer = "unavailable"
    return None if _codeformer == "unavailable" else _codeformer


# ──────────────────────── face validation ────────────────────────

def _face_is_valid(face, img_shape) -> bool:
    """
    Swap üçün yararlı üzləri filter edir.
    Eynəkli üzlər üçün yumşaldılmış yoxlama: bütün kps-in frame daxilində
    olması tələb edilmir, ən azı 3-ü yetər.
    """
    if face is None:
        return False

    h, w = img_shape[:2]
    MARGIN = 15  # kənar payı piksel
    try:
        x1, y1, x2, y2 = face.bbox
    except Exception:
        return False
    if x1 < -MARGIN or y1 < -MARGIN or x2 > w + MARGIN or y2 > h + MARGIN:
        return False
    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return False

    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 5:
        return False
    kps_arr = np.array(kps, dtype=np.float32)
    if kps_arr.size == 0:
        return False
    # Ən azı 3 keypoint frame daxilindədir (eynəkli üzlər)
    in_frame = (
        (kps_arr[:, 0] >= -MARGIN) & (kps_arr[:, 0] <= w + MARGIN) &
        (kps_arr[:, 1] >= -MARGIN) & (kps_arr[:, 1] <= h + MARGIN)
    )
    if np.sum(in_frame) < 3:
        return False

    emb = getattr(face, "normed_embedding", None)
    if emb is None or (hasattr(emb, "size") and emb.size == 0):
        return False

    return True


def _safe_detect(img: np.ndarray) -> list:
    """Detect + validate — yalnız swap üçün yararlı üzlər."""
    try:
        faces = get_face_app().get(img)
    except Exception as e:
        log.warning("detect_faces xətası: %s", e)
        return []
    return [f for f in (faces or []) if _face_is_valid(f, img.shape)]


# ──────────────────────── source face extraction ────────────────────────

def extract_source_face(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Şəkil oxunmadı: {image_path}")

    h, w = img.shape[:2]
    # Kiçik şəkilləri böyüt
    if max(h, w) < 640:
        scale = 640 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_LANCZOS4)

    faces = _safe_detect(img)
    if not faces:
        img2 = cv2.resize(img, (1024, 1024))
        faces = _safe_detect(img2)
        if not faces:
            raise ValueError("Şəkildə istifadəyə yararlı üz tapılmadı. "
                             "Aydın, düz baxışlı üz şəkli seçin.")

    return max(faces,
               key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


# ──────────────────────── alignment helpers ────────────────────────

def _similarity_transform(src_pts: np.ndarray,
                           dst_pts: np.ndarray) -> Optional[np.ndarray]:
    """
    Optimal similarity (scale+rotate+translate) transform hesabla.
    cv2.estimateAffinePartial2D  →  (2×3) matrix
    """
    M, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.LMEDS,
        ransacReprojThreshold=5.0,
    )
    return M  # None ola bilər


def _align_face(img: np.ndarray, face,
                crop_size: int = 512) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    5 keypoint əsasında üzü `crop_size×crop_size` ArcFace template-ə warp et.
    Eynəkli üzlər üçün: frame kənарında olan kps-lər clamp edilir.
    Returns: (aligned_img, M_2x3)  və ya  (None, None)
    """
    kps = np.array(face.kps[:5], dtype=np.float32)
    dst = _TEMPLATE_112 * (crop_size / 112.0)

    # Eynəkli üzlər: frame kənarındakı kps-ləri clamp et
    h, w = img.shape[:2]
    kps_clamped = kps.copy()
    kps_clamped[:, 0] = np.clip(kps_clamped[:, 0], 0, w - 1)
    kps_clamped[:, 1] = np.clip(kps_clamped[:, 1], 0, h - 1)

    M = _similarity_transform(kps_clamped, dst)
    if M is None:
        return None, None
    aligned = cv2.warpAffine(img, M, (crop_size, crop_size),
                              flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REFLECT)
    return aligned, M


def _compute_face_scale_ratio(src_face, tgt_face) -> float:
    """
    Mənbənin üz ölçüsünü hesabla vs hedef üzünün ölçüsü.
    INSwapper hedefin formasını saxladığı üçün bu ratio ilə düzəltmə edirik.

    Kps-lər arası göz məsafəsi əsasında nisbət tapılır:
      ratio > 1: mənbə daha böyük → swapped üzü frame-də böyüdərək yenidən align et
      ratio < 1: mənbə daha kiçik → kiçilt
    """
    def _eye_dist(face):
        kps = np.array(face.kps[:2], dtype=np.float32)  # sol göz, sağ göz
        return float(np.linalg.norm(kps[0] - kps[1]))

    src_dist = _eye_dist(src_face)
    tgt_dist = _eye_dist(tgt_face)
    if tgt_dist < 1e-3:
        return 1.0
    return src_dist / tgt_dist


def _paste_back(swapped_crop: np.ndarray,
                base_img: np.ndarray,
                M: np.ndarray,
                face_mask_crop: np.ndarray,
                tgt_face=None,
                src_face=None) -> np.ndarray:
    """
    Aligned crop-u inverse warp ilə orijinal frame-ə yapışdır.

    ÜZ ÖLÇÜsü düzəltməsi:
      INSwapper hedefin üz ölçülərini saxlayır. Bunu düzəltmək üçün
      swapped crop-u mənbənin üz ölçüsünə uyğun scale edib yenidən
      alignment matrixi tətbiq edirik.

    OCCLUSION (əl/ağız):
      Hedef frame-indəki üz bbox bölgəsini ağız altından analiz edirik.
      Ağız bölgəsinin aşağısında (çənə) kontrast dəri olmayan region varsa
      (əl, barmaq) o piklser orijinal qalır.
    """
    h, w = base_img.shape[:2]
    crop_size = swapped_crop.shape[0]

    # ── Üz ölçüsü düzəltməsi ──
    # Mənbənin göz məsafəsi / hedefin göz məsafəsi = scale factor
    # Bu scale-i aligned crop üzərindəki template məsafəsinə tətbiq edirik
    if src_face is not None and tgt_face is not None:
        scale_ratio = _compute_face_scale_ratio(src_face, tgt_face)
        # Yalnız 0.7–1.4 aralığında tətbiq et (ekstremal halları yox et)
        scale_ratio = float(np.clip(scale_ratio, 0.70, 1.40))

        if abs(scale_ratio - 1.0) > 0.05:  # fərq 5%-dən böyükdürsə tətbiq et
            # Crop mərkəzindən scale et
            cx, cy = crop_size / 2.0, crop_size / 2.0
            # Scale matrix (mərkəz ətrafında)
            S = np.array([
                [scale_ratio, 0,           cx * (1 - scale_ratio)],
                [0,           scale_ratio, cy * (1 - scale_ratio)],
            ], dtype=np.float64)
            swapped_crop = cv2.warpAffine(
                swapped_crop, S, (crop_size, crop_size),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT,
            )
            face_mask_crop = cv2.warpAffine(
                face_mask_crop, S, (crop_size, crop_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

    M_inv = cv2.invertAffineTransform(M)

    # Swapped crop-u tam frame ölçüsünə warp et
    swapped_full = cv2.warpAffine(
        swapped_crop, M_inv, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT,
    )

    # Maska da geri warp
    mask_full = cv2.warpAffine(
        face_mask_crop, M_inv, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # ── Occlusion aşkarlaması: əl/cisim ağız önündədirsə ──
    # Metod: hedef face bbox-ının aşağı yarısında (ağız+çənə bölgəsi)
    # dəri rəngi olmayan pikselləri tap → əl/cisim var
    if tgt_face is not None:
        try:
            mask_full = _remove_occlusion(base_img, mask_full, tgt_face)
        except Exception as e:
            log.debug("Occlusion removal xətası: %s", e)

    # Kənarları yumşat
    mask_full = cv2.GaussianBlur(mask_full, (25, 25), 10)
    mask_f = mask_full.astype(np.float32) / 255.0

    result = (
        swapped_full.astype(np.float32) * mask_f[..., None]
        + base_img.astype(np.float32) * (1 - mask_f[..., None])
    ).astype(np.uint8)
    return result


def _remove_occlusion(base_img: np.ndarray,
                      mask_full: np.ndarray,
                      tgt_face) -> np.ndarray:
    """
    Hedef frame-indəki üz bölgəsini analiz edib əl/cisim occlusion-u aşkarla.

    Yanaşma:
    1. Üz bbox-ının ağız bölgəsini (aşağı 40%) ayır
    2. YCrCb rəng modelində dəri rəngi olan pikselləri tap
    3. Dəri rəngi olmayan amma maska içindəki pikselləri occlusion say
    4. Həmin bölgəni maskadan çıxar

    Bu üsul hedefin öz dəri rəngini referans götürür, buna görə
    fərqli dəri tonları üçün avtomatik uyğunlaşır.
    """
    h, w = base_img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in tgt_face.bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    bh, bw = y2 - y1, x2 - x1
    if bh < 10 or bw < 10:
        return mask_full

    # ── Dəri rəngini referans bölgədən öyrən (üzün yuxarı 40%) ──
    ref_y1 = y1
    ref_y2 = y1 + int(bh * 0.45)
    ref_roi = base_img[ref_y1:ref_y2, x1:x2]
    if ref_roi.size == 0:
        return mask_full

    # YCrCb-də dəri pikselləri
    ref_ycrcb = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    # Dəri rəngi statistikası (YCrCb)
    skin_mask_ref = (
        (ref_ycrcb[:, :, 1] >= 130) & (ref_ycrcb[:, :, 1] <= 185) &
        (ref_ycrcb[:, :, 2] >= 75)  & (ref_ycrcb[:, :, 2] <= 135)
    )
    if np.sum(skin_mask_ref) < 50:
        # Dəri aşkarlanmadı — referans əldə edilmədi, occlusion etməyək
        return mask_full

    # Dəri pikselləri Cr, Cb ortalama/std
    cr_vals = ref_ycrcb[:, :, 1][skin_mask_ref]
    cb_vals = ref_ycrcb[:, :, 2][skin_mask_ref]
    cr_mean, cr_std = float(np.mean(cr_vals)), float(np.std(cr_vals)) + 8.0
    cb_mean, cb_std = float(np.mean(cb_vals)), float(np.std(cb_vals)) + 8.0

    # ── Ağız+çənə bölgəsini analiz et (aşağı 55%) ──
    occ_y1 = y1 + int(bh * 0.50)
    occ_y2 = y2
    occ_roi = base_img[occ_y1:occ_y2, x1:x2]
    if occ_roi.size == 0:
        return mask_full

    occ_ycrcb = cv2.cvtColor(occ_roi, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    # Bu bölgədə hedefin dəri rənginə uymayan pikselləri tap
    cr_occ = occ_ycrcb[:, :, 1]
    cb_occ = occ_ycrcb[:, :, 2]
    non_skin = (
        (np.abs(cr_occ - cr_mean) > 2.2 * cr_std) |
        (np.abs(cb_occ - cb_mean) > 2.2 * cb_std)
    )

    # Non-skin occlusion xəritəsini tam frame ölçüsünə çevir
    occlusion_map = np.zeros((h, w), dtype=np.uint8)
    non_skin_u8 = non_skin.astype(np.uint8) * 255
    # Kiçik gürültüyü sil
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    non_skin_u8 = cv2.morphologyEx(non_skin_u8, cv2.MORPH_OPEN, kernel_open)
    occlusion_map[occ_y1:occ_y2, x1:x2] = non_skin_u8

    # Yalnız mask bölgəsindəki occlusion-u nəzərə al
    occlusion_in_mask = (occlusion_map > 128) & (mask_full > 64)
    if not np.any(occlusion_in_mask):
        return mask_full

    # Occlusion bölgəsini genişlət və yumşat
    occ_u8 = occlusion_in_mask.astype(np.uint8) * 255
    occ_u8 = cv2.dilate(occ_u8,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                         iterations=2)
    occ_u8 = cv2.GaussianBlur(occ_u8, (21, 21), 8)
    occ_f = occ_u8.astype(np.float32) / 255.0

    # Maskadan çıxar
    mask_f = mask_full.astype(np.float32) / 255.0
    mask_f = mask_f * (1.0 - occ_f)
    return (mask_f * 255).astype(np.uint8)


# ──────────────────────── face mask ────────────────────────

def _bisenet_mask(aligned_img: np.ndarray, crop_size: int) -> Optional[np.ndarray]:
    """
    BiSeNet ilə piksel-dəqiq üz maskası.
    19 sinif: 1=dəri, 2=qaş-sol, 3=qaş-sağ, 4=göz-sol, 5=göz-sağ,
              7=qulaq-sol, 8=qulaq-sağ, 10=burun, 11=diş, 12=üst dodaq,
              13=alt dodaq, 17=saç
    Maskaya daxil edilən siniflər: 1-13, 17 (saçı bir qədər daxil et)
    """
    net = get_bisenet()
    if net is None:
        return None
    try:
        import torch
        from torchvision import transforms

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225)),
        ])
        img_rgb = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
        inp = to_tensor(img_rgb).unsqueeze(0)
        with torch.no_grad():
            out = net(inp)[0]
        parsing = out.squeeze(0).argmax(0).numpy().astype(np.uint8)
        # Üz bölgəsi sinifləri
        face_classes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17}
        mask = np.isin(parsing, list(face_classes)).astype(np.uint8) * 255
        mask = cv2.resize(mask, (crop_size, crop_size),
                          interpolation=cv2.INTER_LINEAR)

        # Morfologiya: dəlikləri bağla, kənarları genişlət
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask
    except Exception as e:
        log.debug("BiSeNet mask xətası: %s", e)
        return None


def _landmark_mask(aligned_img_shape: tuple,
                   aligned_face,
                   crop_size: int) -> np.ndarray:
    """
    Fallback: 106-nöqtə landmark convex hull maskası.
    aligned_face — aligned crop üzərindəki üz.
    """
    mask = np.zeros((crop_size, crop_size), dtype=np.uint8)

    lm = getattr(aligned_face, "landmark_2d_106", None)
    if lm is not None and len(lm) >= 4:
        pts = np.clip(lm.astype(np.int32), 0, crop_size - 1)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        # bbox əsasında ellips
        try:
            x1, y1, x2, y2 = [int(v) for v in aligned_face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(crop_size - 1, x2), min(crop_size - 1, y2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            rx = max(1, int((x2 - x1) * 0.44))
            ry = max(1, int((y2 - y1) * 0.48))
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        except Exception:
            # ən kötü hal: mərkəz ellips
            cx, cy = crop_size // 2, crop_size // 2
            cv2.ellipse(mask, (cx, cy),
                        (crop_size // 3, int(crop_size * 0.38)),
                        0, 0, 360, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _build_face_mask(aligned_img: np.ndarray,
                     aligned_face,
                     crop_size: int) -> np.ndarray:
    """BiSeNet → landmark → ellips sıra ilə ən yaxşı maskı seç."""
    mask = _bisenet_mask(aligned_img, crop_size)
    if mask is not None and np.sum(mask > 0) > (crop_size * crop_size * 0.05):
        return mask
    if aligned_face is not None:
        return _landmark_mask(aligned_img.shape, aligned_face, crop_size)
    # son fallback: mərkəz ellips
    mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
    cx, cy = crop_size // 2, crop_size // 2
    cv2.ellipse(mask, (cx, cy),
                (crop_size // 3, int(crop_size * 0.38)),
                0, 0, 360, 255, -1)
    return mask


# ──────────────────────── color correction ────────────────────────

def _lab_color_match(src: np.ndarray,
                     ref: np.ndarray,
                     mask: np.ndarray) -> np.ndarray:
    """
    LAB rəng transferi.
    src  = swap edilmiş üz (aligned space)
    ref  = orijinal aligned üz (hedef rəng)
    mask = üz maskası (uint8)
    Yalnız mask bölgəsindəki statistika istifadə olunur.
    """
    if not np.any(mask > 20):
        return src

    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    out_lab = src_lab.copy()

    for ch in range(3):
        s_mean, s_std = cv2.meanStdDev(src_lab[:, :, ch], mask=mask)
        r_mean, r_std = cv2.meanStdDev(ref_lab[:, :, ch], mask=mask)
        s_std, r_std = float(s_std), float(r_std)
        s_mean, r_mean = float(s_mean), float(r_mean)
        if s_std < 1e-6 or r_std < 1e-6:
            continue
        ratio = r_std / s_std
        # L kanalı (parlaqlıq): 40% transfer — "plastik" görünüşü önlər
        weight = 0.40 if ch == 0 else 1.0
        transferred = (src_lab[:, :, ch] - s_mean) * ratio + r_mean
        out_lab[:, :, ch] = (
            (1 - weight) * src_lab[:, :, ch] + weight * transferred
        )

    result_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


# ──────────────────────── enhancement ────────────────────────

def enhance_face(img: np.ndarray) -> np.ndarray:
    """CodeFormer → GFPGAN → yüngül sharpen sıra ilə."""
    cf = get_codeformer()
    if cf is not None:
        try:
            import torch
            from basicsr.utils import img2tensor, tensor2img
            from torchvision.transforms.functional import normalize as vtnorm

            t = img2tensor(img / 255.0, bgr2rgb=True, float32=True)
            vtnorm(t, [0.5]*3, [0.5]*3, inplace=True)
            with torch.no_grad():
                out = cf(t.unsqueeze(0), w=0.7, adain=True)[0]
            enh = tensor2img(out, rgb2bgr=True, min_max=(-1, 1))
            if enh is not None:
                return enh.astype(np.uint8)
        except Exception as e:
            log.debug("CodeFormer xətası: %s", e)

    gf = get_gfpgan()
    if gf is not None:
        try:
            _, _, enh = gf.enhance(img,
                                    has_aligned=False,
                                    only_center_face=False,
                                    paste_back=True)
            if enh is not None:
                # Yüngül sharpen
                k = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]],
                              dtype=np.float32) * 0.4
                return cv2.filter2D(enh, -1, k)
        except Exception as e:
            log.debug("GFPGAN xətası: %s", e)

    return img


# ──────────────────────── ana swap funksiyası ────────────────────────

def _swap_single_face(frame: np.ndarray,
                      tgt_face,
                      src_face,
                      crop_size: int = 512) -> np.ndarray:
    """
    Bir hədəf üzü üçün tam pipeline:
    align → detect-in-crop → swap → color-match → paste_back (scale + occlusion)
    """
    swapper = get_swapper()

    # ── 1. Align ──
    aligned, M = _align_face(frame, tgt_face, crop_size)
    if aligned is None or M is None:
        try:
            swapped = swapper.get(frame.copy(), tgt_face, src_face, paste_back=True)
            return swapped if swapped is not None else frame
        except Exception:
            return frame

    # ── 2. Aligned crop-da üz tap (INSwapper üçün) ──
    aligned_faces = _safe_detect(aligned)
    if not aligned_faces:
        try:
            swapped = swapper.get(frame.copy(), tgt_face, src_face, paste_back=True)
            return swapped if swapped is not None else frame
        except Exception:
            return frame

    aligned_tgt = max(aligned_faces,
                      key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # ── 3. INSwapper swap ──
    try:
        swapped_crop = swapper.get(aligned.copy(), aligned_tgt, src_face,
                                   paste_back=True)
    except Exception as e:
        log.warning("swapper.get() xətası: %s", e)
        return frame

    if swapped_crop is None:
        return frame

    # ── 4. Üz maskası ──
    mask = _build_face_mask(aligned, aligned_tgt, crop_size)

    # ── 5. LAB color match (aligned space-də) ──
    swapped_colored = _lab_color_match(swapped_crop, aligned, mask)

    # ── 6. Inverse warp + ölçü düzəltməsi + occlusion-aware blend ──
    result = _paste_back(swapped_colored, frame, M, mask,
                         tgt_face=tgt_face, src_face=src_face)
    return result


def swap_frame(frame: np.ndarray,
               source_face,
               all_faces: bool = False,
               high_quality: bool = False,
               crop_size: int = 512) -> np.ndarray:
    """
    Frame-dəki üzləri dəyişdirən əsas funksiya.
    """
    faces = _safe_detect(frame)
    if not faces:
        return frame

    if all_faces:
        target_faces = faces
    else:
        target_faces = [max(faces,
                            key=lambda f: (f.bbox[2] - f.bbox[0]) *
                                          (f.bbox[3] - f.bbox[1]))]

    result = frame.copy()
    for tgt in target_faces:
        result = _swap_single_face(result, tgt, source_face, crop_size)

    if high_quality:
        result = enhance_face(result)

    return result


# ──────────────────────── public API ────────────────────────

def process_image(source_path: str,
                  target_path: str,
                  output_path: str,
                  all_faces: bool = False) -> str:
    source_face = extract_source_face(source_path)
    target = cv2.imread(target_path)
    if target is None:
        raise ValueError(f"Hədəf şəkil oxunmadı: {target_path}")

    result = swap_frame(target, source_face, all_faces, high_quality=True)

    ext = Path(output_path).suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, 98] if ext in (".jpg", ".jpeg") else []
    cv2.imwrite(output_path, result, params)
    return output_path


def process_video(video_path: str,
                  source_image_path: str,
                  output_path: str,
                  all_faces: bool = False,
                  progress_cb: Optional[Callable] = None) -> str:
    if progress_cb:
        progress_cb(5, "Mənbə üzü analiz edilir...")
    source_face = extract_source_face(source_image_path)

    if progress_cb:
        progress_cb(8, "Video açılır...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video açılmadı.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 9999
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = tempfile.mktemp(suffix="_noaudio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp, fourcc, fps, (w, h))

    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = swap_frame(frame, source_face, all_faces,
                                   high_quality=False)
            writer.write(processed)
            idx += 1
            if progress_cb and idx % 4 == 0:
                pct = 10 + int((idx / max(total, 1)) * 78)
                progress_cb(min(pct, 88), f"Frame {idx}/{total}")
    finally:
        cap.release()
        writer.release()

    if progress_cb:
        progress_cb(90, "Audio birləşdirilir...")

    try:
        _merge_audio(video_path, tmp, output_path)
    except Exception:
        import shutil
        shutil.copy2(tmp, output_path)

    try:
        os.remove(tmp)
    except Exception:
        pass

    if progress_cb:
        progress_cb(100, "Tamamlandı!")
    return output_path


def _merge_audio(orig: str, processed: str, out: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", processed, "-i", orig,
        "-c:v", "libx264", "-preset", "fast", "-crf", "17",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest", out,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg: {r.stderr.decode(errors='replace')}")


def check_model_available() -> bool:
    return _find_swapper_path() is not None


def get_model_info() -> dict:
    path = _find_swapper_path()
    return {
        "swapper_path": path,
        "swapper_exists": path is not None,
        "bisenet_available": get_bisenet() is not None,
        "model_dir": str(MODEL_DIR),
        "insightface_version": insightface.__version__,
    }
