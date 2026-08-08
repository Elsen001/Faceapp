"""
Professional Face Swap Pipeline v3.0
=====================================
1.  SCRFD/RetinaFace (buffalo_l)      — dəqiq üz + 5 KPS
2.  Similarity Transform (112→512)     — 3D geometry alignment
3.  INSwapper 128                      — swap
4.  BiSeNet Face Parsing               — piksel-dəqiq maska (AĞIZ DÜZƏLİŞİ İLƏ)
5.  Face Relighting                    — işıq uyumu
6.  Skin Texture Matching              — dəri toxuması
7.  LAB Color Transfer (mask-daxili)   — rəng uyumu
8.  Occlusion Handling                 — əl/cisim aşkarlanması
9.  Poisson Seamless Blend             — kənar birləşmə
10. GFPGAN Enhancement                 — üz bərpası
11. Temporal Stabilization (video)     — frame sabitliyi
12. Cinsiyyət uyğunluğu (şəkil + video) — eyni cinsə swap
13. Üz kilidi (şəkil + video)          — ilk üzü saxla
14. Ağız maskası düzəlişi              — əşya/əl ağıza toxunanda
"""
import os, cv2, numpy as np, threading, subprocess, tempfile, logging
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import insightface
from insightface.app import FaceAnalysis

log = logging.getLogger(__name__)

try:
    import torch
    _CUDA_OK = torch.cuda.is_available()
except Exception:
    _CUDA_OK = False

# GPU seçilibsə CUDA provider, deyilsə CPU-ya geri düşür
_ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"] if _CUDA_OK else ["CPUExecutionProvider"]
_TORCH_DEVICE = "cuda" if _CUDA_OK else "cpu"
if _CUDA_OK:
    log.info(f"GPU aktivdir: {torch.cuda.get_device_name(0)} (classic swapper CUDA istifadə edəcək)")
else:
    log.info("GPU tapılmadı, classic swapper CPU-da işləyəcək")

_face_app = None
_swapper  = None
_gfpgan   = None
_bisenet  = None
_lock     = threading.Lock()
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

ARCFACE_SRC = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

_prev_faces: List = []
_locked_face_id: Optional[int] = None  # Şəkil üçün üz kilidi


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
                app = FaceAnalysis(name="buffalo_l", root=str(MODEL_DIR),
                                   providers=_ORT_PROVIDERS)
                # ctx_id=0 GPU-nu göstərir (insightface-də); CPU-dursa avtomatik CPU-ya keçir
                app.prepare(ctx_id=0 if _CUDA_OK else -1, det_size=(640, 640))
                _face_app = app
    return _face_app


def get_swapper():
    global _swapper
    if _swapper is None:
        with _lock:
            if _swapper is None:
                path = _find_swapper_path()
                if not path:
                    raise FileNotFoundError("inswapper_128.onnx tapilmadi!")
                _swapper = insightface.model_zoo.get_model(
                    path, providers=_ORT_PROVIDERS)
    return _swapper


def get_bisenet():
    global _bisenet
    if _bisenet is None:
        with _lock:
            if _bisenet is None:
                try:
                    from facexlib.parsing import init_parsing_model
                    net = init_parsing_model(model_name="bisenet", half=False,
                                            device=_TORCH_DEVICE, model_rootpath=str(MODEL_DIR))
                    _bisenet = net
                except Exception as e:
                    log.debug("BiSeNet: %s", e)
                    _bisenet = "unavailable"
    return None if _bisenet == "unavailable" else _bisenet


def get_gfpgan():
    global _gfpgan
    if _gfpgan is None:
        with _lock:
            if _gfpgan is None:
                try:
                    from gfpgan import GFPGANer
                    mp = MODEL_DIR / "GFPGANv1.4.pth"
                    if mp.exists():
                        _gfpgan = GFPGANer(model_path=str(mp), upscale=1,
                                           arch="clean", channel_multiplier=2,
                                           bg_upsampler=None, device=_TORCH_DEVICE)
                    else:
                        _gfpgan = "unavailable"
                except Exception:
                    _gfpgan = "unavailable"
    return None if _gfpgan == "unavailable" else _gfpgan


def _is_valid(face, shape) -> bool:
    if face is None:
        return False
    try:
        x1,y1,x2,y2 = face.bbox
    except Exception:
        return False
    if (x2-x1)<20 or (y2-y1)<20:
        return False
    kps = getattr(face,"kps",None)
    if kps is None or len(kps)<5:
        return False
    if getattr(face,"normed_embedding",None) is None:
        return False
    return True


def safe_detect(img: np.ndarray) -> list:
    try:
        faces = get_face_app().get(img)
        return [f for f in (faces or []) if _is_valid(f, img.shape)]
    except Exception:
        return []


def extract_source_face(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Sekil oxunmadi: {image_path}")
    h,w = img.shape[:2]
    if max(h,w)<512:
        s=512/max(h,w)
        img=cv2.resize(img,(int(w*s),int(h*s)),interpolation=cv2.INTER_LANCZOS4)
    faces = safe_detect(img)
    if not faces:
        faces = safe_detect(cv2.resize(img,(1024,1024)))
        if not faces:
            raise ValueError("Sekilde uz tapilmadi.")
    return max(faces, key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))


def _similarity_transform(kps: np.ndarray, size: int) -> Optional[np.ndarray]:
    dst = ARCFACE_SRC * (size / 112.0)
    M,_ = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst, method=cv2.LMEDS)
    return M


def _align_face(img, face, size=512):
    kps = getattr(face,"kps",None)
    if kps is None:
        return None, None
    M = _similarity_transform(np.array(kps), size)
    if M is None:
        return None, None
    aligned = cv2.warpAffine(img, M, (size,size),
                             flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REFLECT)
    return aligned, M


def _get_mouth_region(h, w) -> np.ndarray:
    """Ağız bölgəsini təxmini olaraq müəyyən edir (BiSeNet olmadıqda fallback)"""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2 + int(h * 0.15)
    # Ağız üçün ellips
    cv2.ellipse(mask, (cx, cy), (int(w * 0.25), int(h * 0.12)), 0, 0, 360, 255, -1)
    return mask


def _bisenet_mask(img_bgr: np.ndarray, include_hair=True, fix_mouth=True) -> np.ndarray:
    """
    BiSeNet ilə üz maskası.
    fix_mouth=True olduqda ağız bölgəsi mütləq maskaya daxil edilir.
    """
    bisenet = get_bisenet()
    h,w = img_bgr.shape[:2]
    if bisenet is None:
        mask = _ellipse_mask(h, w)
        if fix_mouth:
            mouth = _get_mouth_region(h, w)
            mask = cv2.bitwise_or(mask, mouth)
        return mask
    
    try:
        import torch
        from torchvision import transforms
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])(img_rgb).unsqueeze(0)
        with torch.no_grad():
            out = bisenet(t)[0]
        parsing = out.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
        parsing = cv2.resize(parsing,(w,h),interpolation=cv2.INTER_NEAREST)
        
        # Əsas üz hissələri
        parts = list(range(1,13)) + ([13,14,17] if include_hair else [])
        mask = np.zeros((h,w),dtype=np.uint8)
        for p in parts:
            mask[parsing==p] = 255
        
        # Ağız hissəsini mütləq əlavə et (əgər fix_mouth aktivdirsə)
        if fix_mouth:
            # Ağız hissələri: 12 (ağız), 13 (üst dodaq), 14 (alt dodaq)
            mouth_parts = [12, 13, 14]
            for p in mouth_parts:
                if p < 20:  # BiSeNet indeksləri
                    mask[parsing==p] = 255
            
            # Əgər ağız hələ də yoxdursa, təxmini ağız əlavə et
            if np.sum(mask > 0) < 1000:
                mouth = _get_mouth_region(h, w)
                mask = cv2.bitwise_or(mask, mouth)
        
        # Morphology ilə maskanı hamarlaşdır
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask
    except Exception as e:
        log.debug("bisenet mask: %s", e)
        mask = _ellipse_mask(h, w)
        if fix_mouth:
            mouth = _get_mouth_region(h, w)
            mask = cv2.bitwise_or(mask, mouth)
        return mask


def _ellipse_mask(h, w) -> np.ndarray:
    mask = np.zeros((h,w),dtype=np.uint8)
    cx,cy = w//2, h//2
    cv2.ellipse(mask,(cx,cy),(int(w*0.42),int(h*0.50)),0,0,360,255,-1)
    cv2.ellipse(mask,(cx,cy-int(h*0.18)),(int(w*0.36),int(h*0.22)),0,0,360,255,-1)
    return mask


def _estimate_light(img, mask):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = (mask>128).astype(np.float32)
    gx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=5)
    gy = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=5)
    lx = float(np.sum(gx*m))/(np.sum(m)+1e-6)
    ly = float(np.sum(gy*m))/(np.sum(m)+1e-6)
    lz = 128.0
    n = np.sqrt(lx**2+ly**2+lz**2)+1e-6
    return np.array([lx/n, ly/n, lz/n])


def _relight(src, tgt, sm, tm):
    sl = _estimate_light(src, sm)
    tl = _estimate_light(tgt, tm)
    src_lab = cv2.cvtColor(src,cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt,cv2.COLOR_BGR2LAB).astype(np.float32)
    sbool = sm>128; tbool = tm>128
    if not sbool.any() or not tbool.any():
        return src
    res = src_lab.copy()
    sL = src_lab[:,:,0]; tL = tgt_lab[:,:,0]
    sm_v=float(sL[sbool].mean()); ss=float(sL[sbool].std())+1e-6
    tm_v=float(tL[tbool].mean()); ts=float(tL[tbool].std())+1e-6
    ldiff = float(np.dot(sl,tl))
    w = np.clip(0.5+ldiff*0.35, 0.35, 0.75)
    new_L = (sL-sm_v)*(ts/ss)+tm_v
    res[:,:,0] = np.clip(sL*(1-w)+new_L*w, 0, 255)
    return cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _skin_texture(src, tgt, sm, tm, strength=0.30):
    sbool = sm>128
    if not sbool.any():
        return src
    base_src = cv2.GaussianBlur(src,(0,0),3.0)
    base_tgt = cv2.GaussianBlur(tgt,(0,0),3.0)
    det_src = cv2.subtract(src, base_src)
    det_tgt = cv2.subtract(tgt, base_tgt)
    blended_det = (det_src.astype(np.float32)*(1-strength) +
                   det_tgt.astype(np.float32)*strength)
    res_full = np.clip(base_src.astype(np.float32)+blended_det,0,255).astype(np.uint8)
    m3 = sbool[:,:,None]
    return np.where(m3, res_full, src).astype(np.uint8)


def _lab_color(src, ref, mask):
    m = mask>30
    if not m.any():
        return src
    sl = cv2.cvtColor(src,cv2.COLOR_BGR2LAB).astype(np.float32)
    rl = cv2.cvtColor(ref,cv2.COLOR_BGR2LAB).astype(np.float32)
    res = sl.copy()
    l_diff = abs(float(sl[:,:,0][m].mean())-float(rl[:,:,0][m].mean()))
    lw = np.clip(0.55+l_diff/200.0, 0.55, 0.85)
    for c,w in enumerate([lw,0.75,0.75]):
        sv=sl[:,:,c][m]; rv=rl[:,:,c][m]
        sm_v=float(sv.mean()); ss=float(sv.std())+1e-6
        rm_v=float(rv.mean()); rs=float(rv.std())+1e-6
        ch=sl[:,:,c].copy()
        adj=(ch-sm_v)*(rs/ss)+rm_v
        ch[m]=np.clip(ch[m]*(1-w)+adj[m]*w,0,255)
        res[:,:,c]=ch
    return cv2.cvtColor(np.clip(res,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR)


def _occlusion(tgt_frame, face_mask):
    h,w = tgt_frame.shape[:2]
    occ = np.zeros((h,w),dtype=np.uint8)
    fa = face_mask>128
    if not fa.any():
        return occ
    ycrcb = cv2.cvtColor(tgt_frame,cv2.COLOR_BGR2YCrCb)
    skin = ((ycrcb[:,:,1]>=133)&(ycrcb[:,:,1]<=173)&
            (ycrcb[:,:,2]>=77) &(ycrcb[:,:,2]<=127))
    ns = fa & ~skin
    occ[ns] = 255
    k = np.ones((7,7),np.uint8)
    occ = cv2.morphologyEx(occ,cv2.MORPH_OPEN,k)
    occ = cv2.morphologyEx(occ,cv2.MORPH_CLOSE,k)
    return occ


def _poisson(src, dst, mask):
    try:
        ys,xs = np.where(mask>128)
        if len(xs)==0:
            return dst
        cx=int((xs.min()+xs.max())//2)
        cy=int((ys.min()+ys.max())//2)
        _,bm=cv2.threshold(mask,128,255,cv2.THRESH_BINARY)
        return cv2.seamlessClone(src,dst,bm,(cx,cy),cv2.NORMAL_CLONE)
    except Exception:
        soft=cv2.GaussianBlur(mask,(31,31),15)
        m=soft[:,:,None].astype(np.float32)/255.0
        return (src.astype(np.float32)*m+dst.astype(np.float32)*(1-m)).astype(np.uint8)


def _paste_back(swapped, original, face, M, crop_mask, size):
    h,w = original.shape[:2]
    M_inv = cv2.invertAffineTransform(M)
    warped = cv2.warpAffine(swapped, M_inv, (w,h),
                            flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_REPLICATE)
    mask_back = cv2.warpAffine(crop_mask, M_inv, (w,h), flags=cv2.INTER_LINEAR)
    mask_back = cv2.GaussianBlur(mask_back,(15,15),7)
    occ = _occlusion(original, mask_back)
    if occ.any():
        mask_back[occ>0] = 0
    return _poisson(warped, original, mask_back)


def _stabilize(face, prev_faces, smooth=0.55):
    if not prev_faces:
        return face
    x1,y1,x2,y2 = face.bbox
    best_iou=0; best_prev=None
    for pf in prev_faces[-3:]:
        px1,py1,px2,py2 = pf.bbox
        ix1,iy1=max(x1,px1),max(y1,py1)
        ix2,iy2=min(x2,px2),min(y2,py2)
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        union=(x2-x1)*(y2-y1)+(px2-px1)*(py2-py1)-inter
        iou=inter/(union+1e-6)
        if iou>best_iou:
            best_iou=iou; best_prev=pf
    if best_prev is None or best_iou<0.3:
        return face
    if (hasattr(face,'kps') and face.kps is not None and
        hasattr(best_prev,'kps') and best_prev.kps is not None):
        face.kps = face.kps*(1-smooth)+best_prev.kps*smooth
    return face


def _enhance(img):
    gf = get_gfpgan()
    if gf is None:
        blur=cv2.GaussianBlur(img,(0,0),1.5)
        return cv2.addWeighted(img,1.6,blur,-0.6,0)
    try:
        _,_,enh=gf.enhance(img,has_aligned=True,only_center_face=True,paste_back=True)
        return enh if enh is not None else img
    except Exception as e:
        log.debug("GFPGAN: %s",e)
        return img


def _match_gender(source_face, target_faces):
    """
    Cinsiyyət uyğunluğu — mənbə ilə eyni cinsiyyətdəki üzləri seç.
    InsightFace-in gender atributundan istifadə edir.
    0 = qadın, 1 = kişi
    EĞER CİNSİYYƏT TAPILMAZSA — BÜTÜN ÜZLƏRİ QAYTAR
    """
    src_gender = getattr(source_face, 'gender', None)
    
    # Əgər mənbə cinsiyyəti yoxdursa, bütün üzləri qaytar
    if src_gender is None:
        return target_faces
    
    # Hər üzün cinsiyyətini yoxla, əgər yoxdursa — burax
    matched = []
    for f in target_faces:
        f_gender = getattr(f, 'gender', None)
        # Əgər üzün cinsiyyəti yoxdursa, onu da daxil et (fallback)
        if f_gender is None:
            matched.append(f)
        elif f_gender == src_gender:
            matched.append(f)
    
    # Heç bir üz uyğun gəlmirsə, bütün üzləri qaytar (fallback)
    return matched if matched else target_faces


def _swap_single(frame, tgt_face, src_face, size=512, high_quality=False):
    swapper = get_swapper()

    aligned, M = _align_face(frame, tgt_face, size)
    if aligned is None or M is None:
        try:
            res = swapper.get(frame.copy(), tgt_face, src_face, paste_back=True)
            return res if res is not None else frame
        except Exception:
            return frame

    al_faces = safe_detect(aligned)
    if not al_faces:
        try:
            res = swapper.get(frame.copy(), tgt_face, src_face, paste_back=True)
            return res if res is not None else frame
        except Exception:
            return frame

    al_tgt = max(al_faces, key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

    try:
        swapped = swapper.get(aligned.copy(), al_tgt, src_face, paste_back=True)
    except Exception as e:
        log.warning("swap xetasi: %s", e)
        return frame
    if swapped is None:
        return frame

    # fix_mouth=True ilə maska yarat — ağız mütləq daxil edilir
    crop_mask = _bisenet_mask(swapped, include_hair=True, fix_mouth=True)
    if crop_mask.max()==0:
        crop_mask = _ellipse_mask(*swapped.shape[:2])
        # Ağız əlavə et
        mouth = _get_mouth_region(*swapped.shape[:2])
        crop_mask = cv2.bitwise_or(crop_mask, mouth)

    al_mask = _bisenet_mask(aligned, include_hair=False, fix_mouth=True)

    # Relighting
    swapped = _relight(swapped, aligned, crop_mask, al_mask)
    # Skin texture
    swapped = _skin_texture(swapped, aligned, crop_mask, al_mask, strength=0.30)
    # LAB color
    swapped = _lab_color(swapped, aligned, crop_mask)
    # GFPGAN
    if high_quality:
        swapped = _enhance(swapped)

    soft_mask = cv2.GaussianBlur(crop_mask,(21,21),10)
    return _paste_back(swapped, frame, tgt_face, M, soft_mask, size)


def swap_frame(frame, source_face, all_faces=False, high_quality=False, video_mode=False):
    global _prev_faces, _locked_face_id
    faces = safe_detect(frame)
    if not faces:
        _prev_faces = []
        _locked_face_id = None
        return frame

    # Cinsiyyətə görə filtr (şəkil üçün də)
    faces = _match_gender(source_face, faces)
    if not faces:
        # Fallback: cinsiyyət uyğun gəlmirsə, bütün üzləri götür
        faces = safe_detect(frame)

    if video_mode:
        faces = [_stabilize(f, _prev_faces) for f in faces]
        _prev_faces = faces

    # Şəkil üçün üz kilidi mexanizmi
    if not video_mode:
        if _locked_face_id is None and faces:
            # İlk dəfədir, ən böyük üzü kilitlə
            _locked_face_id = 0
            target_faces = [max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))]
        elif _locked_face_id is not None and faces:
            # Kilitli üzü saxlamağa çalış
            if not hasattr(swap_frame, '_locked_face_obj'):
                # İlk işləmə - ən böyük üzü götür
                swap_frame._locked_face_obj = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            
            locked_face = swap_frame._locked_face_obj
            best = None
            best_iou = 0.35
            for f in faces:
                iou = _bbox_iou(f.bbox, locked_face.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best = f
            if best is not None:
                target_faces = [best]
                swap_frame._locked_face_obj = best
            else:
                target_faces = [max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))]
                swap_frame._locked_face_obj = target_faces[0]
        else:
            target_faces = faces if all_faces else [
                max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            ]
    else:
        target_faces = faces if all_faces else [
            max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        ]

    result = frame.copy()
    for tgt in target_faces:
        result = _swap_single(result, tgt, source_face, size=512, high_quality=high_quality)
    return result


def _bbox_iou(a, b) -> float:
    """İki bounding box arası IoU hesablayır"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (a_area + b_area - inter)


def process_image(source_path, target_path, output_path, all_faces=False):
    global _locked_face_id
    _locked_face_id = None
    if hasattr(swap_frame, '_locked_face_obj'):
        delattr(swap_frame, '_locked_face_obj')
    
    source_face = extract_source_face(source_path)
    target = cv2.imread(target_path)
    if target is None:
        raise ValueError(f"Hedef oxunmadi: {target_path}")
    if not safe_detect(target):
        raise ValueError("Hedef sekilde uz tapilmadi.")
    result = swap_frame(target, source_face, all_faces, high_quality=True, video_mode=False)
    ext = Path(output_path).suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY,97] if ext in (".jpg",".jpeg") else []
    cv2.imwrite(output_path, result, params)
    return output_path


def process_video(video_path, source_image_path, output_path,
                  all_faces=False, progress_cb=None):
    global _prev_faces, _locked_face_id
    _prev_faces = []
    _locked_face_id = None
    if hasattr(swap_frame, '_locked_face_obj'):
        delattr(swap_frame, '_locked_face_obj')

    if progress_cb:
        progress_cb(2,"Mənbə üzü analiz edilir...")
    source_face = extract_source_face(source_image_path)

    if progress_cb:
        progress_cb(5,"Video acilir...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video acilmadi.")

    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 9999
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp=tempfile.mktemp(suffix="_noaudio.mp4")
    writer=cv2.VideoWriter(tmp,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))

    idx=0
    try:
        while True:
            ret,frame=cap.read()
            if not ret:
                break
            processed=swap_frame(frame,source_face,all_faces,
                                 high_quality=False,video_mode=True)
            writer.write(processed)
            idx+=1
            if progress_cb and idx%3==0:
                pct=10+int((idx/max(total,1))*78)
                progress_cb(min(pct,88),f"Frame {idx}/{total}")
    finally:
        cap.release()
        writer.release()

    if progress_cb:
        progress_cb(90,"Audio birlesdirilir...")

    cmd=["ffmpeg","-y","-i",tmp,"-i",video_path,
         "-c:v","libx264","-preset","fast","-crf","16",
         "-c:a","aac","-b:a","192k",
         "-map","0:v:0","-map","1:a:0?","-shortest",output_path]
    r=subprocess.run(cmd,capture_output=True)
    if r.returncode!=0:
        subprocess.run(["ffmpeg","-y","-i",tmp,
                        "-c:v","libx264","-preset","fast","-crf","16",output_path],
                       capture_output=True,check=True)
    try:
        os.remove(tmp)
    except Exception:
        pass

    if progress_cb:
        progress_cb(100,"Tamamlandi!")
    return output_path


def check_model_available():
    return _find_swapper_path() is not None


def get_model_info():
    path=_find_swapper_path()
    return {
        "swapper_path":path,
        "swapper_exists":path is not None,
        "bisenet_available":get_bisenet() is not None,
        "gfpgan_available":get_gfpgan() is not None,
        "model_dir":str(MODEL_DIR),
        "insightface_version":insightface.__version__,
    }


# ── video_face_swapper.py üçün alias-lar ──────────────────────────────────

def _safe_detect(img):
    return safe_detect(img)

def _swap_single_face(frame, tgt_face, src_face, crop_size=512, enhance=False):
    return _swap_single(frame, tgt_face, src_face, size=crop_size, high_quality=enhance)

def _face_is_valid(face, shape):
    return _is_valid(face, shape)

def enhance_face(img, only_center_face=False):
    return _enhance(img)