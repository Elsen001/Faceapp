"""
Model yükləyici köməkçi skript.
İstifadə: python download_models.py
"""
import os
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ── Əsas model (inswapper) ── FaceFusion açıq mənbəsindən ──────────────────
INSWAPPER_URL = (
    "https://github.com/facefusion/facefusion-assets"
    "/releases/download/models-3.0.0/inswapper_128.onnx"
)
INSWAPPER_FP16_URL = (
    "https://github.com/facefusion/facefusion-assets"
    "/releases/download/models-3.0.0/inswapper_128_fp16.onnx"
)

MODELS = {
    "inswapper_128.onnx": {
        "url": INSWAPPER_URL,
        "size_mb": 555,
        "desc": "INSwapper 128 – ən yüksək keyfiyyət (555 MB)",
        "recommended": True,
    },
    "inswapper_128_fp16.onnx": {
        "url": INSWAPPER_FP16_URL,
        "size_mb": 278,
        "desc": "INSwapper 128 FP16 – daha kiçik, sürətli (278 MB)",
        "recommended": False,
    },
}


def download_with_progress(url: str, dest: Path, label: str):
    print(f"\n⬇  {label}")
    print(f"   Mənbə: FaceFusion GitHub Releases (açıq lisenziya)")
    print(f"   → {dest}")

    def reporthook(block, block_size, total):
        if total > 0:
            done = min(block * block_size, total)
            pct = done * 100 // total
            filled = pct // 5
            bar = "█" * filled + "░" * (20 - filled)
            mb_done = done / 1_000_000
            mb_total = total / 1_000_000
            sys.stdout.write(
                f"\r   [{bar}] {pct:3d}%  {mb_done:.1f}/{mb_total:.1f} MB"
            )
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook)
        print(f"\n   ✓ Tamamlandı! ({dest.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"\n   ✗ Xəta: {e}")
        if dest.exists():
            dest.unlink()
        sys.exit(1)


def main():
    print("=" * 60)
    print("  FaceSwap AI – Model Yükləyici")
    print("  Mənbə: github.com/facefusion/facefusion-assets")
    print("=" * 60)

    # İnswapper modelini yüklə (əsas)
    dest = MODEL_DIR / "inswapper_128.onnx"
    if dest.exists() and dest.stat().st_size > 100_000_000:
        print(f"\n✓ inswapper_128.onnx artıq mövcuddur ({dest.stat().st_size/1e6:.0f} MB)")
    else:
        print("\n📦 inswapper_128.onnx (555 MB) – əsas face swap modeli")
        print("   [TÖVSIYYƏ EDİLƏN]")
        ans = input("   Yükləmək istəyirsiniz? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes", "bəli", "b"):
            download_with_progress(INSWAPPER_URL, dest, "inswapper_128.onnx")
        else:
            # FP16 variant
            dest_fp16 = MODEL_DIR / "inswapper_128_fp16.onnx"
            print("\n📦 inswapper_128_fp16.onnx (278 MB) – daha kiçik variant")
            ans2 = input("   FP16 variantı yükləmək istəyirsiniz? [Y/n]: ").strip().lower()
            if ans2 in ("", "y", "yes", "bəli", "b"):
                download_with_progress(INSWAPPER_FP16_URL, dest_fp16, "inswapper_128_fp16.onnx")
                # face_swapper.py-da fp16 yolunu avtomatik tap
                print("\n⚠  face_swapper.py-da SWAPPER_PATH-i fp16 faylına yönləndir:")
                print(f'   SWAPPER_PATH = "{dest_fp16}"')

    # buffalo_l insightface modeli barədə məlumat
    print("\n\nℹ  buffalo_l face detection modeli serveri ilk dəfə işlətdikdə")
    print("   avtomatik yüklənəcək (~200 MB).")
    print("\n✅ Hazırdır! Serveri başladın:")
    print("   python app.py")
    print()


if __name__ == "__main__":
    main()