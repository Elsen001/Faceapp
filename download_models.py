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

MODELS = {
    "inswapper_128.onnx": {
        "url": "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
        "size_mb": 554,
        "desc": "INSwapper – əsas üz dəyişdirmə modeli (InsightFace)",
    },
}


def download_with_progress(url: str, dest: Path, label: str):
    print(f"\n⬇  {label}")
    print(f"   URL: {url}")
    print(f"   → {dest}")

    def reporthook(block, block_size, total):
        if total > 0:
            done = block * block_size
            pct = min(done * 100 // total, 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            sys.stdout.write(f"\r   [{bar}] {pct}%  ({done/1e6:.1f}/{total/1e6:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook)
        print("\n   ✓ Tamamlandı!")
    except Exception as e:
        print(f"\n   ✗ Xəta: {e}")
        if dest.exists():
            dest.unlink()
        sys.exit(1)


def main():
    print("=" * 60)
    print("  FaceSwap AI – Model Yükləyici")
    print("=" * 60)

    for name, info in MODELS.items():
        dest = MODEL_DIR / name
        if dest.exists():
            print(f"\n✓ {name} artıq mövcuddur ({dest.stat().st_size / 1e6:.0f} MB)")
            continue

        print(f"\n📦 {name}  ({info['size_mb']} MB)")
        print(f"   {info['desc']}")
        ans = input("   Yükləmək istəyirsiniz? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            download_with_progress(info["url"], dest, name)
        else:
            print("   Ötürüldü.")

    # buffalo_l – InsightFace avtomatik yükləyir
    print("\n\nİndi aşağıdakı əmrlə serveri başladın:")
    print("  python app.py\n")


if __name__ == "__main__":
    main()
