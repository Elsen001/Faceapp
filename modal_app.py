import modal

app = modal.App("faceapp")

# Mövcud Dockerfile-ı olduğu kimi istifadə edir (CUDA + torch + FaceFusion + modellər)
image = modal.Image.from_dockerfile("Dockerfile")

# Model fayllarını (inswapper, GFPGAN və s.) build-lər arası saxlamaq üçün.
# Bu, hər dəfə redeploy edəndə modellərin YENİDƏN endirilməsinin qarşısını almır
# (onlar Dockerfile-da build-time endirilir), amma runtime-da yaranan
# uploads/outputs kimi məlumatları saxlamaq üçün faydalıdır.
data_volume = modal.Volume.from_name("faceapp-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=3600,          # 1 saat — uzun video emalı üçün
    memory=16384,
    volumes={"/app/outputs": data_volume},
    min_containers=0,      # istifadə olunmayanda tam bağlanır (xərci azaldır)
    scaledown_window=300,  # 5 dəqiqə boşdursa konteyner söndürülür
)
@modal.web_server(port=7860, startup_timeout=300)
def serve():
    import subprocess
    subprocess.Popen(["python", "app.py"], cwd="/app")
