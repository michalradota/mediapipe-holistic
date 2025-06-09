#!/usr/bin/env python3
import os
from ultralytics import YOLO

# 1) Wymuszenie CPU (usuń tę linię, jeśli masz GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def main():
    # 2) Załaduj pretrenowany model
    model = YOLO('yolov8n.pt')

    # 3) Uruchom trening (200 epok)
    model.train(
        data='data.yaml',      # ścieżka do pliku konfiguracyjnego
        epochs=200,            # liczba epok
        imgsz=640,             # rozmiar wejścia 640×640
        batch=8,               # batch size
        workers=6,             # wątki DataLoadera
        device='cpu',          # CPU-only
        save=True,             # zapisuj best.pt i last.pt
        plots=True,            # generuj wykresy
        cache=True             # cache’uj obrazy w pamięci dla szybszego I/O
    )

    # 4) Walidacja na najlepszych wagach i odczyt najlepszej mAP@0.5
    metrics = model.val(
        data='data.yaml',
        weights='runs/detect/train/weights/best.pt'
    )
    best_map50 = metrics.box.map50  # mAP@0.5 :contentReference[oaicite:2]{index=2}
    print(f"\n✅ Trening zakończony. Najlepsza mAP@0.5: {best_map50:.3f}")

if __name__ == '__main__':
    main()
