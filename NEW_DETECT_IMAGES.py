#!/usr/bin/env python3
"""
detect_images.py

Batchowa detekcja przy użyciu wytrenowanego modelu YOLOv8,
poprawiona obsługa unpackowania Bounding Boxes.
"""

import os
from pathlib import Path
import argparse

import cv2
from ultralytics import YOLO

def detect_folder(weights_path: str,
                  src_folder: str,
                  dst_folder: str,
                  imgsz: int = 640,
                  conf: float = 0.25):
    """
    Wczytuje model z wag, iteruje po wszystkich plikach w src_folder,
    wykonuje detekcję i zapisuje obrazy z naniesionymi boxami do dst_folder.
    """
    src = Path(src_folder)
    dst = Path(dst_folder)
    dst.mkdir(parents=True, exist_ok=True)

    # 1) Załaduj model
    model = YOLO(weights_path)

    # 2) Iteracja po wszystkich obrazach
    for img_path in sorted(src.glob("*.*")):
        # a) Predykcja (zwraca listę obiektów DetPrediction)
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            save=False,
            verbose=False
        )

        # b) Wczytaj oryginał
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Błąd wczytania obrazu {img_path.name}, pomijam.")
            continue

        # c) Rysowanie boxów i etykiet
        #    Używamy r.boxes.data – to tensor Nx6: [x1,y1,x2,y2,score,cls]
        for r in results:
            if not hasattr(r, 'boxes') or r.boxes.data.numel() == 0:
                # brak detekcji na tym obrazie
                continue

            # Zamieniamy tensor na numpy array o kształcie (N,6)
            boxes = r.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, score, cls = box
                cls_id = int(cls)
                label = f"{model.names[cls_id]} {score:.2f}"
                color = (0, 255, 0)  # BGR: zielony

                # prostokąt
                cv2.rectangle(
                    img,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    thickness=2
                )
                # tekst etykiety
                cv2.putText(
                    img,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA
                )

        # d) Zapisz wynik
        out_file = dst / img_path.name
        cv2.imwrite(str(out_file), img)
        print(f"✅ Zapisano: {out_file.name}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batchowa detekcja przy użyciu YOLOv8"
    )
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Ścieżka do pliku .pt z wagami modelu"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="inference/images",
        help="Katalog ze zdjęciami do detekcji"
    )
    parser.add_argument(
        "--dest", "-d",
        type=str,
        default="inference/results",
        help="Katalog, gdzie zapisać obrazy z detekcjami"
    )
    parser.add_argument(
        "--imgsz", "-i",
        type=int,
        default=640,
        help="Rozmiar wejściowy (px). Kwadrat: imgsz x imgsz"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.25,
        help="Próg ufności (confidence threshold) [0.0–1.0]"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(">>> YOLOv8 detect batch")
    print(f" weights: {args.weights}")
    print(f" source : {args.source}")
    print(f" dest   : {args.dest}")
    print(f" imgsz  : {args.imgsz}")
    print(f" conf   : {args.conf}\n")

    detect_folder(
        weights_path=args.weights,
        src_folder=args.source,
        dst_folder=args.dest,
        imgsz=args.imgsz,
        conf=args.conf
    )
