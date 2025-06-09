import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# ------------------------
# 1. Konfiguracja
# ------------------------
mp_holistic = mp.solutions.holistic
mp_pose     = mp.solutions.pose
mp_hands    = mp.solutions.hands
mp_face     = mp.solutions.face_mesh

# Mapowanie nazw części ciała na ID dla YOLO (0–5)
CLASS_MAP = {
    'Head':       0,
    'Torso':      1,
    'Left Hand':  2,
    'Right Hand': 3,
    'Left Leg':   4,
    'Right Leg':  5
}

# Indeksy landmarków dla tułowia i nóg
PART_INDICES = {
    'Torso':     [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                  mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                  mp_pose.PoseLandmark.LEFT_HIP.value,
                  mp_pose.PoseLandmark.RIGHT_HIP.value],
    'Left Leg':  [mp_pose.PoseLandmark.LEFT_HIP.value,
                  mp_pose.PoseLandmark.LEFT_KNEE.value,
                  mp_pose.PoseLandmark.LEFT_ANKLE.value],
    'Right Leg': [mp_pose.PoseLandmark.RIGHT_HIP.value,
                  mp_pose.PoseLandmark.RIGHT_KNEE.value,
                  mp_pose.PoseLandmark.RIGHT_ANKLE.value]
}

# ------------------------
# 2. Funkcja pomocnicza do wyliczenia YOLO-boksa
# ------------------------
def compute_norm_bbox(landmarks):
    """
    landmarks: lista obiektów z atrybutami .x, .y (wartości w [0,1])
    Zwraca: (x_center, y_center, width, height) – wszystkie znormalizowane
    """
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width    = (x_max - x_min)
    height   = (y_max - y_min)
    return x_center, y_center, width, height

# ------------------------
# 3. Przetwarzanie pojedynczego obrazu i zapis etykiet
# ------------------------
def label_image(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠️ Nie można wczytać obrazu: {image_path}")
        return

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # detekcja landmarków
    with mp_holistic.Holistic(static_image_mode=True,
                              model_complexity=2,
                              enable_segmentation=False,
                              min_detection_confidence=0.5) as hol:
        res = hol.process(img_rgb)

    lines = []

    # 3.1 Głowa
    if res.face_landmarks:
        xc, yc, ww, hh = compute_norm_bbox(res.face_landmarks.landmark)
        lines.append(f"{CLASS_MAP['Head']} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

    # 3.2 Tułów i nogi
    if res.pose_landmarks:
        for part_name, idxs in PART_INDICES.items():
            pts = [res.pose_landmarks.landmark[i] for i in idxs]
            xc, yc, ww, hh = compute_norm_bbox(pts)
            lines.append(f"{CLASS_MAP[part_name]} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

    # 3.3 Ręce
    if res.left_hand_landmarks:
        xc, yc, ww, hh = compute_norm_bbox(res.left_hand_landmarks.landmark)
        lines.append(f"{CLASS_MAP['Left Hand']} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
    if res.right_hand_landmarks:
        xc, yc, ww, hh = compute_norm_bbox(res.right_hand_landmarks.landmark)
        lines.append(f"{CLASS_MAP['Right Hand']} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

    # 3.4 Zapis do pliku .txt
    txt_path = image_path.with_suffix('.txt')
    txt_path.write_text("\n".join(lines))
    print(f"✔️ Etykiety zapisane: {txt_path.name}")

# ------------------------
# 4. Pętla po folderze z obrazami
# ------------------------
if __name__ == "__main__":
    raw_dir = Path("raw_images")   # tu trzymaj swoje oryginalne zdjęcia
    image_files = list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpg"))
    if not image_files:
        print("🔍 Nie znaleziono plików *.png ani *.jpg w raw_images/")
    for img_path in image_files:
        label_image(img_path)
