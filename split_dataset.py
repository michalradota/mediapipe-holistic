import random
import shutil
from pathlib import Path

# 1. Wejściowy katalog z obrazami i etykietami
raw_dir = Path('raw_images')
images = list(raw_dir.glob('*.jpg'))
random.seed(42)
random.shuffle(images)

# 2. Ustalenie splitu
split_idx = int(0.8 * len(images))
train_imgs = images[:split_idx]
val_imgs   = images[split_idx:]

# 3. Tworzenie katalogów
for subset in ('train','val'):
    (Path('dataset/images')/subset).mkdir(parents=True, exist_ok=True)
    (Path('dataset/labels')/subset).mkdir(parents=True, exist_ok=True)

# 4. Kopiowanie plików
for img_list, subset in ((train_imgs,'train'), (val_imgs,'val')):
    for img_path in img_list:
        lbl_path = img_path.with_suffix('.txt')
        shutil.copy(img_path, Path('dataset/images')/subset/img_path.name)
        shutil.copy(lbl_path, Path('dataset/labels')/subset/lbl_path.name)
