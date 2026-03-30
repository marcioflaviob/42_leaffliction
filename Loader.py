import json
import numpy as np
from pathlib import Path
from PIL import Image

from Augmentation import balance_directory


def augment_and_balance(input_dir: Path) -> Path:
    print(f"Balancing dataset under '{input_dir}'...")
    balance_directory(input_dir)

    aug_dir = input_dir.parent / "augmented_directory"

    if not aug_dir.exists():
        raise RuntimeError(
            f"Expected augmented_directory at '{aug_dir}' but it was not created."
        )

    print(f"Augmented dataset ready at: {aug_dir}")
    return aug_dir


def encode_labels(aug_dir: Path) -> tuple[dict, dict]:
    classes = sorted([d.name for d in aug_dir.iterdir() if d.is_dir()])

    if not classes:
        raise ValueError(f"No subdirectories found in '{aug_dir}'.")

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    print(f"\nFound {len(classes)} classes:")
    for idx, name in idx_to_class.items():
        print(f"  {idx:>3} → {name}")

    return class_to_idx, idx_to_class


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
TARGET_SIZE = (224, 224)


def load_dataset(aug_dir: Path,
                 class_to_idx: dict,
                 target_size: tuple = TARGET_SIZE
                 ) -> tuple[np.ndarray, np.ndarray]:
    image_paths = []
    labels      = []

    for class_dir in sorted(aug_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label_idx = class_to_idx.get(class_dir.name)
        if label_idx is None:
            print(f"  Warning: '{class_dir.name}' not in class map, skipping.")
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix in IMAGE_EXTENSIONS:
                image_paths.append(img_path)
                labels.append(label_idx)

    total = len(image_paths)
    if total == 0:
        raise ValueError(f"No images found under '{aug_dir}'.")

    print(f"\nLoading {total} images "
          f"(originals + all 6 augmented variants) at {target_size}...")

    X = np.empty((total, *target_size, 3), dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    skipped = []

    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(target_size, Image.LANCZOS)
            X[i] = np.array(img, dtype=np.float32) / 255.0

            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"  {i + 1}/{total} loaded...", end="\r")

        except Exception as e:
            print(f"\n  Warning: skipping '{img_path.name}': {e}")
            skipped.append(i)

    if skipped:
        mask = np.ones(total, dtype=bool)
        mask[skipped] = False
        X = X[mask]
        y = y[mask]
        print(f"\n  Skipped {len(skipped)} unreadable files.")

    print(f"\nDataset ready: {X.shape}  (min={X.min():.3f}, max={X.max():.3f})")
    print(f"Label distribution:")
    for idx, name in sorted((v, k) for k, v in class_to_idx.items()):
        count = int((y == idx).sum())
        print(f"  {idx:>3}  {name:<30} {count:>5} images")

    return X, y


def save_classes_json(idx_to_class: dict, out_path: Path):
    payload = {str(k): v for k, v in idx_to_class.items()}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved classes.json → {out_path}")


if __name__ == "__main__":
    input_dir = Path("./Apple")

    aug_dir = augment_and_balance(input_dir)

    class_to_idx, idx_to_class = encode_labels(aug_dir)

    X, y = load_dataset(aug_dir, class_to_idx)

