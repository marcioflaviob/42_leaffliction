import argparse
import sys
import getpass
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from Loader import val_transform
from Model import LeafCNN

CHECKPOINT = Path("leaf_cnn.pt")


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint  = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    model       = LeafCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names


def get_transformation(image_path: str) -> np.ndarray:
    """
    Apply the plantCV pipeline from Transformation.py and return the ROI image.
    Falls back to CLAHE contrast enhancement if the pipeline fails.
    """
    try:
        from Transformation import (load_and_blur, make_binary_mask,
                                    fill_mask, get_roi_from_mask)
        img, blurred    = load_and_blur(image_path)
        bin_mask, _     = make_binary_mask(blurred)
        filled          = fill_mask(bin_mask)
        roi_img, _      = get_roi_from_mask(filled, img)
        return cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"  Transformation fallback ({e})")
        img_bgr = cv2.imread(image_path)
        lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l       = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)


@torch.no_grad()
def predict(model: LeafCNN, class_names: list, image_path: str,
            device: torch.device) -> tuple[str, float]:
    img    = Image.open(image_path).convert("RGB")
    tensor = val_transform(img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)[0]
    idx    = probs.argmax().item()
    return class_names[idx], probs[idx].item()


def display(image_path: str, transformed: np.ndarray,
            class_name: str, confidence: float):
    original = np.array(Image.open(image_path).convert("RGB"))

    fig = plt.figure(figsize=(12, 7), facecolor="black")

    ax1    = fig.add_axes([0.02, 0.22, 0.46, 0.73])
    ax2    = fig.add_axes([0.52, 0.22, 0.46, 0.73])
    ax_txt = fig.add_axes([0.0,  0.0,  1.0,  0.20])

    ax1.imshow(original)
    ax1.set_title("Original",    color="white", fontsize=13)
    ax1.axis("off")

    ax2.imshow(transformed)
    ax2.set_title("Transformed", color="white", fontsize=13)
    ax2.axis("off")

    ax_txt.set_facecolor("black")
    ax_txt.axis("off")
    ax_txt.text(0.5, 0.72,
                "===    DL classification    ===",
                ha="center", va="center",
                color="white", fontsize=16, fontweight="bold",
                transform=ax_txt.transAxes)
    ax_txt.text(0.5, 0.25,
                f"Class predicted : {class_name}   ({confidence:.1%})",
                ha="center", va="center",
                color="#00ff88", fontsize=14, fontweight="bold",
                transform=ax_txt.transAxes)

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict the disease class of a leaf image.\n\n"
            "Usage:\n"
            "  python predict.py ./images/Apple_healthy/image.JPG\n"
            "  python predict.py ./images/Apple_scab/img.JPG --model leaf_cnn.pt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image",
                        help="Path to the leaf image to classify.")
    parser.add_argument("--model", default=str(CHECKPOINT),
                        help=f"Model checkpoint path (default: {CHECKPOINT}).")
    args = parser.parse_args()

    image_path      = args.image
    checkpoint_path = Path(args.model)

    if not Path(image_path).is_file():
        print(f"Error: image '{image_path}' not found.")
        sys.exit(1)

    if not checkpoint_path.is_file():
        print(f"Error: checkpoint '{checkpoint_path}' not found.")
        print("       Run 'python Train.py <data_dir>' first.")
        sys.exit(1)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device:     {device}")
    print(f"Image:      {image_path}")
    print(f"Checkpoint: {checkpoint_path}")

    model, class_names = load_model(checkpoint_path, device)
    print(f"Classes:    {class_names}")

    print("Applying transformation...")
    transformed = get_transformation(image_path)

    print("Predicting...")
    class_name, confidence = predict(model, class_names, image_path, device)

    print(f"\nClass predicted : {class_name}  ({confidence:.2%})")

    display(image_path, transformed, class_name, confidence)


if __name__ == "__main__":
    main()
