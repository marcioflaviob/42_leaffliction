import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from Loader import load_datasets, val_transform, _TransformSubset
from Model import LeafCNN

CHECKPOINT = Path("leaf_cnn.pt")


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint  = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    model       = LeafCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names


@torch.no_grad()
def evaluate(model, loader, class_names, device):
    n_classes = len(class_names)
    correct_per_class = [0] * n_classes
    total_per_class   = [0] * n_classes

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)

        for label, pred in zip(labels, preds):
            total_per_class[label.item()]  += 1
            if pred == label:
                correct_per_class[label.item()] += 1

    overall_correct = sum(correct_per_class)
    overall_total   = sum(total_per_class)
    overall_acc     = overall_correct / overall_total

    print(f"\n{'Class':<30} {'Correct':>7} {'Total':>7} {'Acc':>7}")
    print("-" * 55)
    for i, name in enumerate(class_names):
        acc = correct_per_class[i] / total_per_class[i] if total_per_class[i] else 0.0
        print(f"{name:<30} {correct_per_class[i]:>7} {total_per_class[i]:>7} {acc:>7.2%}")

    print("-" * 55)
    print(f"{'OVERALL':<30} {overall_correct:>7} {overall_total:>7} {overall_acc:>7.2%}")

    if overall_acc >= 0.90:
        print(f"\nPASS — accuracy {overall_acc:.2%} >= 90%")
    else:
        print(f"\nFAIL — accuracy {overall_acc:.2%} < 90%")

    return overall_acc


def main(data_dir: Path):
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Using device: {device}")
    print(f"Checkpoint:   {CHECKPOINT}")
    print(f"Data dir:     {data_dir}")

    model, class_names = load_model(CHECKPOINT, device)

    # If the data_dir is the same augmented_directory used in training,
    # re-create the fixed-seed val split so we test on the held-out 20%.
    # If it's a separate test directory, load every image in it.
    _, val_loader, _ = load_datasets(data_dir)

    print(f"\nEvaluating on {len(val_loader.dataset)} images...")
    evaluate(model, val_loader, class_names, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", nargs="?",
        default="augmented_directory",
        help="Root of the ImageFolder dataset (default: augmented_directory)"
    )
    args = parser.parse_args()
    main(Path(args.data_dir))
