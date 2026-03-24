import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze leaf disease dataset distribution."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to plant directory (e.g. ./Apple)"
    )
    return parser.parse_args()


def count_images(directory: Path) -> dict:
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    counts = {}

    for subdir in sorted(directory.iterdir()):
        if not subdir.is_dir():
            continue
        n = sum(
            1 for f in subdir.iterdir()
            if f.suffix in IMAGE_EXTENSIONS
        )
        if n > 0:
            counts[subdir.name] = n

    return counts


def plot_distribution(plant_name: str, counts: dict):
    labels = list(counts.keys())
    values = list(counts.values())

    # Strip the plant prefix from labels for readability
    short_labels = [
        label.replace(f"{plant_name}_", "")
        for label in labels
    ]

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{plant_name} — image distribution", fontsize=14)

    ax_pie.pie(
        values,
        labels=short_labels,
        autopct="%1.1f%%",
        startangle=140
    )
    ax_pie.set_title("Proportion per class")

    bars = ax_bar.bar(short_labels, values)
    ax_bar.set_title("Count per class")
    ax_bar.set_ylabel("Number of images")
    ax_bar.set_xlabel("Disease class")

    # Rotate x labels if there are many classes
    plt.setp(ax_bar.get_xticklabels(), rotation=30, ha='right')

    # Annotate exact counts above each bar
    for bar, val in zip(bars, values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            str(val),
            ha='center', va='bottom', fontsize=9
        )

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    directory = Path(args.directory)

    if not directory.is_dir():
        print(f"Error: '{directory}' is not a valid directory.")
        sys.exit(1)

    # The plant name comes from the directory name itself
    plant_name = directory.name

    counts = count_images(directory)

    if not counts:
        print(f"No image subdirectories found in '{directory}'.")
        sys.exit(1)

    total = sum(counts.values())
    print(f"\n{plant_name} — {total} images across {len(counts)} classes:")
    for cls, n in counts.items():
        pct = n / total * 100
        print(f"  {cls:<40} {n:>5}  ({pct:.1f}%)")

    plot_distribution(plant_name, counts)


if __name__ == "__main__":
    main()