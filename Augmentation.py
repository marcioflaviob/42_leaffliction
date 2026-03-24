import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#  The 6 augmentation functions 
def augment_flip(img: Image.Image) -> Image.Image:
    """Horizontal mirror."""
    return ImageOps.mirror(img)


def augment_rotate(img: Image.Image, angle: float = 30.0) -> Image.Image:
    """Rotate by a fixed angle, expand canvas to avoid cropping."""
    return img.rotate(angle, expand=True, resample=Image.BICUBIC)


def augment_skew(img: Image.Image) -> Image.Image:
    """
    Perspective skew — shifts the top-left and top-right corners inward,
    simulating a camera tilt. Uses PIL's transform with PERSPECTIVE.
    """
    w, h = img.size
    # Coefficients: (x0,y0, x1,y1, x2,y2, x3,y3) are the four destination
    # corners mapped from the original unit square.
    skew_factor = 0.15
    coeffs = _find_perspective_coeffs(
        [(0, 0), (w, 0), (w, h), (0, h)],                        # dst
        [(w * skew_factor, 0), (w * (1 - skew_factor), 0),       # src top
         (w, h), (0, h)]                                          # src bottom
    )
    return img.transform(
        (w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC
    )


def augment_shear(img: Image.Image, shear: float = 0.2) -> Image.Image:
    """
    Horizontal affine shear using PIL's AFFINE transform.
    Matrix: | 1  shear  0 |
            | 0    1    0 |
    """
    w, h = img.size
    # Shift origin so shear is applied around the centre
    x_shift = abs(shear) * h / 2
    new_w = w + int(x_shift)
    affine = (1, shear, -x_shift if shear > 0 else 0,
              0, 1, 0)
    return img.transform(
        (new_w, h), Image.AFFINE, affine, Image.BICUBIC
    )


def augment_crop(img: Image.Image, crop_pct: float = 0.15) -> Image.Image:
    """
    Centre crop by removing crop_pct from each edge, then resize back
    to the original dimensions.
    """
    w, h = img.size
    margin_x = int(w * crop_pct)
    margin_y = int(h * crop_pct)
    cropped = img.crop((margin_x, margin_y, w - margin_x, h - margin_y))
    return cropped.resize((w, h), Image.LANCZOS)


def augment_distortion(img: Image.Image, grid: int = 4,
                       magnitude: float = 12.0) -> Image.Image:
    """
    Elastic / barrel distortion via a randomised mesh warp.
    Divides the image into a grid of cells and randomly displaces
    each grid vertex, then uses PIL's MESH transform.
    """
    w, h = img.size
    cell_w = w // grid
    cell_h = h // grid

    rng = np.random.default_rng(42)   # fixed seed → reproducible output
    mesh = []

    for row in range(grid):
        for col in range(grid):
            # Destination rectangle (where we want pixels to land)
            x0, y0 = col * cell_w, row * cell_h
            x1, y1 = x0 + cell_w, y0 + cell_h

            # Source quad: each corner is slightly displaced
            def jitter(v, limit):
                return int(v + rng.uniform(-magnitude, magnitude))

            src = [
                jitter(x0, w), jitter(y0, h),
                jitter(x1, w), jitter(y0, h),
                jitter(x1, w), jitter(y1, h),
                jitter(x0, w), jitter(y1, h),
            ]
            mesh.append(((x0, y0, x1, y1), src))

    return img.transform((w, h), Image.MESH, mesh, Image.BICUBIC)

def _find_perspective_coeffs(dst_pts, src_pts):
    """
    Solve for the 8 coefficients of a projective (perspective) transform
    mapping src_pts → dst_pts.
    PIL's PERSPECTIVE transform needs these 8 values.
    """
    matrix = []
    for (x, y), (X, Y) in zip(dst_pts, src_pts):
        matrix.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y])
        matrix.append([0, 0, 0, X, Y, 1, -y * X, -y * Y])
    A = np.matrix(matrix, dtype=float)
    B = np.array([x for pt in dst_pts for x in pt], dtype=float)
    res = np.linalg.solve(A, B)
    return np.array(res).flatten().tolist()


#  apply all 6 augmentations, save, display

AUGMENTATIONS = [
    ("Flip",       augment_flip),
    ("Rotate",     augment_rotate),
    ("Skew",       augment_skew),
    ("Shear",      augment_shear),
    ("Crop",       augment_crop),
    ("Distortion", augment_distortion),
]


def process_image(image_path: Path, save: bool = True, show: bool = True):
    """
    Load one image, produce all 6 augmented variants, optionally save
    them to the same directory and display a 7-panel matplotlib figure.
    """
    img = Image.open(image_path).convert("RGB")
    stem = image_path.stem        # e.g. 'image (1)'
    suffix = image_path.suffix    # e.g. '.JPG'
    parent = image_path.parent    # e.g. Path('./Apple/apple_healthy')

    results = []
    for name, fn in AUGMENTATIONS:
        augmented = fn(img)
        results.append((name, augmented))
        if save:
            out_path = parent / f"{stem}_{name}{suffix}"
            augmented.save(out_path)
            print(f"  Saved: {out_path}")

    if show:
        _display_grid(img, results, stem)

    return results


def _display_grid(original: Image.Image, results: list, title: str):
    """Show original + 6 augmented images in a 2-row × 4-col grid."""
    all_images = [("Original", original)] + results  # 7 total

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(title, fontsize=12)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.1)

    for idx, (name, im) in enumerate(all_images):
        row, col = divmod(idx, 4)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(im)
        ax.set_title(name, fontsize=9)
        ax.axis("off")

    # Hide the unused 8th cell (2×4 grid = 8 slots, we use 7)
    ax_empty = fig.add_subplot(gs[1, 3])
    ax_empty.axis("off")

    plt.tight_layout()
    plt.show()


#  Balancing

def balance_directory(root: Path):
    """
    Walk all subdirectories of root, find the max image count,
    and augment the smaller classes until they match.

    Augmented files are placed inside the existing subdirectory.
    A copy of the whole tree is written to root/../augmented_directory/.
    """
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    # Count originals per class
    class_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    counts = {}
    for d in class_dirs:
        imgs = [f for f in d.iterdir() if f.suffix in IMAGE_EXTENSIONS]
        counts[d] = imgs

    if not counts:
        print("No image subdirectories found.")
        return

    max_count = max(len(v) for v in counts.values())
    print(f"\nTarget count per class: {max_count}")

    for class_dir, images in counts.items():
        current = len(images)
        needed = max_count - current
        if needed == 0:
            print(f"  {class_dir.name}: already balanced ({current} images)")
            continue

        print(f"  {class_dir.name}: {current} → need {needed} more")
        aug_cycle = list(images) * ((needed // current) + 2)  # enough to cycle
        generated = 0

        for source_img in aug_cycle:
            if generated >= needed:
                break
            for aug_name, aug_fn in AUGMENTATIONS:
                if generated >= needed:
                    break
                img = Image.open(source_img).convert("RGB")
                augmented = aug_fn(img)
                out_name = f"{source_img.stem}_{aug_name}{source_img.suffix}"
                out_path = class_dir / out_name
                # Avoid overwriting if file exists
                counter = 1
                while out_path.exists():
                    out_path = class_dir / f"{source_img.stem}_{aug_name}_{counter}{source_img.suffix}"
                    counter += 1
                augmented.save(out_path)
                generated += 1

        print(f"    Generated {generated} images for {class_dir.name}")

    # Copy balanced dataset to augmented_directory/
    import shutil
    aug_root = root.parent / "augmented_directory"
    if aug_root.exists():
        shutil.rmtree(aug_root)
    shutil.copytree(root, aug_root)
    print(f"\nBalanced dataset saved to: {aug_root}")


# main
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Augmentation.py — produce 6 augmented variants of a leaf image,\n"
            "or balance an entire dataset directory.\n\n"
            "Usage examples:\n"
            "  Single image:  ./Augmentation.py ./Apple/apple_healthy/image\\ (1).JPG\n"
            "  Balance dir:   ./Augmentation.py --balance ./Apple\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a single image OR a plant directory (with --balance)"
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance all subdirectory classes under PATH to the same count"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip the matplotlib display (useful in headless environments)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.path)

    if args.balance:
        if not path.is_dir():
            print(f"Error: '{path}' is not a directory.")
            sys.exit(1)
        balance_directory(path)
        return

    # Single-image mode
    if not path.is_file():
        print(f"Error: '{path}' is not a file.")
        sys.exit(1)

    print(f"Processing: {path}")
    process_image(path, save=True, show=not args.no_display)
    print("Done.")


if __name__ == "__main__":
    main()