from plantcv import plantcv as pcv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

def load_and_blur(image_path, ksize=5):
    pcv.params.debug = None
    img, path, filename = pcv.readimage(filename=image_path)
    a_channel = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    a_inv = cv2.bitwise_not(a_channel)
    blurred = pcv.gaussian_blur(img=a_inv, ksize=(ksize, ksize), sigma_x=0, sigma_y=0)
    return img, blurred


def make_binary_mask(blurred, threshold=None):
    if threshold is None:
        otsu_val, _ = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = int(otsu_val)
        print(f"  Otsu threshold auto-selected: {threshold}")

    bin_mask = pcv.threshold.binary(
        gray_img=blurred,
        threshold=threshold,
        object_type='light'
    )
    return bin_mask, threshold


def fill_mask(bin_mask, fill_size=200):
    filled = pcv.fill(bin_img=bin_mask, size=fill_size)
    kernel = np.ones((25, 25), np.uint8)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
    filled = pcv.fill(bin_img=filled.astype(np.uint8), size=fill_size // 2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        filled = np.where(labels == largest, 255, 0).astype(np.uint8)
    return filled


def get_roi_from_mask(filled_mask, original_img):
    contours, _ = cv2.findContours(
        filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No contours found — check your mask.")
    leaf_contour = max(contours, key=cv2.contourArea)
    viz = original_img.copy()
    cv2.drawContours(viz, [leaf_contour], -1, (0, 255, 0), thickness=3)
    x, y, w, h = cv2.boundingRect(leaf_contour)
    cv2.rectangle(viz, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)
    return viz, (x, y, w, h)


def draw_pseudolandmarks(original_img, filled_mask):
    viz = original_img.copy()
    top_x, bottom_x, center_x = pcv.homology.x_axis_pseudolandmarks(
        img=original_img, mask=filled_mask
    )
    top_y, bottom_y, center_y = pcv.homology.y_axis_pseudolandmarks(
        img=original_img, mask=filled_mask
    )

    def draw_points(img, points, color, shape='circle'):
        if points is None:
            return
        pts = np.array(points).reshape(-1, 2)
        for (px, py) in pts:
            px, py = int(px), int(py)
            if shape == 'circle':
                cv2.circle(img, (px, py), radius=5, color=color, thickness=-1)
            else:
                cv2.rectangle(img, (px-4, py-4), (px+4, py+4), color=color, thickness=-1)

    draw_points(viz, top_x,    color=(0,   0,   255), shape='circle')
    draw_points(viz, bottom_x, color=(255, 0,   0  ), shape='circle')
    draw_points(viz, center_x, color=(0,   255, 0  ), shape='circle')
    draw_points(viz, top_y,    color=(0,   0,   255), shape='square')
    draw_points(viz, bottom_y, color=(255, 0,   0  ), shape='square')
    draw_points(viz, center_y, color=(0,   255, 0  ), shape='square')

    return viz, {
        'x_top': top_x, 'x_bottom': bottom_x, 'x_center': center_x,
        'y_top': top_y, 'y_bottom': bottom_y, 'y_center': center_y,
    }


def analyze_object(original_img, filled_mask):
    viz = original_img.copy()
    contours, _ = cv2.findContours(
        filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No contours found.")
    leaf_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(viz, [leaf_contour], -1, (255, 0, 0), thickness=2)
    hull = cv2.convexHull(leaf_contour)
    cv2.drawContours(viz, [hull], -1, (255, 0, 255), thickness=2)
    M = cv2.moments(leaf_contour)
    area = M['m00']
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    if len(leaf_contour) >= 5:
        ellipse = cv2.fitEllipse(leaf_contour)
        (ex, ey), (minor_ax, major_ax), angle = ellipse
        half_major = int(major_ax / 2)
        half_minor = int(minor_ax / 2)
        angle_rad = np.deg2rad(angle)
        x1_maj = int(cx + half_major * np.sin(angle_rad))
        y1_maj = int(cy - half_major * np.cos(angle_rad))
        x2_maj = int(cx - half_major * np.sin(angle_rad))
        y2_maj = int(cy + half_major * np.cos(angle_rad))
        x1_min = int(cx + half_minor * np.cos(angle_rad))
        y1_min = int(cy + half_minor * np.sin(angle_rad))
        x2_min = int(cx - half_minor * np.cos(angle_rad))
        y2_min = int(cy - half_minor * np.sin(angle_rad))
        cv2.line(viz, (x1_maj, y1_maj), (x2_maj, y2_maj), (255, 0, 255), thickness=2)
        cv2.line(viz, (x1_min, y1_min), (x2_min, y2_min), (255, 0, 255), thickness=2)
        cv2.circle(viz, (cx, cy), radius=6, color=(255, 0, 255), thickness=-1)
    else:
        major_ax, minor_ax, angle = None, None, None

    hull_area = cv2.contourArea(hull)
    solidity  = area / hull_area if hull_area > 0 else 0
    perimeter = cv2.arcLength(leaf_contour, closed=True)

    measurements = {
        'area_px':           int(area),
        'perimeter_px':      round(perimeter, 1),
        'convex_hull_area':  round(hull_area, 1),
        'solidity':          round(solidity, 4),
        'major_axis_length': round(major_ax, 1) if major_ax else None,
        'minor_axis_length': round(minor_ax, 1) if minor_ax else None,
        'centroid':          (cx, cy),
        'angle_deg':         round(angle, 2) if angle else None,
    }
    return viz, measurements


# ── PIPELINE ──────────────────────────────────────────────────────────────────
def run_pipeline(image_path, blur_ksize=5, threshold=None, fill_size=200,
                 save_path=None):
    """
    Run the full pipeline on a single image.
    If save_path is provided the figure is saved there instead of displayed.
    """
    original_img, blurred    = load_and_blur(image_path, ksize=blur_ksize)
    bin_mask, used_threshold = make_binary_mask(blurred, threshold=threshold)
    filled_mask              = fill_mask(bin_mask, fill_size=fill_size)
    roi_img, (x, y, w, h)   = get_roi_from_mask(filled_mask, original_img)
    landmark_img, landmarks  = draw_pseudolandmarks(original_img, filled_mask)
    analysis_img, measures   = analyze_object(original_img, filled_mask)

    print(f"  ROI → x:{x}  y:{y}  w:{w}  h:{h}")

    def add_landmark_legend(ax):
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0],[0], marker='o', color='w', markerfacecolor='red',
                   markersize=7, label='Top'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='blue',
                   markersize=7, label='Bottom'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='lime',
                   markersize=7, label='Center'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='black', markersize=7, label='● x-axis  ■ y-axis'),
        ], loc='lower left', fontsize=6, framealpha=0.7)

    def add_analysis_legend(ax):
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0],[0], color='blue',    linewidth=2, label='Leaf perimeter'),
            Line2D([0],[0], color='magenta', linewidth=2, label='Convex hull'),
            Line2D([0],[0], color='magenta', linewidth=2, linestyle='--',
                   label='Major / Minor axis'),
        ], loc='lower left', fontsize=6, framealpha=0.7)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(Path(image_path).name, fontsize=11, y=1.01)

    axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Figure IV.1: Original")

    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title("Figure IV.2: LAB-a inverted + Blur")

    axes[0, 2].imshow(bin_mask, cmap='gray')
    axes[0, 2].set_title(f"Figure IV.3: Binary Mask (thresh={used_threshold})")

    axes[1, 0].imshow(filled_mask, cmap='gray')
    axes[1, 0].set_title("Figure IV.4: Filled Mask")

    axes[1, 1].imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Figure IV.5: ROI Objects")

    axes[1, 2].imshow(cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Figure IV.6: Pseudolandmarks")
    add_landmark_legend(axes[1, 2])

    axes[2, 0].imshow(cv2.cvtColor(analysis_img, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title("Figure IV.7: Analyze Object")
    add_analysis_legend(axes[2, 0])

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)   # free memory
        print(f"  Saved → {save_path}")
    else:
        plt.show()

    return filled_mask, (x, y, w, h), landmarks, measures


# ── DIRECTORY BATCH PROCESSING ────────────────────────────────────────────────
def process_directory(input_dir, output_dir, blur_ksize, threshold, fill_size):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    candidates = sorted(input_path.iterdir())
    images     = [f for f in candidates
                  if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not images:
        print(f"No supported image files found in '{input_dir}'.")
        print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return

    print(f"Found {len(images)} image(s) in '{input_dir}'.")
    print(f"Results will be saved to '{output_dir}'.\n")

    succeeded, skipped = 0, 0

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] Processing: {img_path.name}")
        out_file = output_path / (img_path.stem + "_analysis.png")

        try:
            run_pipeline(
                image_path=str(img_path),
                blur_ksize=blur_ksize,
                threshold=threshold,
                fill_size=fill_size,
                save_path=str(out_file),
            )
            succeeded += 1

        except KeyboardInterrupt:
            raise

        except Exception as e:
            print(f"  ⚠  Skipped '{img_path.name}': {e}")
            skipped += 1

    print(f"\nDone. {succeeded} processed, {skipped} skipped.")


def validate_args(args):
    input_path = Path(args.image_path)

    if not input_path.exists():
        print(f"Error: input path '{args.image_path}' does not exist.")
        sys.exit(1)

    if input_path.is_dir():
        if not args.out:
            print("Error: --out <output_directory> is required when the input is a directory.")
            sys.exit(1)

        out_path = Path(args.out)

        if out_path.exists() and not out_path.is_dir():
            print(f"Error: --out '{args.out}' exists but is not a directory.")
            sys.exit(1)

        out_path.mkdir(parents=True, exist_ok=True)

    else:
        # Single-file mode: if --out was given it must be a directory
        if args.out:
            out_path = Path(args.out)
            if out_path.exists() and not out_path.is_dir():
                print(f"Error: --out '{args.out}' exists but is not a directory.")
                sys.exit(1)
            out_path.mkdir(parents=True, exist_ok=True)

        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Warning: '{input_path.name}' has an unrecognised extension. "
                  f"Will attempt to process anyway.")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Leaf morphology pipeline — single image or batch directory."
    )
    parser.add_argument(
        "image_path",
        help="Path to an image file OR a directory of images."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for saved figures (required when input is a directory)."
    )
    parser.add_argument("--blur_ksize", type=int, default=5,
                        help="Gaussian blur kernel size (default: 5).")
    parser.add_argument("--threshold",  type=int, default=None,
                        help="Binary threshold (default: Otsu auto-select).")
    parser.add_argument("--fill_size",  type=int, default=200,
                        help="Minimum blob area to keep during fill (default: 200).")
    args = parser.parse_args()

    validate_args(args)

    input_path = Path(args.image_path)

    try:
        if input_path.is_dir():
            process_directory(
                input_dir=str(input_path),
                output_dir=args.out,
                blur_ksize=args.blur_ksize,
                threshold=args.threshold,
                fill_size=args.fill_size,
            )
        else:
            # Single image
            save_path = None
            if args.out:
                out_file  = Path(args.out) / (input_path.stem + "_analysis.png")
                save_path = str(out_file)

            run_pipeline(
                image_path=str(input_path),
                blur_ksize=args.blur_ksize,
                threshold=args.threshold,
                fill_size=args.fill_size,
                save_path=save_path,
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C). Exiting cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main()