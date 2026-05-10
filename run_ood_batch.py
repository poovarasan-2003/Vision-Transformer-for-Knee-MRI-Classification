"""
run_ood_batch.py — Out-of-Distribution batch inference
=======================================================
Runs the trained model on a folder of external MRI images (downloaded from
the internet) and prints a results table comparing predictions to the
MRNet validation baseline.

Usage
-----
    python run_ood_batch.py --model_path best_model.pth --image_dir ood_images/

Expected folder layout:
    ood_images/
        acl_tear_1.jpg
        acl_tear_2.png
        meniscal_tear_1.jpg
        meniscal_tear_2.png
        normal_knee_1.jpg
        normal_knee_2.png

The script accepts any common image format (jpg, jpeg, png, bmp, tiff).
Images can be named anything — you will be prompted to supply the ground
truth label for each image so results can be compared properly.

Outputs
-------
- Printed results table (per-image sigmoid probabilities + binary predictions)
- ood_results.csv   — machine-readable results saved to disk
- ood_comparison.png — bar chart comparing OOD vs MRNet val performance
"""

import os
import sys
import argparse
import csv
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import get_model

# Supported image extensions
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# MRNet validation AUC baseline (from your trained model — update these after training)
MRNET_VAL_AUC = {
    'Abnormal': 0.871,
    'ACL':      0.834,
    'Meniscal': 0.798,
}

CONDITIONS = ['Abnormal', 'ACL', 'Meniscal']


def get_transform():
    """Same preprocessing pipeline as MRNet validation set."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def load_image(path: str) -> torch.Tensor:
    """Load an image file and return a (1, 1, 3, 224, 224) tensor.

    OOD images are single 2D images, so we treat them as a sequence of
    length 1 — consistent with how visualize.py handles --image_path mode.
    """
    img = Image.open(path).convert('RGB')
    tensor = get_transform()(img)            # (3, 224, 224)
    return tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 224, 224)


def run_inference(model, image_tensor: torch.Tensor, device) -> np.ndarray:
    """Return sigmoid probabilities for the three conditions."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
    return probs  # shape (3,)


def collect_image_paths(image_dir: str) -> list:
    paths = []
    for fname in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTS:
            paths.append(os.path.join(image_dir, fname))
    return paths


def ask_ground_truth(fname: str) -> str:
    """Prompt the user for the ground truth label of an image."""
    print(f"\n  Image: {fname}")
    print("  What is the ground truth label?")
    print("  [1] Abnormal (unspecified)  [2] ACL tear  [3] Meniscal tear  [4] Normal  [5] Unknown")
    mapping = {'1': 'Abnormal', '2': 'ACL', '3': 'Meniscal', '4': 'Normal', '5': 'Unknown'}
    while True:
        choice = input("  Enter 1–5: ").strip()
        if choice in mapping:
            return mapping[choice]
        print("  Please enter a number between 1 and 5.")


def print_table(results: list):
    header = f"{'Image':<32} {'GT':<10} {'P(Abn)':>7} {'P(ACL)':>7} {'P(Men)':>7}  {'Pred':<20}"
    print("\n" + "="*len(header))
    print(header)
    print("="*len(header))
    for r in results:
        pred_flags = [c for c, p in zip(CONDITIONS, r['probs']) if p > 0.5] or ['Normal']
        pred_str = ', '.join(pred_flags)
        print(f"{r['name']:<32} {r['gt']:<10} "
              f"{r['probs'][0]:>7.3f} {r['probs'][1]:>7.3f} {r['probs'][2]:>7.3f}  {pred_str:<20}")
    print("="*len(header))


def save_csv(results: list, out_path: str):
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'ground_truth', 'p_abnormal', 'p_acl', 'p_meniscal',
                         'pred_abnormal', 'pred_acl', 'pred_meniscal'])
        for r in results:
            writer.writerow([
                r['name'], r['gt'],
                f"{r['probs'][0]:.4f}", f"{r['probs'][1]:.4f}", f"{r['probs'][2]:.4f}",
                int(r['probs'][0] > 0.5), int(r['probs'][1] > 0.5), int(r['probs'][2] > 0.5)
            ])
    print(f"\nResults saved to: {out_path}")


def compute_ood_accuracy(results: list) -> dict:
    """
    Simple per-condition accuracy against ground truth labels.
    Only counts images where ground truth is known (not 'Unknown').
    """
    gt_map = {
        'Abnormal': [1, 0, 0],
        'ACL':      [1, 1, 0],
        'Meniscal': [1, 0, 1],
        'Normal':   [0, 0, 0],
    }
    correct = [0, 0, 0]
    total = 0
    for r in results:
        if r['gt'] not in gt_map:
            continue
        gt = gt_map[r['gt']]
        pred = [int(p > 0.5) for p in r['probs']]
        for i in range(3):
            correct[i] += int(pred[i] == gt[i])
        total += 1

    if total == 0:
        return {c: None for c in CONDITIONS}
    return {c: correct[i] / total for i, c in enumerate(CONDITIONS)}


def plot_comparison(ood_acc: dict, save_path: str):
    """Bar chart: MRNet val AUC vs OOD accuracy side by side."""
    conds = CONDITIONS
    mrnet_vals = [MRNET_VAL_AUC[c] for c in conds]
    ood_vals = [ood_acc.get(c) for c in conds]

    # Filter out conditions with no OOD ground truth
    valid = [(c, m, o) for c, m, o in zip(conds, mrnet_vals, ood_vals) if o is not None]
    if not valid:
        print("No ground truth labels provided — skipping comparison plot.")
        return

    conds_v, mrnet_v, ood_v = zip(*valid)
    x = np.arange(len(conds_v))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, mrnet_v, width, label='MRNet val (AUC)', color='#4f8ef7', zorder=3)
    bars2 = ax.bar(x + width/2, ood_v,   width, label='OOD accuracy',    color='#f7934f', zorder=3)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('In-distribution (MRNet val AUC) vs Out-of-Distribution accuracy', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(conds_v, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = get_model(pretrained=False)
    if not os.path.exists(args.model_path):
        print(f"ERROR: model not found at '{args.model_path}'")
        sys.exit(1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    print(f"Loaded model from: {args.model_path}")

    # Collect images
    image_paths = collect_image_paths(args.image_dir)
    if not image_paths:
        print(f"No images found in '{args.image_dir}'. "
              f"Supported formats: {', '.join(IMAGE_EXTS)}")
        sys.exit(1)
    print(f"\nFound {len(image_paths)} image(s) in '{args.image_dir}'")

    # Run inference on each image
    results = []
    for path in image_paths:
        fname = os.path.basename(path)
        try:
            image_tensor = load_image(path)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
            continue

        probs = run_inference(model, image_tensor, device)

        # Ground truth — interactive or auto-parsed from filename
        if args.auto_label:
            # Try to infer label from filename keywords (convenient for scripted runs)
            fname_lower = fname.lower()
            if 'normal' in fname_lower and 'abnormal' not in fname_lower:
                gt = 'Normal'
            elif 'acl' in fname_lower:
                gt = 'ACL'
            elif 'menisc' in fname_lower:
                gt = 'Meniscal'
            elif 'abnormal' in fname_lower:
                gt = 'Abnormal'
            else:
                gt = 'Unknown'
            print(f"  {fname}  ->  auto-label: {gt}  |  "
                  f"P(Abn)={probs[0]:.3f}  P(ACL)={probs[1]:.3f}  P(Men)={probs[2]:.3f}")
        else:
            gt = ask_ground_truth(fname)

        results.append({'name': fname, 'gt': gt, 'probs': probs})

    if not results:
        print("No images were processed.")
        sys.exit(1)

    # Print summary table
    print_table(results)

    # Compute OOD accuracy vs ground truth
    ood_acc = compute_ood_accuracy(results)
    print("\nOOD per-condition accuracy (vs ground truth labels):")
    for cond, acc in ood_acc.items():
        if acc is not None:
            print(f"  {cond:<10}: {acc:.3f}")
        else:
            print(f"  {cond:<10}: n/a (no labeled examples)")

    print("\nMRNet validation AUC (in-distribution baseline):")
    for cond, auc in MRNET_VAL_AUC.items():
        print(f"  {cond:<10}: {auc:.3f}")

    # Save outputs
    save_csv(results, args.output_csv)
    plot_comparison(ood_acc, args.output_plot)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch OOD inference on external MRI images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--model_path',  type=str, default='best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--image_dir',   type=str, default='ood_images',
                        help='Directory containing external MRI images')
    parser.add_argument('--output_csv',  type=str, default='ood_results.csv',
                        help='Path to save CSV results table')
    parser.add_argument('--output_plot', type=str, default='ood_comparison.png',
                        help='Path to save comparison bar chart')
    parser.add_argument('--auto_label',  action='store_true',
                        help='Infer ground truth from filename keywords '
                             '(acl/menisc/normal/abnormal) instead of prompting')

    main(parser.parse_args())
