# Standard Library
import os           # File path operations
import json         # Exporting metrics to JSON for report integration
import argparse     # CLI argument parsing for flexible script usage

# Scientific Computing
import numpy as np  # Array manipulation for predictions and label stacking

# Deep Learning
import torch        # Core PyTorch for model inference and tensor ops

# Sklearn Metrics
# All metrics are computed per-task (abnormal / acl / meniscus)
from sklearn.metrics import (
    roc_auc_score,          # Primary metric used in MRNet paper (AUC)
    confusion_matrix,       # To derive TP, TN, FP, FN for sens/spec
    f1_score,               # Harmonic mean of precision and recall
    accuracy_score,         # Fraction of correct binary predictions
    roc_curve               # TPR/FPR points for ROC curve plot
)

# Visualisation
import matplotlib
matplotlib.use('Agg')           # Non-interactive backend - safe for Colab/server
import matplotlib.pyplot as plt  # Plotting confusion matrix and ROC curves
import seaborn as sns           # Heatmap for styled confusion matrix

# Project Modules
# Import model factory and dataset class written by teammates
from model import get_model          # Deepak's MRNetViT architecture
from dataset import MRNetDataset     # Kishore's dataset loader
from torchvision import transforms   # Image preprocessing pipeline
from torch.utils.data import DataLoader  # Batch iteration over the dataset

# Constants
# Task names align with the three MRNet label columns: abnormal, acl, meniscus
TASK_NAMES = ['Abnormal', 'ACL', 'Meniscus']

# Task indices map each label column to its position in the model output tensor
# model output shape: (batch, 3) → [0]=abnormal, [1]=acl, [2]=meniscus
TASK_INDEX = {'abnormal': 0, 'acl': 1, 'meniscus': 2}

# ImageNet normalisation constants - same as used in train.py and dataset.py
# Ensures evaluation preprocessing is consistent with training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# SECTION 1: DATA LOADING

def build_eval_transform() -> transforms.Compose:
    """
    Builds the validation transform pipeline.

    Matches the val_transform defined in dataset.py to ensure there is no
    data preprocessing mismatch between training and evaluation.
    No augmentation is applied during evaluation - only resize + normalise.

    Returns:
        transforms.Compose: Preprocessing pipeline for evaluation.
    """
    return transforms.Compose([
        transforms.ToPILImage(),                        # Convert numpy uint8 array → PIL Image
        transforms.Resize((224, 224)),                  # ViT-S/16 expects 224×224 input
        transforms.ToTensor(),                          # Convert PIL → float tensor [0,1]
        transforms.Normalize(                           # Normalise with ImageNet stats
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def get_eval_loader(data_dir: str, plane: str, num_slices: int,
                    batch_size: int, num_workers: int) -> DataLoader:
    """
    Constructs the DataLoader for the validation split.

    Uses Kishore's MRNetDataset, which loads per-exam .npy volumes, samples
    `num_slices` uniformly across the depth axis, normalises each slice,
    and returns a (num_slices, 3, 224, 224) tensor per exam alongside its labels.

    Args:
        data_dir  : Root directory of MRNet-v1.0 (contains train/ and valid/).
        plane     : MRI plane to load - 'sagittal', 'coronal', or 'axial'.
        num_slices: Number of slices to uniformly sample per volume.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of parallel data loading workers.

    Returns:
        DataLoader: Iterable over (images, labels) validation batches.
    """
    # Point to the 'valid' split inside the MRNet root directory
    valid_dir = os.path.join(data_dir, 'valid')

    # Instantiate the dataset with the evaluation transform (no augmentation)
    dataset = MRNetDataset(
        data_dir=valid_dir,
        plane=plane,
        transform=build_eval_transform(),
        num_slices=num_slices
    )

    # shuffle=False is critical for evaluation - preserves index alignment
    # between predictions and ground-truth labels
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True   # Speeds up CPU→GPU transfer when CUDA is available
    )

    print(f"[Dataset] Loaded {len(dataset)} validation exams | "
          f"Plane: {plane} | Slices/exam: {num_slices}")

    return loader

# SECTION 2: MODEL LOADING

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads the trained MRNetViT model from a saved checkpoint.

    The checkpoint is saved by train.py as `model.state_dict()` whenever
    validation AUC improves. We load with strict=True to ensure no weight
    mismatch goes undetected.

    Args:
        checkpoint_path : Path to the saved .pth checkpoint file.
        device          : Target device (CPU or CUDA).

    Returns:
        model: MRNetViT loaded with trained weights, set to eval mode.
    """
    # Instantiate architecture - pretrained=False because we load our own weights
    model = get_model(pretrained=False)

    # Load state dict; map_location handles GPU→CPU loading on CPU-only machines
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # Move model to the target device (GPU if available)
    model = model.to(device)

    # Set to eval mode: disables dropout and batch norm training behaviour
    model.eval()

    print(f"[Model] Checkpoint loaded from: {checkpoint_path}")
    print(f"[Model] Running inference on: {device}")

    return model

# SECTION 3: INFERENCE

def run_inference(model: torch.nn.Module, loader: DataLoader,
                  device: torch.device):
    """
    Runs forward passes over the entire validation set and collects predictions.

    The model outputs raw logits of shape (batch, 3). We apply sigmoid to
    convert logits to probabilities in [0,1] for each of the three tasks.
    No gradients are computed during evaluation (torch.no_grad()).

    Args:
        model  : Trained MRNetViT in eval mode.
        loader : Validation DataLoader.
        device : Inference device.

    Returns:
        all_probs  (np.ndarray): Shape (N, 3) - sigmoid probabilities.
        all_labels (np.ndarray): Shape (N, 3) - ground-truth binary labels.
    """
    all_probs  = []   # Accumulate per-batch probabilities
    all_labels = []   # Accumulate per-batch ground-truth labels

    with torch.no_grad():   # Disable gradient tracking - saves memory and time
        for batch_idx, (images, labels) in enumerate(loader):
            # Move inputs to GPU/CPU
            images = images.to(device)   # Shape: (B, num_slices, 3, 224, 224)
            labels = labels.to(device)   # Shape: (B, 3) - float32 binary labels

            # Forward pass through MRNetViT
            # logits shape: (B, 3) - one score per task per exam
            logits = model(images)

            # Sigmoid converts unbounded logits → probabilities in [0, 1]
            # Threshold at 0.5 for binary prediction
            probs = torch.sigmoid(logits)

            # Move to CPU and convert to numpy for sklearn metric computation
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Progress feedback every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"  [Inference] Processed {batch_idx + 1}/{len(loader)} batches...")

    # Stack list of (B, 3) arrays → (N, 3) arrays over the full validation set
    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    print(f"[Inference] Complete - {all_probs.shape[0]} exams evaluated.")
    return all_probs, all_labels

# SECTION 4: METRIC COMPUTATION

def compute_metrics(probs: np.ndarray, labels: np.ndarray,
                    task_idx: int, threshold: float = 0.5) -> dict:
    """
    Computes all evaluation metrics for a single task (column).

    Metrics computed:
      - Accuracy    : (TP+TN) / N
      - Sensitivity : TP / (TP+FN) - also known as Recall; crucial for medical imaging
      - Specificity : TN / (TN+FP) - ability to correctly identify negatives
      - F1-Score    : 2*(Precision*Recall)/(Precision+Recall)
      - ROC-AUC     : Area Under the ROC Curve - primary metric in MRNet paper

    Sensitivity and Specificity are derived from the confusion matrix
    rather than sklearn directly, for precise control over the calculation.

    Args:
        probs    : (N, 3) probability array from sigmoid output.
        labels   : (N, 3) ground-truth binary label array.
        task_idx : Column index to evaluate (0=abnormal, 1=acl, 2=meniscus).
        threshold: Decision boundary for converting probability → binary prediction.

    Returns:
        dict: Metric name → float value (4 decimal places).
    """
    # Extract this task's probabilities and ground-truth labels (1D arrays)
    task_probs  = probs[:, task_idx]    # Shape: (N,)
    task_labels = labels[:, task_idx]   # Shape: (N,)

    # Apply threshold to convert probabilities to binary predictions (0 or 1)
    task_preds = (task_probs >= threshold).astype(int)

    # Confusion Matrix
    # sklearn returns [[TN, FP], [FN, TP]] for binary classification
    cm = confusion_matrix(task_labels, task_preds, labels=[0, 1])

    # Safely unpack confusion matrix elements with .ravel() for flat access
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity (Recall)
    # Proportion of actual positives correctly identified
    # Critical in medical imaging - false negatives (missed injuries) are costly
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity
    # Proportion of actual negatives correctly identified
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Accuracy
    # Overall fraction of correct predictions across positives and negatives
    accuracy = accuracy_score(task_labels, task_preds)

    # F1-Score
    # Harmonic mean of precision and recall; robust to class imbalance
    # zero_division=0 handles edge cases where no positive predictions exist
    f1 = f1_score(task_labels, task_preds, zero_division=0)

    # ROC-AUC
    # Probability-based metric - does not depend on threshold choice
    # Returns 0.5 for degenerate cases (all labels same class)
    try:
        auc = roc_auc_score(task_labels, task_probs)
    except ValueError:
        # Raised when only one class is present in labels
        auc = 0.5

    # Round all metrics to 4 decimal places for clean reporting
    return {
        'Accuracy'   : round(float(accuracy),    4),
        'Sensitivity': round(float(sensitivity),  4),
        'Specificity': round(float(specificity),  4),
        'F1-Score'   : round(float(f1),           4),
        'ROC-AUC'    : round(float(auc),          4),
    }


def evaluate_all_tasks(probs: np.ndarray, labels: np.ndarray,
                       task: str = 'all', threshold: float = 0.5) -> dict:
    """
    Evaluates metrics for one or all tasks and returns a structured results dict.

    When task='all', metrics are computed for every task and mean AUC is reported.
    When a specific task is given (e.g., 'acl'), only that task is evaluated.

    Args:
        probs    : (N, 3) sigmoid probability outputs.
        labels   : (N, 3) ground-truth labels.
        task     : One of 'abnormal', 'acl', 'meniscus', or 'all'.
        threshold: Binary decision threshold (default 0.5).

    Returns:
        results (dict): Task name → metrics dict; includes 'Mean_AUC' if task='all'.
    """
    results = {}

    if task == 'all':
        # Evaluate each of the three tasks independently
        auc_scores = []
        for name, idx in TASK_INDEX.items():
            metrics = compute_metrics(probs, labels, idx, threshold)
            results[name.capitalize()] = metrics
            auc_scores.append(metrics['ROC-AUC'])

        # Mean AUC across all three tasks - the headline metric for this project
        results['Mean_AUC'] = round(float(np.mean(auc_scores)), 4)

    else:
        # Single task evaluation - task string maps to column index via TASK_INDEX
        if task not in TASK_INDEX:
            raise ValueError(f"Invalid task '{task}'. Choose from: {list(TASK_INDEX.keys())} or 'all'.")
        idx = TASK_INDEX[task]
        results[task.capitalize()] = compute_metrics(probs, labels, idx, threshold)

    return results

# SECTION 5: VISUALISATION

def plot_confusion_matrix(probs: np.ndarray, labels: np.ndarray,
                          task_idx: int, task_name: str,
                          output_dir: str, threshold: float = 0.5):
    """
    Plots and saves a styled confusion matrix heatmap for a given task.

    The confusion matrix shows TP, TN, FP, FN counts with a colour gradient.
    Saved as a PNG to output_dir for inclusion in the IEEE report.

    Args:
        probs      : (N, 3) probability array.
        labels     : (N, 3) ground-truth labels.
        task_idx   : Column index for this task.
        task_name  : Display label (e.g., 'ACL').
        output_dir : Directory to save the PNG file.
        threshold  : Decision boundary for binary prediction.
    """
    # Extract and threshold predictions for this task
    task_probs  = probs[:, task_idx]
    task_labels = labels[:, task_idx]
    task_preds  = (task_probs >= threshold).astype(int)

    # Compute confusion matrix - rows=actual, cols=predicted
    cm = confusion_matrix(task_labels, task_preds, labels=[0, 1])

    # Create figure with seaborn heatmap for polished styling
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,         # Print counts inside each cell
        fmt='d',            # Integer format (no decimal places)
        cmap='Blues',       # Blue gradient - standard for confusion matrices
        xticklabels=['Negative', 'Positive'],   # Predicted class labels
        yticklabels=['Negative', 'Positive'],   # Actual class labels
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_title(f'Confusion Matrix - {task_name}', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

    plt.tight_layout()   # Prevent label clipping

    # Save to output directory as PNG with 150 DPI for report quality
    save_path = os.path.join(output_dir, f'confusion_matrix_{task_name.lower()}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Release memory

    print(f"[Plot] Confusion matrix saved → {save_path}")


def plot_roc_curves(probs: np.ndarray, labels: np.ndarray,
                    output_dir: str):
    """
    Plots and saves ROC curves for all three tasks on a single figure.

    Each curve shows the True Positive Rate (Sensitivity) vs False Positive Rate
    at varying thresholds. AUC is annotated in the legend for each task.
    The diagonal dashed line represents a random classifier (AUC = 0.5).

    Args:
        probs      : (N, 3) probability array.
        labels     : (N, 3) ground-truth labels.
        output_dir : Directory to save the combined ROC curve PNG.
    """
    # Colour palette - one distinct colour per task for visual clarity
    colours = ['#2196F3', '#F44336', '#4CAF50']

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (name, colour) in enumerate(zip(TASK_NAMES, colours)):
        task_probs  = probs[:, i]
        task_labels = labels[:, i]

        try:
            # roc_curve returns FPR, TPR arrays and threshold values
            fpr, tpr, _ = roc_curve(task_labels, task_probs)
            auc = roc_auc_score(task_labels, task_probs)
        except ValueError:
            # Degenerate case: all labels belong to one class
            continue

        # Plot each task's ROC curve with its AUC annotated in the legend
        ax.plot(fpr, tpr, color=colour, lw=2,
                label=f'{name} (AUC = {auc:.3f})')

    # Diagonal line - represents a random (no-skill) classifier
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    ax.set_title('ROC Curves - MRNetViT Evaluation', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    save_path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[Plot] ROC curves saved → {save_path}")


def print_results_table(results: dict):
    """
    Prints a formatted ASCII table of evaluation metrics to stdout.

    Provides an at-a-glance summary during the evaluation run. The table
    matches the metrics reported in the IEEE report's results section.

    Args:
        results (dict): Output of evaluate_all_tasks().
    """
    # Column widths for consistent alignment
    col_w = 14

    # Header row
    header = f"{'Task':<14}" + "".join(
        [f"{m:>{col_w}}" for m in ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'ROC-AUC']]
    )

    divider = "─" * len(header)
    print("\n" + divider)
    print(" EVALUATION RESULTS - MRNetViT (Vision Transformer for Knee MRI)")
    print(divider)
    print(header)
    print(divider)

    for task_name, metrics in results.items():
        if task_name == 'Mean_AUC':
            continue   # Mean AUC printed separately below the table

        row = f"{task_name:<14}" + "".join(
            [f"{metrics[m]:>{col_w}.4f}" for m in ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'ROC-AUC']]
        )
        print(row)

    print(divider)

    # Print mean AUC if available (only when task='all')
    if 'Mean_AUC' in results:
        print(f"{'Mean AUC':<14}{'':>{col_w * 4}}{results['Mean_AUC']:>{col_w}.4f}")
        print(divider)

    print()

# SECTION 6: RESULTS EXPORT

def save_metrics_json(results: dict, output_dir: str):
    """
    Exports all evaluation metrics to a JSON file for programmatic access.

    The JSON file can be referenced in the report, parsed by other scripts,
    or used to compare runs across different checkpoints or planes.

    Args:
        results    : Output of evaluate_all_tasks().
        output_dir : Directory to save metrics.json.
    """
    os.makedirs(output_dir, exist_ok=True)   # Create output dir if it doesn't exist

    save_path = os.path.join(output_dir, 'metrics.json')

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)   # indent=4 for human-readable formatting

    print(f"[Export] Metrics saved → {save_path}")

# SECTION 7: MAIN EVALUATION PIPELINE

def evaluate(args):
    """
    Main orchestration function. Runs the full evaluation pipeline:
      1. Device selection
      2. Model loading
      3. DataLoader construction
      4. Inference
      5. Metric computation
      6. Visualisation (confusion matrices + ROC curves)
      7. Metric export

    Args:
        args: Parsed argparse.Namespace with all CLI arguments.
    """
    # Device Selection
    # Use GPU if available - inference is significantly faster on CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Setup] Device: {device}")

    # Output Directory
    # All plots and JSON are saved here
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Model
    model = load_model(args.checkpoint, device)

    # Build DataLoader
    loader = get_eval_loader(
        data_dir=args.data_root,
        plane=args.plane,
        num_slices=args.num_slices,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Run Inference
    print("\n[Inference] Running forward passes on validation set...")
    probs, labels = run_inference(model, loader, device)

    # Compute Metrics
    print("\n[Metrics] Computing evaluation metrics...")
    results = evaluate_all_tasks(probs, labels, task=args.task, threshold=args.threshold)

    # Print Table
    print_results_table(results)

    # Visualisations
    print("[Plots] Generating visualisations...")

    # Determine which tasks to plot confusion matrices for
    tasks_to_plot = list(TASK_INDEX.keys()) if args.task == 'all' else [args.task]

    for task_name in tasks_to_plot:
        idx = TASK_INDEX[task_name]
        plot_confusion_matrix(
            probs, labels,
            task_idx=idx,
            task_name=task_name.capitalize(),
            output_dir=args.output_dir,
            threshold=args.threshold
        )

    # ROC curves plotted for all tasks regardless of --task flag
    # (probabilities for all tasks are always available)
    plot_roc_curves(probs, labels, output_dir=args.output_dir)

    # Export Metrics
    save_metrics_json(results, args.output_dir)

    print(f"\n[Done] Evaluation complete. All outputs saved to: {args.output_dir}\n")

# SECTION 8: ARGUMENT PARSING & ENTRY POINT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained MRNetViT on the MRNet validation set.'
    )

    # Path to the MRNet-v1.0 root directory (contains train/ and valid/ subdirs)
    parser.add_argument(
        '--data_root', type=str, required=True,
        help='Path to MRNet-v1.0 root directory (e.g., ./MRNet-v1.0)'
    )

    # Path to the trained model checkpoint saved by train.py
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to trained model checkpoint .pth file (e.g., ./best_model.pth)'
    )

    # MRI plane - must match the plane used during training
    parser.add_argument(
        '--plane', type=str, default='sagittal',
        choices=['sagittal', 'coronal', 'axial'],
        help='MRI plane to evaluate on (default: sagittal)'
    )

    # Specific task or evaluate all three tasks
    parser.add_argument(
        '--task', type=str, default='all',
        choices=['abnormal', 'acl', 'meniscus', 'all'],
        help='Task to evaluate: one of the three conditions or all (default: all)'
    )

    # Number of slices sampled per volume - must match training setting
    parser.add_argument(
        '--num_slices', type=int, default=24,
        help='Number of slices to sample per MRI volume (default: 24)'
    )

    # Batch size for the DataLoader during inference
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size for evaluation DataLoader (default: 1)'
    )

    # Number of DataLoader workers
    parser.add_argument(
        '--num_workers', type=int, default=2,
        help='Number of DataLoader worker processes (default: 2)'
    )

    # Decision threshold for converting sigmoid probabilities → binary predictions
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Decision threshold for binary classification (default: 0.5)'
    )

    # Directory where plots (confusion matrix, ROC) and metrics.json are saved
    parser.add_argument(
        '--output_dir', type=str, default='./eval_outputs',
        help='Directory to save evaluation outputs (default: ./eval_outputs)'
    )

    args = parser.parse_args()

    # Launch Evaluation
    evaluate(args)
