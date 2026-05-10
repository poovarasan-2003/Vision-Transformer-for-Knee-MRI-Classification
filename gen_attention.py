import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
from PIL import Image

from model import get_model
from dataset import MRNetDataset

CHECKPOINT   = "best_model.pth"        # path to your trained checkpoint
DATA_ROOT    = "MRNet-v1.0/valid"      # validation split root
PLANE        = "sagittal"              # plane to use
SAMPLE_IDX   = 10                      # which validation exam to visualise
OUT_PATH     = "eval_outputs/sota_sagittal_10_visualization.png"
TASK_NAMES   = ["Abnormal", "ACL Tear", "Meniscal Tear"]

def main():
    os.makedirs("eval_outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = get_model(pretrained=False)
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval().to(device)
    print(f"Loaded checkpoint: {CHECKPOINT}")

    # Load dataset (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = MRNetDataset(
        data_dir=DATA_ROOT,
        plane=PLANE,
        transform=val_transform,
        num_slices=24
    )

    if SAMPLE_IDX >= len(dataset):
        raise IndexError(f"SAMPLE_IDX={SAMPLE_IDX} out of range (dataset has {len(dataset)} exams)")

    # sequence_tensor: (num_slices, 3, 224, 224)
    # labels:          (3,)
    sequence_tensor, labels = dataset[SAMPLE_IDX]
    print(f"Exam index: {SAMPLE_IDX} | Labels: "
          f"Abnormal={int(labels[0])}  ACL={int(labels[1])}  Meniscus={int(labels[2])}")

    # Pick the most informative slice (middle of sequence)
    mid_slice_idx = sequence_tensor.shape[0] // 2
    x_slice = sequence_tensor[mid_slice_idx]          # (3, 224, 224)
    x_slice_dev = x_slice.unsqueeze(0).to(device)     # (1, 3, 224, 224)

    # Get model prediction
    x_input = sequence_tensor.unsqueeze(0).to(device)  # (1, num_slices, 3, 224, 224)
    with torch.no_grad():
        logits = model(x_input)
        probs  = torch.sigmoid(logits).cpu().numpy().squeeze()  # (3,)

    pred_labels = [TASK_NAMES[i] for i, p in enumerate(probs) if p > 0.5] or ["Normal"]
    print(f"Probabilities: Abnormal={probs[0]:.3f}  ACL={probs[1]:.3f}  Meniscus={probs[2]:.3f}")
    print(f"Predictions:   {', '.join(pred_labels)}")

    # Extract spatial attention map from last ViT block
    # get_spatial_attention returns (1, N-1) = (1, 196) - CLS attn over 14x14 patches
    with torch.no_grad():
        cls_attn = model.get_spatial_attention(x_slice_dev)  # (1, 196)

    attn_map = cls_attn.squeeze().cpu().numpy()   # (196,)
    attn_map = attn_map.reshape(14, 14)           # (14, 14) patch grid

    # Normalise to [0, 1]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Upsample to 224x224
    attn_pil    = Image.fromarray((attn_map * 255).astype(np.uint8))
    attn_up     = attn_pil.resize((224, 224), Image.BICUBIC)
    attn_up_arr = np.array(attn_up) / 255.0      # (224, 224) in [0,1]

    # Reconstruct original slice for display
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    raw  = x_slice.permute(1, 2, 0).numpy()       # (224, 224, 3)
    raw  = (raw * std + mean).clip(0, 1)

    # Build heatmap overlay
    heatmap = cm.jet(attn_up_arr)[..., :3]        # (224, 224, 3) RGB
    overlay = (0.55 * raw + 0.45 * heatmap).clip(0, 1)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor("#1a1a2e")

    panel_data = [
        (raw,      "Original MRI Slice",       "gray"),
        (heatmap,  "CLS Attention Map",         None),
        (overlay,  "Attention Overlay",         None),
    ]

    for ax, (img, title, cmap) in zip(axes, panel_data):
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(title, color="white", fontsize=12, pad=8, fontweight="bold")
        ax.axis("off")

    # Colourbar for the attention panel
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Attention weight", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    title_str = (f"Sample #{SAMPLE_IDX} | Plane: {PLANE.capitalize()} | "
                 f"GT: {', '.join([TASK_NAMES[i] for i,l in enumerate(labels) if l==1]) or 'Normal'} | "
                 f"Pred: {', '.join(pred_labels)}")
    fig.suptitle(title_str, color="white", fontsize=11, y=1.02)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
