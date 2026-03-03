import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import (
    CLASS_SPECS,
    COLOR_MAP,
    DEVICE,
    IGNORE_INDEX,
    IMG_DIM,
    IMG_DIR,
    MASK_DIR,
    N_CLASSES,
    OUTPUT_DIR,
    PARALLEL,
    R,
    SAVED_WEIGHTS,
    SEED,
    SIZE,
    SPLIT_DIR,
    USE_LORA,
)
from core.data import HystoDataset, class_labels
from core.metrics import compute_iou
from core.model_loader import load_dinov2_lora
from core.utils import set_seed


def evaluate_split(model, device, split, batch_size=1):
    dataset = HystoDataset(
        split=split,
        mask_dir=MASK_DIR,
        img_dir=IMG_DIR,
        split_dir=SPLIT_DIR,
        color_map=COLOR_MAP,
        crop_dim=IMG_DIM[0],
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    iou_per_class = []
    indicator_per_class = []
    for images, masks in tqdm(loader, desc=f"Evaluating {split}"):
        images = images.float().to(device)
        masks = masks.long().to(device)

        with torch.no_grad():
            logits = model(images)
            preds = logits.argmax(dim=1)

        labels_np = masks.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()

        iou_batch, indicator_batch = compute_iou(
            preds_np, labels_np, num_classes=N_CLASSES, ignore_index=IGNORE_INDEX
        )
        iou_per_class.append(iou_batch)
        indicator_per_class.append(indicator_batch)

    iou_per_class = np.vstack(iou_per_class)
    indicator_per_class = np.vstack(indicator_per_class)
    masked_iou = np.ma.masked_where(indicator_per_class == 0, iou_per_class)
    mean_iou = masked_iou.mean(axis=0).data
    return mean_iou

def save_iou_plot(val_iou, test_iou):
    labels = class_labels(CLASS_SPECS, IGNORE_INDEX)
    x = np.arange(N_CLASSES)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(x, val_iou, marker="o", color="#2A6EA6", label="Validation")
    ax.plot(x, test_iou, marker="o", color="#C56A2A", label="Test")
    ax.set_title(f"Per-class IoU")
    ax.set_ylabel("Mean IoU")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(True, color="lightgray", linestyle="--", linewidth=0.7)
    ax.legend(loc="lower right")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(OUTPUT_DIR, "val_test_iou.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path

def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------------------------
    # LOAD DINOv2
    # ---------------------------------------------------------------------------

    model, device, _ = load_dinov2_lora(
        size=SIZE,
        r=R,
        use_lora=USE_LORA,
        img_dim=IMG_DIM,
        parallel=PARALLEL,
        dev=DEVICE,
        weights_path=SAVED_WEIGHTS,
    )
    model.eval()

    # ---------------------------------------------------------------------------
    # EVALUATE AND PLOT IoU ON VAL AND TEST SETS
    # ---------------------------------------------------------------------------

    val_iou = evaluate_split(model, device, split="val", batch_size=1)
    test_iou = evaluate_split(model, device, split="test", batch_size=1)

    plot_path = save_iou_plot(val_iou, test_iou)
    print(f"IoU plot saved to {plot_path}")


if __name__ == "__main__":
    main()
