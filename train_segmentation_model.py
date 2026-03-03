import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    BATCH_SIZE,
    COLOR_MAP,
    COLOR_THRESHOLD,
    DEVICE,
    EPOCHS,
    NO_THRESHOLD_LIST,
    IGNORE_INDEX,
    IMG_DIM,
    IMG_DIR,
    LR,
    MASK_DIR,
    MODEL_SAVE_DIR,
    N_CLASSES,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    OUTPUT_DIR,
    PARALLEL,
    R,
    SEED,
    SIZE,
    SPLIT_DIR,
    STRICT_COLOR_LIST,
    STRICT_THRESHOLD,
    USE_LORA,
)
from core.data import HystoDataset, compute_class_weights
from core.metrics import compute_iou
from core.model_loader import load_dinov2_lora
from core.utils import save_config_snapshot, set_seed


def validate_epoch(
    dino_lora: nn.Module,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    metrics: dict,
    device: torch.device,
) -> None:
    val_loss = 0.0
    val_iou = 0.0
    num_batches = 0
    skipped_batches = 0

    dino_lora.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.float().to(device)
            masks = masks.long().to(device)

            # Skip empty mask batches
            if (masks != IGNORE_INDEX).sum() == 0:
                skipped_batches += 1
                continue

            logits = dino_lora(images)

            if torch.isnan(logits).any():
                print("NaN in logits detected during validation! Skipping batch...")
                skipped_batches += 1
                continue

            loss = criterion(logits, masks)
            if torch.isnan(loss):
                print("NaN in loss detected during validation! Skipping batch...")
                skipped_batches += 1
                continue

            val_loss += loss.item()

            preds = logits.argmax(dim=1).detach().cpu().numpy()
            labels = masks.detach().cpu().numpy()

            iou_per_class, indicator = compute_iou(preds, labels, num_classes=N_CLASSES, ignore_index=IGNORE_INDEX)
            # Mean over classes that are present in this batch
            if indicator.sum() > 0:
                mean_iou = iou_per_class[indicator > 0].mean()
                val_iou += mean_iou
            else:
                skipped_batches += 1
                continue

            num_batches += 1

    if num_batches > 0:
        metrics["val_loss"].append(val_loss / num_batches)
        metrics["val_iou"].append(val_iou / num_batches)
    else:
        metrics["val_loss"].append(float('nan'))
        metrics["val_iou"].append(float('nan'))

    total_batches = num_batches + skipped_batches
    print(f"Validation: Skipped {skipped_batches}/{total_batches} batches ({(skipped_batches/total_batches)*100:.1f}%) due to empty masks or NaNs.")

def main() -> None:
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    config_snapshot_path = save_config_snapshot(OUTPUT_DIR)
    print(f"Config snapshot saved to {config_snapshot_path}")

    # ---------------------------------------------------------------------------
    # LOAD DATASET
    # ---------------------------------------------------------------------------

    train_dataset = HystoDataset(
        "train",
        mask_dir=MASK_DIR,
        img_dir=IMG_DIR,
        split_dir=SPLIT_DIR,
        color_map=COLOR_MAP,
        crop_dim=IMG_DIM[0],
        normalize_mean=NORMALIZE_MEAN,
        normalize_std=NORMALIZE_STD,
        color_threshold=COLOR_THRESHOLD,
        strict_threshold=STRICT_THRESHOLD,
        strict_color_list=STRICT_COLOR_LIST,
        ignore_color_list=NO_THRESHOLD_LIST,
    )
    val_dataset = HystoDataset(
        "val",
        mask_dir=MASK_DIR,
        img_dir=IMG_DIR,
        split_dir=SPLIT_DIR,
        color_map=COLOR_MAP,
        crop_dim=IMG_DIM[0],
        normalize_mean=NORMALIZE_MEAN,
        normalize_std=NORMALIZE_STD,
        color_threshold=COLOR_THRESHOLD,
        strict_threshold=STRICT_THRESHOLD,
        strict_color_list=STRICT_COLOR_LIST,
        ignore_color_list=NO_THRESHOLD_LIST,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    # ---------------------------------------------------------------------------
    # LOAD DINOv2
    # ---------------------------------------------------------------------------

    dino_lora, device, _ = load_dinov2_lora(
        size=SIZE,
        r=R,
        use_lora=USE_LORA,
        img_dim=IMG_DIM,
        parallel=PARALLEL,
        dev=DEVICE,
        weights_path=None,
    )

    # ---------------------------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------------------------

    class_weights = compute_class_weights(train_loader, n_classes=N_CLASSES, ignore_index=IGNORE_INDEX).to(device)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX).to(device)
    optimizer = optim.AdamW(dino_lora.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
    }
    best_epoch = None
    best_val_iou = float("-inf")
    best_model_path = f"{MODEL_SAVE_DIR}/dinov2_lora.pt"

    for epoch in range(EPOCHS):
        dino_lora.train()
        start_epoch_time = time.time()

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.float().to(device)
            masks = masks.long().to(device)
            optimizer.zero_grad()

            logits = dino_lora(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            print(
                f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}: "
                f"Loss = {loss.item():.4f}, "
            )

        validate_epoch(dino_lora, val_loader, criterion, metrics, device)

        current_val_iou = metrics["val_iou"][-1]
        if not np.isnan(current_val_iou) and current_val_iou > best_val_iou:
            best_val_iou = float(current_val_iou)
            best_epoch = epoch + 1
            dino_lora.save_parameters(best_model_path)
            print(f"New best model saved to {best_model_path} (epoch {best_epoch}, IoU {best_val_iou:.4f})")

        scheduler.step(metrics["val_loss"][-1])

        elapsed_epoch_time = time.time() - start_epoch_time
        print(f"Epoch {epoch + 1} completed in {elapsed_epoch_time:.2f} seconds")
        print(
            f"Validation Loss = {metrics['val_loss'][-1]:.4f}, "
            f"Validation IoU = {metrics['val_iou'][-1]:.4f}"
        )

        model_save_path = f"{MODEL_SAVE_DIR}/dinov2_lora_epoch{epoch+1:02d}.pt"
        dino_lora.save_parameters(model_save_path)
        print(f"Model saved to {model_save_path}")

    metrics["best_epoch"] = best_epoch
    metrics["best_val_iou"] = best_val_iou if best_epoch is not None else None
    with open(f"{OUTPUT_DIR}/val_metrics.json", "w") as f:
        json.dump(metrics, f)

    epochs_axis = np.arange(1, len(metrics["val_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(epochs_axis, metrics["val_loss"], color="tab:red", label="Val Loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True)
    ax1.set_xticks(np.arange(1, len(epochs_axis) + 1, 2))

    ax2 = ax1.twinx()
    ax2.set_ylabel("IoU", color="tab:blue")
    ax2.plot(epochs_axis, metrics["val_iou"], color="tab:blue", label="Val IoU")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("Validation Loss and IoU")
    fig.tight_layout()
    plot_path = f"{OUTPUT_DIR}/val_loss_iou.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Validation plot saved to {plot_path}")


if __name__ == "__main__":
    main()
