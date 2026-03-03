import os
import numpy as np
import torch
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix

from config import N_CLASSES, NORMALIZE_MEAN, NORMALIZE_STD, OUTPUT_DIR, IGNORE_INDEX
from core.anomaly_detection import compute_maha_plus_scores
from core.model.dino_v2 import _get_patch_size


def compute_avg_preds_and_features(
    model,
    device,
    emb_dim,
    img_dim=(252, 252),
    shift_step=84,
    normalize_features=True,
    outputs=("preds", "features"),
    feature_level="pixel",
    n_classes=None,
    img=None,
    file_name=None,
    img_dir=None,
):
    """Compute shift-averaged predictions/features with selectable outputs.

    Args:
        outputs: Tuple/list with any of ``"preds"``, ``"features"``.
        feature_level: ``"pixel"`` returns features with shape
            ``(img_dim[0] * img_dim[1], emb_dim)``; ``"latent"`` returns
            ``(latent_h * latent_w, emb_dim)``.
        img: PIL image object. If omitted, ``file_name`` and ``img_dir`` are used.

    Returns:
        dict[str, np.ndarray | torch.Tensor]:
            keys included according to ``outputs``.
    """
    need_preds = "preds" in outputs
    need_features = "features" in outputs
    if not (need_preds or need_features):
        raise ValueError("outputs must include at least one of: 'preds', 'features'.")
    if need_preds and n_classes is None:
        raise ValueError("n_classes must be provided when 'preds' is requested.")
    if img is None:
        if file_name is None or img_dir is None:
            raise ValueError("Provide either img, or both file_name and img_dir.")
        img_path = f"{img_dir}/{file_name}.png"
        img = Image.open(img_path)

    img_np = np.array(img)
    ext_h, ext_w = img_np.shape[:2]
    patch_h, patch_w = img_dim
    patch_size = _get_patch_size(model)
    latent_h = patch_h // patch_size
    latent_w = patch_w // patch_size

    img_transform = transforms.Compose([
        transforms.Resize(img_dim),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    # Result tensors for the central region
    if need_features:
        feature_sum = torch.zeros((emb_dim, latent_h, latent_w), device=device)
        feature_count = torch.zeros((1, latent_h, latent_w), device=device)
    if need_preds:
        softmax_sum = torch.zeros((n_classes, patch_h, patch_w), device=device)
        softmax_count = torch.zeros((1, patch_h, patch_w), device=device)

    # Define the bounds of the central region
    central_y_start = (ext_h - patch_h) // 2
    central_y_end = central_y_start + patch_h
    central_x_start = (ext_w - patch_w) // 2
    central_x_end = central_x_start + patch_w

    # For every shift inside extended tile
    for y in range(0, ext_h - patch_h + 1, shift_step):
        for x in range(0, ext_w - patch_w + 1, shift_step):
            patch = img_np[y:y + patch_h, x:x + patch_w, :]
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            patch_tensor = patch_tensor.unsqueeze(0).to(device)
            patch_tensor = img_transform(patch_tensor)

            with torch.no_grad():
                if need_preds:
                    logits = model(patch_tensor)
                    softmaxes = torch.nn.functional.interpolate(
                        torch.softmax(logits, dim=1),
                        size=(patch_h, patch_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                if need_features:
                    features = model.get_patch_embeddings(patch_tensor)
                    features = features.reshape(1, latent_h, latent_w, emb_dim).squeeze(0).permute(2, 0, 1)

            # Determine overlap with the central region
            y_start = max(y, central_y_start)
            x_start = max(x, central_x_start)
            y_end = min(y + patch_h, central_y_end)
            x_end = min(x + patch_w, central_x_end)

            patch_y_start = y_start - y
            patch_x_start = x_start - x
            patch_y_end = patch_y_start + (y_end - y_start)
            patch_x_end = patch_x_start + (x_end - x_start)

            dest_y_start = y_start - central_y_start
            dest_x_start = x_start - central_x_start
            dest_y_end = dest_y_start + (y_end - y_start)
            dest_x_end = dest_x_start + (x_end - x_start)

            # Scale coordinates to latent space
            scale_y = latent_h / patch_h
            scale_x = latent_w / patch_w
            fy_start, fy_end = int(dest_y_start * scale_y), int(dest_y_end * scale_y)
            fx_start, fx_end = int(dest_x_start * scale_x), int(dest_x_end * scale_x)
            py_start, py_end = int(patch_y_start * scale_y), int(patch_y_end * scale_y)
            px_start, px_end = int(patch_x_start * scale_x), int(patch_x_end * scale_x)

            # Aggregate outputs
            if need_preds:
                softmax_sum[:, dest_y_start:dest_y_end, dest_x_start:dest_x_end] += softmaxes[
                    :, patch_y_start:patch_y_end, patch_x_start:patch_x_end
                ]
                softmax_count[:, dest_y_start:dest_y_end, dest_x_start:dest_x_end] += 1
            if need_features:
                feature_sum[:, fy_start:fy_end, fx_start:fx_end] += features[:, py_start:py_end, px_start:px_end]
                feature_count[:, fy_start:fy_end, fx_start:fx_end] += 1

    result = {}
    if need_preds:
        averaged_softmaxes = softmax_sum / softmax_count
        result["preds"] = averaged_softmaxes.argmax(dim=0).cpu().numpy()
    if need_features:
        features_latent_map = feature_sum / feature_count
        if feature_level == "latent":
            features_out = features_latent_map.permute(1, 2, 0).reshape(latent_h * latent_w, emb_dim)
        elif feature_level == "pixel":
            features_pixel = torch.nn.functional.interpolate(
                features_latent_map.unsqueeze(0),
                size=img_dim,
                mode="bilinear",
                align_corners=False,
            )
            features_out = features_pixel.squeeze(0).permute(1, 2, 0).reshape(img_dim[0] * img_dim[1], emb_dim)
        else:
            raise ValueError("feature_level must be 'pixel' or 'latent'.")
        if normalize_features:
            features_out = torch.nn.functional.normalize(features_out, p=2, dim=1)
        result["features"] = features_out

    return result


def get_classwise_outputs(img, mask, img_dim, emb_dim,
                          model, device, n_classes, class_specs,
                          class_means=None, cov_matrix=None,
                          shift_step=84, normalize_features=True):
    """Compute shift-averaged classwise outputs for a single tile image.

    Returns:
        dict[str, list[np.ndarray]]:
            - ``'preds'``: list of length ``len(class_specs)``; each element is a
              1D array of predicted class indices for pixels of that GT class.
            - ``'maha_plus'``: list of length ``len(class_specs)``; each element
              is a 1D array of Maha+ scores for pixels of that GT class.
    """
    if class_means is None or cov_matrix is None:
        raise ValueError("class_means and cov_matrix must be provided.")

    mask = transforms.CenterCrop(img_dim)(mask)
    mask = np.array(mask)

    avg_output = compute_avg_preds_and_features(
        model=model,
        device=device,
        emb_dim=emb_dim,
        img_dim=img_dim,
        shift_step=shift_step,
        normalize_features=normalize_features,
        outputs=("preds", "features"),
        feature_level="latent",
        n_classes=n_classes,
        img=img,
    )

    avg_preds = avg_output["preds"]
    class_preds_per_image_list = []
    for spec in class_specs.values():
        target_color = spec["color"]
        target_color_mask = np.all(mask == target_color, axis=-1)
        class_preds_per_image_list.append(avg_preds[target_color_mask].reshape(-1))

    avg_features = avg_output["features"]
    mm_scores = compute_maha_plus_scores(
        avg_features,
        class_means,
        cov_matrix,
        avg_preds,
        n_classes,
        device,
    )
    class_scores_per_image_list = []
    for spec in class_specs.values():
        target_color = spec["color"]
        target_color_mask = np.all(mask == target_color, axis=-1)
        class_scores_per_image_list.append(mm_scores[target_color_mask].reshape(-1))

    output_dict = {
        "preds": class_preds_per_image_list,
        "maha_plus": class_scores_per_image_list,
    }

    return output_dict


def custom_confusion_matrix(true_labels, pred_labels, n_id_classes, normalize):
    """Extended confusion matrix (ID classes plus one joint OOD class).
        - Healthy class is in the first row/column.
        - All OOD classes are in the last row/column.

    Args:
        true_labels: Ground-truth class indices after remapping.
        pred_labels: Predicted class indices after remapping.
        n_id_classes: Number of in-distribution classes (e.g. ``N_CLASSES``).
        normalize: Passed through to ``sklearn.metrics.confusion_matrix``.
    """
    n_classes = n_id_classes + 1
    all_classes = np.arange(n_classes)
    conf_mat = confusion_matrix(true_labels, pred_labels, labels=all_classes, normalize=normalize)

    padded_conf_mat = np.zeros((n_classes, n_classes))

    for i, true_class in enumerate(all_classes):
        for j, pred_class in enumerate(all_classes):
            if true_class < conf_mat.shape[0] and pred_class < conf_mat.shape[1]:
                padded_conf_mat[true_class, pred_class] = conf_mat[true_class, pred_class]

    present_classes = np.unique(true_labels)
    for i in range(n_classes):
        if i not in present_classes:
            padded_conf_mat[i, :] = 0
            padded_conf_mat[i, i] = 0

    return padded_conf_mat


def plot_custom_conf_mat(true_labels, pred_labels, str_labels):
    """Plot normalized confusion matrix and save it to OUTPUT_DIR.

    Returns:
        str: Path to the saved figure.
    """
    n_id_classes = len(str_labels) - 1
    conf_mat = custom_confusion_matrix(
        true_labels,
        pred_labels,
        n_id_classes=n_id_classes,
        normalize='true',
    ) * 100

    df_cf = pd.DataFrame(conf_mat, index=str_labels, columns=str_labels)

    plt.figure(figsize=(8.3, 7))
    ax = sn.heatmap(
        df_cf,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        cbar_kws={"shrink": 0.8}
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "conf_matrix.png")
    plt.savefig(out_path, dpi=200)


def remapped_confusion_labels(class_specs, healthy_key="normal"):
    """Build confusion-matrix labels matching the remapping used in evaluation.

    Remapped order is:
    0 -> healthy class (``healthy_key``),
    1..N -> remaining ID classes (sorted by original class index),
    last -> joint OOD class.
    """
    if healthy_key not in class_specs:
        raise KeyError(f"CLASS_SPECS must contain '{healthy_key}'.")

    healthy_index = class_specs[healthy_key]["index"]
    id_classes = [
        (name, spec["index"])
        for name, spec in class_specs.items()
        if spec["index"] != IGNORE_INDEX and spec["index"] != healthy_index
    ]
    id_classes.sort(key=lambda x: x[1])

    labels = [healthy_key] + [name for name, _ in id_classes] + ["OOD"]
    return labels


def sort_results(preds, scores):
    """Sort scores/preds into fixed per-predicted-class ID and OOD buckets."""
    keys = list(scores.keys())
    scores_list = [scores[k] for k in keys]
    preds_list = [preds[k] for k in keys]
    if len(scores_list) < N_CLASSES:
        raise ValueError(
            f"Expected at least {N_CLASSES} class slots in scores, got {len(scores_list)}."
        )
    n_ood_slots = max(0, len(scores_list) - N_CLASSES)
    labels_list = [
        np.full(len(scores_list[i]), IGNORE_INDEX if i < n_ood_slots else i - n_ood_slots, dtype=int)
        for i in range(len(scores_list))
    ]

    scores_all = np.concatenate(scores_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)
    preds_all = np.concatenate(preds_list, axis=0)
    is_ood_all = labels_all == IGNORE_INDEX

    scores_pred_id_list, labels_pred_id_list, preds_pred_id_list = [], [], []
    scores_ood_list, labels_ood_list, preds_ood_list = [], [], []

    for k in range(N_CLASSES):
        m_k = preds_all == k
        s_arr = scores_all[m_k]
        l_arr = labels_all[m_k]
        p_arr = preds_all[m_k]
        m_ood = is_ood_all[m_k]
        m_pred = ~m_ood

        scores_pred_id_list.append(s_arr[m_pred])
        labels_pred_id_list.append(l_arr[m_pred])
        preds_pred_id_list.append(p_arr[m_pred])

        scores_ood_list.append(s_arr[m_ood])
        labels_ood_list.append(l_arr[m_ood])
        preds_ood_list.append(p_arr[m_ood])

    return (
        scores_pred_id_list,
        scores_ood_list,
        labels_pred_id_list,
        labels_ood_list,
        preds_pred_id_list,
        preds_ood_list,
    )


def load_and_sort(preds_trainval_path, scores_trainval_path):
    preds = np.load(preds_trainval_path)
    scores = np.load(scores_trainval_path)
    return sort_results(preds, scores)
