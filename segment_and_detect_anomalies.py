import os
from io import BytesIO
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms

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
    PERC,
    REFERENCE_SPLIT,
    NORMALIZE,
    R,
    SAVED_WEIGHTS,
    SEED,
    SHIFT_STEP,
    SIZE,
    SPLIT_DIR,
    USE_LORA,
)
from core.data import collect_files_from_split, load_mask
from core.anomaly_detection import ood_threshold, compute_maha_plus_scores
from core.model_loader import load_dinov2_lora
from core.prediction_maker import (
    compute_avg_preds_and_features,
    custom_confusion_matrix,
    plot_custom_conf_mat,
    remapped_confusion_labels,
    get_classwise_outputs,
    load_and_sort,
)
from core.utils import set_seed


def compute_class_stats(model, device, emb_dim, split_name, normalize=True):
    """Compute class means and shared covariance matrix based on averaged features for Mahalanobis scoring.

    The function iterates over files listed in ``SPLIT_DIR/<split_name>.txt``,
    extracts per-pixel features, optionally L2-normalizes them, and accumulates:
    - class-wise feature sums to compute class means,
    - class-wise centered outer products to compute one shared covariance matrix.

    Saves:
        - ``OUTPUT_DIR/mahalanobis/class_means_<split_name>.npy``
        - ``OUTPUT_DIR/mahalanobis/cov_matrix_<split_name>.npy``

    Returns:
        tuple[np.ndarray, np.ndarray]:
            ``(class_means, cov_matrix)`` where class_means has shape
            ``(N_CLASSES, emb_dim)`` and cov_matrix has shape ``(emb_dim, emb_dim)``.
    """
    split_path = os.path.join(SPLIT_DIR, f"{split_name}.txt")
    files = collect_files_from_split(split_path, MASK_DIR)

    # Preallocate tensors
    features_classwise_sum = torch.zeros((N_CLASSES, emb_dim), dtype=torch.float32, device=device)
    class_counts = torch.zeros(N_CLASSES, dtype=torch.float32, device=device)

    for file_name in tqdm(files, desc=f"Computing class means ({split_name})"):
        mask = load_mask(
            file_name=file_name,
            device=device,
            mask_dir=MASK_DIR,
            color_map=COLOR_MAP,
            img_dim=IMG_DIM,
        )
        img_features = compute_avg_preds_and_features(
            file_name=file_name,
            model=model,
            device=device,
            emb_dim=emb_dim,
            img_dir=IMG_DIR,
            img_dim=IMG_DIM,
            shift_step=SHIFT_STEP,
            normalize_features=normalize,
            outputs=("features",),
            feature_level="pixel",
        )["features"]
        # Compute feature sums and counts
        for i in range(N_CLASSES):
            class_mask = mask == i
            if not class_mask.any():
                continue
            features_classwise_sum[i] += img_features[class_mask].sum(dim=0)
            class_counts[i] += class_mask.sum()

    # Compute class means
    class_means = torch.zeros_like(features_classwise_sum)
    valid = class_counts > 0
    class_means[valid] = features_classwise_sum[valid] / class_counts[valid].unsqueeze(1)
    class_means = class_means.cpu().numpy()

    # Preallocate tensors
    outer_products_classwise_sum = torch.zeros((N_CLASSES, emb_dim, emb_dim), device=device)
    class_counts = torch.zeros(N_CLASSES, dtype=torch.float32, device=device)

    # Compute outer product sums and counts
    for file_name in tqdm(files, desc=f"Computing covariance matrix ({split_name})"):
        mask = load_mask(
            file_name=file_name,
            device=device,
            mask_dir=MASK_DIR,
            color_map=COLOR_MAP,
            img_dim=IMG_DIM,
        )
        img_features = compute_avg_preds_and_features(
            file_name=file_name,
            model=model,
            device=device,
            emb_dim=emb_dim,
            img_dir=IMG_DIR,
            img_dim=IMG_DIM,
            shift_step=SHIFT_STEP,
            normalize_features=normalize,
            outputs=("features",),
            feature_level="pixel",
        )["features"]
        # Compute outer product sums and counts
        for i in range(N_CLASSES):
            class_mask = mask == i
            if not class_mask.any():
                continue
            diff = img_features[class_mask] - class_means[i]
            outer_products_classwise_sum[i] += diff.T @ diff
            class_counts[i] += class_mask.sum()

    # Compute shared covariance matrix
    outer_products_sum = outer_products_classwise_sum.sum(dim=0)
    pixel_counts = class_counts.sum().item()
    cov_matrix = outer_products_sum.cpu().numpy() / pixel_counts

    class_means_path, cov_matrix_path = class_stats_paths(
        OUTPUT_DIR,
        split_name=split_name,
    )
    np.save(class_means_path, class_means)
    np.save(cov_matrix_path, cov_matrix)
    print(f"Saved class stats:\n- {class_means_path}\n- {cov_matrix_path}")

    return class_means, cov_matrix


def class_stats_paths(output_dir, split_name):
    maha_dir = os.path.join(output_dir, "mahalanobis")
    os.makedirs(maha_dir, exist_ok=True)
    class_means_path = os.path.join(maha_dir, f"class_means_{split_name}.npy")
    cov_matrix_path = os.path.join(maha_dir, f"cov_matrix_{split_name}.npy")
    return class_means_path, cov_matrix_path


def load_or_compute_class_stats(model, device, emb_dim, split_name, normalize=True, force_recompute=False):
    class_means_path, cov_matrix_path = class_stats_paths(
        OUTPUT_DIR,
        split_name=split_name,
    )
    stats_exist = os.path.exists(class_means_path) and os.path.exists(cov_matrix_path)

    if stats_exist and not force_recompute:
        print(f"Using saved class stats:\n- {class_means_path}\n- {cov_matrix_path}")
        return np.load(class_means_path), np.load(cov_matrix_path)

    print("Computing class stats (class means + covariance matrix)...")
    class_means, cov_matrix = compute_class_stats(
        model=model,
        device=device,
        emb_dim=emb_dim,
        split_name=split_name,
        normalize=normalize,
    )
    return class_means, cov_matrix


def compute_outputs(model, device, emb_dim, class_means, cov_matrix, split_name):
    """Compute classwise predictions and anomaly scores for one split and save them.

    For each file listed in ``SPLIT_DIR/<split_name>.txt``, this function computes:
    - classwise predicted labels,
    - classwise Maha+ anomaly scores.
    It aggregates per-image outputs across the split:
    - ``class_preds`` is a list of 1D NumPy arrays with array per class in CLASS_SPECS,
    - ``class_scores`` is a list of 1D NumPy arrays with array per class in CLASS_SPECS,
    
    Saves:
        - ``OUTPUT_DIR/classwise_<split_name>_preds.npz``,
        - ``OUTPUT_DIR/classwise_<split_name>_scores.npz``.
    """
    split_path = os.path.join(SPLIT_DIR, f"{split_name}.txt")
    files = collect_files_from_split(split_path, MASK_DIR)

    class_preds_list = []
    class_scores_list = []
    for file_name in tqdm(files, desc=f"Computing outputs ({split_name})"):
        mask = Image.open(os.path.join(MASK_DIR, file_name + ".png"))
        img = Image.open(os.path.join(IMG_DIR, file_name + ".png"))
        output_dict = get_classwise_outputs(
            img,
            mask,
            IMG_DIM,
            emb_dim,
            model,
            device,
            N_CLASSES,
            CLASS_SPECS,
            class_means=class_means,
            cov_matrix=cov_matrix,
            shift_step=SHIFT_STEP,
            normalize_features=NORMALIZE,
        )
        class_preds_list.append(output_dict["preds"])
        class_scores_list.append(output_dict["maha_plus"])

    n_class_specs = len(CLASS_SPECS)
    class_preds = [
        np.concatenate(arrays) if arrays else np.array([])
        for i in range(n_class_specs)
        for arrays in [[row[i] for row in class_preds_list]]
    ]
    class_scores = [
        np.concatenate(arrays) if arrays else np.array([])
        for i in range(n_class_specs)
        for arrays in [[row[i] for row in class_scores_list]]
    ]

    preds_path = os.path.join(OUTPUT_DIR, f"classwise_{split_name}_preds.npz")
    scores_path = os.path.join(OUTPUT_DIR, f"classwise_{split_name}_scores.npz")
    np.savez_compressed(preds_path, *class_preds)
    np.savez_compressed(scores_path, *class_scores)
    print(f"Saved outputs:\n- {preds_path}\n- {scores_path}")


def outputs_paths(split_name):
    preds_path = os.path.join(OUTPUT_DIR, f"classwise_{split_name}_preds.npz")
    scores_path = os.path.join(OUTPUT_DIR, f"classwise_{split_name}_scores.npz")
    return preds_path, scores_path


def load_or_compute_outputs(model, device, emb_dim, class_means, cov_matrix, split_name, force_recompute=False):
    preds_path, scores_path = outputs_paths(split_name)
    outputs_exist = os.path.exists(preds_path) and os.path.exists(scores_path)

    if outputs_exist and not force_recompute:
        print(f"Using existing outputs:\n- {preds_path}\n- {scores_path}")
        return

    compute_outputs(
        model=model,
        device=device,
        emb_dim=emb_dim,
        class_means=class_means,
        cov_matrix=cov_matrix,
        split_name=split_name,
    )


def compute_thresholds(method_name="adaptive", perc=PERC, threshold_split="trainval"):
    """Compute OOD score thresholds from a reference split.

    Loads classwise prediction/score files for ``threshold_split`` and estimates
    threshold(s) with ``ood_threshold``:
    - ``standard``: one global threshold across all predicted ID classes.
    - ``adaptive``: one threshold per predicted ID class.

    Args:
        method_name: ``"standard"`` or ``"adaptive"``.
        perc: Target fraction kept as ID (TPR-like parameter).
        threshold_split: Split name used for threshold calibration.

    Returns:
        list[float]: Thresholds of length ``N_CLASSES``.
    """
    preds_threshold_path, scores_threshold_path = outputs_paths(threshold_split)
    scores_id_list_threshold, *_ = load_and_sort(preds_threshold_path, scores_threshold_path)

    if method_name == "standard":
        scores_id_threshold = np.concatenate(scores_id_list_threshold)
        t = ood_threshold(scores_id_threshold, perc)
        return [t] * N_CLASSES
    if method_name == "adaptive":
        return [ood_threshold(scores_id_list_threshold[i], perc) for i in range(N_CLASSES)]
    raise ValueError(f"Unknown method_name: {method_name}")


def compute_metrics(t_list, eval_split="test", perc=PERC, method_name="adaptive", threshold_split="trainval"):
    """Evaluate anomaly metrics on a split using precomputed thresholds.

    Loads classwise preds/scores for ``eval_split``, applies classwise score
    thresholds (values below threshold become OOD prediction ``-1``), then
    builds the extended confusion matrix (healthy + ID classes + joint OOD)
    and derives the reported rates.

    Args:
        t_list: List of thresholds (length ``N_CLASSES``), typically from
            ``compute_thresholds``.
        eval_split: Split name to evaluate (default: ``"test"``).

    Returns:
        dict[str, float]: Metrics in percent (FNR, FPR, BER and per-group
        misclassification rates). Metrics are also
        saved to ``OUTPUT_DIR/ad_metrics.txt``.
    """
    preds_eval_path, scores_eval_path = outputs_paths(eval_split)
    (scores_id_list_test,
     scores_ood_list_test,
     labels_id_list_test,
     labels_ood_list_test,
     preds_id_list_test,
     preds_ood_list_test) = load_and_sort(preds_eval_path, scores_eval_path)

     # Arrays from lists
    labels_id_test = np.concatenate(labels_id_list_test)
    labels_ood_test = np.concatenate(labels_ood_list_test)
    labels_test = np.concatenate([labels_id_test, labels_ood_test])

    # Flag OOD by threshold and concatenate
    for i in range(len(scores_id_list_test)):
        mask = scores_id_list_test[i] < t_list[i]
        preds_id_list_test[i][mask] = IGNORE_INDEX
    for i in range(len(scores_ood_list_test)):
        mask = scores_ood_list_test[i] < t_list[i]
        preds_ood_list_test[i][mask] = IGNORE_INDEX
    preds_id_test = np.concatenate(preds_id_list_test)
    preds_ood_test = np.concatenate(preds_ood_list_test)
    preds_test = np.concatenate([preds_id_test, preds_ood_test])

    if "normal" not in CLASS_SPECS:
        raise KeyError("CLASS_SPECS must contain a 'normal' entry for healthy-class remapping.")
    healthy_index = CLASS_SPECS["normal"]["index"]

    # Apply remapping
    preds_test_confm = preds_test.copy()
    labels_test_confm = labels_test.copy()
    preds_test_confm[preds_test == IGNORE_INDEX] = N_CLASSES  # all OOD classes are now in the last row/column
    labels_test_confm[labels_test == IGNORE_INDEX] = N_CLASSES
    preds_test_confm[preds_test == healthy_index] = 0  # healthy class is now in the first row/column
    labels_test_confm[labels_test == healthy_index] = 0
    mask_preds = (preds_test >= 0) & (preds_test < N_CLASSES) & (preds_test != healthy_index)  # shift all other ID classes by +1 so row/col 0 stays reserved for healthy
    preds_test_confm[mask_preds] = preds_test_confm[mask_preds] + 1
    mask_labels = (labels_test >= 0) & (labels_test < N_CLASSES) & (labels_test != healthy_index)
    labels_test_confm[mask_labels] = labels_test_confm[mask_labels] + 1

    # Extended confusion matrix
    conf_mat = custom_confusion_matrix(labels_test_confm, preds_test_confm, N_CLASSES, normalize=None)
    str_labels = remapped_confusion_labels(CLASS_SPECS, healthy_key="normal")
    plot_custom_conf_mat(labels_test_confm, preds_test_confm, str_labels)

    # Metrics
    fp = conf_mat[0, 1:].sum()
    tn = conf_mat[0, 0]
    tpr_c = np.mean([conf_mat[k + 1, 1:].sum() / (conf_mat[k + 1, 1:].sum() + conf_mat[k + 1, 0].sum()) for k in range(len(conf_mat) - 1)])
    fpr = fp / (tn + fp)
    # Normal
    n_misclass_as_id_anomaly = (np.sum(conf_mat[0, :]) - conf_mat[0, 0] - conf_mat[0, -1]) / np.sum(conf_mat[0, :])
    n_misclass_as_ood_anomaly = conf_mat[0, -1] / np.sum(conf_mat[0, :])
    # ID Anomalies
    acc_id = np.mean([conf_mat[k + 1, k + 1] / np.sum(conf_mat[k + 1, :]) for k in range(len(conf_mat) - 2)])
    id_misclass_as_id_anomaly = np.mean([
        (np.sum(conf_mat[k + 1, :]) - conf_mat[k + 1, k + 1] - conf_mat[k + 1, 0] - conf_mat[k + 1, -1]) / np.sum(conf_mat[k + 1, :])
        for k in range(len(conf_mat) - 2)
    ])
    id_misclass_as_ood_anomaly = np.mean([conf_mat[k + 1, -1] / np.sum(conf_mat[k + 1, :]) for k in range(len(conf_mat) - 2)])
    id_misclass_as_normal = np.mean([conf_mat[k + 1, 0] / np.sum(conf_mat[k + 1, :]) for k in range(len(conf_mat) - 2)])
    # OOD Anomalies
    acc_ood = conf_mat[-1, -1] / np.sum(conf_mat[-1, :])
    ood_misclass_as_id_anomaly = (np.sum(conf_mat[-1, :]) - conf_mat[-1, -1] - conf_mat[-1, 0]) / np.sum(conf_mat[-1, :])
    ood_misclass_as_normal = conf_mat[-1, 0] / np.sum(conf_mat[-1, :])

    metrics = {
        "FNR": (1 - tpr_c) * 100,
        "FPR": fpr * 100,
        "Healthy: Misclass. as ID anomaly": n_misclass_as_id_anomaly * 100,
        "Healthy: Misclass. as OOD anomaly": n_misclass_as_ood_anomaly * 100,
        "ID-Anomalies: Misclassified": (1 - acc_id) * 100,
        "ID-Anomalies: Misclass. as ID anomaly": id_misclass_as_id_anomaly * 100,
        "ID-Anomalies: Misclass. as OOD anomaly": id_misclass_as_ood_anomaly * 100,
        "ID-Anomalies: Misclass. as normal": id_misclass_as_normal * 100,
        "OOD-Anomalies: Misclassified": (1 - acc_ood) * 100,
        "OOD-Anomalies: Misclass. as ID anomaly": ood_misclass_as_id_anomaly * 100,
        "OOD-Anomalies: Misclass. as normal": ood_misclass_as_normal * 100,
    }

    perc_str = f"{perc}"
    metrics_path = os.path.join(OUTPUT_DIR, "ad_metrics.txt")
    with open(metrics_path, "w") as f:
        print("=" * 60, file=f)
        print(f"p = {perc_str}", file=f)
        print(f"METHOD: Maha+", file=f)
        print(f"NORMALIZE: {NORMALIZE}", file=f)
        print(f"STRATEGY: {method_name}", file=f)
        print(f"THRESHOLD SPLIT: {threshold_split}", file=f)
        print(f"EVAL SPLIT: {eval_split}", file=f)
        print("=" * 60, file=f)
        label_width = max(len(name) for name in metrics)
        for name, value in metrics.items():
            print(f"{name:<{label_width}} = {value:.2f}", file=f)
    print(f"Saved metrics:\n- {metrics_path}")

    return metrics


def visualize_preds_and_scores(
    model,
    device,
    emb_dim,
    class_means,
    cov_matrix,
    thresholds,
    split_name="test",
):
    """Create per-tile visualization with GT overlay, OOD-thresholded prediction, and Maha+ score heatmap.

    OOD pixels are rendered in yellow with RGB color code ``(255, 255, 0)``.
    """
    split_path = os.path.join(SPLIT_DIR, f"{split_name}.txt")
    files = collect_files_from_split(split_path, MASK_DIR)

    output_subdir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(output_subdir, exist_ok=True)
    center_crop = transforms.CenterCrop(IMG_DIM)
    thresholds_array_base = np.asarray(thresholds)

    output_color_map = {
        spec["index"]: spec["color"]
        for spec in CLASS_SPECS.values()
        if spec["index"] != IGNORE_INDEX
    }
    class_dirs = {
        class_name: os.path.join(output_subdir, class_name)
        for class_name in CLASS_SPECS
    }
    for class_dir in class_dirs.values():
        os.makedirs(class_dir, exist_ok=True)

    for file_name in tqdm(files, desc=f"Visualizing tiles ({split_name})"):
        img = Image.open(os.path.join(IMG_DIR, file_name + ".png"))
        mask = Image.open(os.path.join(MASK_DIR, file_name + ".png"))
        img_crop = center_crop(img)
        mask_crop = center_crop(mask)
        mask_rgb = np.array(mask_crop)

        avg_outputs = compute_avg_preds_and_features(
            img=img,
            img_dim=IMG_DIM,
            emb_dim=emb_dim,
            model=model,
            device=device,
            n_classes=N_CLASSES,
            shift_step=SHIFT_STEP,
            normalize_features=NORMALIZE,
            outputs=("preds", "features"),
            feature_level="latent",
        )
        pred_mask_array = avg_outputs["preds"]
        averaged_features = avg_outputs["features"]

        m_scores = compute_maha_plus_scores(
            averaged_features=averaged_features,
            class_means=class_means,
            cov_matrix=cov_matrix,
            preds=pred_mask_array,
            n_classes=N_CLASSES,
            device=device,
        )

        thresholds_array = thresholds_array_base[pred_mask_array]
        score_map = m_scores.squeeze()
        pred_mask_array_new = pred_mask_array.copy()
        pred_mask_array_new[score_map < thresholds_array] = IGNORE_INDEX
        pred_mask_ood = np.zeros((pred_mask_array.shape[0], pred_mask_array.shape[1], 3), dtype=np.uint8)
        for label, color in output_color_map.items():
            pred_mask_ood[pred_mask_array_new == label] = color
        ood_color = [255, 255, 0]
        pred_mask_ood[pred_mask_array_new == IGNORE_INDEX] = ood_color

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={"width_ratios": [0.92, 0.92, 0.92, 1]})

        axes[0].imshow(img_crop, cmap="gray")
        axes[0].set_title("Original Image", fontsize=18)
        axes[0].axis("off")

        axes[1].imshow(img_crop, cmap="gray")
        axes[1].imshow(mask_crop, alpha=0.75)
        axes[1].set_title("Ground Truth", fontsize=18)
        axes[1].axis("off")

        axes[2].imshow(img_crop, cmap="gray")
        axes[2].imshow(pred_mask_ood, alpha=0.75)
        axes[2].set_title("Prediction", fontsize=18)
        axes[2].axis("off")

        im = axes[3].imshow(score_map, cmap="viridis", vmin=-10000, vmax=0)
        axes[3].set_title("Maha+ Score", fontsize=18)
        axes[3].axis("off")
        fig.colorbar(im, ax=axes[3], orientation="vertical", fraction=0.046, pad=0.04)

        plt.tight_layout()

        tile_name = os.path.basename(file_name)
        present_class_names = []
        class_masks = {}
        for class_name, spec in CLASS_SPECS.items():
            class_color = np.array(spec["color"], dtype=np.uint8)
            class_mask = np.all(mask_rgb == class_color, axis=-1)
            class_present = np.any(class_mask)
            if class_present:
                present_class_names.append(class_name)
                class_masks[class_name] = class_mask

        render_buf = BytesIO()
        fig.savefig(render_buf, format="png", bbox_inches="tight")
        figure_bytes = render_buf.getvalue()
        render_buf.close()

        for class_name in present_class_names:
            class_dir = class_dirs[class_name]
            class_avg_score = float(m_scores[class_masks[class_name]].mean())
            avg_name = f"{class_avg_score:.2f}"
            image_file = f"{avg_name}_{tile_name}.png"
            output_path = os.path.join(class_dir, image_file)
            with open(output_path, "wb") as f:
                f.write(figure_bytes)
        plt.close(fig)


def main():
    reference_split = REFERENCE_SPLIT
    eval_split = "test"

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------------------------
    # LOAD DINOv2
    # ---------------------------------------------------------------------------

    model, device, emb_dim = load_dinov2_lora(
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
    # COMPUTE CLASS STATS (CLASS MEANS AND SHARED COVARIANCE MATRIX) 
    # BASED ON AVERAGED FEATURES
    # ---------------------------------------------------------------------------

    class_means, cov_matrix = load_or_compute_class_stats(
        model=model,
        device=device,
        emb_dim=emb_dim,
        split_name=reference_split,
        normalize=NORMALIZE,
        force_recompute=False,
    )

    # ---------------------------------------------------------------------------
    # COMPUTE AVERAGED PREDICTIONS AND MAHA+ SCORES
    # ---------------------------------------------------------------------------
     
    for split_name in {reference_split, eval_split}:
        load_or_compute_outputs(
            model=model,
            device=device,
            emb_dim=emb_dim,
            class_means=class_means,
            cov_matrix=cov_matrix,
            split_name=split_name,
            force_recompute=False,
        )

    # ---------------------------------------------------------------------------
    # COMPUTE THRESHOLDS ON REFERENCE SPLIT AND METRICS ON EVALUATION SPLIT
    # ---------------------------------------------------------------------------

    thresholds = compute_thresholds(
        method_name="adaptive",
        perc=PERC,
        threshold_split=reference_split,
    )
    compute_metrics(
        t_list=thresholds,
        eval_split=eval_split,
        perc=PERC,
        method_name="adaptive",
        threshold_split=reference_split,
    )

    # ---------------------------------------------------------------------------
    # VISUALIZE (ORIG. IMAGES, GROUND-TRUTH MASKS, PREDICTIONS AND MAHA+ SCORES)
    # ---------------------------------------------------------------------------
    
    visualize_preds_and_scores(
        model=model,
        device=device,
        emb_dim=emb_dim,
        class_means=class_means,
        cov_matrix=cov_matrix,
        thresholds=thresholds,
        split_name=eval_split,
    )

if __name__ == "__main__":
    main()
