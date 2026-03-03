import numpy as np
import torch


def ood_threshold(values_in: np.ndarray, tpr: float) -> float:
    """Compute score threshold so approximately ``tpr`` fraction stays ID.

    Uses the ``(1 - tpr)`` quantile of ID-reference scores. Samples below this
    threshold are treated as OOD at evaluation time.
    """
    if len(values_in) == 0:
        return np.NAN
    t = np.quantile(values_in, (1 - tpr))
    return t


def mahalanobis(img_features, class_means, cov_matrix, dev):
    """Compute squared Mahalanobis distances with shared covariance matrix.

    Inputs:
        img_features: ``(N, D)`` feature vectors (NumPy array or torch tensor).
        class_means: ``(C, D)`` class-mean vectors (NumPy array or torch tensor).
        cov_matrix: ``(D, D)`` shared covariance matrix (NumPy array or torch tensor).
        dev: torch device where computation is performed.

    Outputs:
        - ``-m_distances``: ``(N, C)`` negated squared Mahalanobis distances.
    Additional outputs:
        - ``-m_scores``: ``(N,)`` negated Mahalanobis score (original implementation by Lee et al.) per sample.
        - ``min_indices``: ``(N,)`` index of class with minimum distance per sample.

    Negated distances/scores are returned so higher values mean more ID.
    """
    if isinstance(img_features, np.ndarray):
        img_features = torch.tensor(img_features, dtype=torch.float32)
    if isinstance(class_means, np.ndarray):
        class_means = torch.tensor(class_means, dtype=torch.float32)
    if isinstance(cov_matrix, np.ndarray):
        cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32)

    img_features = img_features.to(dev)
    class_means = class_means.to(dev)
    cov_matrix = cov_matrix.to(dev)

    diff = img_features.unsqueeze(1) - class_means
    cov_matrix_inv = torch.linalg.inv(cov_matrix)

    m_distances = torch.einsum("ijk,kl,ijl->ij", diff, cov_matrix_inv, diff)
    m_scores, min_indices = torch.min(m_distances, dim=1)

    return -m_distances, -m_scores, min_indices


def compute_maha_plus_scores(
    averaged_features,
    class_means,
    cov_matrix,
    preds,
    n_classes,
    device,
):
    """Compute per-pixel Maha+ scores for predicted classes.

    Inputs:
        averaged_features: ``(N, D)`` feature vectors.
        class_means: ``(C, D)`` class-mean vectors.
        cov_matrix: ``(D, D)`` shared covariance matrix.
        preds: ``(H, W)`` predicted class index map in output resolution.
        n_classes: Number of in-distribution classes ``C``.
        device: torch device where computation is performed.

    Outputs:
        np.ndarray of shape ``(H, W)`` with Maha+ score per pixel, where each
        pixel score is taken from the Mahalanobis distance map of its predicted
        class.
    """
    out_h, out_w = preds.shape
    m_distances, _, _ = mahalanobis(averaged_features, class_means, cov_matrix, device)
    emb_size = int(np.sqrt(averaged_features.shape[0]))
    m_distances = m_distances.view(emb_size, emb_size, n_classes)
    m_distances = m_distances.permute(2, 0, 1).unsqueeze(0)
    m_distances = torch.nn.functional.interpolate(
        m_distances,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )
    m_distances = m_distances.squeeze(0).permute(1, 2, 0)
    m_scores = m_distances[torch.arange(out_h).unsqueeze(1), torch.arange(out_w), preds]
    return m_scores.cpu().numpy()
