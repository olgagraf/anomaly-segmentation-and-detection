import numpy as np
from config import IGNORE_INDEX


def compute_iou(preds, labels, num_classes, ignore_index=IGNORE_INDEX):
    """Compute the Intersection over Union metric for the predictions and labels."""
    iou = np.zeros(num_classes)
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    indicator = np.zeros(num_classes)

    for c in range(num_classes):
        mask = labels != ignore_index
        true_class_mask = (labels == c) & mask
        predicted_class_mask = (preds == c) & mask

        tp[c] = np.sum(true_class_mask & predicted_class_mask)
        fp[c] = np.sum(predicted_class_mask & ~true_class_mask)
        fn[c] = np.sum(~predicted_class_mask & true_class_mask)

        indicator[c] = 1 if np.sum(true_class_mask) > 0 else 0

        iou[c] = tp[c] / (tp[c] + fp[c] + fn[c]) if (tp[c] + fp[c] + fn[c]) > 0 else 0

    return iou, indicator
