import numpy as np


def compute_metrics(pred_mask, gt_mask):
    """
    计算预测 mask 和真实标注的评估指标。

    Args:
        pred_mask (np.ndarray): 预测的二值化 mask (0 or 255)
        gt_mask (np.ndarray): 真实标注的二值化 mask (0 or 255)

    Returns:
        dict: 包含 MAE, F-measure, Precision, Recall, IoU 的指标
    """
    # 确保输入为二值化 mask (0 或 255)
    pred_mask = (pred_mask > 127).astype(np.uint8)
    gt_mask = (gt_mask > 127).astype(np.uint8)

    # MAE
    mae = np.mean(np.abs(pred_mask - gt_mask) / 255.0)

    # True Positives, False Positives, False Negatives
    TP = np.sum((pred_mask == 1) & (gt_mask == 1))
    FP = np.sum((pred_mask == 1) & (gt_mask == 0))
    FN = np.sum((pred_mask == 0) & (gt_mask == 1))
    TN = np.sum((pred_mask == 0) & (gt_mask == 0))

    # Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # F-measure (β=1)
    beta = 1.0
    fmeasure = ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall) if (
                                                                                                        precision + recall) > 0 else 0.0

    # IoU
    intersection = TP
    union = TP + FP + FN
    iou = intersection / union if union > 0 else 0.0

    return {
        'mae': mae,
        'fmeasure': fmeasure,
        'precision': precision,
        'recall': recall,
        'iou': iou
    }