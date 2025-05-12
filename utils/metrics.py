import numpy as np
from typing import Tuple
from sklearn.metrics import roc_curve, auc


def confidence_histogram(
        confidences: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    计算置信度直方图
    :param confidences:预测的置信度
    :param predictions:预测的类别
    :param targets:真实的类别
    :param n_bins:bin的数量
    :return:(bin准确率, bin置信度, bin大小, bin边界)
    '''
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_sizes = np.zeros(n_bins)

    for i in range(n_bins):
        bin_mask = np.logical_and(confidences > bin_edges[i], confidences <= bin_edges[i + 1])
        if np.sum(bin_mask) > 0:
            bin_accs[i] = np.mean(predictions[bin_mask] == targets[bin_mask])
            bin_confs[i] = np.mean(confidences[bin_mask])
            bin_sizes[i] = np.sum(bin_mask) / len(confidences)

    return bin_accs, bin_confs, bin_sizes, bin_edges


def expected_calibration_error(
        confidences: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10
) -> float:
    '''
    计算期望校准误差 (Expected Calibration Error)
    :param confidences: 预测的置信度
    :param predictions: 预测的类别
    :param targets:真实的类别
    :param n_bins:bin的数量
    :return:ECE值
    '''
    bin_indices = np.minimum(
        np.floor(confidences * n_bins).astype(int),
        n_bins - 1
    )

    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(len(confidences)):
        bin_idx = bin_indices[i]
        bin_counts[bin_idx] += 1
        bin_accuracies[bin_idx] += predictions[i] == targets[i]
        bin_confidences[bin_idx] += confidences[i]

    # 避免除以零
    non_empty_bins = bin_counts > 0
    bin_accuracies[non_empty_bins] /= bin_counts[non_empty_bins]
    bin_confidences[non_empty_bins] /= bin_counts[non_empty_bins]

    # 计算ECE
    ece = np.sum(
        bin_counts[non_empty_bins] / len(confidences) *
        np.abs(bin_accuracies[non_empty_bins] - bin_confidences[non_empty_bins])
    )

    return float(ece)


def compute_roc_curve(
        y_true: np.ndarray,
        y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    计算ROC曲线和AUC
    :param y_true:真实标签
    :param y_score:预测分数
    :return:(假阳性率, 真阳性率, AUC)
    '''
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc
