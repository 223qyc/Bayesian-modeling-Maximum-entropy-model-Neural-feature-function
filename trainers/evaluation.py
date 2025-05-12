import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict,Optional,Any
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from models.baseline_models import NNClassifier, MCDropoutClassifier, DeepEnsembleClassifier
from models.vb_menn import VBMENN

def evaluate_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_type: str = "standard"
) -> Dict[str, float]:
    '''
    评估分类器在给定数据集上的性能
    :param model:模型
    :param data_loader:数据加载器
    :param device:使用的设备
    :param model_type:模型类型，可选值：'standard', 'mc_dropout', 'deep_ensemble', 'vb_menn'
    :return:包含各种指标的字典
    '''
    model = model.to(device)

    # 初始化指标字典
    metrics = {
        "accuracy": 0.0,
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
        "precision_macro": 0.0,
        "recall_macro": 0.0,
        "nll": 0.0,
        "inference_time": 0.0,
    }

    all_targets = []
    all_predictions = []
    all_probs = []

    # 评估模式
    model.eval()

    # 计时开始
    start_time = time.time()

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            # 根据模型类型获取预测
            if model_type == "standard":
                logits = model(X)
                probs = F.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)

            elif model_type == "mc_dropout":
                mean_probs, _ = model.predict_proba_mc(X, n_samples=20)
                _, preds = torch.max(mean_probs, 1)
                probs = mean_probs

            elif model_type == "deep_ensemble":
                mean_probs, _ = model.predict_proba(X)
                _, preds = torch.max(mean_probs, 1)
                probs = mean_probs

            elif model_type == "vb_menn":
                mean_probs, _, _ = model.predict_proba_vi(X, n_samples=20)
                _, preds = torch.max(mean_probs, 1)
                probs = mean_probs

            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

            # 收集结果
            all_targets.append(y.cpu())
            all_predictions.append(preds.cpu())
            all_probs.append(probs.cpu())

    # 计时结束
    end_time = time.time()
    metrics["inference_time"] = end_time - start_time

    # 整合所有批次的结果
    all_targets = torch.cat(all_targets).numpy()
    all_predictions = torch.cat(all_predictions).numpy()
    all_probs = torch.cat(all_probs).numpy()

    # 计算指标
    metrics["accuracy"] = accuracy_score(all_targets, all_predictions)
    metrics["f1_macro"] = f1_score(all_targets, all_predictions, average="macro")
    metrics["f1_weighted"] = f1_score(all_targets, all_predictions, average="weighted")
    metrics["precision_macro"] = precision_score(all_targets, all_predictions, average="macro")
    metrics["recall_macro"] = recall_score(all_targets, all_predictions, average="macro")

    # 计算负对数似然 (NLL)
    nll = 0.0
    for i in range(len(all_targets)):
        nll -= np.log(all_probs[i, all_targets[i]] + 1e-10)
    metrics["nll"] = nll / len(all_targets)

    return metrics

def evaluate_uncertainty(
    model: nn.Module,
    in_dist_loader: DataLoader,
    ood_loader: Optional[DataLoader] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_type: str = "standard",
    n_samples: int = 20
) -> Dict[str, Any]:
    '''
    评估模型的不确定性估计
    :param model:模型
    :param in_dist_loader:分布内数据加载器
    :param ood_loader:分布外数据加载器（可选）
    :param device:使用的设备
    :param model_type:模型类型，可选值：'standard', 'mc_dropout', 'deep_ensemble', 'vb_menn'
    :param n_samples:采样次数（用于mc_dropout, vb_menn）
    :return:包含不确定性指标的字典
    '''

    model = model.to(device)

    # 初始化结果字典
    results = {
        "in_dist_entropy": [],
        "in_dist_variance": [],
        "in_dist_correct": [],
        "in_dist_confidence": [],
        "ood_entropy": [],
        "ood_variance": [],
        "ood_confidence": [],
        "auroc_entropy": None,
        "auroc_variance": None
    }

    # 评估模式
    model.eval()

    # 处理分布内数据
    with torch.no_grad():
        for X, y in in_dist_loader:
            X, y = X.to(device), y.to(device)

            # 根据模型类型获取预测和不确定性
            if model_type == "standard":
                logits = model(X)
                probs = F.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)

                # 计算熵和方差
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                variance = torch.zeros_like(entropy)  # 标准模型无法估计方差

            elif model_type == "mc_dropout":
                mean_probs, std_probs = model.predict_proba_mc(X, n_samples=n_samples)
                _, preds = torch.max(mean_probs, 1)

                # 计算熵和方差
                entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
                variance = torch.mean(std_probs, dim=1)

            elif model_type == "deep_ensemble":
                mean_probs, std_probs = model.predict_proba(X)
                _, preds = torch.max(mean_probs, 1)

                # 计算熵和方差
                entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
                variance = torch.mean(std_probs, dim=1)

            elif model_type == "vb_menn":
                mean_probs, std_probs, logits_vars = model.predict_proba_vi(X, n_samples=n_samples)
                _, preds = torch.max(mean_probs, 1)

                # 计算熵和方差
                entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
                variance = torch.mean(std_probs, dim=1)

            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

            # 收集结果
            results["in_dist_entropy"].append(entropy.cpu().numpy())
            results["in_dist_variance"].append(variance.cpu().numpy())
            results["in_dist_correct"].append((preds == y).cpu().numpy())

            # 每个样本的最大预测概率（置信度）
            confidence, _ = torch.max(mean_probs if model_type != "standard" else probs, dim=1)
            results["in_dist_confidence"].append(confidence.cpu().numpy())

    # 整合所有批次的结果
    results["in_dist_entropy"] = np.concatenate(results["in_dist_entropy"])
    results["in_dist_variance"] = np.concatenate(results["in_dist_variance"])
    results["in_dist_correct"] = np.concatenate(results["in_dist_correct"])
    results["in_dist_confidence"] = np.concatenate(results["in_dist_confidence"])

    # 处理分布外数据（如果提供）
    if ood_loader is not None:
        ood_entropy = []
        ood_variance = []
        ood_confidence = []

        with torch.no_grad():
            for X, _ in ood_loader:  # OOD数据可能没有标签，或标签不重要
                X = X.to(device)

                # 根据模型类型获取预测和不确定性
                if model_type == "standard":
                    logits = model(X)
                    probs = F.softmax(logits, dim=1)

                    # 计算熵和方差
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    variance = torch.zeros_like(entropy)

                    # 置信度
                    confidence, _ = torch.max(probs, dim=1)

                elif model_type == "mc_dropout":
                    mean_probs, std_probs = model.predict_proba_mc(X, n_samples=n_samples)

                    # 计算熵和方差
                    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
                    variance = torch.mean(std_probs, dim=1)

                    # 置信度
                    confidence, _ = torch.max(mean_probs, dim=1)

                elif model_type == "deep_ensemble":
                    mean_probs, std_probs = model.predict_proba(X)

                    # 计算熵和方差
                    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
                    variance = torch.mean(std_probs, dim=1)

                    # 置信度
                    confidence, _ = torch.max(mean_probs, dim=1)

                elif model_type == "vb_menn":
                    mean_probs, std_probs, logits_vars = model.predict_proba_vi(X, n_samples=n_samples)

                    # 计算熵和方差
                    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
                    variance = torch.mean(std_probs, dim=1)

                    # 置信度
                    confidence, _ = torch.max(mean_probs, dim=1)

                # 收集结果
                ood_entropy.append(entropy.cpu().numpy())
                ood_variance.append(variance.cpu().numpy())
                ood_confidence.append(confidence.cpu().numpy())

        # 整合所有批次的结果
        results["ood_entropy"] = np.concatenate(ood_entropy)
        results["ood_variance"] = np.concatenate(ood_variance)
        results["ood_confidence"] = np.concatenate(ood_confidence)

        # 计算OOD检测的AUROC
        # 将in-dist标记为0，ood标记为1
        y_true = np.concatenate([np.zeros_like(results["in_dist_entropy"]), np.ones_like(results["ood_entropy"])])

        # entropy越高，越可能是OOD
        y_score_entropy = np.concatenate([results["in_dist_entropy"], results["ood_entropy"]])
        results["auroc_entropy"] = roc_auc_score(y_true, y_score_entropy)

        # variance越高，越可能是OOD
        y_score_variance = np.concatenate([results["in_dist_variance"], results["ood_variance"]])
        if not np.all(y_score_variance == 0):  # 确保方差不全为0
            results["auroc_variance"] = roc_auc_score(y_true, y_score_variance)

    return results

def compute_calibration_error(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_type: str = "standard",
    n_bins: int = 10
) -> Dict[str, float]:
    '''
    计算模型的校准误差（ECE: Expected Calibration Error）
    :param model:模型
    :param data_loader:数据加载器
    :param device:使用的设备
    :param model_type:模型类型
    :param n_bins:用于计算ECE的bin数量
    :return: 包含校准指标的字典
    '''

    model = model.to(device)

    # 初始化结果
    confidences = []
    predictions = []
    targets = []

    # 评估模式
    model.eval()

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            # 根据模型类型获取预测
            if model_type == "standard":
                logits = model(X)
                probs = F.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)

            elif model_type == "mc_dropout":
                mean_probs, _ = model.predict_proba_mc(X, n_samples=20)
                _, preds = torch.max(mean_probs, 1)
                probs = mean_probs

            elif model_type == "deep_ensemble":
                mean_probs, _ = model.predict_proba(X)
                _, preds = torch.max(mean_probs, 1)
                probs = mean_probs

            elif model_type == "vb_menn":
                mean_probs, _, _ = model.predict_proba_vi(X, n_samples=20)
                _, preds = torch.max(mean_probs, 1)
                probs = mean_probs

            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

            # 获取预测类别的概率（置信度）
            batch_confidences = torch.gather(probs, 1, preds.unsqueeze(1)).squeeze()

            # 收集结果
            confidences.append(batch_confidences.cpu().numpy())
            predictions.append(preds.cpu().numpy())
            targets.append(y.cpu().numpy())

    # 整合所有批次的结果
    confidences = np.concatenate(confidences)
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    # 计算ECE
    ece = _expected_calibration_error(confidences, predictions, targets, n_bins)

    # 计算每个bin的准确率和置信度
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_sizes = []

    for i in range(n_bins):
        bin_mask = np.logical_and(confidences > bin_edges[i], confidences <= bin_edges[i + 1])
        if np.sum(bin_mask) > 0:
            bin_accs.append(np.mean(predictions[bin_mask] == targets[bin_mask]))
            bin_confs.append(np.mean(confidences[bin_mask]))
            bin_sizes.append(np.sum(bin_mask) / len(confidences))
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_sizes.append(0)

    return {
        "ece": ece,
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_sizes": bin_sizes,
        "bin_edges": bin_edges
    }

def _expected_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> float:
    '''
    计算期望校准误差 (Expected Calibration Error)
    :param confidences:预测的置信度
    :param predictions:预测的类别
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



