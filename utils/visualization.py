import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List,Optional,Any
import os
from .metrics import compute_roc_curve

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.set_theme(style="whitegrid")


# 为不同模型类型指定颜色和标记
MODEL_COLORS = {
    "standard": "blue",
    "mc_dropout": "green",
    "deep_ensemble": "orange",
    "vb_menn": "red"
}

MODEL_MARKERS = {
    "standard": "o",
    "mc_dropout": "^",
    "deep_ensemble": "s",
    "vb_menn": "D"
}

MODEL_NAMES = {
    "standard": "标准神经网络",
    "mc_dropout": "MC Dropout",
    "deep_ensemble": "Deep Ensemble",
    "vb_menn": "VB-MENN"
}

def plot_learning_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    '''
    绘制学习曲线
    :param history:训练历史记录
    :param save_path:图像保存路径
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 绘制损失曲线
    ax1.plot(history["train_loss"], label="训练集")
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="验证集")
    ax1.set_title("损失曲线")
    ax1.set_xlabel("轮数")
    ax1.set_ylabel("损失")
    ax1.legend()

    # 绘制准确率曲线
    ax2.plot(history["train_acc"], label="训练集")
    if "val_acc" in history:
        ax2.plot(history["val_acc"], label="验证集")
    ax2.set_title("准确率曲线")
    ax2.set_xlabel("轮数")
    ax2.set_ylabel("准确率")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_sample_size_vs_metrics(
    results: Dict[str, Any],
    save_dir: str
) -> None:
    '''
    绘制样本大小与性能指标的关系图
    :param results:实验结果
    :param save_dir: 图像保存目录
    '''
    os.makedirs(save_dir, exist_ok=True)

    # 获取样本大小和模型类型
    sample_sizes = results["config"]["sample_sizes"]
    model_types = results["config"]["model_types"]
    summary = results["summary"]

    # 绘制准确率图
    plt.figure(figsize=(10, 6))

    for model_type in model_types:
        acc_means = [summary[model_type][size]["accuracy_mean"] for size in sample_sizes]
        acc_stds = [summary[model_type][size]["accuracy_std"] for size in sample_sizes]

        plt.errorbar(
            sample_sizes, acc_means, yerr=acc_stds,
            label=MODEL_NAMES[model_type],
            marker=MODEL_MARKERS[model_type],
            color=MODEL_COLORS[model_type],
            capsize=4
        )

    plt.title("样本大小对准确率的影响")
    plt.xlabel("每类样本数")
    plt.ylabel("准确率")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_vs_sample_size.png"), dpi=300, bbox_inches='tight')

    # 绘制F1分数图
    plt.figure(figsize=(10, 6))

    for model_type in model_types:
        f1_means = [summary[model_type][size]["f1_macro_mean"] for size in sample_sizes]
        f1_stds = [summary[model_type][size]["f1_macro_std"] for size in sample_sizes]

        plt.errorbar(
            sample_sizes, f1_means, yerr=f1_stds,
            label=MODEL_NAMES[model_type],
            marker=MODEL_MARKERS[model_type],
            color=MODEL_COLORS[model_type],
            capsize=4
        )

    plt.title("样本大小对F1分数的影响")
    plt.xlabel("每类样本数")
    plt.ylabel("F1分数 (宏平均)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "f1_vs_sample_size.png"), dpi=300, bbox_inches='tight')

    # 绘制ECE图
    plt.figure(figsize=(10, 6))

    for model_type in model_types:
        ece_means = [summary[model_type][size]["ece_mean"] for size in sample_sizes]
        ece_stds = [summary[model_type][size]["ece_std"] for size in sample_sizes]

        plt.errorbar(
            sample_sizes, ece_means, yerr=ece_stds,
            label=MODEL_NAMES[model_type],
            marker=MODEL_MARKERS[model_type],
            color=MODEL_COLORS[model_type],
            capsize=4
        )

    plt.title("样本大小对ECE的影响")
    plt.xlabel("每类样本数")
    plt.ylabel("期望校准误差 (ECE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "ece_vs_sample_size.png"), dpi=300, bbox_inches='tight')

def plot_noise_ratio_vs_metrics(
    results: Dict[str, Any],
    save_dir: str
) -> None:
    '''
    绘制噪声比例与性能指标的关系图
    :param results:实验结果
    :param save_dir:图像保存目录
    '''
    os.makedirs(save_dir, exist_ok=True)

    # 获取噪声比例和模型类型
    noise_ratios = results["config"]["noise_ratios"]
    model_types = results["config"]["model_types"]
    summary = results["summary"]

    # 将噪声比例转换为百分比
    noise_percentages = [ratio * 100 for ratio in noise_ratios]

    # 绘制准确率图
    plt.figure(figsize=(10, 6))

    for model_type in model_types:
        acc_means = [summary[model_type][ratio]["accuracy_mean"] for ratio in noise_ratios]
        acc_stds = [summary[model_type][ratio]["accuracy_std"] for ratio in noise_ratios]

        plt.errorbar(
            noise_percentages, acc_means, yerr=acc_stds,
            label=MODEL_NAMES[model_type],
            marker=MODEL_MARKERS[model_type],
            color=MODEL_COLORS[model_type],
            capsize=4
        )

    plt.title("噪声比例对准确率的影响")
    plt.xlabel("噪声比例 (%)")
    plt.ylabel("准确率")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_vs_noise.png"), dpi=300, bbox_inches='tight')

    # 绘制F1分数图
    plt.figure(figsize=(10, 6))

    for model_type in model_types:
        f1_means = [summary[model_type][ratio]["f1_macro_mean"] for ratio in noise_ratios]
        f1_stds = [summary[model_type][ratio]["f1_macro_std"] for ratio in noise_ratios]

        plt.errorbar(
            noise_percentages, f1_means, yerr=f1_stds,
            label=MODEL_NAMES[model_type],
            marker=MODEL_MARKERS[model_type],
            color=MODEL_COLORS[model_type],
            capsize=4
        )

    plt.title("噪声比例对F1分数的影响")
    plt.xlabel("噪声比例 (%)")
    plt.ylabel("F1分数 (宏平均)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "f1_vs_noise.png"), dpi=300, bbox_inches='tight')

    # 绘制ECE图
    plt.figure(figsize=(10, 6))

    for model_type in model_types:
        ece_means = [summary[model_type][ratio]["ece_mean"] for ratio in noise_ratios]
        ece_stds = [summary[model_type][ratio]["ece_std"] for ratio in noise_ratios]

        plt.errorbar(
            noise_percentages, ece_means, yerr=ece_stds,
            label=MODEL_NAMES[model_type],
            marker=MODEL_MARKERS[model_type],
            color=MODEL_COLORS[model_type],
            capsize=4
        )

    plt.title("噪声比例对ECE的影响")
    plt.xlabel("噪声比例 (%)")
    plt.ylabel("期望校准误差 (ECE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "ece_vs_noise.png"), dpi=300, bbox_inches='tight')

def plot_reliability_diagrams(
    results: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    '''
    绘制可靠性图
    :param results:实验结果
    :param save_path:图像保存路径
    '''
    model_types = results["config"]["model_types"]
    calibration_metrics = results["calibration_metrics"]

    # 创建子图
    fig, axes = plt.subplots(1, len(model_types), figsize=(5 * len(model_types), 5))

    # 如果只有一个模型类型，需要调整axes为数组
    if len(model_types) == 1:
        axes = [axes]

    # 绘制每个模型的可靠性图
    for i, model_type in enumerate(model_types):
        ax = axes[i]
        metrics = calibration_metrics[model_type]

        bin_accs = metrics["bin_accs"]
        bin_confs = metrics["bin_confs"]
        bin_edges = metrics["bin_edges"]
        ece = metrics["ece"]

        # 绘制可靠性图
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(
            bin_centers, bin_accs, width=(bin_edges[1] - bin_edges[0]),
            alpha=0.8, label="Outputs", edgecolor="black"
        )

        # 绘制对角线 (Perfect Calibration)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="完美校准")

        # 绘制置信度
        ax.plot(bin_centers, bin_confs, marker="o", color="red", label="置信度")

        ax.set_title(f"{MODEL_NAMES[model_type]}\nECE: {ece:.4f}")
        ax.set_xlabel("置信度")
        ax.set_ylabel("准确率")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_uncertainty_histograms(
    results: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    '''
    绘制不确定性直方图
    :param results:实验结果
    :param save_path:图像保存路径
    '''
    model_types = results["config"]["model_types"]
    uncertainty_metrics = results["uncertainty_metrics"]

    # 创建子图
    fig, axes = plt.subplots(len(model_types), 2, figsize=(12, 5 * len(model_types)))

    # 如果只有一个模型类型，需要调整axes为2D数组
    if len(model_types) == 1:
        axes = axes.reshape(1, -1)

    # 绘制每个模型的不确定性直方图
    for i, model_type in enumerate(model_types):
        metrics = uncertainty_metrics[model_type]

        # 提取正确和错误预测的不确定性
        in_dist_entropy = np.array(metrics["in_dist_entropy"])
        in_dist_correct = np.array(metrics["in_dist_correct"])

        correct_entropy = in_dist_entropy[in_dist_correct]
        incorrect_entropy = in_dist_entropy[~in_dist_correct]

        # 绘制熵直方图
        ax1 = axes[i, 0]
        ax1.hist(
            correct_entropy, bins=20, alpha=0.7, label="正确预测",
            color="green", density=True
        )
        ax1.hist(
            incorrect_entropy, bins=20, alpha=0.7, label="错误预测",
            color="red", density=True
        )

        ax1.set_title(f"{MODEL_NAMES[model_type]}: 预测熵直方图")
        ax1.set_xlabel("预测熵")
        ax1.set_ylabel("密度")
        ax1.legend()

        # 绘制方差直方图（如果可用）
        ax2 = axes[i, 1]

        if "in_dist_variance" in metrics and not np.all(np.array(metrics["in_dist_variance"]) == 0):
            in_dist_variance = np.array(metrics["in_dist_variance"])

            correct_variance = in_dist_variance[in_dist_correct]
            incorrect_variance = in_dist_variance[~in_dist_correct]

            ax2.hist(
                correct_variance, bins=20, alpha=0.7, label="正确预测",
                color="green", density=True
            )
            ax2.hist(
                incorrect_variance, bins=20, alpha=0.7, label="错误预测",
                color="red", density=True
            )

            ax2.set_title(f"{MODEL_NAMES[model_type]}: 预测方差直方图")
            ax2.set_xlabel("预测方差")
            ax2.set_ylabel("密度")
            ax2.legend()
        else:
            ax2.text(
                0.5, 0.5, "方差不可用\n(标准模型无法估计方差)",
                ha="center", va="center", transform=ax2.transAxes
            )
            ax2.set_title(f"{MODEL_NAMES[model_type]}: 预测方差直方图")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_ood_detection_curves(
    results: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    '''
    绘制OOD检测ROC曲线
    :param results:实验结果
    :param save_path:图像保存路径
    '''
    model_types = results["config"]["model_types"]
    uncertainty_metrics = results["uncertainty_metrics"]

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制每个模型的ROC曲线
    for model_type in model_types:
        metrics = uncertainty_metrics[model_type]

        if "auroc_entropy" in metrics and metrics["auroc_entropy"] is not None:
            # 计算ROC曲线
            in_dist_entropy = np.array(metrics["in_dist_entropy"])
            ood_entropy = np.array(metrics["ood_entropy"])

            # 构建标签（0表示in-distribution，1表示OOD）
            y_true = np.concatenate([
                np.zeros(len(in_dist_entropy)),
                np.ones(len(ood_entropy))
            ])

            # 构建分数（熵越高，越有可能是OOD）
            y_score = np.concatenate([in_dist_entropy, ood_entropy])

            # 计算ROC曲线
            fpr, tpr, auroc = compute_roc_curve(y_true, y_score)

            # 绘制ROC曲线
            plt.plot(
                fpr, tpr,
                label=f"{MODEL_NAMES[model_type]} (熵, AUC = {auroc:.4f})",
                color=MODEL_COLORS[model_type],
                linestyle="-"
            )

            # 如果有方差信息，也绘制方差的ROC曲线
            if "auroc_variance" in metrics and metrics["auroc_variance"] is not None:
                in_dist_variance = np.array(metrics["in_dist_variance"])
                ood_variance = np.array(metrics["ood_variance"])

                # 构建分数（方差越高，越有可能是OOD）
                y_score_var = np.concatenate([in_dist_variance, ood_variance])

                # 计算ROC曲线
                fpr_var, tpr_var, auroc_var = compute_roc_curve(y_true, y_score_var)

                # 绘制ROC曲线
                plt.plot(
                    fpr_var, tpr_var,
                    label=f"{MODEL_NAMES[model_type]} (方差, AUC = {auroc_var:.4f})",
                    color=MODEL_COLORS[model_type],
                    linestyle="--"
                )

    # 绘制随机分类器的基准线
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray", label="随机分类")

    plt.title("OOD检测ROC曲线")
    plt.xlabel("假阳性率")
    plt.ylabel("真阳性率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')