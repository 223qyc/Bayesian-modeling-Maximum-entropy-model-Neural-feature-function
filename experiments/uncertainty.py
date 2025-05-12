import torch
import torch.nn as nn
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
from tqdm import tqdm
import time
import random

from data import load_dataset, preprocess_text, prepare_data_loaders
from models import NNClassifier, MCDropoutClassifier, DeepEnsembleClassifier, VBMENN
from trainers import train_nn_classifier, train_mc_dropout, train_deep_ensemble, train_vb_menn
from trainers import evaluate_classifier, evaluate_uncertainty, compute_calibration_error
from utils import plot_uncertainty_histograms, plot_reliability_diagrams, plot_ood_detection_curves


def run_uncertainty_experiment(
        dataset_name: str = "imdb",
        ood_dataset_name: Optional[str] = "ag_news",
        model_types: List[str] = ["standard", "mc_dropout", "deep_ensemble", "vb_menn"],
        hidden_dims: List[int] = [128, 64],
        bert_model: str = "bert-base-uncased",
        batch_size: int = 16,
        n_epochs: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        results_dir: str = "results/uncertainty",
        seed: int = 42,
        verbose: bool = True
) -> Dict[str, Any]:
    '''
    运行不确定性评估实验
    :param dataset_name: 主数据集名称（用于训练和in-distribution测试）
    :param ood_dataset_name:分布外数据集名称（可选，用于OOD检测）
    :param hidden_dims:要评估的模型类型列表
    :param bert_model:BERT模型名称
    :param batch_size:批次大小
    :param n_epochs:训练轮数
    :param device:使用的设备
    :param results_dir:结果保存目录
    :param seed:随机种子
    :param verbose:是否显示详细信息
    :return:实验结果
    '''
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)

    # 加载主数据集
    print(f"加载{dataset_name}数据集...")
    train_dataset = load_dataset(dataset_name, split="train")
    test_dataset = load_dataset(dataset_name, split="test")

    # 处理训练集和测试集
    print("处理主数据集...")
    X_train, y_train = preprocess_text(
        train_dataset,
        model_name=bert_model,
        batch_size=batch_size,
        device=device,
        dataset_name=dataset_name,
        split="train",
        dataset_type="uncertainty"
    )

    X_test, y_test = preprocess_text(
        test_dataset,
        model_name=bert_model,
        batch_size=batch_size,
        device=device,
        dataset_name=dataset_name,
        split="test",
        dataset_type="uncertainty"
    )

    # 加载OOD数据集（如果提供）
    ood_data = None
    if ood_dataset_name:
        print(f"加载OOD数据集 {ood_dataset_name}...")
        ood_dataset = load_dataset(ood_dataset_name, split="test")

        # 处理OOD数据集
        X_ood, y_ood = preprocess_text(
            ood_dataset,
            model_name=bert_model,
            batch_size=batch_size,
            device=device,
            dataset_name=ood_dataset_name,
            split="test",
            dataset_type="uncertainty_ood"
        )
        ood_data = (X_ood, y_ood)

    # 确定输入维度和输出维度
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train.numpy()))

    print(f"输入维度: {input_dim}, 输出维度: {output_dim}")

    # 创建验证集（从训练集中划分）
    indices = torch.randperm(len(X_train))
    train_size = int(0.8 * len(X_train))

    X_train_split = X_train[indices[:train_size]]
    y_train_split = y_train[indices[:train_size]]

    X_val = X_train[indices[train_size:]]
    y_val = y_train[indices[train_size:]]

    # 准备数据加载器
    loaders = prepare_data_loaders(
        (X_train_split, y_train_split),
        (X_val, y_val),
        (X_test, y_test),
        batch_size=batch_size
    )

    # 如果有OOD数据，则创建OOD数据加载器
    if ood_data:
        ood_loader = prepare_data_loaders(
            ood_data, batch_size=batch_size
        )["train"]
    else:
        ood_loader = None

    # 存储结果
    results = {
        "config": {
            "dataset": dataset_name,
            "ood_dataset": ood_dataset_name,
            "model_types": model_types,
            "hidden_dims": hidden_dims,
            "bert_model": bert_model,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "seed": seed
        },
        "classification_metrics": {},
        "calibration_metrics": {},
        "uncertainty_metrics": {}
    }

    # 对每种模型类型进行训练和评估
    for model_type in model_types:
        print(f"\n=== 训练和评估模型类型: {model_type} ===")

        # 创建模型
        if model_type == "standard":
            model = NNClassifier(input_dim, hidden_dims, output_dim)
            training_fn = train_nn_classifier

        elif model_type == "mc_dropout":
            model = MCDropoutClassifier(input_dim, hidden_dims, output_dim, dropout_prob=0.1)
            training_fn = train_mc_dropout

        elif model_type == "deep_ensemble":
            model = DeepEnsembleClassifier(input_dim, hidden_dims, output_dim, n_estimators=5)
            training_fn = train_deep_ensemble

        elif model_type == "vb_menn":
            model = VBMENN(input_dim, hidden_dims, output_dim, prior_std=1.0)
            training_fn = train_vb_menn

        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 训练模型
        history = training_fn(
            model,
            loaders["train"],
            loaders["val"],
            n_epochs=n_epochs,
            device=device,
            verbose=verbose
        )

        # 评估分类性能
        print("\n评估分类性能...")
        classification_metrics = evaluate_classifier(
            model, loaders["test"], device=device, model_type=model_type
        )

        # 计算校准误差
        print("计算校准误差...")
        calibration_metrics = compute_calibration_error(
            model, loaders["test"], device=device, model_type=model_type
        )

        # 评估不确定性
        print("评估不确定性...")
        uncertainty_metrics = evaluate_uncertainty(
            model, loaders["test"], ood_loader, device=device, model_type=model_type
        )

        # 保存结果
        results["classification_metrics"][model_type] = classification_metrics
        results["calibration_metrics"][model_type] = calibration_metrics
        results["uncertainty_metrics"][model_type] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in uncertainty_metrics.items()
        }

        # 输出结果
        print(f"测试准确率: {classification_metrics['accuracy']:.4f}")
        print(f"测试F1分数 (macro): {classification_metrics['f1_macro']:.4f}")
        print(f"测试ECE: {calibration_metrics['ece']:.4f}")

        if ood_loader:
            print(f"OOD检测AUROC (熵): {uncertainty_metrics['auroc_entropy']:.4f}")
            if uncertainty_metrics["auroc_variance"] is not None:
                print(f"OOD检测AUROC (方差): {uncertainty_metrics['auroc_variance']:.4f}")

    # 保存结果
    results_file = os.path.join(results_dir, f"{dataset_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n结果已保存到 {results_file}")

    # 绘制结果图表
    plots_dir = os.path.join(results_dir, f"{dataset_name}_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 绘制可靠性图
    plot_reliability_diagrams(results, os.path.join(plots_dir, "reliability_diagrams.png"))

    # 绘制不确定性直方图
    plot_uncertainty_histograms(results, os.path.join(plots_dir, "uncertainty_histograms.png"))

    # 如果有OOD数据，绘制OOD检测曲线
    if ood_loader:
        plot_ood_detection_curves(results, os.path.join(plots_dir, "ood_detection_curves.png"))

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="运行不确定性评估实验")
    parser.add_argument("--dataset", type=str, default="imdb", help="主数据集名称")
    parser.add_argument("--ood_dataset", type=str, default="ag_news", help="OOD数据集名称")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--n_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--results_dir", type=str, default="results/uncertainty", help="结果保存目录")

    args = parser.parse_args()

    run_uncertainty_experiment(
        dataset_name=args.dataset,
        ood_dataset_name=args.ood_dataset,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        seed=args.seed,
        results_dir=args.results_dir
    )
