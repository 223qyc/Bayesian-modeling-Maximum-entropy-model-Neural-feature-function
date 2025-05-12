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

from data import load_dataset, add_label_noise, preprocess_text, prepare_data_loaders
from models import NNClassifier, MCDropoutClassifier, DeepEnsembleClassifier, VBMENN
from trainers import train_nn_classifier, train_mc_dropout, train_deep_ensemble, train_vb_menn
from trainers import evaluate_classifier, compute_calibration_error
from utils import plot_learning_curves, plot_noise_ratio_vs_metrics


def run_noisy_labels_experiment(
        dataset_name: str = "imdb",
        noise_ratios: List[float] = [0.0, 0.1, 0.2, 0.3],
        model_types: List[str] = ["standard", "mc_dropout", "deep_ensemble", "vb_menn"],
        n_runs: int = 3,
        hidden_dims: List[int] = [128, 64],
        bert_model: str = "bert-base-uncased",
        batch_size: int = 16,
        n_epochs: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        results_dir: str = "results/noisy_labels",
        seed: int = 42,
        verbose: bool = True
) -> Dict[str, Any]:
    '''
    运行噪声标签实验
    :param dataset_name:数据集名称
    :param noise_ratios:噪声比例列表
    :param model_types:要评估的模型类型列表
    :param n_runs:每个配置的运行次数（用于计算平均值和标准差）
    :param hidden_dims:隐藏层维度列表
    :param bert_model:BERT模型名称
    :param batch_size:批次大小
    :param n_epochs: 训练轮数
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

    # 加载数据集
    print(f"加载{dataset_name}数据集...")
    train_dataset = load_dataset(dataset_name, split="train")
    test_dataset = load_dataset(dataset_name, split="test")

    # 处理测试集
    print("处理测试集...")
    X_test, y_test = preprocess_text(
        test_dataset,
        model_name=bert_model,
        batch_size=batch_size,
        device=device,
        dataset_name=dataset_name,
        split="test"
    )
    test_data = (X_test, y_test)

    # 确定输入维度和输出维度
    input_dim = X_test.shape[1]
    output_dim = len(np.unique(y_test.numpy()))

    print(f"输入维度: {input_dim}, 输出维度: {output_dim}")

    # 存储结果
    results = {
        "config": {
            "dataset": dataset_name,
            "noise_ratios": noise_ratios,
            "model_types": model_types,
            "n_runs": n_runs,
            "hidden_dims": hidden_dims,
            "bert_model": bert_model,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "seed": seed
        },
        "metrics": {model_type: {ratio: [] for ratio in noise_ratios} for model_type in model_types}
    }

    # 对每个噪声比例进行实验
    for noise_ratio in noise_ratios:
        print(f"\n=== 噪声比例: {noise_ratio * 100:.1f}% ===")

        # 多次运行以计算平均值和标准差
        for run in range(n_runs):
            print(f"\n--- 运行 {run + 1}/{n_runs} ---")

            # 添加标签噪声
            if noise_ratio > 0:
                print(f"添加{noise_ratio * 100:.1f}%噪声...")
                noisy_train_dataset = add_label_noise(
                    train_dataset,
                    noise_ratio,
                    seed=seed + run,
                    dataset_name=dataset_name,
                    split="train",
                    save_local=True
                )
            else:
                noisy_train_dataset = train_dataset

            # 处理训练集
            print("处理训练集...")
            X_train, y_train = preprocess_text(
                noisy_train_dataset,
                model_name=bert_model,
                batch_size=batch_size,
                device=device,
                dataset_name=dataset_name,
                split="train",
                dataset_type=f"noisy_{int(noise_ratio * 100)}_run_{run}"
            )

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
                test_data,
                batch_size=batch_size
            )

            # 对每种模型类型进行训练和评估
            for model_type in model_types:
                print(f"\n训练模型类型: {model_type}")

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

                # 评估模型
                metrics = evaluate_classifier(
                    model, loaders["test"], device=device, model_type=model_type
                )

                # 计算校准误差
                calibration = compute_calibration_error(
                    model, loaders["test"], device=device, model_type=model_type
                )

                # 合并指标
                metrics.update({"ece": calibration["ece"]})

                # 保存结果
                results["metrics"][model_type][noise_ratio].append(metrics)

                # 输出结果
                print(f"测试准确率: {metrics['accuracy']:.4f}")
                print(f"测试F1分数 (macro): {metrics['f1_macro']:.4f}")
                print(f"测试ECE: {metrics['ece']:.4f}")

    # 计算平均值和标准差
    summary = {model_type: {ratio: {} for ratio in noise_ratios} for model_type in model_types}

    for model_type in model_types:
        for ratio in noise_ratios:
            # 提取所有运行的指标
            accuracies = [run["accuracy"] for run in results["metrics"][model_type][ratio]]
            f1_scores = [run["f1_macro"] for run in results["metrics"][model_type][ratio]]
            eces = [run["ece"] for run in results["metrics"][model_type][ratio]]

            # 计算平均值和标准差
            summary[model_type][ratio] = {
                "accuracy_mean": np.mean(accuracies),
                "accuracy_std": np.std(accuracies),
                "f1_macro_mean": np.mean(f1_scores),
                "f1_macro_std": np.std(f1_scores),
                "ece_mean": np.mean(eces),
                "ece_std": np.std(eces)
            }

    # 将结果添加到总结果中
    results["summary"] = summary

    # 保存结果
    results_file = os.path.join(results_dir, f"{dataset_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n结果已保存到 {results_file}")

    # 绘制结果图表
    plot_noise_ratio_vs_metrics(results, os.path.join(results_dir, f"{dataset_name}_plots"))

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="运行噪声标签实验")
    parser.add_argument("--dataset", type=str, default="imdb", help="数据集名称")
    parser.add_argument("--noise_ratios", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3], help="噪声比例列表")
    parser.add_argument("--n_runs", type=int, default=3, help="每个配置的运行次数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--n_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--results_dir", type=str, default="results/noisy_labels", help="结果保存目录")

    args = parser.parse_args()

    run_noisy_labels_experiment(
        dataset_name=args.dataset,
        noise_ratios=args.noise_ratios,
        n_runs=args.n_runs,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        seed=args.seed,
        results_dir=args.results_dir
    )
