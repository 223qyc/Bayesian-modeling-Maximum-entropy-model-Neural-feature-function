import numpy as np
import torch
from datasets import load_dataset as hf_load_dataset
import random
import os
from typing import Dict, List, Tuple, Optional, Union
import pickle

# 设置数据集存储目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_files")
os.makedirs(DATA_DIR, exist_ok=True)


def load_dataset(
        dataset_name: str,
        split: str = "train",
        cache_dir: Optional[str] = None
) -> Dict:
    """
    用于加载指定的数据集
    参数:
        dataset_name: 数据集名称，支持 'imdb', 'ag_news', 'trec'
        split: 数据集分片，如 'train', 'test', 'validation'
        cache_dir: 缓存目录，默认为None
    返回:
        加载的数据集
    """
    supported_datasets = {
        "imdb": ("imdb", None),
        "ag_news": ("ag_news", None),
        "trec": ("trec", None),
    }

    if dataset_name not in supported_datasets:
        raise ValueError(f"不支持的数据集: {dataset_name}，支持的数据集有: {list(supported_datasets.keys())}")

    # 本地数据文件路径
    local_file_path = os.path.join(DATA_DIR, f"{dataset_name}_{split}.pkl")

    # 如果本地已有数据文件，直接加载
    if os.path.exists(local_file_path):
        print(f"从本地加载数据集: {local_file_path}")
        with open(local_file_path, 'rb') as f:
            return pickle.load(f)

    # 如果本地没有，则从Hugging Face下载并保存到本地
    print(f"从Hugging Face下载数据集 {dataset_name} 的 {split} 分割...")
    dataset_id, subset = supported_datasets[dataset_name]
    dataset = hf_load_dataset(dataset_id, subset, split=split, cache_dir=cache_dir)

    # 保存到本地
    print(f"保存数据集到本地: {local_file_path}")
    with open(local_file_path, 'wb') as f:
        pickle.dump(dataset, f)

    return dataset


def create_small_sample(
        dataset: Dict,
        n_samples_per_class: int,
        label_column: str = "label",
        seed: int = 42,
        dataset_name: str = None,
        split: str = None,
        save_local: bool = True
) -> Dict:
    """
    从数据集中抽取每个类别固定数量的样本，创建小样本数据集
    参数:
        dataset: 原始数据集
        n_samples_per_class: 每个类别的样本数
        label_column: 标签列名
        seed: 随机种子
        dataset_name: 数据集名称，用于保存本地文件
        split: 数据分割名称，用于保存本地文件
        save_local: 是否保存到本地
    返回:
        小样本数据集
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 获取所有唯一的类别
    all_labels = dataset[label_column]
    unique_labels = np.unique(all_labels)

    # 为每个类别选择样本
    selected_indices = []
    for label in unique_labels:
        label_indices = np.where(np.array(all_labels) == label)[0]
        if len(label_indices) <= n_samples_per_class:
            selected_indices.extend(label_indices)
        else:
            selected_indices.extend(
                np.random.choice(label_indices, n_samples_per_class, replace=False)
            )

    # 创建小样本数据集
    small_sample = dataset.select(selected_indices)

    print(f"原始数据集大小: {len(dataset)}")
    print(f"小样本数据集大小: {len(small_sample)}")
    print(f"每个类别的样本数: {n_samples_per_class}")

    # 如果提供了数据集名称和分割，保存到本地
    if save_local and dataset_name and split:
        local_file_path = os.path.join(DATA_DIR, f"{dataset_name}_{split}_small_{n_samples_per_class}.pkl")
        print(f"保存小样本数据集到本地: {local_file_path}")
        with open(local_file_path, 'wb') as f:
            pickle.dump(small_sample, f)

    return small_sample


def add_label_noise(
        dataset: Dict,
        noise_ratio: float,
        label_column: str = "label",
        seed: int = 42,
        dataset_name: str = None,
        split: str = None,
        save_local: bool = True
) -> Dict:
    """
       向数据集的标签添加噪声
       参数:
           dataset: 原始数据集
           noise_ratio: 噪声比例 (0.0 到 1.0 之间)
           label_column: 标签列名
           seed: 随机种子
           dataset_name: 数据集名称，用于保存本地文件
           split: 数据分割名称，用于保存本地文件
           save_local: 是否保存到本地
       返回:
           带有噪声标签的数据集
       """
    if noise_ratio < 0.0 or noise_ratio > 1.0:
        raise ValueError("噪声比例必须在 0.0 到 1.0 之间")

    random.seed(seed)
    np.random.seed(seed)

    # 获取所有标签和唯一标签
    all_labels = dataset[label_column]
    unique_labels = np.unique(all_labels)

    # 计算要添加噪声的样本数
    num_samples = len(dataset)
    num_noisy_samples = int(num_samples * noise_ratio)

    # --- 新增的检查 ---
    if num_noisy_samples == 0:
        print("噪声比例为 0 或计算出的噪声样本数为 0。返回原始数据集。")
        # 如果噪声样本数为0，直接返回原始数据集，不需要进行后续的remove/add列操作
        # 也不保存带有'_noisy_0'的文件，除非你有明确需求要标记无噪声的情况
        return dataset
    # --- 检查结束 ---


    # 随机选择要添加噪声的样本索引
    noisy_indices = np.random.choice(
        np.arange(num_samples), num_noisy_samples, replace=False
    )

    # 创建新的标签列表
    new_labels = all_labels.copy() # 复制原始标签列表进行修改
    for idx in noisy_indices:
        current_label = all_labels[idx]
        # 从其他标签中随机选择一个作为噪声标签
        other_labels = [l for l in unique_labels if l != current_label]
        # 确保有其他标签可以选择 (处理极端情况，如数据集中只有一个类别)
        if not other_labels:
             continue # 如果只有一个类别，无法添加“不同”的噪声，跳过此样本
        new_label = np.random.choice(other_labels)
        new_labels[idx] = new_label

    # 更新数据集中的标签
    # 先从原始数据集移除标签列，这会返回一个新数据集对象
    noisy_dataset = dataset.remove_columns(label_column)
    # 再向新的数据集对象中添加修改后的标签列，这会再次返回一个新数据集对象
    noisy_dataset = noisy_dataset.add_column(label_column, new_labels)


    print(f"原始数据集大小: {num_samples}")
    print(f"添加噪声的样本数: {num_noisy_samples} ({noise_ratio * 100:.1f}%)")

    # 如果提供了数据集名称和分割，保存到本地
    # 注意：此处的 save_local 块现在只在 num_noisy_samples > 0 时才会执行
    if save_local and dataset_name and split:
        local_file_path = os.path.join(DATA_DIR, f"{dataset_name}_{split}_noisy_{int(noise_ratio * 100)}.pkl")
        print(f"保存带噪声数据集到本地: {local_file_path}")
        with open(local_file_path, 'wb') as f:
            pickle.dump(noisy_dataset, f) # noisy_dataset 在此处已被赋值

    return noisy_dataset # noisy_dataset 在此处已被赋值