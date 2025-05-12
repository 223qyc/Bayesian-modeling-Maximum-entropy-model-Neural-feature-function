import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import tqdm
import os
import pickle

# 设置数据集存储目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_files")
os.makedirs(DATA_DIR, exist_ok=True)


def preprocess_text(
        dataset: Dict,
        text_column: str = "text",
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dataset_name: str = None,
        split: str = None,
        dataset_type: str = None,
        save_local: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预处理文本数据，获取BERT编码

    参数:
        dataset: 数据集
        text_column: 文本列的名称
        model_name: 使用的预训练模型名称
        max_length: 最大序列长度
        batch_size: 批处理大小
        device: 使用的设备
        dataset_name: 数据集名称，用于保存本地文件
        split: 数据分割名称，用于保存本地文件
        dataset_type: 数据集类型（如small_10, noisy_20等），用于保存本地文件
        save_local: 是否保存到本地

    返回:
        (文本编码, 标签)
    """
    # 如果提供了数据集信息，检查本地是否有已处理的数据
    if save_local and dataset_name and split:
        # 构建文件名
        file_suffix = f"_{dataset_type}" if dataset_type else ""
        embeddings_file = os.path.join(DATA_DIR, f"{dataset_name}_{split}{file_suffix}_embeddings.pkl")

        # 如果本地已有处理好的数据，直接加载
        if os.path.exists(embeddings_file):
            print(f"从本地加载预处理数据: {embeddings_file}")
            with open(embeddings_file, 'rb') as f:
                return pickle.load(f)

    # 加载tokenizer和模型
    print(f"加载预训练模型 {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()  # 设置为评估模式

    # 准备空tensor来存储结果
    embeddings = []
    labels = []

    # 创建数据加载器
    texts = dataset[text_column]
    dataset_labels = dataset["label"]

    # 分批处理
    print(f"使用{model_name}处理文本...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_labels = dataset_labels[i:i + batch_size]

        # Tokenize
        encoded_input = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        # 获取BERT输出
        with torch.no_grad():
            outputs = model(**encoded_input)

        # 使用注意力掩码进行平均池化（忽略padding tokens）
        last_hidden_states = outputs.last_hidden_state
        attention_mask = encoded_input["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        batch_embeddings = (sum_embeddings / sum_mask).cpu()

        embeddings.append(batch_embeddings)
        labels.append(torch.tensor(batch_labels))

    # 连接所有批次的结果
    text_embeddings = torch.cat(embeddings, dim=0)
    label_tensor = torch.cat(labels, dim=0)

    print(f"文本嵌入形状: {text_embeddings.shape}")
    print(f"标签形状: {label_tensor.shape}")

    # 如果提供了数据集信息，保存处理结果到本地
    if save_local and dataset_name and split:
        file_suffix = f"_{dataset_type}" if dataset_type else ""
        embeddings_file = os.path.join(DATA_DIR, f"{dataset_name}_{split}{file_suffix}_embeddings.pkl")
        print(f"保存预处理数据到本地: {embeddings_file}")
        with open(embeddings_file, 'wb') as f:
            pickle.dump((text_embeddings, label_tensor), f)

    return text_embeddings, label_tensor


def prepare_data_loaders(
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        batch_size: int = 32,
        shuffle: bool = True
) -> Dict[str, DataLoader]:
    """
    准备数据加载器

    参数:
        train_data: 训练数据 (特征, 标签)
        val_data: 验证数据 (特征, 标签)
        test_data: 测试数据 (特征, 标签)
        batch_size: 批处理大小
        shuffle: 是否打乱训练数据

    返回:
        包含数据加载器的字典
    """
    data_loaders = {}

    # 训练数据加载器
    X_train, y_train = train_data
    train_dataset = TensorDataset(X_train, y_train)
    data_loaders["train"] = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )

    # 验证数据加载器
    if val_data is not None:
        X_val, y_val = val_data
        val_dataset = TensorDataset(X_val, y_val)
        data_loaders["val"] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

    # 测试数据加载器
    if test_data is not None:
        X_test, y_test = test_data
        test_dataset = TensorDataset(X_test, y_test)
        data_loaders["test"] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

    return data_loaders