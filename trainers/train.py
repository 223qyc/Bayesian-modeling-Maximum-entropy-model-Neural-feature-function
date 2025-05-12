import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import time
import pyro

from models.baseline_models import NNClassifier, MCDropoutClassifier, DeepEnsembleClassifier
from models.vb_menn import VBMENN


def train_nn_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    criterion: Callable = nn.CrossEntropyLoss(),
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    n_epochs: int = 100,
    early_stopping: bool = True,
    patience: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True
) -> Dict[str, List[float]]:
    '''
    训练标准神经网络分类器
    :param model:模型
    :param train_loader:训练数据加载器
    :param val_loader:验证数据加载器
    :param criterion:损失函数
    :param optimizer:优化器（如果为None，则使用Adam）
    :param lr:学习率
    :param weight_decay:权重衰减系数
    :param n_epochs:训练轮数
    :param early_stopping:是否使用早停
    :param patience:早停耐心值
    :param device:使用的设备
    :param verbose:是否显示训练进度
    :return:训练历史记录
    '''

    # 将模型移至指定设备
    model = model.to(device)

    # 如果没有提供优化器，则创建一个Adam优化器
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    # 初始化训练历史记录
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # 初始化早停变量
    best_val_loss = float("inf")
    no_improve_epochs = 0

    # 训练循环
    for epoch in range(n_epochs):
        # 训练模式
        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        # 使用tqdm显示进度条
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}") if verbose else train_loader

        # 训练一个epoch
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(X)

            # 计算损失
            loss = criterion(outputs, y)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 累计损失
            train_loss += loss.item() * X.size(0)

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        # 计算平均训练损失和准确率
        train_loss = train_loss / total
        train_acc = correct / total

        # 添加到历史记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # 如果有验证集，则进行验证
        if val_loader is not None:
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

            # 添加到历史记录
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # 检查早停
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if verbose:
                print(f"Epoch {epoch + 1}/{n_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            if verbose:
                print(f"Epoch {epoch + 1}/{n_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    return history


def train_mc_dropout(
        model: MCDropoutClassifier,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Callable = nn.CrossEntropyLoss(),
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        n_epochs: int = 100,
        early_stopping: bool = True,
        patience: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True
) -> Dict[str, List[float]]:
    '''
    训练MC Dropout分类器,与标准神经网络分类器训练相同，只是确保模型包含dropout
    参数与train_nn_classifier相同
    :return:训练历史记录
    '''

    # 确保dropout_prob > 0
    if model.dropout_prob <= 0:
        raise ValueError("MC Dropout分类器的dropout_prob必须大于0")

    # 使用标准训练过程
    return train_nn_classifier(
        model, train_loader, val_loader, criterion, optimizer,
        lr, weight_decay, n_epochs, early_stopping, patience,
        device, verbose
    )


def train_deep_ensemble(
        model: DeepEnsembleClassifier,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Callable = nn.CrossEntropyLoss(),
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        n_epochs: int = 100,
        early_stopping: bool = True,
        patience: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True
) -> Dict[str, List[float]]:
    '''
    训练Deep Ensemble分类器
    参数与train_nn_classifier基本相同
    :return:训练历史记录
    '''

    # 将模型移至指定设备
    model = model.to(device)

    # 如果没有提供优化器，则创建一个Adam优化器
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    # 初始化训练历史记录
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # 初始化早停变量
    best_val_loss = float("inf")
    no_improve_epochs = 0

    # 训练循环
    for epoch in range(n_epochs):
        # 训练模式
        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        # 使用tqdm显示进度条
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}") if verbose else train_loader

        # 训练一个epoch
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 每个估计器的损失和梯度
            ensemble_loss = 0.0
            ensemble_correct = 0

            # 对每个估计器进行前向传播和梯度计算
            for est in model.estimators:
                # 前向传播
                outputs = est(X)

                # 计算损失
                loss = criterion(outputs, y)

                # 反向传播
                loss.backward()

                # 累计损失
                ensemble_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                ensemble_correct += (predicted == y).sum().item()

            # 取平均
            ensemble_loss /= len(model.estimators)
            ensemble_correct /= len(model.estimators)

            # 更新参数
            optimizer.step()

            # 累计损失和准确率
            train_loss += ensemble_loss * X.size(0)
            correct += ensemble_correct
            total += y.size(0)

        # 计算平均训练损失和准确率
        train_loss = train_loss / total
        train_acc = correct / total

        # 添加到历史记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # 如果有验证集，则进行验证
        if val_loader is not None:
            val_loss = 0.0
            correct = 0
            total = 0

            # 评估模式
            model.eval()

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)

                    # 获取集成的平均预测概率
                    mean_probs, _ = model.predict_proba(X)

                    # 计算损失 (使用平均预测的交叉熵)
                    loss = -torch.sum(
                        F.one_hot(y, num_classes=mean_probs.size(1)).float() * torch.log(mean_probs + 1e-10)) / X.size(
                        0)

                    # 获取预测类别
                    _, predicted = torch.max(mean_probs, 1)

                    # 累计
                    val_loss += loss.item() * X.size(0)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

            # 计算平均验证损失和准确率
            val_loss = val_loss / total
            val_acc = correct / total

            # 添加到历史记录
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # 检查早停
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if verbose:
                print(f"Epoch {epoch + 1}/{n_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            if verbose:
                print(f"Epoch {epoch + 1}/{n_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    return history


def train_vb_menn(
        model: VBMENN,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 0.01,
        n_epochs: int = 100,
        n_samples: int = 5,  # 每批次的MC样本数
        early_stopping: bool = True,
        patience: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True
) -> Dict[str, List[float]]:
    '''
    使用变分推断训练VB-MENN模型
    :param model:VB-MENN模型
    :param train_loader:训练数据加载器
    :param val_loader:验证数据加载器
    :param lr:学习率
    :param n_epochs:训练轮数
    :param n_samples:每批次的MC样本数
    :param early_stopping:是否使用早停
    :param patience:早停耐心值
    :param device: 使用的设备
    :param verbose:是否显示训练进度
    :return:训练历史记录
    '''

    # 将模型移至指定设备
    model = model.to(device)

    # 设置变分推断
    model.setup_vi(lr=lr)

    # 初始化训练历史记录
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # 初始化早停变量
    best_val_loss = float("inf")
    no_improve_epochs = 0

    # 训练循环
    for epoch in range(n_epochs):
        # 训练模式
        model.train()

        total_loss = 0.0
        train_preds = []
        train_targets = []

        # 使用tqdm显示进度条
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}") if verbose else train_loader

        # 训练一个epoch
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)

            # SVI步骤（使用多个MC样本）
            batch_loss = 0.0
            for _ in range(n_samples):
                # 计算损失并更新参数
                loss = model.svi.step(X, y)
                batch_loss += loss

            # 取平均损失
            batch_loss /= n_samples
            total_loss += batch_loss

            # 进行一次前向传播以获取预测
            with torch.no_grad():
                logits = model(X)
                _, preds = torch.max(logits, 1)

            train_preds.append(preds.cpu())
            train_targets.append(y.cpu())

        # 计算平均训练损失
        avg_loss = total_loss / len(train_loader)

        # 计算训练准确率
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_acc = (train_preds == train_targets).float().mean().item()

        # 添加到历史记录
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)

        # 如果有验证集，则进行验证
        if val_loader is not None:
            val_loss, val_preds, val_targets = 0.0, [], []

            # 评估模式
            model.eval()

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)

                    # 计算验证损失
                    loss = model.svi.evaluate_loss(X, y)
                    val_loss += loss

                    # 进行前向传播以获取预测
                    logits = model(X)
                    _, preds = torch.max(logits, 1)

                    val_preds.append(preds.cpu())
                    val_targets.append(y.cpu())

            # 计算平均验证损失
            val_loss = val_loss / len(val_loader)

            # 计算验证准确率
            val_preds = torch.cat(val_preds)
            val_targets = torch.cat(val_targets)
            val_acc = (val_preds == val_targets).float().mean().item()

            # 添加到历史记录
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # 检查早停
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if verbose:
                print(f"Epoch {epoch + 1}/{n_epochs} - "
                      f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            if verbose:
                print(f"Epoch {epoch + 1}/{n_epochs} - "
                      f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")

    return history


def _evaluate(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Callable,
        device: str
) -> Tuple[float, float]:
    '''
    评估模型在给定数据集上的性能
    :param model:模型
    :param data_loader:数据加载器
    :param criterion:损失函数
    :param device:使用的设备
    :return:(平均损失, 准确率)
    '''

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            # 前向传播
            outputs = model(X)

            # 计算损失
            loss = criterion(outputs, y)

            # 累计损失
            val_loss += loss.item() * X.size(0)

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    # 计算平均验证损失和准确率
    val_loss = val_loss / total
    val_acc = correct / total

    return val_loss, val_acc

