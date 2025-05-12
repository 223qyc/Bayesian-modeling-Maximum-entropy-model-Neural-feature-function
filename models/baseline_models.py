import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Dict

class NNClassifier(nn.Module):
    '''
    标准的神经网络分类器(也就是最大熵模型特征函数的神经化构造-无贝叶斯)
    '''
    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            output_dim: int,
            dropout_prob: float = 0.0
    ):
        '''
        :param input_dim:输入维度
        :param hidden_dims:表示各隐藏层维度的列表
        :param output_dim:输出维度
        :param dropout_prob:Dropou率，这里的默认是没有使用
        '''
        super(NNClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        # 构建网络层
        layers = []

        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))

        # 隐藏层间连接
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))

        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        :param x:输入张量
        :return:输出logits
        '''
        return self.model(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        '''
        :param x:输入张量
        :return:概率分布
        '''
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs


class MCDropoutClassifier(NNClassifier):
    '''
    Monte Carlo Dropout分类器,在预测时保持dropout开启，通过多次采样估计不确定性
    '''
    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            output_dim: int,
            dropout_prob: float = 0.1
    ):
        '''
        :param input_dim: 输入维度
        :param hidden_dims: 隐藏层的维度列表
        :param output_dim: 输出维度
        :param dropout_prob: Dropout概率
        '''
        super(MCDropoutClassifier, self).__init__(
            input_dim, hidden_dims, output_dim, dropout_prob
        )

        # dropout_prob > 0
        if dropout_prob <= 0:
            raise ValueError("MC Dropout分类器的dropout_prob必须大于0")

    def predict_proba_mc(
            self,
            x: torch.Tensor,
            n_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        采用蒙特卡洛采样进行预测
        :param x:输入张量
        :param n_samples:采样次数
        :return: (平均概率, 概率标准差)
        '''
        self.train()  # 训练模式要启用dropout
        all_probs = []
        for _ in range(n_samples):
            with torch.no_grad():
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)

        # 堆叠所有预测
        all_probs = torch.stack(all_probs, dim=0)

        # 计算平均概率和标准差
        mean_probs = torch.mean(all_probs, dim=0)
        std_probs = torch.std(all_probs, dim=0)

        return mean_probs, std_probs

class DeepEnsembleClassifier:
    '''
    深度集成分类器,训练多个具有不同初始化的分类器并集成预测
    '''

    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            output_dim: int,
            n_estimators: int = 5,
            dropout_prob: float = 0.0
    ):
        '''
        初始化深度集成分类器
        :param input_dim:输入维度
        :param hidden_dims: 隐藏层维度列表
        :param output_dim: 输出维度（类别数）
        :param n_estimators:估计器数量
        :param dropout_prob:每个估计器的Dropout概率
        '''
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.n_estimators = n_estimators
        self.dropout_prob = dropout_prob

        # 创建多个分类器
        self.estimators = [
            NNClassifier(input_dim, hidden_dims, output_dim, dropout_prob)
            for _ in range(n_estimators)
        ]

    def parameters(self):
        """
        返回所有估计器的参数
        """
        for est in self.estimators:
            for param in est.parameters():
                yield param

    def train(self):
        """
        设置所有估计器为训练模式
        """
        for est in self.estimators:
            est.train()

    def eval(self):
        """
        设置所有估计器为评估模式
        """
        for est in self.estimators:
            est.eval()

    def to(self, device):
        """
        将所有估计器移到指定设备
        """
        for est in self.estimators:
            est.to(device)
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        '''
        前向传播，返回所有估计器的输出
        :param x:输入张量
        :return:所有估计器的logits列表
        '''
        return [est.forward(x) for est in self.estimators]

    def predict_proba(
            self,
            x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        进行概率预测
        :param x:输入张量
        :return:(平均概率, 概率标准差)
        '''
        all_probs = []

        for est in self.estimators:
            est.eval()  # 设置为评估模式
            with torch.no_grad():
                logits = est.forward(x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)

        # 堆叠所有预测
        all_probs = torch.stack(all_probs, dim=0)

        # 计算平均概率和标准差
        mean_probs = torch.mean(all_probs, dim=0)
        std_probs = torch.std(all_probs, dim=0)

        return mean_probs, std_probs

