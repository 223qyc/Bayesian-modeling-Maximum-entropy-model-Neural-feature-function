import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.optim
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from typing import List, Tuple, Dict, Optional, Union
import numpy as np

from .layers import BayesianLinear


class VBMENN(nn.Module):
    '''
    变分贝叶斯最大熵神经网络 (VB-MENN)
    我构建的该网络的特点是使用贝叶斯建模最大熵模型的神经特征函数，并展示其在小样本和噪声中的优势
    '''
    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            output_dim: int,
            prior_std: float = 1.0,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        '''
        初始化

        :param input_dim:输入维度
        :param hidden_dims:隐藏层维度列表
        :param output_dim:输出维度（类别数）
        :param prior_std:先验分布的标准差
        :param device:使用的设备
        '''
        super(VBMENN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.prior_std = prior_std
        self.device = device

        # 创建贝叶斯网络层
        self.layers = nn.ModuleList()

        # 输入层到第一个隐藏层
        self.layers.append(
            BayesianLinear(
                input_dim, hidden_dims[0],
                prior_std=prior_std,
                name="layer_0"
            )
        )

        # 隐藏层间连接
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                BayesianLinear(
                    hidden_dims[i], hidden_dims[i + 1],
                    prior_std=prior_std,
                    name=f"layer_{i + 1}"
                )
            )

        # 最后一个隐藏层到输出层
        self.layers.append(
            BayesianLinear(
                hidden_dims[-1], output_dim,
                prior_std=prior_std,
                name=f"layer_{len(hidden_dims)}"
            )
        )

        # 初始化SVI
        self.svi = None
        self.elbo = None
        self.optim = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        定义向前传播
        :param x: 输入张量
        :return:logits: 输出logits
        '''
        activation = x
        for i, layer in enumerate(self.layers[:-1]):
            activation = F.relu(layer(activation))

        logits = self.layers[-1](activation)
        return logits


    def model(
            self,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        '''
        变分推断的模型
        :param x:输入特征
        :param y: 目标值 (类别标签)
        :return:模型输出 (logits)
        '''
        priors = {}

        # 设置网络参数的先验
        for i, layer in enumerate(self.layers):
            priors[f"layer_{i}.weight"] = dist.Normal(
                torch.zeros_like(layer.weight_mu),
                self.prior_std * torch.ones_like(layer.weight_mu)
            ).to_event(2)

            priors[f"layer_{i}.bias"] = dist.Normal(
                torch.zeros_like(layer.bias_mu),
                self.prior_std * torch.ones_like(layer.bias_mu)
            ).to_event(1)

        # 采样参数
        lifted_module = pyro.random_module("module", self, priors)
        lifted_reg_model = lifted_module()

        # 计算前向传播
        logits = lifted_reg_model(x)

        # 设置似然
        with pyro.plate("data", x.shape[0]):
            pyro.sample(
                "obs",
                dist.Categorical(logits=logits),
                obs=y
            )
        return logits


    def guide(
            self,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        '''
        变分推断的向导
        :param x:输入特征
        :param y:目标值 (类别标签)
        :return:前向传播输出
        '''
        dists = {}

        # 设置网络参数的变分后验
        for i, layer in enumerate(self.layers):
            weight_sigma = torch.log1p(torch.exp(layer.weight_rho))
            dists[f"layer_{i}.weight"] = dist.Normal(
                layer.weight_mu, weight_sigma
            ).to_event(2)

            bias_sigma = torch.log1p(torch.exp(layer.bias_rho))
            dists[f"layer_{i}.bias"] = dist.Normal(
                layer.bias_mu, bias_sigma
            ).to_event(1)

        # 采样参数
        lifted_module = pyro.random_module("module", self, dists)
        lifted_reg_model = lifted_module()

        # 计算前向传播
        return lifted_reg_model(x)


    def setup_vi(
        self,
        lr: float = 0.01
    ):
        '''
        设置变分推断
        :param lr:学习率
        '''
        # 清除参数存储
        pyro.clear_param_store()

        # 设置优化器、损失函数和SVI
        self.optim = pyro.optim.Adam({"lr": lr})
        self.elbo = Trace_ELBO()
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)


    def kl_divergence(self) -> torch.Tensor:
        '''
        计算整个网络的KL散度
        :return: KL散度之和
        '''
        kl_sum = 0
        for layer in self.layers:
            kl_sum += layer.kl_divergence()
        return kl_sum


    def predict_proba(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        '''
         预测类别概率
        :param x: 输入特征
        :return:类别概率
        '''
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs


    def predict_proba_vi(
            self,
            x: torch.Tensor,
            n_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        使用变分推断采样进行预测
        :param x:输入特征
        :param n_samples:采样次数
        :return:(平均概率, 概率标准差, logit方差)
        '''
        probs_samples = []
        logits_samples = []

        # 多次采样
        for _ in range(n_samples):
            # 向导采样
            guide_trace = pyro.poutine.trace(self.guide).get_trace(x)
            # 从向导轨迹中提取采样的参数
            sampled_model = pyro.poutine.replay(self.model, guide_trace)

            # 运行采样的模型（不提供y）
            with torch.no_grad():
                logits = sampled_model(x)
                probs = F.softmax(logits, dim=1)

            probs_samples.append(probs)
            logits_samples.append(logits)

        # 堆叠样本
        probs_samples = torch.stack(probs_samples)
        logits_samples = torch.stack(logits_samples)

        # 计算平均值和标准差
        mean_probs = torch.mean(probs_samples, dim=0)
        std_probs = torch.std(probs_samples, dim=0)
        logits_vars = torch.var(logits_samples, dim=0)

        return mean_probs, std_probs, logits_vars


    def get_predictive_entropy(
            self,
            probs: torch.Tensor
    ) -> torch.Tensor:
        '''
        计算预测熵
        :param probs:概率分布
        :return:预测熵
        '''
        # 避免log(0)
        eps = 1e-10
        return -torch.sum(probs * torch.log(probs + eps), dim=1)


    def get_aleatoric_uncertainty(
            self,
            mean_probs: torch.Tensor
    ) -> torch.Tensor:
        '''
        计算偶然不确定性（随机不确定性）
        :param mean_probs:平均概率分布
        :return:偶然不确定性
        '''
        return self.get_predictive_entropy(mean_probs)


    def get_epistemic_uncertainty(
            self,
            mean_probs: torch.Tensor,
            probs_samples: torch.Tensor
    ) -> torch.Tensor:
        '''
        计算认知不确定性（模型不确定性)
        :param mean_probs:平均概率分布
        :param probs_samples:概率分布样本 [n_samples, batch_size, n_classes]
        :return:认知不确定性
        '''

        aleatoric = self.get_aleatoric_uncertainty(mean_probs)

        # 计算平均熵
        sample_entropies = []
        for i in range(probs_samples.shape[0]):
            entropy_i = self.get_predictive_entropy(probs_samples[i])
            sample_entropies.append(entropy_i)

        mean_entropy = torch.mean(torch.stack(sample_entropies), dim=0)

        # 认知不确定性 = 总不确定性 - 偶然不确定性
        return aleatoric - mean_entropy






