import  torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import math
from typing import Tuple

class BayesianLinear(nn.Module):
    '''
    这个类定义贝叶斯线性层：使用权重和偏置的分布来表示参数的不确定性
    '''

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            prior_std: float = 1.0,
            name: str = "",
            device: str = None
    ):
        '''
        初始化相关
        :param in_features: 输入特征维度
        :param out_features:输出特征维度
        :param bias:是否使用偏置项
        :param prior_std:先验分布的标准差
        :param name:层的名称
        :param device:设备选择
        '''
        super(BayesianLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.prior_std = prior_std
        self.name = name if name else f"bayesian_linear_{id(self)}"
        self.device = device

        # 注册参数
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        # 初始化参数
        self.reset_parameters()


    def reset_parameters(self):
        '''
        初始化参数
        '''
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        # 将 rho 初始化为小的负值，以便初始的标准差接近零
        nn.init.constant_(self.weight_rho, -3.0)
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        前向传播
        :param x: 输入张量
        :return: 输出张量
        '''
        # 采样权重
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        # 计算输出
        out = F.linear(x, weight)
        # 如果使用偏置，采样偏置
        if self.bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
            out = out + bias

        return out

    def model(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        '''
        贝叶斯线性层的模型-用于Pyro的变分推断
        :param x: 输入张量
        :param y:目标张量
        :return:输出张量
        '''
        # 权重先验
        weight_prior = dist.Normal(0, self.prior_std)
        weight = pyro.sample(
            f"{self.name}.weight",
            dist.Normal(
                0, self.prior_std
            ).expand([self.out_features, self.in_features]).to_event(2)
        )
        # 偏置先验
        if self.bias:
            bias_prior = dist.Normal(0, self.prior_std)
            bias = pyro.sample(
                f"{self.name}.bias",
                dist.Normal(
                    0, self.prior_std
                ).expand([self.out_features]).to_event(1)
            )
        else:
            bias = None

        # 计算线性输出
        output = F.linear(x, weight, bias)
        # 如果提供了y，则计算似然
        if y is not None:
            pyro.sample(
                f"{self.name}.output",
                dist.Normal(output, 0.1).to_event(1),
                obs=y
            )

        return output

    def guide(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        '''
        贝叶斯线性层的向导-用于Pyro的变分推断
        :param x:输入张量
        :param y:目标张量
        :return:输出张量
        '''
        # 权重变分后验
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight = pyro.sample(
            f"{self.name}.weight",
            dist.Normal(
                self.weight_mu, weight_sigma
            ).to_event(2)
        )

        # 偏置变分后验
        if self.bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = pyro.sample(
                f"{self.name}.bias",
                dist.Normal(
                    self.bias_mu, bias_sigma
                ).to_event(1)
            )
        else:
            bias = None

        # 计算线性输出
        return F.linear(x, weight, bias)

    def sample_reparameterize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        使用重参数化技巧采样权重
        :return:(权重, 偏置)
        '''
        # 采样权重
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)

        # 采样偏置
        if self.bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        else:
            bias = None

        return weight, bias

    def kl_divergence(self) -> torch.Tensor:
        '''
        用于计算KL散度
        :return:KL散度值
        '''
        # 计算权重的KL散度
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        kl_weight = self._kl_normal(
            self.weight_mu, weight_sigma,
            torch.zeros_like(self.weight_mu), self.prior_std * torch.ones_like(self.weight_mu)
        )

        # 计算偏置的KL散度
        if self.bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            kl_bias = self._kl_normal(
                self.bias_mu, bias_sigma,
                torch.zeros_like(self.bias_mu), self.prior_std * torch.ones_like(self.bias_mu)
            )
        else:
            kl_bias = 0

        return kl_weight + kl_bias

    def _kl_normal(
            self,
            mu_q: torch.Tensor,
            sigma_q: torch.Tensor,
            mu_p: torch.Tensor,
            sigma_p: torch.Tensor
    ) -> torch.Tensor:
        '''
        计算两个正态分布之间的KL散度
        :param mu_q:后验均值
        :param sigma_q:后验标准差
        :param mu_p:先验均值
        :param sigma_p:先验标准差
        :return:KL散度值
        '''
        var_p = sigma_p ** 2
        var_q = sigma_q ** 2

        kl = torch.log(sigma_p / sigma_q) + (var_q + (mu_q - mu_p) ** 2) / (2 * var_p) - 0.5
        return torch.sum(kl)








