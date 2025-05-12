# VB-MENN实验项目文档
## 说在前面
这个仓库存放的是我一个idea的实现过程，基本思想是使用变分贝叶斯建模最大熵模型的神经特征函数，验证其在某些情况下的优势。

目前实测后发现并不能做到超过基线效果太多，可能存在错误或仍待后续继续研究开发，但由于最近较忙，所以将项目封存在此。

如果有更好的想法也欢迎提出。
## 项目概述

这个项目可以看作是我本人的一次独立科研经历，虽然碰壁，但是让我学到了很多。

本项目实现了变分贝叶斯最大熵神经网络（VB-MENN）及多种基线模型，用于比较不同神经网络方法在小样本学习、噪声标签处理和不确定性量化方面的性能。该项目主要关注自然语言处理任务，使用BERT进行特征提取，并通过多种实验展示VB-MENN在挑战性场景下的优势。

该项目解决的核心问题：
1. **小样本学习**：当标注数据有限时，如何训练出性能良好的模型
2. **噪声标签**：当训练数据存在标签错误时，如何保持模型的鲁棒性
3. **不确定性量化**：如何准确估计模型预测的不确定性，区分认知不确定性（epistemic uncertainty）和偶然不确定性（aleatoric uncertainty）

## 项目结构

```
VB-MENN-Experiment/
├── data/                      # 数据处理相关模块
│   ├── dataset_files/         # 数据集文件存储目录
│   ├── dataset_utils.py       # 数据集加载、小样本创建、标签噪声添加
│   ├── text_encoder.py        # 文本编码和数据加载器准备
│   └── __init__.py            # 模块导出接口
├── experiments/               # 实验定义模块
│   ├── small_sample.py        # 小样本学习实验
│   ├── noisy_labels.py        # 噪声标签实验
│   ├── uncertainty.py         # 不确定性评估实验
│   └── __init__.py            # 模块导出接口
├── models/                    # 模型定义模块
│   ├── baseline_models.py     # 基线模型实现（标准NN、MCDropout、深度集成）
│   ├── vb_menn.py             # 变分贝叶斯最大熵神经网络实现
│   ├── layers.py              # 自定义网络层（如贝叶斯线性层）
│   └── __init__.py            # 模块导出接口
├── trainers/                  # 训练和评估模块
│   ├── train.py               # 各类模型的训练函数
│   ├── evaluation.py          # 模型评估和不确定性量化函数
│   └── __init__.py            # 模块导出接口
├── utils/                     # 辅助工具
│   ├── metrics.py             # 评估指标计算
│   ├── visualization.py       # 实验结果可视化
│   └── __init__.py            # 模块导出接口
├── main.py                    # 主程序入口
└── README.md                  # 项目文档（本文件）
```

## 模型详解

本项目实现并比较了以下四种模型：

### 1. 标准神经网络分类器 (NNClassifier)

基础的前馈神经网络，作为比较的基准模型。具有以下特点：
- 使用ReLU激活函数
- 具有可配置的隐藏层维度
- 使用交叉熵损失函数训练

### 2. Monte Carlo Dropout分类器 (MCDropoutClassifier)

利用Dropout作为贝叶斯近似的分类器：
- 在训练和预测时都保持Dropout开启
- 通过多次采样估计预测的不确定性
- 可以同时捕获认知不确定性和偶然不确定性
- 实现简单但效果较好

### 3. 深度集成分类器 (DeepEnsembleClassifier)

训练多个具有不同随机初始化的神经网络模型：
- 默认使用5个估计器组成集成
- 每个估计器是独立训练的标准神经网络
- 预测时合并多个模型的输出
- 可以评估预测的方差作为不确定性度量
- 对抗过拟合，提高泛化能力

### 4. 变分贝叶斯最大熵神经网络 (VBMENN)

项目的核心模型，结合贝叶斯方法和最大熵原则：
- 使用变分推断学习网络参数的后验分布
- 使用Pyro框架实现变分推断
- 包含贝叶斯线性层，可捕获参数的不确定性
- 能够有效区分认知不确定性和偶然不确定性
- 在小样本和噪声标签场景下展现出更好的鲁棒性
- 通过KL散度正则化防止过拟合

## 实验类型详解

项目支持三种主要实验类型，每种实验都有特定的目标和评估方法：

### 1. 小样本学习实验 (Small Sample Experiment)

目标：评估模型在训练数据有限的情况下的学习能力。

特点：
- 从完整数据集中随机采样少量样本（每类5-100个样本）
- 使用相同的测试集评估所有模型
- 重复多次运行以计算性能的平均值和标准差
- 测量准确率、F1分数、校准误差等指标
- 绘制样本大小与性能指标的关系图

### 2. 噪声标签实验 (Noisy Labels Experiment)

目标：评估模型对训练数据中标签噪声的鲁棒性。

特点：
- 在训练数据中人为添加不同比例的标签噪声（0%-30%）
- 保持测试集标签不变
- 测量模型性能随噪声比例的变化
- 评估不同模型在噪声存在时的校准性能
- 重复多次运行以获得稳定结果

### 3. 不确定性评估实验 (Uncertainty Experiment)

目标：评估模型量化预测不确定性的能力，区分分布内(ID)和分布外(OOD)样本。

特点：
- 在一个数据集上训练模型（如IMDB）
- 在同分布测试集和不同分布数据集（如AG News）上评估
- 计算预测熵、认知不确定性和偶然不确定性
- 评估OOD检测能力（如AUROC、AUPR等）
- 绘制不确定性直方图和OOD检测曲线

## 环境配置

### 依赖项

本项目需要以下主要依赖：

```
torch>=1.8.0
pyro-ppl>=1.8.0
numpy>=1.19.0
matplotlib>=3.3.0
pandas>=1.1.0
transformers>=4.5.0
datasets>=1.5.0
scikit-learn>=0.24.0
tqdm>=4.45.0
```

### 安装方法

1. 克隆项目仓库：
   ```bash
   git clone https://github.com/yourusername/VB-MENN-Experiment.git
   cd VB-MENN-Experiment
   ```

2. 创建虚拟环境（可选但推荐）：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate  # Windows
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用指南

### 参数说明

`main.py`支持以下命令行参数：

#### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--experiment` | str | 必需 | 实验类型：`small_sample`, `noisy_labels`, `uncertainty` |
| `--dataset` | str | "imdb" | 数据集名称 |
| `--model_types` | list[str] | ["standard", "mc_dropout", "deep_ensemble", "vb_menn"] | 要评估的模型类型 |
| `--batch_size` | int | 16 | 批次大小 |
| `--n_epochs` | int | 50 | 训练轮数 |
| `--seed` | int | 42 | 随机种子 |
| `--device` | str | "cuda"或"cpu" | 使用的设备 |
| `--results_dir` | str | "results" | 结果保存目录的基础路径 |

#### 小样本实验参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--sample_sizes` | list[int] | [5, 10, 20, 50, 100] | 每个类别的样本数列表 |
| `--n_runs` | int | 3 | 每个配置的运行次数 |

#### 噪声标签实验参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--noise_ratios` | list[float] | [0.0, 0.1, 0.2, 0.3] | 噪声比例列表 |
| `--n_runs` | int | 3 | 每个配置的运行次数 |

#### 不确定性评估实验参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ood_dataset` | str | "ag_news" | 分布外数据集名称 |

### 实验运行指令

以下是各种实验配置的详细运行指令：

#### 小样本实验指令

1. 基本小样本实验（使用默认参数）：
   ```bash
   python main.py --experiment small_sample
   ```

2. 使用特定数据集和模型：
   ```bash
   python main.py --experiment small_sample --dataset sst2 --model_types standard vb_menn
   ```

3. 自定义样本大小：
   ```bash
   python main.py --experiment small_sample --sample_sizes 5 10 25 50 100 200
   ```

4. 增加运行次数以获得更稳定的结果：
   ```bash
   python main.py --experiment small_sample --n_runs 5
   ```

5. 调整训练参数：
   ```bash
   python main.py --experiment small_sample --batch_size 32 --n_epochs 100
   ```

6. 仅评估VBMENN与标准模型的对比：
   ```bash
   python main.py --experiment small_sample --model_types standard vb_menn --sample_sizes 10 50 100
   ```

7. 完整配置的小样本实验：
   ```bash
   python main.py --experiment small_sample --dataset imdb --model_types standard mc_dropout deep_ensemble vb_menn --sample_sizes 5 10 20 50 100 --n_runs 5 --batch_size 16 --n_epochs 50 --seed 42 --device cuda --results_dir results/small_sample_full
   ```

#### 噪声标签实验指令

1. 基本噪声标签实验（使用默认参数）：
   ```bash
   python main.py --experiment noisy_labels
   ```

2. 使用特定数据集和模型：
   ```bash
   python main.py --experiment noisy_labels --dataset sst2 --model_types standard vb_menn
   ```

3. 自定义噪声比例：
   ```bash
   python main.py --experiment noisy_labels --noise_ratios 0.0 0.1 0.2 0.3 0.4 0.5
   ```

4. 增加运行次数以获得更稳定的结果：
   ```bash
   python main.py --experiment noisy_labels --n_runs 5
   ```

5. 调整训练参数：
   ```bash
   python main.py --experiment noisy_labels --batch_size 32 --n_epochs 100
   ```

6. 仅对比MC Dropout和VBMENN：
   ```bash
   python main.py --experiment noisy_labels --model_types mc_dropout vb_menn --noise_ratios 0.0 0.2 0.4
   ```

7. 完整配置的噪声标签实验：
   ```bash
   python main.py --experiment noisy_labels --dataset imdb --model_types standard mc_dropout deep_ensemble vb_menn --noise_ratios 0.0 0.1 0.2 0.3 0.4 --n_runs 5 --batch_size 16 --n_epochs 50 --seed 42 --device cuda --results_dir results/noisy_labels_full
   ```

#### 不确定性评估实验指令

1. 基本不确定性评估实验（使用默认参数）：
   ```bash
   python main.py --experiment uncertainty
   ```

2. 使用特定数据集和OOD数据集：
   ```bash
   python main.py --experiment uncertainty --dataset sst2 --ood_dataset yahoo_answers
   ```

3. 仅评估特定模型：
   ```bash
   python main.py --experiment uncertainty --model_types mc_dropout vb_menn
   ```

4. 调整训练参数：
   ```bash
   python main.py --experiment uncertainty --batch_size 32 --n_epochs 100
   ```

5. 完整配置的不确定性评估实验：
   ```bash
   python main.py --experiment uncertainty --dataset imdb --ood_dataset ag_news --model_types standard mc_dropout deep_ensemble vb_menn --batch_size 16 --n_epochs 50 --seed 42 --device cuda --results_dir results/uncertainty_full
   ```

### 多数据集对比实验

以下是在多个数据集上运行相同实验的示例：

1. 在多个数据集上运行小样本实验：
   ```bash
   # IMDB数据集
   python main.py --experiment small_sample --dataset imdb --results_dir results/small_sample/imdb
   
   # SST-2数据集
   python main.py --experiment small_sample --dataset sst2 --results_dir results/small_sample/sst2
   
   # AG News数据集
   python main.py --experiment small_sample --dataset ag_news --results_dir results/small_sample/ag_news
   ```

2. 在多个数据集上运行噪声标签实验：
   ```bash
   # IMDB数据集
   python main.py --experiment noisy_labels --dataset imdb --results_dir results/noisy_labels/imdb
   
   # SST-2数据集
   python main.py --experiment noisy_labels --dataset sst2 --results_dir results/noisy_labels/sst2
   ```

### 高级使用场景

1. 使用不同的随机种子运行多次实验并取平均值：
   ```bash
   # 使用种子1
   python main.py --experiment small_sample --seed 1 --results_dir results/seed1
   
   # 使用种子2
   python main.py --experiment small_sample --seed 2 --results_dir results/seed2
   
   # 使用种子3
   python main.py --experiment small_sample --seed 3 --results_dir results/seed3
   ```

2. 仅使用CPU运行（用于没有GPU的环境）：
   ```bash
   python main.py --experiment small_sample --device cpu
   ```

3. 使用较小的批次大小（适用于内存有限的情况）：
   ```bash
   python main.py --experiment small_sample --batch_size 8
   ```

## 结果说明

实验结果将保存在指定的结果目录中（默认为`results/实验名称/`）。对于每种实验，将生成以下文件：

1. **JSON结果文件**：包含详细的实验配置和性能指标
   - 文件名格式：`experiment_results.json`

2. **性能指标图表**：
   - 小样本实验：样本大小vs准确率/F1分数
   - 噪声标签实验：噪声比例vs准确率/校准误差
   - 不确定性实验：不确定性直方图、ROC曲线、PR曲线

3. **学习曲线**：展示训练过程中的损失和准确率变化

### 结果解读指南

1. **小样本实验**：
   - 观察曲线斜率：较陡的斜率表示模型能更有效地利用增加的样本
   - 比较低样本量下的性能：更高的准确率表示模型在小样本场景下更有效
   - 观察误差条：较小的误差条表示模型在多次运行中更稳定

2. **噪声标签实验**：
   - 曲线的平缓程度表示对噪声的鲁棒性
   - 在高噪声比例下保持较高准确率的模型更鲁棒
   - 校准误差（ECE）较低的模型在噪声环境中有更可靠的不确定性估计

3. **不确定性实验**：
   - AUROC和AUPR值越高，模型越能区分分布内和分布外样本
   - 不确定性直方图中的清晰分离表示良好的OOD检测能力
   - 认知不确定性（epistemic uncertainty）应该在OOD样本上较高

## 示例应用场景

本项目的技术和模型适用于多种实际应用场景：

1. **医疗诊断**：
   - 小样本学习适用于罕见疾病的诊断
   - 不确定性量化可以指示何时应由人类专家进行进一步检查
   - 贝叶斯方法提供可解释的置信度估计

2. **金融风险评估**：
   - 对异常交易的高不确定性可以触发进一步审查
   - 模型的校准性能对风险评估至关重要
   - 对噪声的鲁棒性有助于处理金融数据中的异常值

3. **智能客服**：
   - 不确定性检测可以决定何时将查询转发给人工客服
   - 小样本学习可用于快速适应新产品或服务类别
   - 噪声鲁棒性有助于处理用户输入中的错误或模糊查询

4. **科学研究**：
   - 在数据收集成本高的领域（如材料科学）中应用小样本学习
   - 不确定性量化可以指导实验设计和资源分配
   - 认知不确定性可以识别需要更多研究的领域

## 常见问题与解决方案

1. **内存不足错误**
   - 减小批次大小：`--batch_size 8`或更小
   - 使用较小的模型配置
   - 处理较小的数据集

2. **训练过程中的过拟合**
   - 增加正则化强度（调整贝叶斯先验）
   - 减少训练轮数：`--n_epochs 30`
   - 使用早停策略（代码中已实现）

3. **模型性能不稳定**
   - 增加运行次数：`--n_runs 5`或更多
   - 尝试不同的随机种子
   - 数据预处理和特征提取可能需要优化

4. **OOD检测性能不佳**
   - 尝试不同的OOD数据集
   - 调整不确定性阈值
   - 考虑结合多种不确定性度量

## 扩展与定制

1. **添加新模型**：
   - 在`models/`目录下创建新的模型文件
   - 在`models/__init__.py`中导出模型
   - 在`trainers/train.py`中添加训练函数
   - 在`main.py`中添加模型选项

2. **支持新数据集**：
   - 在`data/dataset_utils.py`中添加数据集加载和处理逻辑
   - 确保与现有的预处理管道兼容

3. **自定义实验**：
   - 在`experiments/`目录下创建新的实验文件
   - 在`experiments/__init__.py`中导出实验函数
   - 在`main.py`中添加新的实验选项

## 最后

如果有任何问题或建议，请提交问题或贡献代码，帮助改进这个项目。 