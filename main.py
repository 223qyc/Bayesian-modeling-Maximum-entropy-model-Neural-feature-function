import argparse
import os
import torch

from experiments import run_small_sample_experiment, run_noisy_labels_experiment, run_uncertainty_experiment

def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="VB-MENN实验")
    
    # 添加实验类型参数
    parser.add_argument(
        "--experiment", type=str, required=True,
        choices=["small_sample", "noisy_labels", "uncertainty"],
        help="要运行的实验类型"
    )
    
    # 通用参数
    parser.add_argument("--dataset", type=str, default="imdb", help="数据集名称")
    parser.add_argument(
        "--model_types", type=str, nargs="+",
        default=["standard", "mc_dropout", "deep_ensemble", "vb_menn"],
        help="要评估的模型类型"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--n_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="使用的设备")
    parser.add_argument("--results_dir", type=str, default="results", help="结果保存目录的基础路径")
    
    # 小样本实验参数
    parser.add_argument(
        "--sample_sizes", type=int, nargs="+", default=[5, 10, 20, 50, 100],
        help="每个类别的样本数列表（用于小样本实验）"
    )
    parser.add_argument(
        "--n_runs", type=int, default=3,
        help="每个配置的运行次数（用于小样本和噪声标签实验）"
    )
    
    # 噪声标签实验参数
    parser.add_argument(
        "--noise_ratios", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3],
        help="噪声比例列表（用于噪声标签实验）"
    )
    
    # 不确定性评估实验参数
    parser.add_argument(
        "--ood_dataset", type=str, default="ag_news",
        help="分布外数据集名称（用于不确定性评估实验）"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建结果目录
    results_dir = os.path.join(args.results_dir, args.experiment)
    os.makedirs(results_dir, exist_ok=True)
    
    # 根据实验类型调用相应的函数
    if args.experiment == "small_sample":
        print("=== 运行小样本实验 ===")
        run_small_sample_experiment(
            dataset_name=args.dataset,
            sample_sizes=args.sample_sizes,
            model_types=args.model_types,
            n_runs=args.n_runs,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            device=args.device,
            results_dir=results_dir,
            seed=args.seed
        )
    
    elif args.experiment == "noisy_labels":
        print("=== 运行噪声标签实验 ===")
        run_noisy_labels_experiment(
            dataset_name=args.dataset,
            noise_ratios=args.noise_ratios,
            model_types=args.model_types,
            n_runs=args.n_runs,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            device=args.device,
            results_dir=results_dir,
            seed=args.seed
        )
    
    elif args.experiment == "uncertainty":
        print("=== 运行不确定性评估实验 ===")
        run_uncertainty_experiment(
            dataset_name=args.dataset,
            ood_dataset_name=args.ood_dataset,
            model_types=args.model_types,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            device=args.device,
            results_dir=results_dir,
            seed=args.seed
        )

if __name__ == "__main__":
    main()
