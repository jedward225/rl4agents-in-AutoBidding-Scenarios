"""
BC数据收集工具（9维观测）
使用 BCTrainer 内置的数据准备功能
"""
import sys
import os

# 确保可以导入auction_sim模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from auction_sim.bc_trainer import BCTrainer

def main():
    """主函数：收集9维BC数据"""
    print("=" * 80)
    print("BC数据收集工具（9维观测）")
    print("=" * 80)
    print("\n提示: 使用 BCTrainer 内置的数据准备功能")
    print("该功能也被 bc_ippo_trainer.py 自动调用，确保数据一致性\n")
    
    # 创建BC训练器（9维观测）
    bc_trainer = BCTrainer(obs_dim=9, lr=1e-3)
    
    # 准备BC数据（自动检查维度并在需要时收集）
    dataset_path = "auction_sim/bc_dataset.pkl"
    success = bc_trainer.prepare_bc_data(
        dataset_path=dataset_path,
        n_episodes=50,
        force_recollect=False  # 如果已有9维数据则使用，否则重新收集
    )
    
    print("\n" + "=" * 80)
    if success:
        print("✅ BC数据准备成功！")
        print("\n现在可以开始训练了：")
        print("  cd jjjj && python bc_ippo_trainer.py")
        print("\n或者强制重新收集数据：")
        print("  修改此脚本，设置 force_recollect=True")
    else:
        print("❌ BC数据准备失败")
    print("=" * 80)

if __name__ == "__main__":
    main()
