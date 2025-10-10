#!/usr/bin/env python3
"""
重新训练BC模型（使用新的[0,1.5]动作空间）
修复BC和IPPO的兼容性问题
"""
import sys
sys.path.insert(0, '.')

import os
from auction_sim.bc_trainer import BCTrainer

def main():
    print("="*70)
    print("重新训练BC模型")
    print("="*70)
    
    # 检查BC数据是否存在
    bc_data_path = "auction_sim/bc_dataset.pkl"
    bc_model_path = "auction_sim/models/bc_pretrained.pth"
    
    if not os.path.exists(bc_data_path):
        print(f"\n❌ BC数据不存在: {bc_data_path}")
        print("请先运行: python auction_sim/collect_bc_data_9dim.py")
        return
    
    # 检查旧模型
    if os.path.exists(bc_model_path):
        print(f"\n⚠️  发现旧的BC模型: {bc_model_path}")
        backup_path = bc_model_path + ".old"
        os.rename(bc_model_path, backup_path)
        print(f"    已备份到: {backup_path}")
    
    # 创建BC训练器（使用新的网络结构）
    print(f"\n🔧 创建BC训练器（obs_dim=9, 动作空间=[0,1.5]）...")
    bc_trainer = BCTrainer(obs_dim=9, action_dim=1, lr=1e-3)
    
    # 训练BC模型
    print(f"\n🚀 开始BC训练...")
    print(f"   - 数据集: {bc_data_path}")
    print(f"   - Epochs: 30")
    print(f"   - Batch size: 64")
    print(f"   - 保存路径: {bc_model_path}")
    
    results = bc_trainer.train(
        dataset_path=bc_data_path,
        n_epochs=30,
        batch_size=64,
        save_path=bc_model_path,
        n_collect_episodes=0,  # 不重新收集数据
        auto_prepare_data=False  # 使用现有数据
    )
    
    print(f"\n" + "="*70)
    print("BC训练完成！")
    print("="*70)
    
    if results:
        print(f"最终训练损失: {results.get('final_train_loss', 'N/A'):.4f}")
        print(f"最终验证损失: {results.get('final_val_loss', 'N/A'):.4f}")
        print(f"模型保存位置: {bc_model_path}")
    
    print(f"\n✅ 下一步:")
    print(f"   运行 python bc_ippo_trainer.py 开始IPPO训练")
    print(f"   IPPO将自动加载新的BC预训练权重")

if __name__ == "__main__":
    main()



