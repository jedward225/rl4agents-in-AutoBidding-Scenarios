#!/usr/bin/env python3
"""
BC + DDPG (TD3) 完整训练流程主运行脚本

这个脚本负责执行从行为克隆预训练到DDPG强化学习的完整训练流程。
基于bb.md文档的指导，提供了一个清晰、易用的训练入口。

使用方法:
    python run_bc_ddpg.py

作者: AI Assistant
日期: 2024
"""

import numpy as np
import torch
import os
import sys
import time
from datetime import datetime

def main():
    """
    主函数，用于配置并启动 BC + DDPG 的完整训练流程。
    """
    # ----------------------------------------------------
    # 1. 导入必要的模块
    # ----------------------------------------------------
    print("🔧 正在导入必要的模块...")
    try:
        from bc_ddpg_trainer import BCDDPGPipelineTrainer
        from auction_sim.config import CurriculumStage
        print("✅ 模块导入成功")
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确保您在项目的根目录下运行此脚本，或者项目路径已正确添加到 PYTHONPATH。")
        print("当前工作目录:", os.getcwd())
        print("Python路径:", sys.path[:3])
        return False

    # ----------------------------------------------------
    # 2. 设置随机种子以保证实验可复现
    # ----------------------------------------------------
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print("\n" + "="*80)
    print("🚀 开始执行 BC + DDPG (TD3) 完整训练流程 🚀")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎲 实验种子: {seed}")
    print(f"🖥️  设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*80)

    # ----------------------------------------------------
    # 3. 配置训练参数
    # ----------------------------------------------------
    print("\n📋 配置训练参数...")
    
    # 您可以在这里轻松调整所有关键超参数
    training_config = {
        "use_bc_pretraining": True,         # 是否启用行为克隆预训练
        "bc_episodes": 10,                  # BC 数据收集的回合数
        "bc_epochs": 20,                    # BC 训练的轮数
        "ddpg_episodes": 500,               # DDPG 强化学习的训练回合数
        "target_stage": CurriculumStage.STAGE_8, # 目标训练环境 (8个智能体的复杂环境)
        "save_dir": "auction_sim/models/bc_ddpg_final",   # 模型保存目录
        "results_dir": "auction_sim/results/bc_ddpg_final", # 结果与图表保存目录
        "auto_save": True                   # 是否自动保存检查点和最终模型
    }
    
    # 打印配置信息
    print("  训练配置:")
    print(f"    - BC预训练: {'启用' if training_config['use_bc_pretraining'] else '禁用'}")
    print(f"    - BC数据收集回合数: {training_config['bc_episodes']}")
    print(f"    - BC训练轮数: {training_config['bc_epochs']}")
    print(f"    - DDPG训练回合数: {training_config['ddpg_episodes']}")
    print(f"    - 目标环境: {training_config['target_stage'].name}")
    print(f"    - 模型保存目录: {training_config['save_dir']}")
    print(f"    - 结果保存目录: {training_config['results_dir']}")
    
    # ----------------------------------------------------
    # 4. 初始化并运行训练流程管理器
    # ----------------------------------------------------
    print("\n🏗️  初始化训练流程管理器...")
    
    try:
        # 创建 BCDDPGPipelineTrainer 实例
        pipeline_trainer = BCDDPGPipelineTrainer(
            use_bc_pretraining=training_config["use_bc_pretraining"],
            bc_episodes=training_config["bc_episodes"],
            bc_epochs=training_config["bc_epochs"],
            ddpg_episodes=training_config["ddpg_episodes"],
            target_stage=training_config["target_stage"],
            save_dir=training_config["save_dir"],
            results_dir=training_config["results_dir"],
            auto_save=training_config["auto_save"]
        )
        print("✅ 训练流程管理器初始化成功")
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行完整的训练流程
        print("\n🎯 开始执行完整训练流程...")
        results = pipeline_trainer.train_complete_pipeline()
        
        # 计算总训练时间
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        # ----------------------------------------------------
        # 5. 打印最终结果
        # ----------------------------------------------------
        print("\n" + "="*80)
        print("🎉 训练流程全部完成！🎉")
        print(f"⏱️  总训练时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"📅 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        if results:
            print("\n📊 最终性能摘要 (基于最后50个回合的平均值):")
            final_roi = results.get('final_roi', 0.0)
            final_win_rate = results.get('final_win_rate', 0.0)
            
            print(f"  - 最终平均投资回报率 (ROI): {final_roi * 100:.2f}%")
            print(f"  - 最终平均胜率: {final_win_rate * 100:.2f}%")
            
            # 性能评估
            if final_roi > 0.1:  # ROI > 10%
                print("  🟢 ROI表现: 优秀")
            elif final_roi > 0.05:  # ROI > 5%
                print("  🟡 ROI表现: 良好")
            else:
                print("  🔴 ROI表现: 需要改进")
                
            if final_win_rate > 0.6:  # 胜率 > 60%
                print("  🟢 胜率表现: 优秀")
            elif final_win_rate > 0.4:  # 胜率 > 40%
                print("  🟡 胜率表现: 良好")
            else:
                print("  🔴 胜率表现: 需要改进")
            
            # 文件路径信息
            fig_path = os.path.join(
                training_config["results_dir"],
                f"bc_ddpg_training_{training_config['target_stage'].name.lower()}.png"
            )
            model_path = os.path.join(training_config["save_dir"], "bc_ddpg_td3.pth")
            
            print(f"\n📈 详细训练曲线图已保存至: {fig_path}")
            print(f"💾 最终模型已保存至: {model_path}")
            
            # 保存训练摘要
            summary_path = os.path.join(training_config["results_dir"], "training_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"BC + DDPG 训练摘要\n")
                f.write(f"="*50 + "\n")
                f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总训练时长: {hours:02d}:{minutes:02d}:{seconds:02d}\n")
                f.write(f"目标环境: {training_config['target_stage'].name}\n")
                f.write(f"BC预训练: {'启用' if training_config['use_bc_pretraining'] else '禁用'}\n")
                f.write(f"BC回合数: {training_config['bc_episodes']}\n")
                f.write(f"BC训练轮数: {training_config['bc_epochs']}\n")
                f.write(f"DDPG回合数: {training_config['ddpg_episodes']}\n")
                f.write(f"\n最终性能:\n")
                f.write(f"  - ROI: {final_roi * 100:.2f}%\n")
                f.write(f"  - 胜率: {final_win_rate * 100:.2f}%\n")
            
            print(f"📄 训练摘要已保存至: {summary_path}")
            
        else:
            print("⚠️ 训练未返回有效结果。")
            return False
            
        print("\n🎯 建议下一步:")
        print("  1. 查看训练曲线图分析训练过程")
        print("  2. 使用保存的模型进行测试和评估")
        print("  3. 根据性能结果调整超参数进行进一步优化")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_usage():
    """打印使用说明"""
    print("BC + DDPG 训练流程脚本")
    print("="*40)
    print("使用方法:")
    print("  python run_bc_ddpg.py")
    print("\n功能:")
    print("  - 自动执行BC预训练（如果需要）")
    print("  - 执行DDPG强化学习训练")
    print("  - 生成训练曲线和性能报告")
    print("  - 保存训练好的模型")
    print("\n配置:")
    print("  - 可在main()函数中的training_config字典中调整参数")
    print("  - 支持自定义训练回合数、保存路径等")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        sys.exit(0)
    
    # 执行主函数
    success = main()
    
    if success:
        print("\n✅ 脚本执行成功完成！")
        sys.exit(0)
    else:
        print("\n❌ 脚本执行失败！")
        sys.exit(1)