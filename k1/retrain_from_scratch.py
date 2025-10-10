"""
一键式重新训练脚本
修复所有已知问题后，从头开始训练
"""
import os
import sys
import shutil
from datetime import datetime

print("=" * 80)
print("🔧 准备重新训练 - 修复所有已知问题")
print("=" * 80)

# Step 1: 检查BC数据
print("\n📊 Step 1: 检查BC数据集...")
bc_data_path = "auction_sim/bc_dataset.pkl"
if os.path.exists(bc_data_path):
    import pickle
    with open(bc_data_path, 'rb') as f:
        data = pickle.load(f)
    
    if data:
        actions = [d['action'] for d in data]
        action_min = min(actions)
        action_max = max(actions)
        action_mean = sum(actions) / len(actions)
        
        print(f"  BC数据样本数: {len(data)}")
        print(f"  动作范围: [{action_min:.4f}, {action_max:.4f}]")
        print(f"  动作均值: {action_mean:.4f}")
        
        # 检查动作范围是否合理
        if action_min < -0.1 or action_max > 1.3:
            print(f"  ⚠️ 警告: 动作范围可能不正确！期望 [0.0, 1.2]")
            print(f"  建议: 重新收集BC数据")
            
            response = input("\n是否重新收集BC数据? (y/n): ")
            if response.lower() == 'y':
                print("\n  正在重新收集BC数据...")
                from auction_sim.bc_data_collector import BCDataCollector
                collector = BCDataCollector()
                collector.collect_and_save(
                    n_episodes=50,
                    dataset_path=bc_data_path
                )
                print("  ✅ BC数据收集完成！")
            else:
                print("  ⏭️ 跳过BC数据收集")
        else:
            print("  ✅ BC数据范围正常")
    else:
        print("  ❌ BC数据为空！")
        print("  正在收集BC数据...")
        from auction_sim.bc_data_collector import BCDataCollector
        collector = BCDataCollector()
        collector.collect_and_save(
            n_episodes=50,
            dataset_path=bc_data_path
        )
        print("  ✅ BC数据收集完成！")
else:
    print("  ❌ BC数据集不存在！")
    print("  正在收集BC数据...")
    from auction_sim.bc_data_collector import BCDataCollector
    collector = BCDataCollector()
    collector.collect_and_save(
        n_episodes=50,
        dataset_path=bc_data_path
    )
    print("  ✅ BC数据收集完成！")

# Step 2: 备份旧模型
print("\n💾 Step 2: 备份旧模型...")
model_dir = "auction_sim/models/bc_ippo"
if os.path.exists(model_dir):
    backup_name = f"bc_ippo_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_path = f"auction_sim/models/{backup_name}"
    
    response = input(f"是否备份当前模型到 {backup_name}? (y/n): ")
    if response.lower() == 'y':
        shutil.copytree(model_dir, backup_path)
        print(f"  ✅ 模型已备份到: {backup_path}")
    else:
        print("  ⏭️ 跳过模型备份")
else:
    print("  ℹ️ 没有找到旧模型，无需备份")

# Step 3: 清除best_reward记录
print("\n🗑️ Step 3: 清除best_reward记录...")
best_reward_file = "auction_sim/models/bc_ippo/best_reward.txt"
if os.path.exists(best_reward_file):
    os.remove(best_reward_file)
    print("  ✅ best_reward.txt 已删除")
else:
    print("  ℹ️ best_reward.txt 不存在")

# Step 4: 检查关键修复
print("\n🔍 Step 4: 验证关键修复...")

# 检查agents.py的动作clip
print("  检查 agents.py 动作clip...")
with open("auction_sim/agents.py", 'r', encoding='utf-8') as f:
    agents_content = f.read()
    if "np.clip(action, 0.0, 1.2)" in agents_content:
        print("    ✅ 动作clip正确 [0.0, 1.2]")
    else:
        print("    ⚠️ 动作clip可能不正确")

# 检查exp_runner.py的预算检查
print("  检查 exp_runner.py 预算检查...")
with open("exp_runner.py", 'r', encoding='utf-8') as f:
    runner_content = f.read()
    if "can_afford_bid(bid_price)" in runner_content:
        print("    ✅ 预算检查正确")
    else:
        print("    ⚠️ 预算检查可能不正确")

# 检查ROI计算
print("  检查 agents.py ROI计算...")
with open("auction_sim/agents.py", 'r', encoding='utf-8') as f:
    agents_content = f.read()
    if "total_cost < 1.0" in agents_content:
        print("    ✅ ROI计算已修复（防止除零）")
    else:
        print("    ⚠️ ROI计算可能未修复")

# Step 5: 开始训练
print("\n" + "=" * 80)
print("🚀 准备完成！开始训练...")
print("=" * 80)
print("\n训练参数:")
print("  - BC Episodes: 50")
print("  - IPPO Episodes: 300")
print("  - 预期ROI: 200-400%")
print("  - 预期Win Rate: 30-50%")
print("\n")

response = input("开始训练? (y/n): ")
if response.lower() == 'y':
    print("\n开始训练...\n")
    os.system("python bc_ippo_trainer.py")
    
    print("\n" + "=" * 80)
    print("✅ 训练完成！")
    print("=" * 80)
    
    # Step 6: 运行评估
    response = input("\n运行评估? (y/n): ")
    if response.lower() == 'y':
        print("\n开始评估...\n")
        os.system("python exp_runner.py")
else:
    print("\n⏭️ 取消训练")
    print("你可以稍后手动运行: python bc_ippo_trainer.py")

print("\n" + "=" * 80)
print("✅ 完成！")
print("=" * 80)


