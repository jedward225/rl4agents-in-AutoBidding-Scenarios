#!/usr/bin/env python3
"""
Pure Curriculum Learning: 纯课程学习训练器
从零开始，无BC预训练，通过完整的9阶段课程学习训练智能体
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from collections import defaultdict, deque

# Add auction_sim to path
sys.path.append('.')

from auction_sim import config
from auction_sim.config import CurriculumStage, CURRICULUM_CONFIGS, CURRICULUM_SUCCESS_CRITERIA
from auction_sim.ma_runner import create_rule_agents, create_learning_agents
from auction_sim.ma_environment import MultiAgentAuctionEnv
from auction_sim.ma_trainer import MAPPOTrainer
from auction_sim.curriculum_controller import CurriculumController

class PureCurriculumTrainer:
    """纯课程学习训练器 - 从零开始的9阶段渐进训练"""
    
    def __init__(self, 
                 start_stage: CurriculumStage = CurriculumStage.STAGE_0,
                 target_final_stage: CurriculumStage = CurriculumStage.STAGE_8,
                 max_total_episodes: int = 2000):
        """
        Args:
            start_stage: 开始阶段
            target_final_stage: 目标最终阶段
            max_total_episodes: 最大总训练episodes
        """
        self.start_stage = start_stage
        self.target_final_stage = target_final_stage
        self.max_total_episodes = max_total_episodes
        
        # 模型保存路径
        self.models_dir = "auction_sim/models/pure_curriculum"
        os.makedirs(self.models_dir, exist_ok=True)
        
        print(f"Pure Curriculum Trainer初始化:")
        print(f"  开始阶段: {start_stage.name}")
        print(f"  目标阶段: {target_final_stage.name}")
        print(f"  最大episodes: {max_total_episodes}")

    def train_curriculum_learning(self):
        """执行完整的课程学习训练"""
        print("\n" + "="*70)
        print("纯课程学习训练")
        print("="*70)
        
        # 初始化课程控制器
        curriculum_controller = CurriculumController(start_stage=self.start_stage)
        
        # 全局训练跟踪
        total_episodes = 0
        global_metrics = []
        stage_transitions = []
        
        print(f"开始阶段: {curriculum_controller.current_stage.name}")
        
        while (curriculum_controller.current_stage.value <= self.target_final_stage.value and 
               total_episodes < self.max_total_episodes):
            
            current_stage = curriculum_controller.current_stage
            print(f"\n" + "="*60)
            print(f"进入 {current_stage.name}")
            print("="*60)
            
            # 获取当前阶段配置
            stage_config = CURRICULUM_CONFIGS[current_stage]
            stage_criteria = CURRICULUM_SUCCESS_CRITERIA[current_stage]
            
            print(f"阶段描述: {stage_config['description']}")
            print(f"智能体配置: {stage_config['n_learning']}L + {stage_config['n_truthful']}T + "
                  f"{stage_config['n_conservative']}C + {stage_config['n_aggressive']}A")
            print(f"成功标准: 胜率≥{stage_criteria['min_win_rate']:.1%}, "
                  f"ROI≥{stage_criteria['min_roi']:.1f}%, "
                  f"预算使用≥{stage_criteria['min_budget_usage']:.1%}")
            print(f"最少episodes: {stage_criteria['min_episodes']}")
            
            # 创建当前阶段的环境和智能体
            rule_agents = create_rule_agents(current_stage)
            learning_agents, learning_agent_ids = create_learning_agents(current_stage)
            
            env = MultiAgentAuctionEnv(
                learning_agent_ids,
                rule_agents,
                curriculum_stage=current_stage
            )
            
            # 创建trainer - 如果是第一个阶段，创建新的；否则继承前一阶段
            if current_stage == self.start_stage or 'trainer' not in locals():
                trainer = MAPPOTrainer(
                    obs_dim=9,  # 修复：从7更新到9（增强观测特征）
                    action_dim=1,
                    n_agents=len(learning_agents),
                    lr=5e-4,  # 适中的学习率
                    gamma=0.95,
                    gae_lambda=0.9,
                    clip_ratio=0.2,
                    vf_coef=0.5,
                    ent_coef=0.05,  # 较高的探索率，因为从零开始
                    max_grad_norm=0.5,
                    use_curriculum=True
                )
                print("✅ 创建新的trainer")
            else:
                # 调整网络数量以匹配当前阶段
                current_agents = len(learning_agents)
                if current_agents != trainer.n_agents:
                    print(f"调整网络数量: {trainer.n_agents} -> {current_agents}")
                    trainer.adjust_agent_count(current_agents)
                print("✅ 继承前阶段trainer")
            
            # 连接模型到智能体
            for i, agent in enumerate(learning_agents):
                agent_id = f"Learning_{i}"
                
                class ModelWrapper:
                    def __init__(self, network, trainer, agent_id):
                        self.network = network
                        self.trainer = trainer
                        self.agent_id = agent_id
                    
                    def predict(self, obs, deterministic=False):
                        action, _ = self.trainer.get_action(self.agent_id, obs, deterministic)
                        return np.array([action]), None
                
                model_wrapper = ModelWrapper(trainer.networks[agent_id], trainer, agent_id)
                agent.set_model(model_wrapper)
            
            # 当前阶段训练
            stage_episodes = 0
            stage_metrics = []
            stage_best_reward = float('-inf')
            
            # 阶段训练循环
            stage_pbar = tqdm(desc=f"{current_stage.name} Training", 
                            total=stage_criteria['min_episodes'])
            
            while (stage_episodes < stage_criteria.get('max_episodes', 500) and 
                   total_episodes < self.max_total_episodes):
                
                # 训练一个episode
                max_steps = stage_config['max_rounds']
                episode_rewards, episode_wins = trainer.train_episode(env, max_steps=max_steps)
                
                # 计算详细统计
                episode_stats = self._calculate_episode_stats(
                    env, learning_agent_ids, stage_config, episode_rewards, episode_wins, max_steps
                )
                
                # 更新课程控制器
                curriculum_controller.update_metrics(episode_stats)
                stage_metrics.append(episode_stats)
                global_metrics.append({
                    'stage': current_stage.name,
                    'episode': total_episodes + 1,
                    'stage_episode': stage_episodes + 1,
                    **episode_stats
                })
                
                # 保存最佳模型
                total_reward = episode_stats.get('total_reward', 0)
                if total_reward > stage_best_reward:
                    stage_best_reward = total_reward
                    trainer.save_models(f"{self.models_dir}/{current_stage.name}_best")
                
                stage_episodes += 1
                total_episodes += 1
                stage_pbar.update(1)
                stage_pbar.set_postfix({
                    'WinRate': f"{episode_stats['avg_win_rate']:.1%}",
                    'ROI': f"{episode_stats['avg_roi']:.0f}%",
                    'Budget': f"{episode_stats['budget_usage_ratio']:.1%}",
                    'Reward': f"{total_reward:.0f}"
                })
                
                # 每20个episodes打印详细信息
                if stage_episodes % 20 == 0 and len(stage_metrics) >= 10:
                    self._print_stage_progress(current_stage, stage_episodes, stage_metrics[-10:])
                
                # 检查是否可以晋级
                if stage_episodes >= stage_criteria['min_episodes'] and curriculum_controller.should_advance():
                    print(f"\n🎉 {current_stage.name} 完成！满足晋级条件")
                    break
                
                # 检查是否在当前阶段停滞太久
                if stage_episodes >= stage_criteria.get('max_episodes', 500):
                    print(f"\n⏰ {current_stage.name} 达到最大episodes限制")
                    # 可以选择放松标准或强制晋级
                    if stage_episodes >= stage_criteria['min_episodes']:
                        print("达到最少episodes要求，强制晋级")
                        break
                    else:
                        print("未达到最少episodes要求，继续训练")
            
            stage_pbar.close()
            
            # 保存当前阶段的最终模型
            trainer.save_models(f"{self.models_dir}/{current_stage.name}_final")
            
            # 记录阶段转换
            stage_summary = {
                'from_stage': current_stage.name,
                'episodes_trained': stage_episodes,
                'final_metrics': stage_metrics[-1] if stage_metrics else {},
                'best_reward': stage_best_reward,
                'total_episodes': total_episodes
            }
            stage_transitions.append(stage_summary)
            
            # 尝试推进到下一阶段
            if current_stage.value < self.target_final_stage.value:
                next_stage_value = current_stage.value + 1
                try:
                    next_stage = CurriculumStage(next_stage_value)
                    curriculum_controller.current_stage = next_stage
                    curriculum_controller.reset_stage_metrics()
                    print(f"✅ 晋级到 {next_stage.name}")
                except ValueError:
                    print(f"❌ 无法找到下一阶段 (value={next_stage_value})")
                    break
            else:
                print(f"✅ 到达目标阶段 {current_stage.name}")
                break
        
        # 绘制完整训练曲线
        self._plot_curriculum_curves(global_metrics, stage_transitions)
        
        # 最终总结
        print("\n" + "="*70)
        print("纯课程学习训练完成")
        print("="*70)
        
        print(f"总训练episodes: {total_episodes}")
        print(f"完成阶段数: {len(stage_transitions)}")
        print(f"最终阶段: {curriculum_controller.current_stage.name}")
        
        if stage_transitions:
            print("\n阶段总结:")
            for transition in stage_transitions:
                print(f"  {transition['from_stage']}: {transition['episodes_trained']} episodes, "
                      f"最佳奖励 {transition['best_reward']:.0f}")
        
        if global_metrics:
            final_metrics = global_metrics[-20:]  # 最后20个episodes
            final_win_rate = np.mean([m['avg_win_rate'] for m in final_metrics])
            final_roi = np.mean([m['avg_roi'] for m in final_metrics])
            final_budget = np.mean([m['budget_usage_ratio'] for m in final_metrics])
            
            print(f"\n最终性能 (最后20 episodes平均):")
            print(f"  胜率: {final_win_rate:.1%}")
            print(f"  ROI: {final_roi:.1f}%")
            print(f"  预算使用: {final_budget:.1%}")
        
        return {
            'trainer': trainer,
            'curriculum_controller': curriculum_controller,
            'global_metrics': global_metrics,
            'stage_transitions': stage_transitions,
            'total_episodes': total_episodes
        }

    def _calculate_episode_stats(self, env, learning_agent_ids, stage_config, episode_rewards, episode_wins, max_steps):
        """计算episode统计数据"""
        stats = {
            'avg_win_rate': 0.0,
            'avg_roi': 0.0,
            'budget_usage_ratio': 0.0,
            'avg_bid_ratio': 1.0,
            'total_reward': sum(episode_rewards.values()),
            'individual_rewards': episode_rewards,
            'individual_wins': episode_wins,
            'individual_stats': {}
        }
        
        # 计算平均胜率
        total_wins = sum(episode_wins.values())
        stats['avg_win_rate'] = total_wins / (len(learning_agent_ids) * max_steps) if learning_agent_ids else 0.0
        
        for agent_id in learning_agent_ids:
            agent_stats = {
                'win_rate': episode_wins.get(agent_id, 0) / max_steps if max_steps > 0 else 0.0,
                'roi': -100.0,
                'budget_usage': 0.0,
                'bid_ratio': 1.0,
                'reward': episode_rewards.get(agent_id, 0)
            }
            
            # 计算预算使用率
            initial_budget = stage_config['budget']
            remaining = env.agent_budgets.get(agent_id, initial_budget)
            used = initial_budget - remaining
            agent_stats['budget_usage'] = used / initial_budget
            
            # 计算ROI和出价比率
            if hasattr(env, 'agent_histories') and agent_id in env.agent_histories:
                history = env.agent_histories[agent_id]
                if history:
                    total_profit = sum(h.get('profit', 0) for h in history)
                    total_cost = sum(h.get('cost', 0) for h in history)
                    
                    if total_cost > 0:
                        agent_stats['roi'] = (total_profit / total_cost) * 100
                    
                    # 计算出价比率
                    bids = [h.get('bid', 0) for h in history if h.get('perceived_value', 0) > 0]
                    values = [h.get('perceived_value', 1) for h in history if h.get('perceived_value', 0) > 0]
                    if bids and values:
                        agent_stats['bid_ratio'] = np.mean([b/v for b, v in zip(bids, values) if v > 0])
            
            stats['individual_stats'][agent_id] = agent_stats
            
            # 累加到平均值
            stats['avg_roi'] += agent_stats['roi']
            stats['budget_usage_ratio'] += agent_stats['budget_usage'] 
            stats['avg_bid_ratio'] += agent_stats['bid_ratio']
        
        # 计算平均值
        n_agents = len(learning_agent_ids)
        if n_agents > 0:
            stats['avg_roi'] /= n_agents
            stats['budget_usage_ratio'] /= n_agents
            stats['avg_bid_ratio'] /= n_agents
        
        return stats

    def _print_stage_progress(self, current_stage, stage_episodes, recent_metrics):
        """打印阶段训练进度"""
        avg_win_rate = np.mean([m['avg_win_rate'] for m in recent_metrics])
        avg_roi = np.mean([m['avg_roi'] for m in recent_metrics])
        avg_budget = np.mean([m['budget_usage_ratio'] for m in recent_metrics])
        avg_reward = np.mean([m['total_reward'] for m in recent_metrics])
        
        print(f"\n{current_stage.name} Episode {stage_episodes} (最近10集平均):")
        print(f"  胜率: {avg_win_rate:.1%}")
        print(f"  ROI: {avg_roi:.1f}%")
        print(f"  预算使用: {avg_budget:.1%}")
        print(f"  总奖励: {avg_reward:.0f}")
        
        # 显示个体差异
        if recent_metrics:
            last_stats = recent_metrics[-1]['individual_stats']
            print("  个体表现:")
            for agent_id, stats in last_stats.items():
                print(f"    {agent_id}: 胜率{stats['win_rate']:.1%}, ROI{stats['roi']:.0f}%, "
                      f"预算{stats['budget_usage']:.1%}")

    def _plot_curriculum_curves(self, global_metrics, stage_transitions):
        """绘制完整课程学习曲线"""
        if not global_metrics:
            print("没有足够的数据绘制训练曲线")
            return
        
        plt.figure(figsize=(20, 15))
        
        # 提取数据
        episodes = [m['episode'] for m in global_metrics]
        stages = [m['stage'] for m in global_metrics]
        win_rates = [m['avg_win_rate'] for m in global_metrics]
        rois = [m['avg_roi'] for m in global_metrics]
        budget_usage = [m['budget_usage_ratio'] for m in global_metrics]
        rewards = [m['total_reward'] for m in global_metrics]
        
        # 创建阶段分隔线
        stage_boundaries = []
        for transition in stage_transitions:
            stage_boundaries.append(transition['total_episodes'])
        
        # 滑动平均函数
        def moving_average(data, window=20):
            return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]
        
        ma_win_rates = moving_average(win_rates)
        ma_rois = moving_average(rois)
        ma_budget = moving_average(budget_usage)
        ma_rewards = moving_average(rewards)
        
        # 绘制子图
        plt.subplot(3, 2, 1)
        plt.plot(episodes, win_rates, alpha=0.3, color='green', label='Raw')
        plt.plot(episodes, ma_win_rates, color='green', linewidth=2, label='MA-20')
        for boundary in stage_boundaries:
            plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
        plt.title('Win Rate Throughout Curriculum')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 2, 2)
        plt.plot(episodes, rois, alpha=0.3, color='blue', label='Raw')
        plt.plot(episodes, ma_rois, color='blue', linewidth=2, label='MA-20')
        for boundary in stage_boundaries:
            plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
        plt.title('ROI Throughout Curriculum')
        plt.xlabel('Episode')
        plt.ylabel('ROI (%)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 2, 3)
        plt.plot(episodes, budget_usage, alpha=0.3, color='orange', label='Raw')
        plt.plot(episodes, ma_budget, color='orange', linewidth=2, label='MA-20')
        for boundary in stage_boundaries:
            plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
        plt.title('Budget Usage Throughout Curriculum')
        plt.xlabel('Episode')
        plt.ylabel('Budget Usage Ratio')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 2, 4)
        plt.plot(episodes, rewards, alpha=0.3, color='purple', label='Raw')
        plt.plot(episodes, ma_rewards, color='purple', linewidth=2, label='MA-20')
        for boundary in stage_boundaries:
            plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
        plt.title('Episode Rewards Throughout Curriculum')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)
        
        # 阶段分布图
        plt.subplot(3, 2, 5)
        stage_counts = {}
        for stage in stages:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        stage_names = list(stage_counts.keys())
        stage_episodes = list(stage_counts.values())
        
        bars = plt.bar(stage_names, stage_episodes, alpha=0.7)
        plt.title('Episodes per Curriculum Stage')
        plt.xlabel('Stage')
        plt.ylabel('Number of Episodes')
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar, count in zip(bars, stage_episodes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom')
        
        # 综合性能图
        plt.subplot(3, 2, 6)
        # 简化的综合评分
        composite_scores = []
        for i, (wr, roi, budget) in enumerate(zip(win_rates, rois, budget_usage)):
            score = 0.4 * wr + 0.3 * max(0, roi/100) + 0.3 * budget
            composite_scores.append(score)
        
        ma_composite = moving_average(composite_scores)
        plt.plot(episodes, composite_scores, alpha=0.3, color='black', label='Raw')
        plt.plot(episodes, ma_composite, color='black', linewidth=2, label='MA-20')
        for boundary in stage_boundaries:
            plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
        plt.title('Composite Performance Score')
        plt.xlabel('Episode')
        plt.ylabel('Composite Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('auction_sim/results/pure_curriculum_training_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"课程学习曲线已保存: auction_sim/results/pure_curriculum_training_curves.png")

def main():
    """主函数 - 执行纯课程学习"""
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("开始Pure Curriculum Learning实验...")
    
    # 创建trainer
    trainer = PureCurriculumTrainer(
        start_stage=CurriculumStage.STAGE_0,
        target_final_stage=CurriculumStage.STAGE_8,
        max_total_episodes=3000  # 较大的episode限制
    )
    
    # 执行课程学习训练
    results = trainer.train_curriculum_learning()
    
    print("\n" + "="*80)
    print("PURE CURRICULUM LEARNING 实验完成！")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()