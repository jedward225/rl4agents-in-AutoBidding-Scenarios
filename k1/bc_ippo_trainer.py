#!/usr/bin/env python3
"""
BC + IPPO Training: 行为克隆预训练后使用独立PPO训练
每个智能体独立训练，不共享参数，解决不对称问题
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
from auction_sim.config import CurriculumStage, CURRICULUM_CONFIGS
from auction_sim.bc_trainer import BCTrainer
from auction_sim.ippo_trainer import IPPOTrainer, IPPOActorCriticNetwork
from auction_sim.ma_environment import MultiAgentAuctionEnv
from auction_sim.agents import TruthfulAgent, ConservativeAgent, AggressiveAgent, MultiAgentLearningAgent

class BCIPPOTrainer:
    """BC预训练 + 独立PPO训练的混合trainer"""
    
    def __init__(self,
                 use_bc_pretraining: bool = True,
                 bc_episodes: int = 10,
                 ippo_episodes: int = 300,
                 target_stage: CurriculumStage = CurriculumStage.STAGE_8,
                 clean_previous_best: bool = True,
                 force_collect_bc_data: bool = False):
        """
        Args:
            use_bc_pretraining: 是否使用BC预训练
            bc_episodes: BC数据收集episodes
            ippo_episodes: IPPO训练episodes  
            target_stage: 目标环境阶段
            clean_previous_best: 是否清理之前的最佳模型记录（默认False，保留累积改进）
            force_collect_bc_data: 是否强制重新收集BC数据（即使数据集已存在）
        """
        self.use_bc_pretraining = use_bc_pretraining
        self.bc_episodes = bc_episodes
        self.ippo_episodes = ippo_episodes
        self.target_stage = target_stage
        self.clean_previous_best = clean_previous_best
        self.force_collect_bc_data = force_collect_bc_data
        
        # 模型路径
        self.bc_model_path = "auction_sim/models/bc_pretrained.pth"
        self.models_dir = "auction_sim/models/bc_ippo"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 如果需要清理之前的最佳模型记录
        if clean_previous_best:
            best_reward_file = f"{self.models_dir}/best_reward.txt"
            if os.path.exists(best_reward_file):
                os.remove(best_reward_file)
                print(f"🗑️  已清理之前的最佳模型记录，将从头开始")
        
        print(f"BC + IPPO Trainer初始化:")
        print(f"  BC预训练: {use_bc_pretraining}")
        print(f"  BC数据收集: {bc_episodes} episodes")
        print(f"  IPPO训练: {ippo_episodes} episodes")
        print(f"  目标环境: {target_stage.name}")
        print(f"  强制重新收集数据: {force_collect_bc_data}")
        print(f"  关键特性: 独立网络，不共享参数")

    def step1_bc_pretraining(self):
        """步骤1: BC预训练（如果需要）"""
        if not self.use_bc_pretraining:
            return None, None, None
            
        print("\n" + "="*70)
        print("步骤1: BC预训练")
        print("="*70)
        
        bc_data_path = "auction_sim/bc_dataset.pkl"
        
        # 检查并收集BC数据（如果需要）
        if not os.path.exists(bc_data_path) or self.force_collect_bc_data:
            if self.force_collect_bc_data and os.path.exists(bc_data_path):
                print(f"强制重新收集BC数据，将覆盖: {bc_data_path}")
            else:
                print(f"未找到BC数据集: {bc_data_path}")
            
            print("开始收集BC数据...")
            
            from auction_sim.bc_data_collector import BCDataCollector
            
            collector = BCDataCollector()
            dataset = collector.collect_dataset(
                n_episodes=self.bc_episodes,
                save_path=bc_data_path
            )
            
            print(f"✅ BC数据收集完成: {len(dataset['data'])} 个样本")
        else:
            print(f"✅ 发现现有BC数据集: {bc_data_path}")
        
        # 检查现有BC模型
        if os.path.exists(self.bc_model_path):
            print(f"发现已存在的BC模型: {self.bc_model_path}")
            
            try:
                # 简单检查BC模型是否可以加载
                bc_state_dict = torch.load(self.bc_model_path)
                print(f"✅ BC模型加载成功，包含 {len(bc_state_dict)} 个参数")
                print("跳过BC预训练，直接使用现有模型")
                return None, {'status': 'loaded_existing'}, {'model_path': self.bc_model_path}
            except Exception as e:
                print(f"❌ BC模型加载失败: {e}")
                print("将重新进行BC预训练")
        
        # BC训练 - 使用7维观测空间（与新的适配保持一致）
        print("\n开始BC预训练...")
        bc_trainer = BCTrainer(obs_dim=7, lr=1e-3)  # 改为7维观测空间
        bc_results = bc_trainer.train(
            dataset_path=bc_data_path,
            n_epochs=30,  # 增加到30个epoch以获得更好的BC预训练效果
            batch_size=64,
            save_path=self.bc_model_path
            # 移除不支持的参数：n_collect_episodes 和 auto_prepare_data
        )
        
        bc_metrics = {
            'accuracy': bc_results.get('final_accuracy', 'N/A'),
            'rmse': bc_results.get('final_rmse', 'N/A'),
            'model_path': self.bc_model_path
        }
        
        print(f"\nBC预训练完成:")
        print(f"  准确率: {bc_metrics['accuracy']}")
        print(f"  RMSE: {bc_metrics['rmse']}")
        
        return bc_trainer, bc_results, bc_metrics

    def step2_ippo_training(self):
        """步骤2: 独立PPO训练"""
        print("\n" + "="*70)
        print("步骤2: 独立PPO训练 (IPPO)")
        print("="*70)
        
        # 获取环境配置
        stage_config = CURRICULUM_CONFIGS[self.target_stage]
        print(f"目标环境: {stage_config['description']}")
        print(f"智能体配置: {stage_config['n_learning']}L + {stage_config['n_truthful']}T + "
              f"{stage_config['n_conservative']}C + {stage_config['n_aggressive']}A")
        
        # 创建规则智能体
        rule_agents = self._create_rule_agents(stage_config)
        
        # 创建学习智能体
        learning_agents = []
        learning_agent_ids = []
        for i in range(stage_config['n_learning']):
            agent_id = f"Learning_{i}"
            agent = MultiAgentLearningAgent(
                agent_id=agent_id,
                budget=stage_config['budget'],
                perception_noise_std=config.AGENT_PERCEPTION_NOISE_STD,
                model=None,
                is_training=True
            )
            learning_agents.append(agent)
            learning_agent_ids.append(agent_id)
        
        print(f"创建了 {len(rule_agents)} 个规则智能体")
        print(f"创建了 {len(learning_agents)} 个学习智能体: {learning_agent_ids}")
        
        # 创建环境
        env = MultiAgentAuctionEnv(
            learning_agent_ids,
            rule_agents,
            curriculum_stage=self.target_stage
        )
        
        # 创建IPPO trainer - 关键：每个智能体独立的网络
        # 使用7维观测空间（与BC数据收集器兼容）
        trainer = IPPOTrainer(
            obs_dim=7,  # 改为7维观测空间
            action_dim=1,
            n_agents=len(learning_agents),
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=0.5
        )
        
        print("\n✨ IPPO关键特性:")
        print("  - 每个智能体完全独立的Actor-Critic网络")
        print("  - 独立的优化器和经验缓冲区")
        print("  - 无参数共享，避免梯度冲突")
        
        # 🔧 BC权重加载（网络结构已对齐）
        if self.use_bc_pretraining and os.path.exists(self.bc_model_path):
            print("\n📥 加载BC预训练权重...")
            print("    注意：请确保BC模型是用新的[0,1.5]范围训练的")
            print("    如果是旧模型，请先运行：python auction_sim/bc_data_collector.py")
            
            try:
                success_count = trainer.load_bc_weights(self.bc_model_path, strict=False)
                if success_count > 0:
                    print(f"✅ BC权重加载成功，{success_count}个智能体初始化完成")
                else:
                    print("⚠️  BC权重加载失败，将从随机初始化开始")
            except Exception as e:
                print(f"⚠️  BC权重加载出错: {e}")
                print("    将从随机初始化开始训练")
        
        # 连接模型到智能体
        for i, agent in enumerate(learning_agents):
            agent_id = f"Learning_{i}"
            
            class IPPOModelWrapper:
                def __init__(self, network):
                    self.network = network
                
                def predict(self, obs, deterministic=False):
                    """Predict action from observation"""
                    import torch
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    with torch.no_grad():
                        action, _, _ = self.network.get_action(obs_tensor, deterministic)
                    return action.cpu().numpy(), None
            
            model_wrapper = IPPOModelWrapper(trainer.networks[agent_id])
            agent.set_model(model_wrapper)
        
        # 训练循环
        print(f"\n开始IPPO独立训练 {self.ippo_episodes} episodes...")
        
        # 性能跟踪
        episode_rewards = []
        episode_wins = []
        episode_metrics = []
        
        # 尝试加载之前的最佳奖励记录（跨训练会话保留）
        best_reward_file = f"{self.models_dir}/best_reward.txt"
        if os.path.exists(best_reward_file):
            try:
                with open(best_reward_file, 'r') as f:
                    best_avg_reward = float(f.read().strip())
                print(f"\n📊 发现之前的最佳模型记录")
                print(f"   最佳平均奖励: {best_avg_reward:.2f}")
                print(f"   ✅ 新训练将在此基础上改进（只有更好的模型才会覆盖）")
            except:
                best_avg_reward = float('-inf')
                print(f"\n📊 最佳奖励记录文件损坏，将重新开始")
        else:
            best_avg_reward = float('-inf')
            print(f"\n📊 首次训练，将记录最佳模型")
        
        # 滑动窗口
        recent_rewards = deque(maxlen=20)
        recent_wins = deque(maxlen=20)
        recent_metrics = deque(maxlen=20)
        
        # 个体性能跟踪 - 关键：分别跟踪每个智能体
        individual_performance = {
            agent_id: {
                'rewards': [],
                'win_rates': [],
                'rois': [],
                'budget_usage': []
            } for agent_id in learning_agent_ids
        }
        
        pbar = tqdm(range(self.ippo_episodes), desc="IPPO Training")
        
        for episode in pbar:
            # 训练一个episode
            max_steps = stage_config['max_rounds']
            ep_rewards, ep_wins = trainer.train_episode(env, max_steps=max_steps)
            
            # 计算统计
            total_reward = sum(ep_rewards.values())
            total_wins = sum(ep_wins.values())
            avg_win_rate = total_wins / (len(learning_agents) * max_steps)
            
            episode_rewards.append(total_reward)
            episode_wins.append(avg_win_rate)
            recent_rewards.append(total_reward)
            recent_wins.append(avg_win_rate)
            
            # 计算详细指标（包括个体指标）
            episode_stats = self._calculate_episode_stats(env, learning_agent_ids, stage_config, ep_rewards, ep_wins)
            episode_metrics.append(episode_stats)
            recent_metrics.append(episode_stats)
            
            # 计算并记录Economic Value
            for agent_id in learning_agent_ids:
                if hasattr(env, 'cum_profit') and hasattr(env, 'cum_cost'):
                    total_rounds = stage_config['max_rounds']
                    num_agents = len(learning_agent_ids) + len(rule_agents)
                    target_wins = (total_rounds / num_agents) * 0.8
                    
                    cum_profit = env.cum_profit.get(agent_id, 0.0)
                    cum_cost = env.cum_cost.get(agent_id, 0.0)
                    win_count = ep_wins.get(agent_id, 0)
                    win_rate = win_count / max(1, total_rounds)
                    
                    roi_frac = cum_profit / max(1e-8, cum_cost)
                    economic_value = (
                        0.5 * cum_profit +
                        0.15 * roi_frac * cum_cost +
                        0.35 * win_rate * target_wins
                    )
                    
                    if 'economic_value' not in individual_performance[agent_id]:
                        individual_performance[agent_id]['economic_value'] = []
                    individual_performance[agent_id]['economic_value'].append(economic_value)
            
            # 记录个体性能
            for agent_id in learning_agent_ids:
                if agent_id in episode_stats['individual_stats']:
                    stats = episode_stats['individual_stats'][agent_id]
                    individual_performance[agent_id]['rewards'].append(ep_rewards.get(agent_id, 0))
                    individual_performance[agent_id]['win_rates'].append(stats['win_rate'])
                    individual_performance[agent_id]['rois'].append(stats['roi'])
                    individual_performance[agent_id]['budget_usage'].append(stats['budget_usage'])
            
            # 更新进度条
            if len(recent_metrics) > 0:
                last_metrics = recent_metrics[-1]
                pbar.set_postfix({
                    'Reward': f"{total_reward:.0f}",
                    'WinRate': f"{avg_win_rate:.1%}",
                    'L0_WR': f"{last_metrics['individual_stats'].get('Learning_0', {}).get('win_rate', 0):.1%}",
                    'L1_WR': f"{last_metrics['individual_stats'].get('Learning_1', {}).get('win_rate', 0):.1%}"
                })
            
            # 保存最佳模型（使用滑动窗口平均，更稳定）
            if len(recent_rewards) >= 10:  # 至少10个episodes后才开始比较
                avg_recent_reward = np.mean(recent_rewards)
                if avg_recent_reward > best_avg_reward:
                    best_avg_reward = avg_recent_reward
                    trainer.save_models(f"{self.models_dir}/best")
                    # 保存最佳奖励记录
                    with open(best_reward_file, 'w') as f:
                        f.write(f"{best_avg_reward:.6f}")
                    print(f"\n✨ 新的最佳模型！平均奖励: {best_avg_reward:.2f} (最近{len(recent_rewards)}集平均)")
            
            # 定期checkpoint
            if (episode + 1) % 50 == 0:
                trainer.save_models(f"{self.models_dir}/checkpoint_ep{episode+1}")
            
            # 定期打印详细进度
            if (episode + 1) % 20 == 0 and len(recent_metrics) >= 10:
                self._print_progress(episode + 1, recent_rewards, recent_wins, recent_metrics, individual_performance)
        
        pbar.close()
        
        # 保存最终模型（带时间戳备份）
        trainer.save_models(f"{self.models_dir}/final")
        
        # 额外保存带时间戳的最终模型（便于对比不同训练）
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trainer.save_models(f"{self.models_dir}/final_{timestamp}")
        print(f"📦 带时间戳的最终模型已保存: final_{timestamp}")
        
        # 绘制训练曲线
        self._plot_training_curves(episode_rewards, episode_wins, episode_metrics, individual_performance)
        
        print(f"\nIPPO训练完成！")
        print(f"最佳平均奖励: {best_avg_reward:.0f}")
        
        # 分析不对称性
        self._analyze_asymmetry(individual_performance)
        
        return trainer, learning_agents, rule_agents, episode_metrics, individual_performance

    def _create_rule_agents(self, stage_config):
        """创建规则智能体"""
        rule_agents = []
        agent_id_counter = 0
        
        # Truthful agents
        for _ in range(stage_config.get('n_truthful', 0)):
            agent = TruthfulAgent(
                f"Truthful_{agent_id_counter}",
                stage_config['budget'],
                config.AGENT_PERCEPTION_NOISE_STD
            )
            rule_agents.append(agent)
            agent_id_counter += 1
        
        # Conservative agents
        for _ in range(stage_config.get('n_conservative', 0)):
            agent = ConservativeAgent(
                f"Conservative_{agent_id_counter}",
                stage_config['budget'],
                config.AGENT_PERCEPTION_NOISE_STD,
                stage_config['max_rounds']
            )
            rule_agents.append(agent)
            agent_id_counter += 1
        
        # Aggressive agents
        for _ in range(stage_config.get('n_aggressive', 0)):
            total_agents = (stage_config.get('n_learning', 0) + 
                          stage_config.get('n_truthful', 0) + 
                          stage_config.get('n_conservative', 0) + 
                          stage_config.get('n_aggressive', 0))
            agent = AggressiveAgent(
                f"Aggressive_{agent_id_counter}",
                stage_config['budget'],
                config.AGENT_PERCEPTION_NOISE_STD,
                total_agents
            )
            rule_agents.append(agent)
            agent_id_counter += 1
        
        return rule_agents

    def _calculate_episode_stats(self, env, learning_agent_ids, stage_config, ep_rewards, ep_wins):
        """计算episode统计数据"""
        stats = {
            'avg_win_rate': 0.0,
            'avg_roi': 0.0,
            'budget_usage_ratio': 0.0,
            'avg_bid_ratio': 0.0,  # 修复：从0开始累加，避免偏置
            'individual_stats': {}
        }
        
        for agent_id in learning_agent_ids:
            agent_stats = {
                'win_rate': 0.0,
                'roi': -100.0,
                'budget_usage': 0.0,
                'bid_ratio': 0.0  # 修复：从0开始，避免偏置
            }
            
            # 预算使用率
            initial_budget = stage_config['budget']
            remaining = env.agent_budgets.get(agent_id, initial_budget)
            used = initial_budget - remaining
            agent_stats['budget_usage'] = used / initial_budget
            
            # 胜率和ROI
            if hasattr(env, 'agent_histories') and agent_id in env.agent_histories:
                history = env.agent_histories[agent_id]
                if history:
                    total_profit = sum(h.get('profit', 0) for h in history)
                    total_cost = sum(h.get('cost', 0) for h in history)
                    wins = sum(1 for h in history if h.get('won', False))
                    
                    agent_stats['win_rate'] = wins / len(history) if history else 0.0
                    if total_cost > 0:
                        agent_stats['roi'] = (total_profit / total_cost) * 100
                    
                    # 出价比率
                    bids = [h.get('bid', 0) for h in history if h.get('perceived_value', 0) > 0]
                    values = [h.get('perceived_value', 1) for h in history if h.get('perceived_value', 0) > 0]
                    if bids and values:
                        agent_stats['bid_ratio'] = np.mean([b/v for b, v in zip(bids, values) if v > 0])
            
            stats['individual_stats'][agent_id] = agent_stats
            
            # 累加平均值
            stats['avg_win_rate'] += agent_stats['win_rate']
            stats['avg_roi'] += agent_stats['roi']
            stats['budget_usage_ratio'] += agent_stats['budget_usage']
            stats['avg_bid_ratio'] += agent_stats['bid_ratio']
        
        # 计算平均
        n_agents = len(learning_agent_ids)
        if n_agents > 0:
            stats['avg_win_rate'] /= n_agents
            stats['avg_roi'] /= n_agents
            stats['budget_usage_ratio'] /= n_agents
            stats['avg_bid_ratio'] /= n_agents
        
        return stats

    def _print_progress(self, episode, recent_rewards, recent_wins, recent_metrics, individual_performance):
        """打印训练进度"""
        avg_reward = np.mean(recent_rewards)
        avg_win_rate = np.mean(recent_wins)
        
        avg_roi = np.mean([m['avg_roi'] for m in recent_metrics])
        avg_budget = np.mean([m['budget_usage_ratio'] for m in recent_metrics])
        
        print(f"\nEpisode {episode}/{self.ippo_episodes} (最近20集平均):")
        print(f"  总奖励: {avg_reward:.0f}")
        print(f"  平均胜率: {avg_win_rate:.1%}")
        print(f"  平均ROI: {avg_roi:.1f}%")
        print(f"  预算使用: {avg_budget:.1%}")
        
        # 显示Economic Value（如果有的话）
        if any('economic_value' in perf for perf in individual_performance.values()):
            avg_ev = 0.0
            count = 0
            for agent_id, perf in individual_performance.items():
                if 'economic_value' in perf and len(perf['economic_value']) > 0:
                    recent_ev = perf['economic_value'][-20:] if len(perf['economic_value']) >= 20 else perf['economic_value']
                    avg_ev += np.mean(recent_ev)
                    count += 1
            if count > 0:
                avg_ev /= count
                print(f"  📊 Economic Value: {avg_ev:.2f}")
        
        # 关键：显示个体表现对比
        print("\n  🔍 个体表现对比 (IPPO独立训练):")
        for agent_id in individual_performance.keys():
            recent_wrs = individual_performance[agent_id]['win_rates'][-20:]
            recent_rois = individual_performance[agent_id]['rois'][-20:]
            recent_budgets = individual_performance[agent_id]['budget_usage'][-20:]
            
            if recent_wrs:
                print(f"    {agent_id}:")
                print(f"      胜率: {np.mean(recent_wrs):.1%}")
                print(f"      ROI: {np.mean(recent_rois):.0f}%")
                print(f"      预算: {np.mean(recent_budgets):.1%}")
        
        # 计算不对称度
        if len(individual_performance) == 2:
            agents = list(individual_performance.keys())
            wr_diff = abs(np.mean(individual_performance[agents[0]]['win_rates'][-20:]) - 
                         np.mean(individual_performance[agents[1]]['win_rates'][-20:]))
            print(f"\n  📊 不对称度: 胜率差异 {wr_diff:.1%}")

    def _analyze_asymmetry(self, individual_performance):
        """分析智能体不对称性"""
        print("\n" + "="*70)
        print("不对称性分析 (IPPO vs MAPPO对比)")
        print("="*70)
        
        if len(individual_performance) == 2:
            agents = list(individual_performance.keys())
            
            # 最终性能
            final_episodes = 50  # 最后50个episodes
            
            agent0_final_wr = np.mean(individual_performance[agents[0]]['win_rates'][-final_episodes:])
            agent1_final_wr = np.mean(individual_performance[agents[1]]['win_rates'][-final_episodes:])
            
            agent0_final_roi = np.mean(individual_performance[agents[0]]['rois'][-final_episodes:])
            agent1_final_roi = np.mean(individual_performance[agents[1]]['rois'][-final_episodes:])
            
            print(f"\n最终性能对比 (最后{final_episodes}集):")
            print(f"  {agents[0]}: 胜率 {agent0_final_wr:.1%}, ROI {agent0_final_roi:.0f}%")
            print(f"  {agents[1]}: 胜率 {agent1_final_wr:.1%}, ROI {agent1_final_roi:.0f}%")
            
            wr_diff = abs(agent0_final_wr - agent1_final_wr)
            roi_diff = abs(agent0_final_roi - agent1_final_roi)
            
            print(f"\n不对称度指标:")
            print(f"  胜率差异: {wr_diff:.1%}")
            print(f"  ROI差异: {roi_diff:.0f}%")
            
            # 评估
            if wr_diff < 0.05:  # 5%以内
                print("\n✅ 评估: 不对称性问题基本解决！")
            elif wr_diff < 0.15:  # 15%以内
                print("\n⚠️ 评估: 轻度不对称，可接受")
            else:
                print("\n❌ 评估: 仍存在显著不对称")
            
            # 对比MAPPO结果
            print("\n与MAPPO对比:")
            print("  MAPPO: 胜率差异通常 30-40%")
            print(f"  IPPO:  胜率差异 {wr_diff:.1%}")
            improvement = max(0, 0.35 - wr_diff) / 0.35 * 100
            print(f"  改善程度: {improvement:.0f}%")

    def _plot_training_curves(self, episode_rewards, episode_wins, episode_metrics, individual_performance):
        """绘制训练曲线"""
        # 检查是否需要3行（有EV数据时）
        has_ev_data = any('economic_value' in perf for perf in individual_performance.values())
        n_rows = 3 if has_ev_data else 2
        plt.figure(figsize=(18, 12 if n_rows == 2 else 16))
        
        episodes = range(1, len(episode_rewards) + 1)
        
        # 滑动平均
        def moving_average(data, window=10):
            return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]
        
        # 1. 总奖励
        plt.subplot(n_rows, 3, 1)
        plt.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw')
        plt.plot(episodes, moving_average(episode_rewards), color='blue', linewidth=2, label='MA-10')
        plt.title('Episode Rewards (IPPO)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 平均胜率
        plt.subplot(n_rows, 3, 2)
        plt.plot(episodes, episode_wins, alpha=0.3, color='green', label='Average')
        plt.plot(episodes, moving_average(episode_wins), color='green', linewidth=2, label='MA-10')
        plt.title('Average Win Rate (IPPO)')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 个体胜率对比 - 关键图表
        plt.subplot(n_rows, 3, 3)
        colors = ['red', 'blue']
        for i, (agent_id, perf) in enumerate(individual_performance.items()):
            plt.plot(episodes, perf['win_rates'], alpha=0.3, color=colors[i])
            plt.plot(episodes, moving_average(perf['win_rates']), 
                    color=colors[i], linewidth=2, label=agent_id)
        plt.title('Individual Win Rates (IPPO)')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 个体ROI对比
        plt.subplot(n_rows, 3, 4)
        for i, (agent_id, perf) in enumerate(individual_performance.items()):
            plt.plot(episodes, perf['rois'], alpha=0.3, color=colors[i])
            plt.plot(episodes, moving_average(perf['rois']), 
                    color=colors[i], linewidth=2, label=agent_id)
        plt.title('Individual ROI (IPPO)')
        plt.xlabel('Episode')
        plt.ylabel('ROI (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. 胜率差异度量
        plt.subplot(n_rows, 3, 5)
        if len(individual_performance) == 2:
            agents = list(individual_performance.keys())
            wr_diffs = [abs(individual_performance[agents[0]]['win_rates'][i] - 
                           individual_performance[agents[1]]['win_rates'][i]) 
                       for i in range(len(episode_rewards))]
            plt.plot(episodes, wr_diffs, alpha=0.3, color='purple')
            plt.plot(episodes, moving_average(wr_diffs), color='purple', linewidth=2)
            plt.axhline(y=0.05, color='green', linestyle='--', label='目标 (<5%)')
            plt.axhline(y=0.15, color='orange', linestyle='--', label='可接受 (<15%)')
            plt.title('Win Rate Asymmetry (IPPO)')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate Difference')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 6. 预算使用对比 (如果有EV数据则移到第三行)
        subplot_pos = (3, 3, 6) if has_ev_data else (2, 3, 6)
        plt.subplot(*subplot_pos)
        for i, (agent_id, perf) in enumerate(individual_performance.items()):
            plt.plot(episodes, perf['budget_usage'], alpha=0.3, color=colors[i])
            plt.plot(episodes, moving_average(perf['budget_usage']), 
                    color=colors[i], linewidth=2, label=agent_id)
        plt.title('Budget Usage (IPPO)')
        plt.xlabel('Episode')
        plt.ylabel('Budget Usage Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7-9. Economic Value相关图表（如果有数据）
        if has_ev_data:
            # 7. Economic Value趋势
            plt.subplot(3, 3, 7)
            for i, (agent_id, perf) in enumerate(individual_performance.items()):
                if 'economic_value' in perf:
                    plt.plot(range(len(perf['economic_value'])), perf['economic_value'], 
                            alpha=0.3, color=colors[i])
                    plt.plot(range(len(perf['economic_value'])), 
                            moving_average(perf['economic_value']), 
                            color=colors[i], linewidth=2, label=agent_id)
            plt.title('Economic Value (EV-aligned Reward)')
            plt.xlabel('Episode')
            plt.ylabel('Economic Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 8. 平均Economic Value
            plt.subplot(3, 3, 8)
            avg_ev_per_episode = []
            for ep_idx in range(len(episode_rewards)):
                ev_sum = 0
                ev_count = 0
                for agent_id, perf in individual_performance.items():
                    if 'economic_value' in perf and ep_idx < len(perf['economic_value']):
                        ev_sum += perf['economic_value'][ep_idx]
                        ev_count += 1
                if ev_count > 0:
                    avg_ev_per_episode.append(ev_sum / ev_count)
                else:
                    avg_ev_per_episode.append(0)
            
            plt.plot(episodes, avg_ev_per_episode, alpha=0.3, color='purple', label='Raw')
            plt.plot(episodes, moving_average(avg_ev_per_episode), 
                    color='purple', linewidth=2, label='MA-10')
            plt.title('Average Economic Value')
            plt.xlabel('Episode')
            plt.ylabel('Avg EV')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 9. EV Components分解
            plt.subplot(3, 3, 9)
            # 显示Economic Value的增长趋势信息
            if len(avg_ev_per_episode) > 20:
                early_avg = np.mean(avg_ev_per_episode[:20])
                late_avg = np.mean(avg_ev_per_episode[-20:])
                improvement = ((late_avg - early_avg) / max(abs(early_avg), 1e-8)) * 100
                
                info_text = f"EV Improvement\n"
                info_text += f"Early: {early_avg:.1f}\n"
                info_text += f"Late: {late_avg:.1f}\n"
                info_text += f"Change: {improvement:+.1f}%"
                
                plt.text(0.5, 0.5, info_text, 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=plt.gca().transAxes,
                        fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                plt.title('EV Performance Summary')
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('auction_sim/results/bc_ippo_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存: auction_sim/results/bc_ippo_training_curves.png")

    def train_complete_pipeline(self):
        """执行完整的BC+IPPO训练流程"""
        print("="*80)
        print("BC + IPPO TRAINING PIPELINE")
        print("独立PPO解决不对称问题")
        print("="*80)
        
        # 步骤1: BC预训练
        bc_trainer, bc_results, bc_metrics = self.step1_bc_pretraining()
        
        # 步骤2: IPPO独立训练
        ippo_trainer, learning_agents, rule_agents, training_metrics, individual_performance = self.step2_ippo_training()
        
        # 最终评估
        print("\n" + "="*70)
        print("最终评估")
        print("="*70)
        
        if training_metrics:
            final_metrics = training_metrics[-20:]  # 最后20个episodes
            final_avg_win_rate = np.mean([m['avg_win_rate'] for m in final_metrics])
            final_avg_roi = np.mean([m['avg_roi'] for m in final_metrics])
            final_budget_usage = np.mean([m['budget_usage_ratio'] for m in final_metrics])
            
            print(f"最终性能 (最后20 episodes平均):")
            print(f"  平均胜率: {final_avg_win_rate:.1%}")
            print(f"  平均ROI: {final_avg_roi:.1f}%")
            print(f"  预算使用: {final_budget_usage:.1%}")
            
            # 个体最终性能
            print(f"\n个体最终性能:")
            for agent_id, perf in individual_performance.items():
                final_wr = np.mean(perf['win_rates'][-20:])
                final_roi = np.mean(perf['rois'][-20:])
                print(f"  {agent_id}: 胜率 {final_wr:.1%}, ROI {final_roi:.0f}%")
            
            # 与理论期望对比
            theoretical_win_rate = 2 / 8  # 2个学习智能体在8个总智能体中
            print(f"\n对比分析:")
            print(f"  理论胜率: {theoretical_win_rate:.1%}")
            print(f"  实际平均胜率: {final_avg_win_rate:.1%}")
            print(f"  胜率效率: {final_avg_win_rate/theoretical_win_rate:.1%}")
        
        print(f"\n模型保存位置:")
        print(f"  最佳模型: {self.models_dir}/best")
        print(f"  最终模型: {self.models_dir}/final")
        print(f"  训练曲线: auction_sim/results/bc_ippo_training_curves.png")
        
        return {
            'bc_trainer': bc_trainer,
            'bc_results': bc_results,
            'bc_metrics': bc_metrics,
            'ippo_trainer': ippo_trainer,
            'learning_agents': learning_agents,
            'rule_agents': rule_agents,
            'training_metrics': training_metrics,
            'individual_performance': individual_performance
        }

def main():
    """主函数 - 执行BC+IPPO训练"""
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("开始BC + IPPO Training实验...")
    print("目标: 验证独立训练能否解决智能体不对称问题")
    
    # 创建trainer
    trainer = BCIPPOTrainer(
        use_bc_pretraining=True,    # 使用BC预训练
        bc_episodes=10,             # BC数据收集episodes
        ippo_episodes=300,          # IPPO训练episodes
        target_stage=CurriculumStage.STAGE_8,  # 完整8智能体环境
        force_collect_bc_data=False  # 是否强制重新收集BC数据
    )
    
    # 执行训练
    results = trainer.train_complete_pipeline()
    
    print("\n" + "="*80)
    print("BC + IPPO 实验完成！")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()