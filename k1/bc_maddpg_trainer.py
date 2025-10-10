#!/usr/bin/env python3
"""
BC + MADDPG Training: 行为克隆预训练后使用多智能体DDPG训练
中心化训练 + 分散执行，适合竞争性拍卖环境
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from collections import defaultdict, deque
import random

# Add auction_sim to path
sys.path.append('.')

from auction_sim import config
from auction_sim.config import CurriculumStage, CURRICULUM_CONFIGS
from auction_sim.bc_trainer import BCTrainer
from auction_sim.bc_data_collector import BCDataCollector
from auction_sim.ma_env import MultiAgentAuctionEnv
# from auction_sim.ma_environment import MultiAgentAuctionEnv
from auction_sim.agents import TruthfulAgent, ConservativeAgent, AggressiveAgent, MultiAgentLearningAgent

class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck过程噪声，用于连续动作探索"""
    
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        
    def reset(self):
        self.state = np.ones(self.size) * self.mu
        
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state.copy()

class Actor(nn.Module):
    """分散执行的Actor网络"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # 输出范围[0,1]，后续需要缩放
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 最后一层用小权重初始化
        with torch.no_grad():
            self.network[-2].weight.uniform_(-3e-3, 3e-3)
            self.network[-2].bias.uniform_(-3e-3, 3e-3)
    
    def forward(self, obs):
        out = self.network(obs)
        # 缩放到拍卖环境合理范围 [0.0, 1.5]，与ma_env.action_space一致
        return out * 1.5

class Critic(nn.Module):
    """DDPG的Critic网络，观察单个智能体的状态和动作"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # 直接拼接obs和action的输入维度
        input_dim = obs_dim + action_dim
        
        # Q值网络
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, obs, actions):
        # 拼接观察和动作
        x = torch.cat([obs, actions], dim=-1)  # [batch_size, obs_dim + action_dim]
        return self.network(x)  # [batch_size, 1]

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class MADDPGAgent:
    """单个DDPG智能体（保持类名以兼容现有接口）"""
    
    def __init__(self, agent_id, obs_dim, action_dim, total_obs_dim, total_action_dim,
                 lr_actor=1e-4, lr_critic=3e-4, gamma=0.95, tau=0.01, device=None):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device if device is not None else torch.device('cpu')
        
        # 网络 - 使用DDPG的自我输入Critic
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim, action_dim).to(self.device)  # 改为自我输入
        self.critic_target = Critic(obs_dim, action_dim).to(self.device)  # 改为自我输入
        
        # 复制参数到目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 噪声
        self.noise = OrnsteinUhlenbeckNoise(action_dim)
        
        # 损失记录
        self.actor_losses = []
        self.critic_losses = []
    
    def get_action(self, obs, add_noise=True, noise_scale=1.0):
        """获取动作"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actor(obs_tensor).squeeze(0).cpu().numpy()
            
            if add_noise:
                noise = self.noise.sample() * noise_scale
                action += noise
                action = np.clip(action, 0.0, 1.5)  # 与Actor输出和ma_env.action_space保持一致
            
            return action
    
    def load_bc_weights(self, bc_state_dict):
        """加载BC预训练权重"""
        try:
            # 尝试将BC权重映射到Actor网络
            actor_weights = {}
            for key, value in bc_state_dict.items():
                if 'actor' in key or 'backbone' in key:
                    # 简单的键名映射
                    new_key = key.replace('actor.', 'network.')
                    new_key = new_key.replace('backbone.', 'network.')
                    if new_key.startswith('network.'):
                        actor_weights[new_key] = value
            
            # 部分加载，忽略不匹配的层
            self.actor.load_state_dict(actor_weights, strict=False)
            self.actor_target.load_state_dict(self.actor.state_dict())
            return True
        except Exception as e:
            print(f"  ⚠️ {self.agent_id}: BC权重加载失败 ({e})")
            return False
    
    def soft_update(self):
        """软更新目标网络"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class MADDPGTrainer:
    """DDPG训练器（保持类名以兼容现有接口）"""
    
    def __init__(self, n_agents, obs_dim, action_dim, 
                 lr_actor=1e-4, lr_critic=3e-4, gamma=0.95, tau=0.01,
                 buffer_capacity=100000, batch_size=64):
        # DDPG模式断言
        assert n_agents == 1, "DDPG模式要求单学习体 (n_agents == 1)"
        
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # DDPG模式标志
        self.ddpg_mode = True
        
        # 全局维度（保持兼容性，但在DDPG模式下等于单体维度）
        self.total_obs_dim = obs_dim * n_agents
        self.total_action_dim = action_dim * n_agents
        
        # 创建智能体
        self.agents = {}
        for i in range(n_agents):
            agent_id = f"Learning_{i}"
            self.agents[agent_id] = MADDPGAgent(
                agent_id, obs_dim, action_dim, self.total_obs_dim, self.total_action_dim,
                lr_actor, lr_critic, gamma, tau, device=self.device
            )
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': {agent_id: [] for agent_id in self.agents.keys()},
            'actor_losses': {agent_id: [] for agent_id in self.agents.keys()},
            'critic_losses': {agent_id: [] for agent_id in self.agents.keys()}
        }
    
    def get_action(self, agent_id, obs, add_noise=True, noise_scale=1.0):
        """获取指定智能体的动作"""
        return self.agents[agent_id].get_action(obs, add_noise, noise_scale)
    
    def store_transition(self, states, actions, rewards, next_states, dones):
        """存储转换到经验回放缓冲区（DDPG单智能体模式）"""
        # 获取唯一智能体ID
        agent_id = list(self.agents.keys())[0]
        
        # 在DDPG模式下，只存储唯一智能体的经验
        # 但保持全局格式以兼容现有ReplayBuffer
        state = states[agent_id]  # [obs_dim]
        action = actions[agent_id]  # [action_dim] 或 标量
        reward = rewards[agent_id]  # 标量
        next_state = next_states[agent_id]  # [obs_dim]
        done = dones[agent_id]  # 布尔值
        
        # 确保动作格式正确
        if isinstance(action, np.ndarray) and len(action) > 0:
            action_val = action[0] if action.ndim > 0 else float(action)
        else:
            action_val = float(action)
        
        # 存储为兼容格式（保持原有维度以兼容ReplayBuffer）
        self.replay_buffer.push(
            state,  # [obs_dim]
            np.array([action_val]),  # [action_dim]
            reward,  # 标量
            next_state,  # [obs_dim]
            done  # 布尔值
        )
    
    def update_agents(self):
        """DDPG单智能体更新"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为tensor并移到设备
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)
        
        # 获取唯一智能体
        agent_id = list(self.agents.keys())[0]
        agent = self.agents[agent_id]
        
        # 自我切片（在DDPG模式下，全局状态就是自我状态）
        s_i = states[:, :self.obs_dim]  # [batch_size, obs_dim]
        a_i = actions[:, :self.action_dim]  # [batch_size, action_dim]
        ns_i = next_states[:, :self.obs_dim]  # [batch_size, obs_dim]
        r_i = rewards.view(-1, 1) if rewards.ndim == 1 else rewards[:, :1]  # [batch_size, 1]
        
        # === Critic更新 ===
        with torch.no_grad():
            # 使用目标actor计算下一个动作
            next_action = agent.actor_target(ns_i)  # [batch_size, action_dim]
            # 使用目标critic计算目标Q值
            target_q = agent.critic_target(ns_i, next_action)  # [batch_size, 1]
            # 计算TD目标
            y = r_i + self.gamma * (1 - dones) * target_q  # [batch_size, 1]
        
        # 当前Q值
        current_q = agent.critic(s_i, a_i)  # [batch_size, 1]
        critic_loss = nn.MSELoss()(current_q, y)
        
        # 更新Critic
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        agent.critic_optimizer.step()
        
        # === Actor更新 ===
        # 使用当前actor生成动作
        policy_action = agent.actor(s_i)  # [batch_size, action_dim]
        # 计算actor损失（确定性策略梯度）
        actor_loss = -agent.critic(s_i, policy_action).mean()
        
        # 更新Actor
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
        agent.actor_optimizer.step()
        
        # 软更新目标网络
        agent.soft_update()
        
        # 记录损失
        agent.actor_losses.append(actor_loss.item())
        agent.critic_losses.append(critic_loss.item())
    
    def train_episode(self, env, max_steps=1000):
        """训练一个episode"""
        states = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in self.agents.keys()}
        
        # 重置噪声
        for agent in self.agents.values():
            agent.noise.reset()
        
        for step in range(max_steps):
            # 获取动作
            actions = {}
            for agent_id in self.agents.keys():
                action = self.get_action(agent_id, states[agent_id], add_noise=True, 
                                       noise_scale=max(0.1, 1.0 - step/max_steps))  # 逐渐降低噪声
                # 环境期望数组格式 actions[agent_id][0]
                if isinstance(action, (float, np.float32, np.float64)):
                    actions[agent_id] = np.array([action])
                else:
                    actions[agent_id] = action
            
            # 执行动作
            step_result = env.step(actions)
            if len(step_result) == 5:
                next_states, rewards, terminated, truncated, infos = step_result
                dones = {k: terminated[k] or truncated[k] for k in terminated.keys()}
            else:
                next_states, rewards, dones, infos = step_result
            
            # 存储经验
            self.store_transition(states, actions, rewards, next_states, dones)
            
            # 更新统计
            for agent_id in self.agents.keys():
                episode_rewards[agent_id] += rewards[agent_id]
            
            states = next_states
            
            # 更新网络
            if step % 4 == 0:  # 每4步更新一次
                self.update_agents()
            
            if any(dones.values()):
                break
        
        # 记录episode统计
        for agent_id in self.agents.keys():
            self.training_stats['episode_rewards'][agent_id].append(episode_rewards[agent_id])
        
        # episode_wins不再使用，返回空dict保持接口一致
        return episode_rewards, {agent_id: 0 for agent_id in self.agents.keys()}
    
    def save_models(self, save_dir):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            torch.save(agent.actor.state_dict(), f"{save_dir}/{agent_id}_actor.pth")
            torch.save(agent.critic.state_dict(), f"{save_dir}/{agent_id}_critic.pth")
        
        print(f"MADDPG模型已保存到: {save_dir}")
    
    def load_models(self, save_dir):
        """加载训练好的模型"""
        for agent_id, agent in self.agents.items():
            actor_path = f"{save_dir}/{agent_id}_actor.pth"
            critic_path = f"{save_dir}/{agent_id}_critic.pth"
            
            if os.path.exists(actor_path):
                agent.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                agent.actor_target.load_state_dict(agent.actor.state_dict())
                print(f"已加载 {agent_id} 的 Actor 模型")
            else:
                print(f"警告: 找不到 {agent_id} 的 Actor 模型文件: {actor_path}")
            
            if os.path.exists(critic_path):
                agent.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                agent.critic_target.load_state_dict(agent.critic.state_dict())
                print(f"已加载 {agent_id} 的 Critic 模型")
            else:
                print(f"警告: 找不到 {agent_id} 的 Critic 模型文件: {critic_path}")
        
        print(f"MADDPG模型加载完成: {save_dir}")

class BCMADDPGTrainer:
    """BC预训练 + MADDPG训练的混合trainer"""
    
    def __init__(self,
                 use_bc_pretraining: bool = True,
                 bc_episodes: int = 10,
                 maddpg_episodes: int = 300,
                 target_stage: CurriculumStage = CurriculumStage.STAGE_8):
        
        self.use_bc_pretraining = use_bc_pretraining
        self.bc_episodes = bc_episodes
        self.maddpg_episodes = maddpg_episodes
        self.target_stage = target_stage
        
        # 路径配置
        self.bc_model_path = "auction_sim/models/bc_pretrained.pth"
        self.models_dir = "auction_sim/models/bc_maddpg"
        os.makedirs(self.models_dir, exist_ok=True)
        
        print(f"BC + MADDPG Trainer初始化:")
        print(f"  BC预训练: {use_bc_pretraining}")
        print(f"  MADDPG训练: {maddpg_episodes} episodes")
        print(f"  目标环境: {target_stage.name}")
        print(f"  关键特性: 中心化训练 + 分散执行")
    
    def step1_bc_pretraining(self):
        """步骤1: BC预训练（9维观测）"""
        if not self.use_bc_pretraining:
            return None, None, None
        
        print("\n" + "="*70)
        print("步骤1: BC预训练")
        print("="*70)
        
        bc_data_path = "auction_sim/bc_dataset.pkl"
        
        # 检查现有BC模型
        if os.path.exists(self.bc_model_path):
            print(f"发现已存在的BC模型: {self.bc_model_path}")
            
            try:
                # 创建临时trainer检查数据维度
                temp_trainer = BCTrainer(obs_dim=9, lr=1e-3)
                is_valid, current_dim = temp_trainer._check_data_dimension(bc_data_path)
                
                if is_valid:
                    print(f"✅ BC数据维度匹配（{current_dim}维）")
                    bc_state_dict = torch.load(self.bc_model_path)
                    print(f"✅ BC模型加载成功，包含 {len(bc_state_dict)} 个参数")
                    print("跳过BC预训练，直接使用现有模型")
                    return None, {'status': 'loaded_existing'}, {'model_path': self.bc_model_path}
                else:
                    print(f"⚠️  BC数据维度不匹配（期望9维，实际{current_dim}维），将重新训练")
            except Exception as e:
                print(f"❌ BC模型加载失败: {e}")
                print("将重新进行BC预训练")
        
        # BC训练 - BCTrainer会自动检查和准备数据
        print("\n开始BC预训练...")
        bc_trainer = BCTrainer(obs_dim=9, lr=1e-3)
        bc_results = bc_trainer.train(
            dataset_path=bc_data_path,
            n_epochs=30,  # 增加到30个epoch以获得更好的BC预训练效果
            batch_size=64,
            save_path=self.bc_model_path,
            n_collect_episodes=self.bc_episodes,
            auto_prepare_data=True  # 自动检查和准备9维BC数据
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
    
    def step2_maddpg_training(self):
        """步骤2: MADDPG训练"""
        print("\n" + "="*70)
        print("步骤2: MADDPG训练")
        print("="*70)
        
        # 环境配置
        stage_config = CURRICULUM_CONFIGS[self.target_stage]
        print(f"目标环境: {stage_config['description']}")
        
        # 创建智能体和环境
        rule_agents = self._create_rule_agents(stage_config)
        learning_agent_ids = [f"Learning_{i}" for i in range(stage_config['n_learning'])]
        
        env = MultiAgentAuctionEnv(
            learning_agent_ids,
            rule_agents,
            curriculum_stage=self.target_stage
        )
        
        # 创建DDPG trainer（单智能体模式）
        maddpg = MADDPGTrainer(
            n_agents=1,  # DDPG模式：强制为1
            obs_dim=9,  # 更新为9维观测（包含近期平均赢价和CPC）
            action_dim=1,
            lr_actor=1e-4,
            lr_critic=3e-4,
            gamma=0.95,
            tau=0.01,
            buffer_capacity=100000,
            batch_size=64
        )
        
        print(f"\n✨ DDPG关键特性:")
        print("  - 单智能体Critic：只观察自身状态和动作")
        print("  - 单智能体Actor：独立决策和学习")
        print("  - 连续动作空间：适合拍卖出价场景")
        print("  - OU噪声探索：比随机噪声更适合连续控制")
        
        # BC权重初始化
        if self.use_bc_pretraining and os.path.exists(self.bc_model_path):
            print("\n加载BC预训练权重...")
            bc_state_dict = torch.load(self.bc_model_path)
            
            for agent_id, agent in maddpg.agents.items():
                success = agent.load_bc_weights(bc_state_dict)
                if success:
                    print(f"  ✅ {agent_id}: BC权重加载成功")
        
        # 创建环境接口
        learning_agents = []
        for agent_id in learning_agent_ids:
            agent = MultiAgentLearningAgent(
                agent_id=agent_id,
                budget=stage_config['budget'],
                perception_noise_std=config.AGENT_PERCEPTION_NOISE_STD,
                model=None,
                is_training=True
            )
            
            # 包装MADDPG agent
            class MADDPGWrapper:
                def __init__(self, maddpg_trainer, agent_id):
                    self.maddpg_trainer = maddpg_trainer
                    self.agent_id = agent_id
                
                def predict(self, obs, deterministic=False):
                    action = self.maddpg_trainer.get_action(
                        self.agent_id, obs, 
                        add_noise=not deterministic
                    )
                    # 确保返回数组格式
                    if isinstance(action, (float, np.float32, np.float64)):
                        return np.array([action]), None
                    else:
                        return action, None
            
            agent.set_model(MADDPGWrapper(maddpg, agent_id))
            learning_agents.append(agent)
        
        # 训练循环
        print(f"\n开始MADDPG训练 {self.maddpg_episodes} episodes...")
        
        individual_performance = {
            agent_id: {
                'rewards': [],
                'win_rates': [],
                'rois': [],
                'budget_usage': []
            } for agent_id in learning_agent_ids
        }
        
        episode_metrics = []
        best_avg_reward = float('-inf')
        
        pbar = tqdm(range(self.maddpg_episodes), desc="MADDPG Training")
        
        for episode in pbar:
            # 训练一个episode
            ep_rewards, ep_wins = maddpg.train_episode(env, max_steps=stage_config['max_rounds'])
            
            # 计算统计
            total_reward = sum(ep_rewards.values())
            episode_stats = self._calculate_episode_stats(env, learning_agent_ids, stage_config, ep_rewards, ep_wins)
            episode_metrics.append(episode_stats)
            
            # 记录个体表现
            for agent_id in learning_agent_ids:
                if agent_id in episode_stats['individual_stats']:
                    stats = episode_stats['individual_stats'][agent_id]
                    individual_performance[agent_id]['rewards'].append(ep_rewards.get(agent_id, 0))
                    individual_performance[agent_id]['win_rates'].append(stats['win_rate'])
                    individual_performance[agent_id]['rois'].append(stats['roi'])
                    individual_performance[agent_id]['budget_usage'].append(stats['budget_usage'])
            
            # 更新进度条
            if len(episode_metrics) > 0:
                last = episode_metrics[-1]
                pbar.set_postfix({
                    'Reward': f"{total_reward:.0f}",
                    'WinRate': f"{last['avg_win_rate']:.1%}",
                    'ROI': f"{last['avg_roi']:.0f}%",
                    'Budget': f"{last['budget_usage_ratio']:.1%}",
                    'L0_WR': f"{last['individual_stats'].get('Learning_0', {}).get('win_rate', 0):.1%}"
                })
            
            # 保存最佳模型
            if total_reward > best_avg_reward:
                best_avg_reward = total_reward
                maddpg.save_models(f"{self.models_dir}/best")
            
            # 定期保存
            if (episode + 1) % 50 == 0:
                maddpg.save_models(f"{self.models_dir}/checkpoint_ep{episode+1}")
                self._print_progress(episode + 1, episode_metrics[-20:], individual_performance)
        
        pbar.close()
        
        # 保存最终模型
        maddpg.save_models(f"{self.models_dir}/final")
        
        # === 新增：绘制训练曲线（与 IPPO 对齐） ===
        self._plot_training_curves(episode_metrics, individual_performance, tag="ddpg")
        
        # 生成分析
        self._analyze_results(individual_performance)
        
        return maddpg, learning_agents, rule_agents, episode_metrics, individual_performance
    
    def _create_rule_agents(self, stage_config):
        """创建规则智能体"""
        rule_agents = []
        agent_id_counter = 0
        
        for _ in range(stage_config.get('n_truthful', 0)):
            agent = TruthfulAgent(
                f"Truthful_{agent_id_counter}",
                stage_config['budget'],
                config.AGENT_PERCEPTION_NOISE_STD
            )
            rule_agents.append(agent)
            agent_id_counter += 1
        
        for _ in range(stage_config.get('n_conservative', 0)):
            agent = ConservativeAgent(
                f"Conservative_{agent_id_counter}",
                stage_config['budget'],
                config.AGENT_PERCEPTION_NOISE_STD,
                stage_config['max_rounds']
            )
            rule_agents.append(agent)
            agent_id_counter += 1
        
        for _ in range(stage_config.get('n_aggressive', 0)):
            total_agents = sum([stage_config.get(f'n_{t}', 0) for t in ['learning', 'truthful', 'conservative', 'aggressive']])
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
        """计算episode统计"""
        stats = {
            'avg_win_rate': 0.0,
            'avg_roi': 0.0,
            'budget_usage_ratio': 0.0,
            'individual_stats': {}
        }
        
        for agent_id in learning_agent_ids:
            agent_stats = {
                'win_rate': 0.0,
                'roi': -100.0,
                'budget_usage': 0.0
            }
            
            # 计算预算使用率
            initial_budget = stage_config['budget']
            remaining = env.agent_budgets.get(agent_id, initial_budget)
            used = initial_budget - remaining
            agent_stats['budget_usage'] = used / initial_budget
            
            # 计算胜率和ROI
            if hasattr(env, 'agent_histories') and agent_id in env.agent_histories:
                history = env.agent_histories[agent_id]
                if history:
                    total_profit = sum(h.get('profit', 0) for h in history)
                    total_cost = sum(h.get('cost', 0) for h in history)
                    wins = sum(1 for h in history if h.get('won', False))
                    
                    # 统一胜率计算口径（与IPPO一致）
                    agent_stats['win_rate'] = wins / len(history) if history else 0.0
                    if total_cost > 0:
                        agent_stats['roi'] = (total_profit / total_cost) * 100
            
            stats['individual_stats'][agent_id] = agent_stats
            stats['avg_win_rate'] += agent_stats['win_rate']
            stats['avg_roi'] += agent_stats['roi']
            stats['budget_usage_ratio'] += agent_stats['budget_usage']
        
        # 平均化
        n_agents = len(learning_agent_ids)
        if n_agents > 0:
            stats['avg_win_rate'] /= n_agents
            stats['avg_roi'] /= n_agents
            stats['budget_usage_ratio'] /= n_agents
        
        return stats
    
    def _print_progress(self, episode, recent_metrics, individual_performance):
        """打印训练进度"""
        avg_win_rate = np.mean([m['avg_win_rate'] for m in recent_metrics])
        avg_roi = np.mean([m['avg_roi'] for m in recent_metrics])
        avg_budget = np.mean([m['budget_usage_ratio'] for m in recent_metrics])
        
        print(f"\nEpisode {episode}/{self.maddpg_episodes} (最近20集平均):")
        print(f"  平均胜率: {avg_win_rate:.1%}")
        print(f"  平均ROI: {avg_roi:.1f}%")
        print(f"  预算使用: {avg_budget:.1%}")
        
        # 个体表现
        print("\n  🔍 个体表现对比 (MADDPG中心化训练):")
        for agent_id in individual_performance.keys():
            recent_wrs = individual_performance[agent_id]['win_rates'][-20:]
            recent_rois = individual_performance[agent_id]['rois'][-20:]
            
            if recent_wrs:
                print(f"    {agent_id}: 胜率 {np.mean(recent_wrs):.1%}, ROI {np.mean(recent_rois):.0f}%")
        
        # 不对称度
        if len(individual_performance) == 2:
            agents = list(individual_performance.keys())
            wr_diff = abs(np.mean(individual_performance[agents[0]]['win_rates'][-20:]) - 
                         np.mean(individual_performance[agents[1]]['win_rates'][-20:]))
            print(f"\n  📊 不对称度: 胜率差异 {wr_diff:.1%}")
    
    def _plot_training_curves(self, episode_metrics, individual_performance, tag="ddpg"):
        """绘制训练曲线（与IPPO对齐）"""
        import numpy as np
        import matplotlib.pyplot as plt
        os.makedirs('auction_sim/results', exist_ok=True)

        def moving_average(data, window=10):
            if len(data) == 0:
                return data
            return [np.mean(data[max(0, i-window+1):i+1]) for i in range(len(data))]

        episodes = range(1, len(episode_metrics) + 1)

        # 提取全局曲线
        avg_wr   = [m['avg_win_rate'] for m in episode_metrics]
        avg_roi  = [m['avg_roi'] for m in episode_metrics]
        avg_budg = [m['budget_usage_ratio'] for m in episode_metrics]

        plt.figure(figsize=(18, 10))

        # 1. 平均胜率
        plt.subplot(2, 3, 1)
        plt.plot(episodes, avg_wr, alpha=0.3, label='Raw')
        plt.plot(episodes, moving_average(avg_wr, 10), linewidth=2, label='MA-10')
        plt.title('Average Win Rate (DDPG)')
        plt.xlabel('Episode'); plt.ylabel('Win Rate'); plt.grid(True, alpha=0.3); plt.legend()

        # 2. 平均 ROI
        plt.subplot(2, 3, 2)
        plt.plot(episodes, avg_roi, alpha=0.3, label='Raw')
        plt.plot(episodes, moving_average(avg_roi, 10), linewidth=2, label='MA-10')
        plt.title('Average ROI (DDPG)')
        plt.xlabel('Episode'); plt.ylabel('ROI (%)'); plt.grid(True, alpha=0.3); plt.legend()

        # 3. 平均预算使用
        plt.subplot(2, 3, 3)
        plt.plot(episodes, avg_budg, alpha=0.3, label='Raw')
        plt.plot(episodes, moving_average(avg_budg, 10), linewidth=2, label='MA-10')
        plt.title('Average Budget Usage (DDPG)')
        plt.xlabel('Episode'); plt.ylabel('Usage Ratio'); plt.grid(True, alpha=0.3); plt.legend()

        # 4~6. 个体曲线（若有多个学习体，也能画；单体就一条）
        colors = ['C0', 'C1', 'C2', 'C3']
        # 个体胜率
        plt.subplot(2, 3, 4)
        for i, (aid, perf) in enumerate(individual_performance.items()):
            wr = perf['win_rates']
            plt.plot(episodes, wr, alpha=0.3, color=colors[i % len(colors)])
            plt.plot(episodes, moving_average(wr, 10), linewidth=2, color=colors[i % len(colors)], label=aid)
        plt.title('Individual Win Rates (DDPG)'); plt.xlabel('Episode'); plt.ylabel('Win Rate'); plt.grid(True, alpha=0.3); plt.legend()

        # 个体 ROI
        plt.subplot(2, 3, 5)
        for i, (aid, perf) in enumerate(individual_performance.items()):
            rois = perf['rois']
            plt.plot(episodes, rois, alpha=0.3, color=colors[i % len(colors)])
            plt.plot(episodes, moving_average(rois, 10), linewidth=2, color=colors[i % len(colors)], label=aid)
        plt.title('Individual ROI (DDPG)'); plt.xlabel('Episode'); plt.ylabel('ROI (%)'); plt.grid(True, alpha=0.3); plt.legend()

        # 个体预算使用
        plt.subplot(2, 3, 6)
        for i, (aid, perf) in enumerate(individual_performance.items()):
            bu = perf['budget_usage']
            plt.plot(episodes, bu, alpha=0.3, color=colors[i % len(colors)])
            plt.plot(episodes, moving_average(bu, 10), linewidth=2, color=colors[i % len(colors)], label=aid)
        plt.title('Budget Usage (DDPG)'); plt.xlabel('Episode'); plt.ylabel('Usage Ratio'); plt.grid(True, alpha=0.3); plt.legend()

        out_path = f'auction_sim/results/bc_{tag}_training_curves.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存: {out_path}")

    def _analyze_results(self, individual_performance):
        """分析MADDPG结果"""
        print("\n" + "="*70)
        print("MADDPG结果分析")
        print("="*70)
        
        if len(individual_performance) == 2:
            agents = list(individual_performance.keys())
            
            # 最终性能
            final_episodes = 50
            agent0_final_wr = np.mean(individual_performance[agents[0]]['win_rates'][-final_episodes:])
            agent1_final_wr = np.mean(individual_performance[agents[1]]['win_rates'][-final_episodes:])
            
            agent0_final_roi = np.mean(individual_performance[agents[0]]['rois'][-final_episodes:])
            agent1_final_roi = np.mean(individual_performance[agents[1]]['rois'][-final_episodes:])
            
            print(f"\n最终性能对比 (最后{final_episodes}集):")
            print(f"  {agents[0]}: 胜率 {agent0_final_wr:.1%}, ROI {agent0_final_roi:.0f}%")
            print(f"  {agents[1]}: 胜率 {agent1_final_wr:.1%}, ROI {agent1_final_roi:.0f}%")
            
            wr_diff = abs(agent0_final_wr - agent1_final_wr)
            print(f"\n不对称度指标:")
            print(f"  胜率差异: {wr_diff:.1%}")
            
            # 算法对比
            print("\n算法对比:")
            print("  MAPPO: 胜率差异 ~35%")
            print("  IPPO:  胜率差异 20.1%")
            print(f"  MADDPG: 胜率差异 {wr_diff:.1%}")
            
            if wr_diff < 0.15:
                print("\n✅ MADDPG在解决不对称问题方面表现良好!")
            else:
                print("\n⚠️ MADDPG仍需进一步调优")
    
    def train_complete_pipeline(self):
        """执行完整训练流程"""
        print("="*80)
        print("BC + MADDPG TRAINING PIPELINE")
        print("中心化训练 + 分散执行")
        print("="*80)
        
        # 步骤1: BC预训练
        bc_trainer, bc_results, bc_metrics = self.step1_bc_pretraining()
        
        # 步骤2: MADDPG训练
        results = self.step2_maddpg_training()
        
        return {
            'bc_metrics': bc_metrics,
            'maddpg_results': results
        }

def main():
    """主函数"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("开始BC + MADDPG Training实验...")
    print("目标: 测试中心化训练能否更好地解决不对称问题")
    
    trainer = BCMADDPGTrainer(
        use_bc_pretraining=True,
        bc_episodes=10,
        maddpg_episodes=300,
        target_stage=CurriculumStage.STAGE_8
    )
    
    results = trainer.train_complete_pipeline()
    
    print("\n" + "="*80)
    print("BC + MADDPG 实验完成！")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()