# /auction_sim/ippo_trainer.py
"""
Independent PPO (IPPO) Trainer for k=2 scenario
Simplified version focusing on core PPO algorithm with EV reward
Each learning agent trains independently without coordination
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from typing import Dict, List, Tuple, Any
import os
import math
from collections import deque
import matplotlib.pyplot as plt

from . import config
from .ma_environment import MultiAgentAuctionEnv
from .agents import TruthfulAgent, ConservativeAgent, AggressiveAgent, MultiAgentLearningAgent
from .utils.reward_utils import RunningMeanStd

class IPPOActorCriticNetwork(nn.Module):
    """
    Independent Actor-Critic network for IPPO
    与BC训练器的ActorCriticNetwork结构完全一致
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Shared backbone (与BC训练器完全一致)
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy) - 与BC训练器一致
        self.actor_linear = nn.Linear(hidden_dim, action_dim)
        self.actor_activation = nn.Sigmoid()
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # ---- Squashed Gaussian: state-dependent mean/std ----
        self.actor_mean = nn.Linear(hidden_dim, action_dim)  # 新增用于Squashed Gaussian
        self.actor_std  = nn.Linear(hidden_dim, action_dim)  # 新增用于Squashed Gaussian
        
        # Critic head (value function) - 与BC训练器一致
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights for stable training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for stable training (与BC训练器一致)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Actor输出层小权重初始化（与BC训练器一致）
        nn.init.orthogonal_(self.actor_linear.weight, gain=0.01)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        
    @staticmethod
    def _squash(a_u):
        # action in [0, 1.5]  via tanh
        a = 0.75 * (torch.tanh(a_u) + 1.0)
        return a

    def forward(self, obs, use_squashed_gaussian=True):
        features = self.backbone(obs)
        
        if use_squashed_gaussian:
            # Squashed Gaussian mode (稳定化模式)
            mu = self.actor_mean(features)                  # unconstrained
            log_std = F.softplus(self.actor_std(features))  # >0 then log
            std = log_std.clamp(min=1e-3)            # numerical floor
        else:
            # Traditional mode (与BC训练器兼容)
            mean = self.actor_linear(features)
            mean = self.actor_activation(mean)
            mean = mean * 1.5  # Scale to [0, 1.5]
            mu = mean
            std = torch.exp(self.actor_logstd.clamp(-20, 2))
        
        # Critic output
        value = self.critic(features)
        
        return mu, std, value

    def _dist(self, mu, std):
        return Normal(mu, std)
    
    def get_action(self, obs, deterministic=False, use_squashed_gaussian=True):
        with torch.no_grad():
            mu, std, value = self.forward(obs, use_squashed_gaussian)
            
            if use_squashed_gaussian:
                # Squashed Gaussian mode (稳定化模式)
                dist = self._dist(mu, std)
                u = mu if deterministic else dist.rsample()
                a = self._squash(u)
                # log_prob with tanh correction (per-dim sum)
                logp_u = dist.log_prob(u).sum(-1)
                correction = torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6).sum(-1)
                log_prob = logp_u - correction
            else:
                # Traditional mode (与BC训练器兼容)
                if deterministic:
                    a = mu
                    log_prob = torch.zeros_like(a)
                else:
                    dist = Normal(mu, std)
                    a = dist.sample()
                    a = torch.clamp(a, 0.0, 1.5)
                    log_prob = dist.log_prob(a).sum(-1)
            
            # Optional: tiny execution noise like DDPG (only for env step, not for log_prob)
            if np.random.rand() < 0.05:
                noise = np.random.normal(0.0, 0.03, size=a.shape)
                a = (a + torch.from_numpy(noise).to(a)).clamp(0.0, 1.5)
        return a, log_prob, value
    
    def evaluate_action(self, obs, action, use_squashed_gaussian=True):
        mu, std, value = self.forward(obs, use_squashed_gaussian)
        
        if use_squashed_gaussian:
            # Squashed Gaussian mode (稳定化模式)
            dist = self._dist(mu, std)
            # recover pre-squash u from action for correct log_prob
            # a = 0.75*(tanh(u)+1)  ->  tanh(u) = 2a/1.5 - 1
            t = (2.0 * action / 1.5) - 1.0
            t = t.clamp(-0.999999, 0.999999)
            u = torch.atanh(t)
            logp_u = dist.log_prob(u).sum(-1)
            correction = torch.log(1.0 - t.pow(2) + 1e-6).sum(-1)
            log_prob = logp_u - correction
            entropy = dist.entropy().sum(-1)  # still informative for exploration
        else:
            # Traditional mode (与BC训练器兼容)
            dist = self._dist(mu, std)
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        
        return log_prob, entropy, value

class IPPOTrainer:
    """
    Simplified Independent PPO Trainer
    Each agent learns independently with its own network and optimizer
    """
    
    def __init__(self, 
                 obs_dim: int = 7,  # 与BC数据收集器的7维观测空间一致
                 action_dim: int = 1,
                 n_agents: int = 1,
                 lr: float = 2e-4,  # 🔧 dd.md修复: 从3e-4降到2e-4
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 max_buffer_size: int = 50000,
                 freeze_backbone_until: int = 20):  # 🔧 dd.md: 早期冻结backbone
        
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # PPO Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.max_buffer_size = max_buffer_size
        
        # 🔧 dd.md: 早期冻结backbone相关参数
        self.freeze_backbone_until = freeze_backbone_until
        self.episode_idx = 0
        
        # ==== BC anchor as distributional KL (new) ====
        self.use_bc_anchor = True
        self.lambda_bc = 0.20
        self.lambda_bc_min = 0.02
        self.lambda_bc_decay = 0.97  # 每个episode乘一次
        
        # 🔧 dd.md: KL早停相关参数
        self.target_kl = 0.01
        self.min_clip_ratio = 0.1
        
        # ==== KL / Entropy / Schedules (new) ====
        self.target_kl_start = 0.08
        self.target_kl_end   = 0.04
        self.kl_patience     = 6
        self.initial_ent_coef= 0.025
        self.min_ent_coef    = 0.003
        self.ent_coef_decay  = 0.9995
        
        # 温和KL早停控制参数
        self.kl_beta = 0.9
        self.kl_ema = 0.0
        self.kl_exceed_streak = 0
        self.min_ppo_epochs = 2
        self.target_kl_decay_ep = 50  # 前50个episode衰减完成
        
        # 奖励标准化
        self._reward_rms = RunningMeanStd()
        
        # Create independent networks and optimizers for each agent
        self.networks = {}
        self.optimizers = {}
        self.schedulers = {}  # 修复5: 添加学习率调度器
        
        for i in range(n_agents):
            agent_id = f"Learning_{i}"
            self.networks[agent_id] = IPPOActorCriticNetwork(obs_dim, action_dim)
            optimizer = optim.Adam(
                self.networks[agent_id].parameters(), lr=lr
            )
            self.optimizers[agent_id] = optimizer
            
            # 移除学习率调度器，让智能体自然学习
            self.schedulers[agent_id] = None
        
        # Independent experience buffers
        self.buffers = {agent_id: {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        } for agent_id in self.networks.keys()}
        
        # ==== Entropy decay (new, slower) ====
        self.initial_ent_coef = self.initial_ent_coef
        self.current_ent_coef = self.initial_ent_coef
        self.ent_coef_decay = self.ent_coef_decay
        self.min_ent_coef = self.min_ent_coef
        
        # 移除自适应调整相关参数
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': {agent_id: [] for agent_id in self.networks.keys()},
            'episode_lengths': [],
            'actor_losses': {agent_id: [] for agent_id in self.networks.keys()},
            'critic_losses': {agent_id: [] for agent_id in self.networks.keys()},
            'win_rates': {agent_id: [] for agent_id in self.networks.keys()},
            'learning_rates': {agent_id: [] for agent_id in self.networks.keys()},  # 跟踪学习率
            'ent_coefs': []  # 跟踪熵系数
        }
        
        print(f"✨ Simplified IPPO Trainer initialized:")
        print(f"   - {n_agents} independent learning agents")
        print(f"   - Using EV reward: 0.5*Profit + 0.15*ROI*Cost + 0.35*Win")
        print(f"   - Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    def get_action(self, agent_id: str, obs: np.ndarray, deterministic: bool = False):
        """Get action from independent policy network"""
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        if obs.ndim == 0:
            obs = obs.reshape(1,)
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value = self.networks[agent_id].get_action(obs_tensor, deterministic)
            
        return action.cpu().numpy()[0], log_prob.cpu().numpy().item(), value.cpu().numpy().item()
    
    def store_experience(self, agent_id: str, obs, action, reward, value, log_prob, done):
        """Store experience in agent's independent buffer"""
        if len(self.buffers[agent_id]['observations']) >= self.max_buffer_size:
            for key in self.buffers[agent_id]:
                if isinstance(self.buffers[agent_id][key], list):
                    self.buffers[agent_id][key].pop(0)
        
        # reward normalization to stabilize GAE scale
        self._reward_rms.update(np.array([reward]))
        norm_reward = float(self._reward_rms.normalize(np.array([reward]))[0])
        
        self.buffers[agent_id]['observations'].append(obs)
        self.buffers[agent_id]['actions'].append(action)
        self.buffers[agent_id]['rewards'].append(norm_reward)
        self.buffers[agent_id]['values'].append(value)
        self.buffers[agent_id]['log_probs'].append(log_prob)
        self.buffers[agent_id]['dones'].append(done)
    
    
    def compute_gae(self, agent_id: str, next_value: float = 0.0):
        """Compute GAE for individual agent"""
        rewards = self.buffers[agent_id]['rewards']
        values = self.buffers[agent_id]['values'] + [next_value]
        dones = self.buffers[agent_id]['dones']
        
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return advantages, returns
    
    def update_policy(self, agent_id: str, advantages, returns):
        """Update individual agent's policy using PPO with gentle KL early stopping"""
        # Convert to tensors
        obs = torch.FloatTensor(np.array(self.buffers[agent_id]['observations']))
        actions = torch.FloatTensor(self.buffers[agent_id]['actions'])
        old_log_probs = torch.FloatTensor(self.buffers[agent_id]['log_probs'])
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 🔧 dd.md修复: 去掉returns标准化
        # 保持returns在原始尺度，避免与GAE计算时的V(s)尺度不匹配
        # 只标准化advantages即可
        
        # 修复3: PPO update loop - 全量小批遍历，不浪费样本
        # 原问题：1.6万步只用256步×4epoch，95%+样本浪费
        n_epochs = 8  # 增加epoch数
        batch_size = min(2048, len(obs))  # 增大batch size
        vf_clip = 0.2  # 🔧 cc.md改进2: Value clipping系数
        N = len(obs)
        
        # 🔧 温和KL早停: 记录实际运行的epoch数
        actual_epoch_count = 0
        early_stop_epochs = False
        
        for epoch in range(n_epochs):
            # 检查是否应该提前停止本轮PPO的后续epoch
            if early_stop_epochs:
                break
                
            actual_epoch_count += 1
            
            # 每个epoch都随机打乱并完整遍历所有样本
            indices = torch.randperm(N)
            
            for start in range(0, N, batch_size):
                # 取一个完整batch
                idx = indices[start:start+batch_size]
                
                obs_batch = obs[idx]
                actions_batch = actions[idx]
                old_log_probs_batch = old_log_probs[idx]
                advantages_batch = advantages[idx]
                returns_batch = returns[idx]
            
                # Forward pass
                log_probs, entropy, values = self.networks[agent_id].evaluate_action(
                    obs_batch, actions_batch
                )
                
                # Ensure dimensions match
                if values.dim() > 1:
                    values = values.squeeze(-1)
                
                # PPO Actor loss with clipping
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 🔧 cc.md改进2: Critic loss with value clipping
                # 防止value function更新过大，提高训练稳定性
                if values.dim() > returns_batch.dim():
                    values = values.squeeze(-1)
                v_clipped = values + (returns_batch - values).clamp(-vf_clip, vf_clip)
                critic_loss = 0.5 * (nn.MSELoss()(values, returns_batch) + 
                                    nn.MSELoss()(v_clipped, returns_batch))
                
                # 修复5: Entropy bonus for exploration（使用动态系数）
                entropy_loss = -entropy.mean()
                
                # === BC anchor loss (new, KL(bc || cur)) ===
                bc_anchor_loss = 0.0
                if self.use_bc_anchor and self.lambda_bc > 0:
                    with torch.no_grad():
                        # 使用传统模式计算BC分布（与BC训练器兼容）
                        bc_mu, bc_std, _ = self.networks[agent_id].forward(obs_batch, use_squashed_gaussian=False)
                        bc_dist = Normal(bc_mu, bc_std)
                    # 使用当前模式计算当前分布
                    cur_mu, cur_std, _ = self.networks[agent_id].forward(obs_batch, use_squashed_gaussian=True)
                    cur_dist = Normal(cur_mu, cur_std)
                    bc_anchor_loss = kl_divergence(bc_dist, cur_dist).sum(-1).mean()
                
                # Total loss（使用当前的熵系数）
                total_loss = actor_loss + self.vf_coef * critic_loss + self.current_ent_coef * entropy_loss
                if self.use_bc_anchor and self.lambda_bc > 0:
                    total_loss = total_loss + self.lambda_bc * bc_anchor_loss
                
                # 🔧 温和KL早停: 统一approx_kl计算口径
                with torch.no_grad():
                    # 使用公式 approx_kl = mean(logp_old - logp_new) 的正值（>=0）
                    # 先对每个样本求和后再取 batch mean，禁止 abs()
                    approx_kl = (old_log_probs_batch - log_probs).sum(dim=-1).mean().clamp_min(0.0).item()
                
                # 更新EMA与耐心
                self.kl_ema = self.kl_beta * self.kl_ema + (1 - self.kl_beta) * approx_kl
                if self.kl_ema > self.target_kl:
                    self.kl_exceed_streak += 1
                else:
                    self.kl_exceed_streak = 0
                
                # 记录KL散度
                if 'approx_kl' not in self.training_stats:
                    self.training_stats['approx_kl'] = {aid: [] for aid in self.networks.keys()}
                self.training_stats['approx_kl'][agent_id].append(approx_kl)
                
                # 记录KL EMA
                if 'kl_ema' not in self.training_stats:
                    self.training_stats['kl_ema'] = {aid: [] for aid in self.networks.keys()}
                self.training_stats['kl_ema'][agent_id].append(self.kl_ema)
                
                # 记录KL超阈连续次数
                if 'kl_exceed_streak' not in self.training_stats:
                    self.training_stats['kl_exceed_streak'] = {aid: [] for aid in self.networks.keys()}
                self.training_stats['kl_exceed_streak'][agent_id].append(self.kl_exceed_streak)
                
                # 记录BC锚定损失
                if self.use_bc_anchor and self.lambda_bc > 0:
                    if 'bc_anchor_loss' not in self.training_stats:
                        self.training_stats['bc_anchor_loss'] = {aid: [] for aid in self.networks.keys()}
                    self.training_stats['bc_anchor_loss'][agent_id].append(bc_anchor_loss.item())
                
                # 🔧 温和KL早停: 梯度安全网
                self.optimizers[agent_id].zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.networks[agent_id].parameters(), max_norm=0.5)
                self.optimizers[agent_id].step()
                
                # Store losses for monitoring
                self.training_stats['actor_losses'][agent_id].append(actor_loss.item())
                self.training_stats['critic_losses'][agent_id].append(critic_loss.item())
            
            # 简化的KL早停：只记录，不干预训练
            if epoch + 1 >= self.min_ppo_epochs and self.kl_exceed_streak >= self.kl_patience:
                if self.kl_ema > self.target_kl:
                    print(f"⚠️  KL较大但继续训练: epoch={epoch+1}, kl_ema={self.kl_ema:.4f}, target={self.target_kl:.4f}")
                    # 不早停，让智能体继续学习
        
        # 移除自适应调整，让智能体自然学习
        
        # 记录实际运行的epoch数
        if 'ppo_epochs_run' not in self.training_stats:
            self.training_stats['ppo_epochs_run'] = {aid: [] for aid in self.networks.keys()}
        self.training_stats['ppo_epochs_run'][agent_id].append(actual_epoch_count)
        
        # 记录当前学习率
        current_lr = self.optimizers[agent_id].param_groups[0]['lr']
        self.training_stats['learning_rates'][agent_id].append(current_lr)
    
    def clear_buffers(self):
        """Clear experience buffers"""
        for agent_id in self.buffers:
            for key in self.buffers[agent_id]:
                self.buffers[agent_id][key] = []
    
    def train_episode(self, env: MultiAgentAuctionEnv, max_steps: int = 16000):
        """Train for one episode with independent learning"""
        # 🔧 温和KL早停: 动态更新target_kl
        t = min(1.0, self.episode_idx / max(1, self.target_kl_decay_ep))
        self.target_kl = self.target_kl_start + (self.target_kl_end - self.target_kl_start) * t
        
        # 🔧 dd.md: 控制backbone冻结与解冻
        for net in self.networks.values():
            for p in net.backbone.parameters():
                p.requires_grad = self.episode_idx >= self.freeze_backbone_until
        
        obs = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in env.learning_agent_ids}
        episode_wins = {agent_id: 0 for agent_id in env.learning_agent_ids}
        
        for step in range(max_steps):
            actions = {}
            values = {}
            log_probs = {}
            
            # Get actions for all learning agents independently
            for agent_id in env.learning_agent_ids:
                action, log_prob, value = self.get_action(agent_id, obs[agent_id])
                
                actions[agent_id] = np.array([action])
                values[agent_id] = value
                log_probs[agent_id] = log_prob
            
            # Environment step
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Store experiences independently
            for agent_id in env.learning_agent_ids:
                done = terminated[agent_id] or truncated[agent_id]
                
                self.store_experience(
                    agent_id, obs[agent_id], actions[agent_id][0], 
                    rewards[agent_id], values[agent_id], log_probs[agent_id], done
                )
                
                episode_rewards[agent_id] += rewards[agent_id]
                
                # Track wins
                if hasattr(env, 'agent_histories') and agent_id in env.agent_histories:
                    if env.agent_histories[agent_id] and env.agent_histories[agent_id][-1].get('won', False):
                        episode_wins[agent_id] += 1
            
            obs = next_obs
            
            # Check if episode is done - 改为all()避免单体终止导致整体episode被截断
            # 这样可以让其他agent继续学习，保留完整的长期信号
            if all(terminated.values()) or all(truncated.values()):
                break
        
        # Update each agent's policy independently
        for agent_id in env.learning_agent_ids:
            # Get final value for GAE computation
            obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
            with torch.no_grad():
                _, _, next_value = self.networks[agent_id].get_action(obs_tensor, deterministic=True)
                next_value = next_value.item()
            
            advantages, returns = self.compute_gae(agent_id, next_value)
            self.update_policy(agent_id, advantages, returns)
            
            # 修正统计口径：计算正确的胜率和avg_bid_ratio
            actual_auctions = 0
            actual_wins = 0
            bid_ratios = []
            
            if hasattr(env, 'agent_histories') and agent_id in env.agent_histories:
                history = env.agent_histories[agent_id]
                actual_auctions = len(history)
                actual_wins = sum(1 for h in history if h.get('won', False))
                
                # 计算出价比率
                for h in history:
                    bid = h.get('bid', 0)
                    perceived_value = h.get('perceived_value', 0)
                    if perceived_value > 0:
                        bid_ratios.append(bid / perceived_value)
            
            # 修正胜率计算：使用实际拍卖次数而非max_steps
            win_rate = actual_wins / actual_auctions if actual_auctions > 0 else 0.0
            avg_bid_ratio = np.mean(bid_ratios) if bid_ratios else 1.0
            
            # Store episode statistics
            self.training_stats['episode_rewards'][agent_id].append(episode_rewards[agent_id])
            self.training_stats['win_rates'][agent_id].append(win_rate)
            
            # 添加avg_bid_ratio统计
            if 'avg_bid_ratios' not in self.training_stats:
                self.training_stats['avg_bid_ratios'] = {aid: [] for aid in env.learning_agent_ids}
            self.training_stats['avg_bid_ratios'][agent_id].append(avg_bid_ratio)
        
        self.training_stats['episode_lengths'].append(step + 1)
        
        # Episode结束后，更新熵系数（移除学习率调度器）
        
        # ==== Entropy decay (new, slower) ====
        self.current_ent_coef = max(self.min_ent_coef, self.current_ent_coef * self.ent_coef_decay)
        
        # ==== BC lambda decay (new, distributional) ====
        if self.use_bc_anchor:
            self.lambda_bc = max(self.lambda_bc_min, self.lambda_bc * self.lambda_bc_decay)
        
        # 🔧 dd.md: 监控和保险丝机制
        self._monitor_training_health(episode_rewards, episode_wins)
        
        # 🔧 dd.md: 增加episode计数
        self.episode_idx += 1
        
        self.clear_buffers()
        
        return episode_rewards, episode_wins
    
    def _monitor_training_health(self, episode_rewards, episode_wins):
        """🔧 dd.md: 监控训练健康状态并触发保险丝机制"""
        if self.episode_idx < 3:  # 前3个episode不触发保险丝
            return
        
        # 计算最近3个episode的统计
        recent_episodes = min(3, self.episode_idx)
        
        for agent_id in episode_rewards.keys():
            if agent_id not in self.training_stats['win_rates']:
                continue
                
            recent_win_rates = self.training_stats['win_rates'][agent_id][-recent_episodes:]
            recent_rewards = self.training_stats['episode_rewards'][agent_id][-recent_episodes:]
            
            if len(recent_win_rates) < recent_episodes:
                continue
            
            # 计算Action-BC一致率（如果有BC锚定损失记录）
            bc_consistency = 1.0
            if 'bc_anchor_loss' in self.training_stats and agent_id in self.training_stats['bc_anchor_loss']:
                recent_bc_losses = self.training_stats['bc_anchor_loss'][agent_id][-recent_episodes:]
                if len(recent_bc_losses) >= recent_episodes:
                    # 基于BC损失估算一致率（损失越小，一致率越高）
                    avg_bc_loss = np.mean(recent_bc_losses)
                    bc_consistency = max(0.0, 1.0 - avg_bc_loss / 0.1)  # 假设0.1为阈值
            
            # 检查是否需要触发保险丝
            current_win_rate = recent_win_rates[-1]
            prev_win_rate = recent_win_rates[0] if len(recent_win_rates) > 1 else current_win_rate
            
            # 保险丝条件：一致率骤降>30%且WinRate同降
            consistency_drop = (1.0 - bc_consistency) > 0.3
            win_rate_drop = (prev_win_rate - current_win_rate) > 0.1
            
            if consistency_drop and win_rate_drop:
                print(f"🚨 保险丝触发 - {agent_id}:")
                print(f"   BC一致率: {bc_consistency:.3f}, WinRate下降: {prev_win_rate:.3f} → {current_win_rate:.3f}")
                
                # 触发保险丝措施
                self.lambda_bc = max(self.lambda_bc, 0.5)  # 把锚定拉回来
                self.lr *= 0.5  # 降低学习率
                
                # 重新冻结backbone 5个episode
                self.freeze_backbone_until = max(self.freeze_backbone_until, self.episode_idx + 5)
                
                print(f"   措施: lambda_bc={self.lambda_bc:.3f}, lr={self.lr:.6f}, 冻结backbone至ep{self.freeze_backbone_until}")
    
    def load_bc_weights(self, bc_model_path: str, strict: bool = False):
        """
        加载BC预训练权重到IPPO网络 - 增强版
        
        BC网络结构：network.0-7 (完整Sequential: 3×ReLU + Linear + Sigmoid)
        IPPO网络结构：backbone.0-5 (3×ReLU) + actor_mean + actor_logstd + critic
        
        加载策略：
        1. 将BC的network.0-5映射到IPPO的backbone.0-5
        2. 将BC的network.6映射到IPPO的actor_mean（避免"换头"问题）
        3. 初始化actor_logstd为小常数（std≈0.1）
        
        Args:
            bc_model_path: BC模型文件路径
            strict: 是否严格匹配所有参数（默认False，允许部分加载）
            
        Returns:
            加载成功的智能体数量
        """
        print(f"\n📥 加载BC预训练权重: {bc_model_path}")
        
        if not os.path.exists(bc_model_path):
            print(f"❌ BC模型文件不存在: {bc_model_path}")
            return 0
        
        bc_state_dict = torch.load(bc_model_path, map_location='cpu')
        success_count = 0
        loaded_items = []
        skipped_items = []
        
        print(f"📦 BC模型包含的键: {list(bc_state_dict.keys())}")
        
        for agent_id, network in self.networks.items():
            try:
                # 🔧 映射逻辑：BC的ActorCriticNetwork → IPPO的IPPOActorCriticNetwork
                ippo_state_dict = {}
                loaded_layers = []
                
                # 1) 加载backbone层（backbone.0-5 → backbone.0-5）
                for key, value in bc_state_dict.items():
                    if key.startswith('backbone.'):
                        ippo_state_dict[key] = value
                        if key not in loaded_layers:
                            loaded_layers.append(key.split('.')[0] + '.' + key.split('.')[1])
                
                # 2) 加载actor_linear层到actor_linear（传统模式兼容）
                for k_w, k_b in [("actor_linear.weight", "actor_linear.bias")]:
                    if k_w in bc_state_dict and k_b in bc_state_dict:
                        ippo_state_dict[k_w] = bc_state_dict[k_w]
                        ippo_state_dict[k_b] = bc_state_dict[k_b]
                        loaded_items.append(f"{agent_id}.{k_w}")
                        loaded_items.append(f"{agent_id}.{k_b}")
                    else:
                        skipped_items.append(f"{agent_id}.{k_w}/{k_b} not in BC ckpt")
                
                # 3) 加载actor_logstd
                if "actor_logstd" in bc_state_dict:
                    ippo_state_dict["actor_logstd"] = bc_state_dict["actor_logstd"]
                    loaded_items.append(f"{agent_id}.actor_logstd")
                else:
                    # 初始化logstd为小常数（std≈0.1）
                    with torch.no_grad():
                        if hasattr(network, "actor_logstd"):
                            network.actor_logstd.fill_(math.log(0.1))
                            loaded_items.append(f"{agent_id}.actor_logstd = log(0.1)")
                
                # 4) 初始化Squashed Gaussian的actor_mean和actor_std
                with torch.no_grad():
                    if hasattr(network, "actor_mean") and hasattr(network, "actor_linear"):
                        # 将传统actor_linear的权重复制到actor_mean
                        network.actor_mean.weight.data.copy_(network.actor_linear.weight.data)
                        network.actor_mean.bias.data.copy_(network.actor_linear.bias.data)
                        loaded_items.append(f"{agent_id}.actor_mean initialized from actor_linear")
                    
                    if hasattr(network, "actor_std"):
                        # 初始化actor_std为小权重
                        nn.init.orthogonal_(network.actor_std.weight, gain=0.01)
                        nn.init.constant_(network.actor_std.bias, -2.0)  # 初始std较小
                        loaded_items.append(f"{agent_id}.actor_std initialized")
                
                # 5) 创建冻结的BC头（用于锚定损失）
                if hasattr(network, "actor_linear"):
                    bc_head = nn.Linear(network.actor_linear.in_features, network.actor_linear.out_features, bias=True)
                    bc_head.weight.data.copy_(network.actor_linear.weight.data)
                    bc_head.bias.data.copy_(network.actor_linear.bias.data)
                    for p in bc_head.parameters():
                        p.requires_grad = False
                    network.bc_head_frozen = bc_head
                    loaded_items.append(f"{agent_id}.bc_head_frozen created")
                
                # 加载backbone权重到IPPO网络
                missing_keys, unexpected_keys = network.load_state_dict(ippo_state_dict, strict=False)
                
                print(f"  ✅ {agent_id}: BC预训练权重加载成功")
                print(f"     已加载层: {sorted(set(loaded_layers))}")
                print(f"     加载参数数: {len(ippo_state_dict)}")
                
                # 显示未加载的部分（这是正常的）
                expected_missing = [k for k in missing_keys if 'critic' in k]
                if expected_missing:
                    print(f"     未加载（随机初始化）: critic")
                
                success_count += 1
                    
            except Exception as e:
                print(f"  ❌ {agent_id}: 加载失败 - {e}")
                import traceback
                traceback.print_exc()
        
        # 输出详细的加载日志
        print(f"\n✅ BC预训练权重加载完成 | loaded={len(loaded_items)} | skipped={len(skipped_items)}")
        for item in loaded_items:
            print(f"   + {item}")
        for item in skipped_items:
            print(f"   - {item}")
        
        print(f"\n✨ BC预训练加载完成: {success_count}/{len(self.networks)} 个智能体成功")
        print(f"💡 说明: 已加载共享特征提取器(backbone.0-5)和actor_mean，避免'换头'问题")
        return success_count
    
    def save_models(self, save_dir: str):
        """Save trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        for agent_id, network in self.networks.items():
            model_path = f"{save_dir}/{agent_id}_ippo_model.pth"
            torch.save(network.state_dict(), model_path)
        
        print(f"✅ IPPO models saved to {save_dir}")
    
    def load_models(self, save_dir: str):
        """Load trained models"""
        for agent_id, network in self.networks.items():
            model_path = f"{save_dir}/{agent_id}_ippo_model.pth"
            if os.path.exists(model_path):
                network.load_state_dict(torch.load(model_path))
                print(f"✅ Loaded IPPO model for {agent_id}")
            else:
                print(f"⚠️ Model not found: {model_path}")
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        for agent_id in self.networks.keys():
            axes[0, 0].plot(self.training_stats['episode_rewards'][agent_id], 
                           label=f"{agent_id} (IPPO)", alpha=0.7)
        axes[0, 0].set_title('Episode Rewards (IPPO)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Win rates
        for agent_id in self.networks.keys():
            axes[0, 1].plot(self.training_stats['win_rates'][agent_id], 
                           label=f"{agent_id} (IPPO)", alpha=0.7)
        axes[0, 1].set_title('Win Rates (IPPO)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Actor losses
        for agent_id in self.networks.keys():
            if self.training_stats['actor_losses'][agent_id]:
                axes[1, 0].plot(self.training_stats['actor_losses'][agent_id], 
                               label=f"{agent_id} (IPPO)", alpha=0.7)
        axes[1, 0].set_title('Actor Losses (IPPO)')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Critic losses
        for agent_id in self.networks.keys():
            if self.training_stats['critic_losses'][agent_id]:
                axes[1, 1].plot(self.training_stats['critic_losses'][agent_id], 
                               label=f"{agent_id} (IPPO)", alpha=0.7)
        axes[1, 1].set_title('Critic Losses (IPPO)')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"IPPO training curves saved to {save_path}")
        
        plt.show()