# /auction_sim/ma_trainer.py
"""
Multi-Agent PPO Trainer for k=2 scenario
Fixed for stable training with corrected reward signals
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Any
import os
from collections import deque
import matplotlib.pyplot as plt

from . import config
from .config import CurriculumStage, CURRICULUM_CONFIGS
from .curriculum_controller import CurriculumController
from .ma_environment import MultiAgentAuctionEnv
from .agents import TruthfulAgent, ConservativeAgent, AggressiveAgent, MultiAgentLearningAgent

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for MAPPO"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_linear = nn.Linear(hidden_dim, action_dim)
        self.actor_activation = nn.Sigmoid()
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        nn.init.orthogonal_(self.actor_linear.weight, gain=0.01)
        
    def forward(self, obs):
        features = self.backbone(obs)
        
        # Actor output
        # 🔧 修复BC兼容性: 改为[0, 1.5]与IPPO一致，移除+0.5偏置
        mean = self.actor_linear(features)
        mean = self.actor_activation(mean)
        mean = mean * 1.5  # Scale to [0, 1.5] - 与IPPO完全一致
        std = torch.exp(self.actor_logstd)
        
        # Critic output
        value = self.critic(features)
        
        return mean, std, value
    
    def get_action(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)
        
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, 0.1, 3.0)
        
        return action, value
    
    def evaluate_action(self, obs, action):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy, value

class MAPPOTrainer:
    """Multi-Agent PPO Trainer for auction environment"""
    
    def __init__(self, 
                 obs_dim: int = 7,
                 action_dim: int = 1,
                 n_agents: int = 2,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 max_buffer_size: int = 50000,
                 use_curriculum: bool = True):
        
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.max_buffer_size = max_buffer_size
        self.use_curriculum = use_curriculum
        
        # Curriculum controller
        if use_curriculum:
            self.curriculum_controller = CurriculumController(CurriculumStage.STAGE_0)
        else:
            self.curriculum_controller = None
        
        # MAPPO: Create single shared network for all agents
        self.shared_network = ActorCriticNetwork(obs_dim, action_dim)
        self.shared_optimizer = optim.Adam(self.shared_network.parameters(), lr=lr)
        
        # All agents share the same network
        self.networks = {}
        self.optimizers = {}
        for i in range(n_agents):
            agent_id = f"Learning_{i}"
            self.networks[agent_id] = self.shared_network  # Point to shared network
            self.optimizers[agent_id] = self.shared_optimizer  # Point to shared optimizer
        
        # Experience buffers
        self.buffers = {agent_id: {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        } for agent_id in self.networks.keys()}
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': {agent_id: [] for agent_id in self.networks.keys()},
            'episode_lengths': [],
            'actor_losses': {agent_id: [] for agent_id in self.networks.keys()},
            'critic_losses': {agent_id: [] for agent_id in self.networks.keys()},
            'win_rates': {agent_id: [] for agent_id in self.networks.keys()}
        }
        
        print(f"MAPPO Trainer initialized with {n_agents} agents")
    
    def add_agent(self, agent_id: str):
        """Dynamically add a new agent to the trainer"""
        if agent_id not in self.networks:
            # Create new network and optimizer
            self.networks[agent_id] = ActorCriticNetwork(self.obs_dim, self.action_dim)
            self.optimizers[agent_id] = optim.Adam(
                self.networks[agent_id].parameters(), lr=self.lr
            )
            
            # Create buffer for new agent
            self.buffers[agent_id] = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'values': [],
                'log_probs': [],
                'dones': []
            }
            
            # Initialize training stats for new agent
            self.training_stats['episode_rewards'][agent_id] = []
            self.training_stats['actor_losses'][agent_id] = []
            self.training_stats['critic_losses'][agent_id] = []
            self.training_stats['win_rates'][agent_id] = []
            
            print(f"Added new agent {agent_id} to trainer")
    
    def get_action(self, agent_id: str, obs: np.ndarray, deterministic: bool = False):
        """Get action from policy network"""
        # Add agent if it doesn't exist
        if agent_id not in self.networks:
            self.add_agent(agent_id)
        
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        if obs.ndim == 0:
            obs = obs.reshape(1,)
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action, value = self.networks[agent_id].get_action(obs_tensor, deterministic)
            
        return action.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def store_experience(self, agent_id: str, obs, action, reward, value, log_prob, done):
        """Store experience in buffer with size limit"""
        # Add agent if it doesn't exist
        if agent_id not in self.buffers:
            self.add_agent(agent_id)
            
        if len(self.buffers[agent_id]['observations']) >= self.max_buffer_size:
            for key in self.buffers[agent_id]:
                if isinstance(self.buffers[agent_id][key], list):
                    self.buffers[agent_id][key].pop(0)
        
        self.buffers[agent_id]['observations'].append(obs)
        self.buffers[agent_id]['actions'].append(action)
        self.buffers[agent_id]['rewards'].append(reward)
        self.buffers[agent_id]['values'].append(value)
        self.buffers[agent_id]['log_probs'].append(log_prob)
        self.buffers[agent_id]['dones'].append(done)
    
    def compute_gae(self, agent_id: str, next_value: float = 0.0):
        """Compute Generalized Advantage Estimation"""
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
        """Update policy using PPO"""
        # Convert to tensors
        obs = torch.FloatTensor(self.buffers[agent_id]['observations'])
        actions = torch.FloatTensor(self.buffers[agent_id]['actions'])
        old_log_probs = torch.FloatTensor(self.buffers[agent_id]['log_probs'])
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 0:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages = advantages - adv_mean
        
        # Multiple epochs of updates
        n_updates = 4
        batch_size = min(256, len(obs))
        
        for _ in range(n_updates):
            # Sample mini-batch
            indices = torch.randperm(len(obs))[:batch_size]
            
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            old_log_probs_batch = old_log_probs[indices]
            advantages_batch = advantages[indices]
            returns_batch = returns[indices]
            
            # Forward pass
            log_probs, entropy, values = self.networks[agent_id].evaluate_action(
                obs_batch, actions_batch
            )
            
            # PPO loss
            ratio = torch.exp(log_probs - old_log_probs_batch)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            if values.dim() > returns_batch.dim():
                values = values.squeeze(-1)
            elif values.dim() < returns_batch.dim():
                returns_batch = returns_batch.squeeze(-1)
            critic_loss = nn.MSELoss()(values, returns_batch)
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss
            
            # Optimize
            self.optimizers[agent_id].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.networks[agent_id].parameters(), self.max_grad_norm)
            self.optimizers[agent_id].step()
            
            # Store losses
            self.training_stats['actor_losses'][agent_id].append(actor_loss.item())
            self.training_stats['critic_losses'][agent_id].append(critic_loss.item())
    
    def update_shared_policy(self, all_advantages, all_returns):
        """Update shared policy using combined experience from all agents"""
        # Combine observations, actions, and log_probs from all agents
        all_obs = []
        all_actions = []
        all_old_log_probs = []
        
        for agent_id in self.buffers.keys():
            all_obs.extend(self.buffers[agent_id]['observations'])
            all_actions.extend(self.buffers[agent_id]['actions'])
            all_old_log_probs.extend(self.buffers[agent_id]['log_probs'])
        
        # Convert to tensors
        obs = torch.FloatTensor(all_obs)
        actions = torch.FloatTensor(all_actions)
        old_log_probs = torch.FloatTensor(all_old_log_probs)
        advantages = torch.FloatTensor(all_advantages)
        returns = torch.FloatTensor(all_returns)
        
        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 0:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages = advantages - adv_mean
        
        # Multiple epochs of updates on shared network
        n_updates = 4
        batch_size = min(256, len(obs))
        
        for _ in range(n_updates):
            # Sample mini-batch
            indices = torch.randperm(len(obs))[:batch_size]
            
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            old_log_probs_batch = old_log_probs[indices]
            advantages_batch = advantages[indices]
            returns_batch = returns[indices]
            
            # Forward pass through shared network
            log_probs, entropy, values = self.shared_network.evaluate_action(obs_batch, actions_batch)
            
            # PPO clipped surrogate objective
            ratio = torch.exp(log_probs - old_log_probs_batch)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            if values.dim() > returns_batch.dim():
                values = values.squeeze(-1)
            elif values.dim() < returns_batch.dim():
                returns_batch = returns_batch.squeeze(-1)
            critic_loss = nn.MSELoss()(values, returns_batch)
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss
            
            # Optimize shared network
            self.shared_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.shared_network.parameters(), self.max_grad_norm)
            self.shared_optimizer.step()
    
    def clear_buffers(self):
        """Clear experience buffers"""
        for agent_id in self.buffers:
            for key in self.buffers[agent_id]:
                self.buffers[agent_id][key] = []
    
    def train_episode(self, env: MultiAgentAuctionEnv, max_steps: int = 16000):
        """Train for one episode"""
        obs = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in env.learning_agent_ids}
        episode_wins = {agent_id: 0 for agent_id in env.learning_agent_ids}
        
        for step in range(max_steps):
            actions = {}
            values = {}
            log_probs = {}
            
            # Get actions for all learning agents
            for agent_id in env.learning_agent_ids:
                action, value = self.get_action(agent_id, obs[agent_id])
                
                # Calculate log probability for storage
                obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
                action_tensor = torch.FloatTensor([float(action)])
                # Ensure agent exists before evaluating
                if agent_id not in self.networks:
                    self.add_agent(agent_id)
                log_prob, _, _ = self.networks[agent_id].evaluate_action(obs_tensor, action_tensor)
                
                actions[agent_id] = np.array([action])
                values[agent_id] = value
                log_probs[agent_id] = log_prob.item()
            
            # Environment step
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Store experiences
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
            
            # Check if episode is done
            if any(terminated.values()) or any(truncated.values()):
                break
        
        # MAPPO: Compute advantages for all agents, then update shared network once
        all_advantages = []
        all_returns = []
        
        for agent_id in env.learning_agent_ids:
            # Get final value for GAE computation
            obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
            with torch.no_grad():
                _, next_value = self.networks[agent_id].get_action(obs_tensor, deterministic=True)
                next_value = next_value.item()
            
            advantages, returns = self.compute_gae(agent_id, next_value)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
            
            # Store episode statistics
            self.training_stats['episode_rewards'][agent_id].append(episode_rewards[agent_id])
            win_rate = episode_wins[agent_id] / max_steps
            self.training_stats['win_rates'][agent_id].append(win_rate)
        
        # Update shared network with combined experience from all agents
        self.update_shared_policy(all_advantages, all_returns)
        
        self.training_stats['episode_lengths'].append(step + 1)
        self.clear_buffers()
        
        return episode_rewards, episode_wins
    
    def save_models(self, save_dir: str):
        """Save shared model"""
        os.makedirs(save_dir, exist_ok=True)
        
        # MAPPO: Save only the shared network
        torch.save(self.shared_network.state_dict(), f"{save_dir}/shared_model.pth")
        
        # For compatibility, also save copies with agent names
        for agent_id in self.networks.keys():
            torch.save(self.shared_network.state_dict(), f"{save_dir}/{agent_id}_model.pth")
        
        print(f"MAPPO shared model saved to {save_dir}")
    
    def load_models(self, save_dir: str):
        """Load shared model"""
        model_path = f"{save_dir}/shared_model.pth"
        if os.path.exists(model_path):
            self.shared_network.load_state_dict(torch.load(model_path))
            print(f"MAPPO shared model loaded from {model_path}")
        else:
            # Fallback: try to load from agent-specific file
            fallback_path = f"{save_dir}/Learning_0_model.pth"
            if os.path.exists(fallback_path):
                self.shared_network.load_state_dict(torch.load(fallback_path))
                print(f"MAPPO model loaded from fallback {fallback_path}")
                print(f"Loaded model for {agent_id}")
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        for agent_id in self.networks.keys():
            axes[0, 0].plot(self.training_stats['episode_rewards'][agent_id], 
                           label=agent_id, alpha=0.7)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Win rates
        for agent_id in self.networks.keys():
            axes[0, 1].plot(self.training_stats['win_rates'][agent_id], 
                           label=agent_id, alpha=0.7)
        axes[0, 1].set_title('Win Rates')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Actor losses
        for agent_id in self.networks.keys():
            if self.training_stats['actor_losses'][agent_id]:
                axes[1, 0].plot(self.training_stats['actor_losses'][agent_id], 
                               label=agent_id, alpha=0.7)
        axes[1, 0].set_title('Actor Losses')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Critic losses
        for agent_id in self.networks.keys():
            if self.training_stats['critic_losses'][agent_id]:
                axes[1, 1].plot(self.training_stats['critic_losses'][agent_id], 
                               label=agent_id, alpha=0.7)
        axes[1, 1].set_title('Critic Losses')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()