# /auction_sim/ma_environment.py
"""
Multi-Agent Auction Environment for k=2 scenario
Fixed reward function to avoid negative training rewards
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any
from . import config
from .config import CurriculumStage, CURRICULUM_CONFIGS
from .auction import GSPAuction
from .agents import Agent, TruthfulAgent, ConservativeAgent, AggressiveAgent

class MultiAgentAuctionEnv:
    """
    Multi-Agent Environment wrapper for auction simulation
    Designed for k=2 learning agents competing with rule-based agents
    """
    
    def __init__(self, learning_agent_ids: List[str], rule_agents: List[Agent], 
                 curriculum_stage: CurriculumStage = CurriculumStage.STAGE_8):
        self.learning_agent_ids = learning_agent_ids
        self.rule_agents = rule_agents
        self.n_learning_agents = len(learning_agent_ids)
        self.curriculum_stage = curriculum_stage
        
        # Get curriculum configuration
        self.stage_config = CURRICULUM_CONFIGS.get(curriculum_stage, {})
        
        # Initialize auction with curriculum-specific settings
        n_slots = self.stage_config.get('n_slots', config.N_SLOTS)
        ctr_positions = self.stage_config.get('ctr_positions', config.CTR_POSITIONS)
        self.auction = GSPAuction(n_slots, ctr_positions, config.CTR_NOISE_STD)
        
        # Environment state
        self.current_round = 0
        self.max_rounds = self.stage_config.get('max_rounds', config.SIMULATION_ROUNDS)
        self.current_true_value = 0.0
        
        # 修复1: 缓存当前观测，避免状态-动作时序错配
        self.cached_observations = {}
        self.cached_perceived_values = {}
        
        # Agent states (for learning agents) with curriculum-specific budget
        curriculum_budget = self.stage_config.get('budget', config.AGENT_BUDGET)
        self.agent_budgets = {aid: curriculum_budget for aid in learning_agent_ids}
        self.initial_budgets = {aid: curriculum_budget for aid in learning_agent_ids}
        self.agent_histories = {aid: [] for aid in learning_agent_ids}
        self.round_history = []  # Track auction results for win rate calculation
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Performance tracking
        self.episode_rewards = {aid: 0.0 for aid in learning_agent_ids}
        
        # 累积统计（用于分析）
        self.cum_profit = {aid: 0.0 for aid in learning_agent_ids}
        self.cum_cost = {aid: 0.0 for aid in learning_agent_ids}
        
    def _setup_spaces(self):
        """Setup observation and action spaces for multi-agent learning"""
        
        # Observation space: 7D (与BC数据收集器保持一致)
        # [perceived_value, budget_ratio, time_ratio, recent_win_rate, recent_profit, 
        #  opponent_win_rate, market_competition]
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Action space: continuous bid multiplier [0, 1.5]
        # 与BC网络输出范围一致，覆盖所有合理策略（保底价要求0.15+）
        self.action_space = spaces.Box(low=0.0, high=1.5, shape=(1,), dtype=np.float32)
        
        print(f"Multi-Agent Environment initialized:")
        print(f"  - Learning agents: {self.n_learning_agents}")
        print(f"  - Rule agents: {len(self.rule_agents)}")
        print(f"  - Observation space: {self.observation_space}")
        print(f"  - Action space: {self.action_space}")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment for new episode"""
        self.current_round = 0
        self.current_true_value = 0.0
        
        # Reset agent states with curriculum-specific budget
        curriculum_budget = self.stage_config.get('budget', config.AGENT_BUDGET)
        self.agent_budgets = {aid: curriculum_budget for aid in self.learning_agent_ids}
        self.initial_budgets = {aid: curriculum_budget for aid in self.learning_agent_ids}
        self.agent_histories = {aid: [] for aid in self.learning_agent_ids}
        
        # Reset rule agents
        for agent in self.rule_agents:
            agent.budget = agent.initial_budget
            agent.history = []
        
        self.episode_rewards = {aid: 0.0 for aid in self.learning_agent_ids}
        
        # 重置累积统计（用于分析）
        self.cum_profit = {aid: 0.0 for aid in self.learning_agent_ids}
        self.cum_cost = {aid: 0.0 for aid in self.learning_agent_ids}
        
        # 修复1: 生成并缓存首轮观测
        # 这样保证trainer收到的obs_0对应的就是step(a_0)将要使用的状态
        initial_obs = self._generate_new_observations()
        self.cached_observations = initial_obs['observations']
        self.cached_perceived_values = initial_obs['perceived_values']
        
        return self.cached_observations
    
    def _generate_new_observations(self) -> Dict[str, Any]:
        """
        修复1: 生成新观测（包含新true_value和perceived_values）
        返回包含observations和perceived_values的字典，用于后续缓存
        """
        # 生成新的true value
        self.current_true_value = np.random.uniform(*config.TRUE_VALUE_RANGE)
        
        observations = {}
        perceived_values = {}
        
        for agent_id in self.learning_agent_ids:
            # Add perception noise
            perceived_value = self.current_true_value + np.random.normal(0, config.AGENT_PERCEPTION_NOISE_STD)
            perceived_value = max(0, perceived_value)
            perceived_values[agent_id] = perceived_value
            
            obs = self._build_observation(agent_id, perceived_value)
            observations[agent_id] = obs
        
        return {'observations': observations, 'perceived_values': perceived_values}
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """兼容旧接口：返回缓存的观测"""
        return self.cached_observations
    
    def _build_observation(self, agent_id: str, perceived_value: float) -> np.ndarray:
        """Build observation vector for a specific learning agent"""
        
        # 1. Normalized perceived value
        norm_perceived_value = (perceived_value - config.TRUE_VALUE_RANGE[0]) / \
                              (config.TRUE_VALUE_RANGE[1] - config.TRUE_VALUE_RANGE[0])
        norm_perceived_value = np.clip(norm_perceived_value, 0.0, 1.0)
        
        # 2. Remaining budget ratio
        budget_ratio = self.agent_budgets[agent_id] / self.initial_budgets[agent_id]
        
        # 3. Time ratio
        time_ratio = (self.max_rounds - self.current_round) / self.max_rounds
        
        # 4. Recent win rate
        recent_wins = 0
        recent_rounds = min(len(self.agent_histories[agent_id]), 100)
        if recent_rounds > 0:
            recent_history = self.agent_histories[agent_id][-recent_rounds:]
            recent_wins = sum(1 for record in recent_history 
                            if record.get('won', False))
            recent_win_rate = recent_wins / recent_rounds
        else:
            recent_win_rate = 0.0
        
        # 5. Recent average profit (normalized)
        recent_profit = 0.0
        if recent_rounds > 0:
            recent_history = self.agent_histories[agent_id][-recent_rounds:]
            profits = [record.get('profit', 0.0) for record in recent_history]
            profits = [float(p) if p is not None else 0.0 for p in profits]
            recent_profit = np.mean(profits) if profits else 0.0
            recent_profit = np.clip(recent_profit / 10.0, -1.0, 1.0)
        
        # 6. Opponent win rate
        opponent_win_rate = 0.0
        if len(self.learning_agent_ids) > 1:
            other_agents = [aid for aid in self.learning_agent_ids if aid != agent_id]
            total_wins = 0
            total_rounds = 0
            
            for other_id in other_agents:
                other_recent = min(len(self.agent_histories[other_id]), 100)
                if other_recent > 0:
                    other_history = self.agent_histories[other_id][-other_recent:]
                    other_wins = sum(1 for record in other_history 
                                   if record.get('won', False))
                    total_wins += other_wins
                    total_rounds += other_recent
            
            opponent_win_rate = total_wins / total_rounds if total_rounds > 0 else 0.0
        
        # 7. Market competition level
        total_agents = len(self.learning_agent_ids) + len(self.rule_agents)
        competition_level = min(total_agents / 10.0, 1.0)
        
        # 7D observation (与BC数据收集器保持一致)
        observation = np.array([
            norm_perceived_value,    # 0: 标准化感知价值
            budget_ratio,           # 1: 剩余预算比例
            time_ratio,             # 2: 时间比例
            recent_win_rate,        # 3: 近期胜率
            recent_profit,          # 4: 近期平均利润
            opponent_win_rate,      # 5: 对手胜率
            competition_level       # 6: 市场竞争水平
        ], dtype=np.float32)
        
        return observation
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        修复1: 使用缓存的观测执行动作，避免状态-动作时序错配
        标准MDP时序: obs_t → action_t → (reward_t, obs_{t+1})
        """
        # 使用缓存的观测（即trainer用来生成action的那个观测）
        observations_t = self.cached_observations
        perceived_values_t = self.cached_perceived_values
        
        # Collect all bids (基于当前轮的perceived values)
        all_bids = {}
        perceived_values = {}
        
        # Learning agents bids
        for agent_id in self.learning_agent_ids:
            if agent_id in actions:
                # 使用缓存的perceived_value（与obs_t一致）
                perceived_value = perceived_values_t[agent_id]
                perceived_values[agent_id] = perceived_value
                
                bid_multiplier = float(actions[agent_id][0])
                bid_price = perceived_value * bid_multiplier
                
                # 获取最低保底价（防止出价过低导致意外高成本）
                min_reserve = min(self.auction.reserve_per_slot) if len(self.auction.reserve_per_slot) > 0 else 0.0
                
                # 检查：1）有足够预算支付出价 2）出价不低于保底价
                # 这确保智能体不会出价过低而在拍卖时被强制支付更高的保底价
                if (self.agent_budgets[agent_id] >= bid_price and 
                    bid_price >= min_reserve):
                    all_bids[agent_id] = bid_price
                # 否则拒绝出价（不参与本轮拍卖）
        
        # Rule agents bids
        for agent in self.rule_agents:
            if agent.budget > 0:
                perceived_value = agent.perceive(self.current_true_value)
                bid_price = agent.bid(perceived_value)
                
                if agent.can_afford_bid(bid_price):
                    all_bids[agent.id] = bid_price
                    perceived_values[agent.id] = perceived_value
        
        # Run auction
        auction_results = {}
        if all_bids:
            auction_results = self.auction.run_auction(all_bids)
        
        # Record round history for competitive analysis
        self.round_history.append(auction_results)
        
        # Calculate rewards and update states
        rewards = {}
        for agent_id in self.learning_agent_ids:
            reward, profit = self._calculate_reward(agent_id, auction_results, perceived_values.get(agent_id, 0))
            rewards[agent_id] = reward
            self.episode_rewards[agent_id] += reward
            
            # Update agent history
            result = auction_results.get(agent_id)
            won = result['won'] if result else False
            cost = 0.0
            cost_per_click = 0.0
            bid = all_bids.get(agent_id, 0.0)
            
            if result and result['won']:
                expected_cost = result['cost_per_click'] * result['slot_ctr']
                cost = min(expected_cost, self.agent_budgets[agent_id])
                cost_per_click = result['cost_per_click']
                self.agent_budgets[agent_id] -= cost
            
            self.agent_histories[agent_id].append({
                'round': self.current_round,
                'won': won,
                'profit': float(profit) if profit is not None else 0.0,
                'cost': float(cost),
                'budget': float(self.agent_budgets[agent_id]),
                'perceived_value': float(perceived_values.get(agent_id, 0.0)),
                'bid': float(bid),
                'cost_per_click': float(cost_per_click)  # 添加CPC记录
            })
        
        # Update rule agents
        for agent in self.rule_agents:
            result = auction_results.get(agent.id)
            if result and result['won']:
                true_value_profit = self.current_true_value * result['slot_ctr']
                expected_cost = result['cost_per_click'] * result['slot_ctr']
                profit = true_value_profit - expected_cost
            else:
                profit = 0.0
            
            agent.update(result, self.current_round)
        
        # 修复1: 完成本轮结算后，递增round并生成下一轮观测
        self.current_round += 1
        
        # 生成下一轮观测并缓存（用于下次step）
        next_obs_data = self._generate_new_observations()
        self.cached_observations = next_obs_data['observations']
        self.cached_perceived_values = next_obs_data['perceived_values']
        next_observations = self.cached_observations
        
        # Check termination conditions
        terminated = {agent_id: False for agent_id in self.learning_agent_ids}
        truncated = {agent_id: self.current_round >= self.max_rounds for agent_id in self.learning_agent_ids}
        
        # Info dict
        info = {agent_id: {
            'episode_reward': self.episode_rewards[agent_id],
            'budget_remaining': self.agent_budgets[agent_id],
            'round': self.current_round
        } for agent_id in self.learning_agent_ids}
        
        return next_observations, rewards, terminated, truncated, info
    
    def _calculate_reward(self, agent_id: str, auction_results: Dict, perceived_value: float) -> Tuple[float, float]:
        """
        使用纯利润作为RL奖励，确保训练信号与性能指标一致（与DDPG对齐）
        使用真实支付成本（考虑预算限制）并添加预算节奏约束
        """
        # 计算纯利润作为主要奖励
        result = auction_results.get(agent_id)
        if result and result.get('won', False):
            true_value_profit = self.current_true_value * result['slot_ctr']
            expected_cost = result['cost_per_click'] * result['slot_ctr']
            # 使用实际支付成本（考虑预算限制）
            current_cost = min(expected_cost, self.agent_budgets[agent_id])
            current_profit = true_value_profit - current_cost
        else:
            current_profit = 0.0
        
        # 预算节奏约束：防止预算超前消耗
        spent_ratio = 1.0 - (self.agent_budgets[agent_id] / self.initial_budgets[agent_id])
        time_ratio = self.current_round / self.max_rounds  # time_ratio 应该是已进行的回合比例
        pace_penalty = -0.05 * max(0.0, spent_ratio - time_ratio)
        
        # 用于学习的奖励 = 真实利润 + 预算惩罚
        reward = current_profit + pace_penalty
        
        # 返回塑形后的 reward 和 纯粹的 current_profit
        return float(reward), float(current_profit)
    
    def _stage0_reward(self, agent_id: str, auction_results: Dict, perceived_value: float) -> Tuple[float, float]:
        """
        Stage 0 (COOPERATIVE) reward - both agents learn together vs Truthful opponents
        """
        result = auction_results.get(agent_id)
        stage_config = CURRICULUM_CONFIGS[self.curriculum_stage]
        reward_weights = stage_config.get('reward_weights', {})
        
        # Calculate basic profit and cost
        if result and result['won']:
            true_profit = self.current_true_value * result['slot_ctr']
            expected_cost = result['cost_per_click'] * result['slot_ctr']
            current_profit = float(true_profit - expected_cost)
            current_cost = float(min(expected_cost, self.agent_budgets[agent_id]))
        else:
            current_profit = 0.0
            current_cost = 0.0
        
        # Base profit reward (scaled)
        profit_reward = current_profit * reward_weights.get('profit', 0.3) / 10.0
        
        # Win bonus
        win_bonus = 0.0
        if result and result['won']:
            win_bonus = reward_weights.get('win_bonus', 0.6)
        
        # Cooperation bonus - reward when both learning agents win
        cooperation_bonus = 0.0
        if result and result['won']:
            other_learning_agents = [aid for aid in self.learning_agent_ids if aid != agent_id]
            if other_learning_agents:
                other_agent_won = any(
                    auction_results.get(other_id, {}).get('won', False) 
                    for other_id in other_learning_agents
                )
                if other_agent_won:
                    cooperation_bonus = reward_weights.get('cooperation', 0.1)
        
        total_reward = profit_reward + win_bonus + cooperation_bonus
        return float(total_reward), float(current_profit)
    
    def _stage1_reward(self, agent_id: str, auction_results: Dict, perceived_value: float) -> Tuple[float, float]:
        """
        Stage 1 (COMPETITIVE) reward - compete with Conservative + Truthful opponents
        """
        result = auction_results.get(agent_id)
        stage_config = CURRICULUM_CONFIGS[self.curriculum_stage]
        reward_weights = stage_config.get('reward_weights', {})
        
        # Calculate profit and cost
        if result and result['won']:
            true_profit = self.current_true_value * result['slot_ctr']
            expected_cost = result['cost_per_click'] * result['slot_ctr']
            current_profit = float(true_profit - expected_cost)
            current_cost = float(min(expected_cost, self.agent_budgets[agent_id]))
        else:
            current_profit = 0.0
            current_cost = 0.0
        
        # Base profit reward
        profit_reward = current_profit * reward_weights.get('profit', 0.5) / 10.0
        
        # Win bonus
        win_bonus = 0.0
        if result and result['won']:
            win_bonus = reward_weights.get('win_bonus', 0.3)
        
        # ROI bonus - introduce efficiency awareness
        roi_bonus = 0.0
        if result and result['won'] and current_cost > 0:
            roi = (current_profit / current_cost) * 100
            if roi > 0:
                roi_bonus = reward_weights.get('roi', 0.2) * np.tanh(roi / 50.0)
            else:
                roi_bonus = -0.05  # Small penalty for unprofitable wins
        
        total_reward = profit_reward + win_bonus + roi_bonus
        return float(np.clip(total_reward, -2.0, 2.0)), float(current_profit)
    
    def _stage2_reward(self, agent_id: str, auction_results: Dict, perceived_value: float) -> Tuple[float, float]:
        """
        Stage 2 (ADVANCED) reward - compete with Aggressive opponents
        """
        result = auction_results.get(agent_id)
        stage_config = CURRICULUM_CONFIGS[self.curriculum_stage]
        reward_weights = stage_config.get('reward_weights', {})
        
        # Calculate profit and cost
        if result and result['won']:
            true_profit = self.current_true_value * result['slot_ctr']
            expected_cost = result['cost_per_click'] * result['slot_ctr']
            current_profit = float(true_profit - expected_cost)
            current_cost = float(min(expected_cost, self.agent_budgets[agent_id]))
        else:
            current_profit = 0.0
            current_cost = 0.0
        
        # Base profit reward
        profit_reward = current_profit * reward_weights.get('profit', 0.4) / 10.0
        
        # Win bonus
        win_bonus = 0.0
        if result and result['won']:
            win_bonus = reward_weights.get('win_bonus', 0.2)
        
        # ROI bonus (increased importance)
        roi_bonus = 0.0
        if result and result['won'] and current_cost > 0:
            roi = (current_profit / current_cost) * 100
            if roi > 0:
                roi_bonus = reward_weights.get('roi', 0.3) * np.tanh(roi / 50.0)
            else:
                roi_bonus = -0.1  # Larger penalty for unprofitable wins
        
        # Competition bonus - extra reward for beating aggressive opponents
        competition_bonus = 0.0
        if result and result['won']:
            # Check if we beat any aggressive agents (simplified)
            competition_bonus = reward_weights.get('competition', 0.1)
        
        total_reward = profit_reward + win_bonus + roi_bonus + competition_bonus
        return float(np.clip(total_reward, -2.0, 3.0)), float(current_profit)
    
    def _default_reward(self, agent_id: str, auction_results: Dict, perceived_value: float) -> Tuple[float, float]:
        """
        Default reward function (original simplified version)
        """
        result = auction_results.get(agent_id)
        
        # Calculate current round profit and cost
        if result and result['won']:
            true_profit = self.current_true_value * result['slot_ctr']
            expected_cost = result['cost_per_click'] * result['slot_ctr']
            current_profit = float(true_profit - expected_cost)
            current_cost = float(min(expected_cost, self.agent_budgets[agent_id]))
        else:
            current_profit = 0.0
            current_cost = 0.0
        
        # Base reward: immediate profit (positive when profitable)
        reward = current_profit
        
        # Simple efficiency bonus for profitable wins
        if result and result['won'] and current_cost > 0:
            immediate_roi = (current_profit / current_cost) * 100.0
            if immediate_roi > 0:
                # Bonus for profitable wins (encourages high ROI)
                efficiency_bonus = 0.1 * np.tanh(immediate_roi / 100.0)
                reward += efficiency_bonus
            else:
                # Small penalty for unprofitable wins
                reward -= 0.05
        
        # Small win bonus to encourage participation
        if result and result['won']:
            reward += 0.02
        
        # Conservative scaling for stable learning
        reward = reward / 2.0
        
        return float(reward), float(current_profit)
    
    def _full_stage_reward(self, agent_id: str, auction_results: Dict, perceived_value: float) -> Tuple[float, float]:
        """
        Stage 3 (FULL_SPECTRUM) reward - smooth transition to final objective
        """
        result = auction_results.get(agent_id)
        stage_config = CURRICULUM_CONFIGS[self.curriculum_stage]
        
        # Calculate profit and cost
        if result and result['won']:
            true_profit = self.current_true_value * result['slot_ctr']
            expected_cost = result['cost_per_click'] * result['slot_ctr']
            current_profit = float(true_profit - expected_cost)
            current_cost = float(min(expected_cost, self.agent_budgets[agent_id]))
        else:
            current_profit = 0.0
            current_cost = 0.0
        
        # Get transition progress (if implemented)
        transition_alpha = getattr(self, 'transition_alpha', 1.0)  # Default to full final objective
        
        if transition_alpha < 1.0:
            # Interpolate between stage 2 and final objective
            stage2_reward, _ = self._stage2_reward(agent_id, auction_results, perceived_value)
            final_reward = self._final_objective_reward(agent_id, current_profit, current_cost)
            
            total_reward = (1 - transition_alpha) * stage2_reward + transition_alpha * final_reward
        else:
            # Use full final objective
            total_reward = self._final_objective_reward(agent_id, current_profit, current_cost)
        
        return float(total_reward), float(current_profit)
    
    def _final_objective_reward(self, agent_id: str, auction_results: Dict, perceived_value: float) -> Tuple[float, float]:
        """
        Economic Value (EV) reward shaping:
        Maps README formula to per-step rewards:
        0.5 * Profit_t + 0.15 * ROI_frac_t * ΔCost_t + 0.35 * Win_t
        """
        # Import EV config
        from . import config
        
        # Calculate basic profit and cost
        result = auction_results.get(agent_id)
        if result and result['won']:
            true_profit = self.current_true_value * result['slot_ctr']
            expected_cost = result['cost_per_click'] * result['slot_ctr']
            current_profit = float(true_profit - expected_cost)
            current_cost = float(min(expected_cost, self.agent_budgets[agent_id]))
        else:
            current_profit = 0.0
            current_cost = 0.0
        
        # 修复4: 更新统计（累积 + 滑动窗口）
        self.cum_profit[agent_id] += current_profit
        self.cum_cost[agent_id] += current_cost
        self.recent_profits[agent_id].append(current_profit)
        self.recent_costs[agent_id].append(current_cost)
        
        if config.USE_EV_SHAPING:
            # EV-aligned reward shaping
            # 1) Profit component (当步利润)
            profit_component = config.EV_W_PROFIT * current_profit
            
            # 修复4: 2) ROI component - 使用滑动窗口ROI替代累积ROI
            # 问题：累积ROI非平稳且噪声后期放大
            # 解决：用最近500步的ROI，稳态化且更responsive
            roi_component = 0.0
            recent_cost_sum = sum(self.recent_costs[agent_id])
            if recent_cost_sum > config.EV_EPS:
                recent_profit_sum = sum(self.recent_profits[agent_id])
                roi_frac = recent_profit_sum / recent_cost_sum  # 滑动窗口ROI
                # 使用tanh限制范围，防止极端值
                roi_component = config.EV_W_ROI * np.tanh(roi_frac) * current_cost
            
            # 3) Win component (当步是否获胜)
            win_component = 0.0
            if result and result['won']:
                win_component = config.EV_W_WIN * 1.0
            
            total_reward = profit_component + roi_component + win_component
            
            # 可选：与原奖励混合
            if config.EV_SHAPING_ALPHA < 1.0:
                # 原始奖励（简化版）
                original_reward = current_profit / 10.0  # 原来的缩放
                total_reward = config.EV_SHAPING_ALPHA * total_reward + (1 - config.EV_SHAPING_ALPHA) * original_reward
        else:
            # 回退到原始奖励
            total_reward = current_profit / 10.0  # 保持原缩放
        
        return float(total_reward), float(current_profit)
    
    def _get_agent_bid(self, agent_id: str) -> float:
        """Helper to get the bid amount for an agent"""
        # This would need to be tracked during the bidding phase
        # For now, return a default
        return 0.0
    
    def render(self):
        """Optional rendering for debugging"""
        print(f"Round {self.current_round}/{self.max_rounds}")
        print(f"True Value: {self.current_true_value:.2f}")
        for agent_id in self.learning_agent_ids:
            budget = self.agent_budgets[agent_id]
            episode_reward = self.episode_rewards[agent_id]
            print(f"  {agent_id}: Budget={budget:.2f}, Episode Reward={episode_reward:.2f}")