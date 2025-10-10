import numpy as np
from abc import ABC, abstractmethod
from typing import Dict
from collections import deque
from . import config

class Agent(ABC):
    """所有智能体的抽象基类"""
    def __init__(self, agent_id: str, budget: float, perception_noise_std: float):
        self.id = agent_id
        self.initial_budget = budget
        self.budget = budget
        self.perception_noise_std = perception_noise_std
        
        # 用于记录历史数据
        self.history = []

    def perceive(self, true_value: float) -> float:
        """模拟对真实价值的感知，加入噪声"""
        perceived_value = true_value + np.random.normal(0, self.perception_noise_std)
        return max(0, perceived_value) # 感知价值不能为负

    @abstractmethod
    def bid(self, perceived_value: float) -> float:
        """
        核心出价方法，由子类实现。
        """
        pass

    def can_afford_bid(self, bid_price: float) -> bool:
        """检查智能体是否有足够预算支付出价"""
        return self.budget >= bid_price

    def update(self, result: Dict, round_num: int, true_value: float = None, profit: float = None):
        """
        根据一轮拍卖的结果更新自身状态。
        """
        if result and result['won']:
            # 计算期望成本 = CPC × CTR，本轮实际花费不得超过剩余预算
            expected_cost = float(result['cost_per_click']) * float(result['slot_ctr'])
            cost = min(expected_cost, self.budget)
            self.budget -= cost
        else:
            cost = 0.0
        
        # 记录本轮数据
        self.history.append({
            'round': round_num,
            'result': result,
            'cost': cost,
            'budget': self.budget,
            'true_value': true_value,
            'profit': profit if profit is not None else 0.0,
        })

    def get_cumulative_profit(self) -> float:
        """计算累计利润"""
        total = 0.0
        for record in self.history:
            profit = record.get('profit', 0.0)
            # Ensure profit is a scalar
            if hasattr(profit, '__len__') and not isinstance(profit, str):
                profit = float(profit[0]) if len(profit) > 0 else 0.0
            total += float(profit)
        return float(total)
    
    def get_total_cost(self) -> float:
        """计算总花费"""
        total = 0.0
        for record in self.history:
            cost = record.get('cost', 0.0)
            # Ensure cost is a scalar
            if hasattr(cost, '__len__') and not isinstance(cost, str):
                cost = float(cost[0]) if len(cost) > 0 else 0.0
            total += float(cost)
        return float(total)
    
    def get_roi(self) -> float:
        """
        计算ROI = 累计利润 / 累计成本 * 100%
        如果成本太小（<1元），返回0避免ROI爆炸
        """
        total_cost = self.get_total_cost()
        # 如果成本太小（<1元），认为没有有效投资，返回0
        if total_cost < 1.0:
            return 0.0
        total_profit = self.get_cumulative_profit()
        return float((total_profit / total_cost) * 100)

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, budget={self.budget:.2f})"

# --- 规则智能体 ---

class TruthfulAgent(Agent):
    """“老实人”智能体：出价等于感知价值"""
    def bid(self, perceived_value: float) -> float:
        return perceived_value

class ConservativeAgent(Agent):
    """“保守派”智能体：根据预算消耗节奏调整出价"""
    def __init__(self, agent_id: str, budget: float, perception_noise_std: float, total_rounds: int):
        super().__init__(agent_id, budget, perception_noise_std)
        self.total_rounds = total_rounds
        self.alpha = 1.0  # 初始调整因子

    def bid(self, perceived_value: float) -> float:
        current_round = len(self.history) + 1
        if current_round <= 1:
            return perceived_value * self.alpha

        # 1. 计算理想花费速度
        ideal_pace = self.initial_budget / self.total_rounds
        
        # 2. 计算实际花费速度
        budget_spent = self.initial_budget - self.budget
        actual_pace = budget_spent / (current_round -1)

        # 3. 计算并平滑更新alpha
        if actual_pace < 1e-6: # 避免除以0
            new_alpha = 1.1 # 如果还没花钱，就稍微激进一点
        else:
            # 使用一个平滑函数来更新alpha，防止剧烈波动
            target_alpha = ideal_pace / actual_pace
            new_alpha = config.CONSERVATIVE_AGENT_SMOOTHING * self.alpha + \
                        (1 - config.CONSERVATIVE_AGENT_SMOOTHING) * target_alpha
        
        self.alpha = max(0.1, min(new_alpha, 3.0)) # 限制alpha范围

        return perceived_value * self.alpha

class AggressiveAgent(Agent):
    """“激进派”智能体：根据近期胜率调整出价"""
    def __init__(self, agent_id: str, budget: float, perception_noise_std: float, n_agents: int):
        super().__init__(agent_id, budget, perception_noise_std)
        self.target_win_rate = 1.0 / n_agents
        self.win_history = deque(maxlen=config.AGGRESSIVE_AGENT_LOOKBACK)
        self.beta = 1.0

    def bid(self, perceived_value: float) -> float:
        if not self.win_history:
            current_win_rate = 0.0
        else:
            current_win_rate = sum(self.win_history) / len(self.win_history)
        
        # 更新好胜因子 beta
        self.beta = 1.0 + config.AGGRESSIVE_AGENT_LAMBDA * (self.target_win_rate - current_win_rate)
        self.beta = max(0.5, min(self.beta, 2.5)) # 限制beta范围

        return perceived_value * self.beta

    def update(self, result: Dict, round_num: int, true_value: float = None, profit: float = None):
        super().update(result, round_num, true_value, profit)
        self.win_history.append(1 if result and result['won'] else 0)

# --- 学习智能体 (框架) ---

class LearningAgent(Agent):
    """“自适应”学习智能体：使用强化学习模型进行出价"""
    def __init__(self, agent_id: str, budget: float, perception_noise_std: float):
        super().__init__(agent_id, budget, perception_noise_std)
        # TODO: 在这里初始化你的RL模型 (e.g., PPO from stable-baselines3)
        # self.model = PPO(...)
        # self.replay_buffer = ReplayBuffer(...)
        print(f"LearningAgent {self.id} initialized.")
    
    def get_state(self) -> np.ndarray:
        """
        构建并返回当前的状态向量，用于输入RL模型。
        这是你需要根据方案详细实现的部分。
        """
        # 这是一个示例状态，你需要填充真实数据
        state = np.array([
            0.0, # perceived_value_t
            0.0, # remaining_budget_ratio
            0.0, # time_ratio
            0.0, # recent_win_rate
            0.0, # recent_avg_profit_per_win
        ])
        return state

    def bid(self, perceived_value: float) -> float:
        """
        使用RL模型预测动作并转化为出价。
        """
        # 1. 获取当前状态
        # state = self.get_state(perceived_value, ...)
        
        # 2. 使用模型预测动作
        # action, _ = self.model.predict(state, deterministic=True)
        # raw_action = action[0] # PPO/DDPG通常返回一个数组
        
        # 3. 将动作缩放到合理的出价范围
        # bid_price = self.scale_action(raw_action, perceived_value)
        
        # [占位符] 目前使用随机出价作为示例
        bid_price = perceived_value * np.random.uniform(0.8, 1.2)
        
        return bid_price

    def scale_action(self, raw_action: float, perceived_value: float) -> float:
        """将模型输出 (-1, 1) 映射到出价。这是关键步骤！"""
        # 示例：将tanh输出映射到 [0, 2 * perceived_value]
        # max_bid_multiplier = 2.0
        # bid_price = (raw_action + 1) / 2 * max_bid_multiplier * perceived_value
        # return bid_price
        pass
    
    def update(self, result: Dict, round_num: int, true_value: float = None, profit: float = None):
        """
        更新智能体状态，并将 (s, a, r, s') 存入经验池。
        """
        super().update(result, round_num, true_value, profit)
        # TODO:
        # 1. 计算奖励 R
        # 2. 获取下一个状态 S'
        # 3. 将 (S, A, R, S') 添加到回放池
        # 4. (可选) 在这里或在runner中调用 model.learn()

class MultiAgentLearningAgent(Agent):
    """
    Multi-Agent Learning Agent for k=2 scenario
    """
    def __init__(self, agent_id: str, budget: float, perception_noise_std: float, 
                 model=None, is_training: bool = True):
        super().__init__(agent_id, budget, perception_noise_std)
        self.model = model
        self.is_training = is_training
        self.last_observation = None
        self.last_action = None
        
        # Multi-agent specific tracking
        self.win_history = deque(maxlen=100)
        self.profit_history = deque(maxlen=100)
        # P1修复：增加bid_history长度以匹配win_history，用于计算近K轮统计
        self.bid_history = deque(maxlen=100)
        self.perceived_value_history = deque(maxlen=100)
        
        print(f"MultiAgentLearningAgent {self.id} initialized (training={is_training})")
    
    def set_model(self, model):
        """Set the RL model for this agent"""
        self.model = model
    
    def get_observation(self, perceived_value: float, current_round: int, max_rounds: int, 
                       opponent_win_rates: Dict[str, float] = None) -> np.ndarray:
        """Build observation vector for multi-agent learning"""
        from . import config
        
        # 1. Normalized perceived value
        norm_perceived_value = (perceived_value - config.TRUE_VALUE_RANGE[0]) / \
                              (config.TRUE_VALUE_RANGE[1] - config.TRUE_VALUE_RANGE[0])
        norm_perceived_value = np.clip(norm_perceived_value, 0.0, 1.0)
        
        # 2. Remaining budget ratio
        budget_ratio = self.budget / self.initial_budget
        
        # 3. Time ratio
        time_ratio = (max_rounds - current_round) / max_rounds
        
        # 4. Recent win rate
        recent_win_rate = sum(self.win_history) / len(self.win_history) if self.win_history else 0.0
        
        # 5. Recent average profit (normalized)
        recent_profit = np.mean(self.profit_history) if self.profit_history else 0.0
        recent_profit = np.clip(recent_profit / 10.0, -1.0, 1.0)
        
        # 6. Opponent win rates (average if multiple opponents)
        opponent_win_rate = 0.0
        if opponent_win_rates:
            opponent_win_rate = np.mean(list(opponent_win_rates.values()))
        
        # 7. Competition level (simplified for now)
        competition_level = 0.6
        
        # 8. P1修复：近K轮平均赢价（归一化）
        recent_avg_bid = 0.0
        if len(self.bid_history) > 0 and len(self.win_history) > 0:
            # 只统计赢标的轮次，使用实际可用的最小长度
            min_len = min(len(self.bid_history), len(self.win_history))
            winning_bids = [bid for bid, won in zip(list(self.bid_history)[-min_len:], list(self.win_history)[-min_len:]) if won]
            if winning_bids:
                # 归一化到[0, 1]范围（假设最大出价约为TRUE_VALUE_RANGE上界的3倍）
                max_bid = config.TRUE_VALUE_RANGE[1] * 3.0 if hasattr(config, 'TRUE_VALUE_RANGE') else 300.0
                recent_avg_bid = np.mean(winning_bids) / max_bid
                recent_avg_bid = np.clip(recent_avg_bid, 0.0, 1.0)
        
        # 9. P1修复：近K轮平均CPC（归一化）
        recent_avg_cpc = 0.0
        if len(self.history) > 0:
            # 从历史记录中提取CPC（只统计赢标的轮次）
            # 注意：cost_per_click存储在result字段内部
            winning_cpcs = []
            for record in self.history[-100:]:
                result = record.get('result')
                if result and result.get('won', False):
                    cpc = result.get('cost_per_click', 0.0)
                    if cpc > 0:
                        winning_cpcs.append(cpc)
            
            if winning_cpcs:
                # 归一化CPC（假设最大CPC约为TRUE_VALUE_RANGE上界）
                max_cpc = config.TRUE_VALUE_RANGE[1] if hasattr(config, 'TRUE_VALUE_RANGE') else 100.0
                recent_avg_cpc = np.mean(winning_cpcs) / max_cpc
                recent_avg_cpc = np.clip(recent_avg_cpc, 0.0, 1.0)
        
        observation = np.array([
            norm_perceived_value,
            budget_ratio,
            time_ratio,
            recent_win_rate,
            recent_profit,
            opponent_win_rate,
            competition_level,
            recent_avg_bid,    # P1修复：近K轮平均赢价
            recent_avg_cpc     # P1修复：近K轮平均CPC
        ], dtype=np.float32)
        
        self.last_observation = observation
        return observation
    
    def bid(self, perceived_value: float, current_round: int = 1, max_rounds: int = 1000, 
            opponent_win_rates: Dict[str, float] = None) -> float:
        """Use RL model to predict bid"""
        if self.model is None:
            # Fallback to random bidding if no model
            bid_multiplier = np.random.uniform(0.8, 1.2)
            return perceived_value * bid_multiplier
        
        # Get observation
        observation = self.get_observation(perceived_value, current_round, max_rounds, opponent_win_rates)
        
        try:
            # Use model to predict action
            if hasattr(self.model, 'predict'):
                action, _ = self.model.predict(observation, deterministic=not self.is_training)
                # 确保action是标量：处理所有可能的数组格式
                if isinstance(action, (np.ndarray, list)):
                    action = float(action.flatten()[0])  # 展平后取第一个元素
                else:
                    action = float(action)
            else:
                action = self.model.act(observation)
                if isinstance(action, (np.ndarray, list)):
                    action = float(action.flatten()[0])
                else:
                    action = float(action)
            
            # P1修复：统一动作域到 [0, 1.5]
            # 与BC网络输出和训练环境保持一致，覆盖所有合理策略
            bid_multiplier = float(np.clip(action, 0.0, 1.5))
            
        except Exception as e:
            print(f"Warning: Model prediction failed for {self.id}: {e}")
            bid_multiplier = 1.0  # Safe fallback
        
        self.last_action = bid_multiplier
        bid_price = float(perceived_value * bid_multiplier)
        
        # Track bidding behavior for enhanced observations
        self.bid_history.append(bid_price)
        self.perceived_value_history.append(perceived_value)
        
        return bid_price
    
    def update(self, result: Dict, round_num: int, true_value: float = None, profit: float = None):
        """Update agent state and learning history"""
        super().update(result, round_num, true_value, profit)
        
        # Update multi-agent specific tracking
        won = result['won'] if result else False
        self.win_history.append(1 if won else 0)
        
        # Update profit history with scalar values only
        profit_value = profit if profit is not None else 0.0
        # Ensure profit_value is a scalar
        if hasattr(profit_value, '__len__') and not isinstance(profit_value, str):
            profit_value = float(profit_value[0]) if len(profit_value) > 0 else 0.0
        else:
            profit_value = float(profit_value)
        self.profit_history.append(profit_value)
        
        # Store experience for learning (if in training mode)
        if self.is_training and hasattr(self, '_store_experience'):
            self._store_experience(result)
    
    def _store_experience(self, result: Dict):
        """Store experience for replay buffer (to be implemented by training framework)"""
        # This will be overridden by the training loop
        pass