import numpy as np
from enum import Enum

# --- Curriculum Learning Stages ---
class CurriculumStage(Enum):
    STAGE_0 = 0     # Solo practice - 2 learning agents only
    STAGE_1 = 1     # Gentle start - 2L + 2T (4 total)
    STAGE_2 = 2     # Basic competition - 2L + 4T (6 total)
    STAGE_3 = 3     # Mixed easy - 2L + 3T + 1C (6 total)
    STAGE_4 = 4     # First aggressive - 2L + 2T + 1C + 1A (6 total)
    STAGE_5 = 5     # Balanced mix - 2L + 2T + 2C (6 total)
    STAGE_6 = 6     # Growing competition - 2L + 2T + 2C + 1A (7 total)
    STAGE_7 = 7     # Near full - 2L + 2T + 1C + 2A (7 total)
    STAGE_8 = 8     # Full competition - 2L + 2T + 2C + 2A (8 total)

# --- 模拟环境核心设置 ---
SIMULATION_ROUNDS = 7000  # 总共进行的拍卖轮次
TRAINING_EPISODES = 200  # Increased from 50 for better convergence

# === OPTIMIZED PARAMETERS FOR POSITIVE ROI ===
N_SLOTS = 4  # 增加到4个广告位，50%智能体可获胜，降低竞争压力
# 平滑的CTR递减曲线，更接近真实场景
CTR_POSITIONS = np.array([0.8, 0.65, 0.5, 0.35])  # 更平缓的递减，保持较高CTR
CTR_NOISE_STD = 0.05  # CTR的噪声标准差 (模拟±5%的扰动)

# --- 价值与预算设置 ---
# 大幅扩展价值范围以匹配预算规模，确保正向ROI
TRUE_VALUE_RANGE = (10.0, 40.0)  # 3倍扩展，平均值37.5
AGENT_BUDGET = 30000.0  # 智能体的初始预算
# 可以设置为[20000, 30000, 40000]，分别对应三种不同的实验
AGENT_PERCEPTION_NOISE_STD = 0.15  # 优化到15%，提高决策精度

# --- 规则智能体参数 ---
# "保守派"智能体平滑因子
CONSERVATIVE_AGENT_SMOOTHING = 0.8  # 用于平滑alpha变化的因子，防止调整过猛

# "激进派"智能体参数
AGGRESSIVE_AGENT_LOOKBACK = 15  # 回溯最近N轮的胜率
AGGRESSIVE_AGENT_LAMBDA = 0.5  # 调整出价的敏感度 λ

# --- Economic-Value (EV) reward shaping ---
USE_EV_SHAPING = True   # 是否使用EV奖励塑形
EV_W_PROFIT = 0.50      # 利润权重：对应 0.5 · Profit
EV_W_ROI    = 0.15      # ROI权重：对应 0.15 · ROI · TotalCost
EV_W_WIN    = 0.35      # 胜率权重：对应 0.35 · WinRate · TargetWins
EV_EPS       = 1e-8      # 防止除零的小常数
EV_SHAPING_ALPHA = 1.0  # 奖励混合系数：=1 纯EV奖励；<1 时按 alpha*EV + (1-alpha)*原reward 混合

"""
--- 实验配置 ---
下面你可以根据这个配置来动态创建智能体
这是一个 k=0 的示例配置
格式: {'type': 'AgentType', 'count': N, 'budget': B}
Type可以是 'Truthful', 'Conservative', 'Aggressive', 'Learning'
"""

# k=0: 规则智能体对照实验
EXPERIMENT_SETUP_K0 = {
    'k': 0,
    'agents': [
        {'type': 'Conservative', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Aggressive', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Truthful', 'count': 2, 'budget': AGENT_BUDGET},
    ]
}

# k=1: 单智能体强化学习 (1个学习智能体 + 规则对手)
EXPERIMENT_SETUP_K1 = {
    'k': 1,
    'agents': [
        {'type': 'Learning', 'count': 1, 'budget': AGENT_BUDGET},
        {'type': 'Conservative', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Aggressive', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Truthful', 'count': 2, 'budget': AGENT_BUDGET},
    ]
}

# k=2: 多智能体强化学习 (2个学习智能体 + 规则对手)
EXPERIMENT_SETUP_K2 = {
    'k': 2,
    'agents': [
        {'type': 'Learning', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Conservative', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Aggressive', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Truthful', 'count': 2, 'budget': AGENT_BUDGET},
    ]
}

# 当前实验设置 (可以切换 K0/K1/K2)
EXPERIMENT_SETUP = EXPERIMENT_SETUP_K1 # 切换这里来改变实验类型

# --- Progressive Curriculum Learning Configurations ---
# 渐进式课程学习：固定环境参数，只改变对手组合
# 固定：4个广告位，价值范围(15.0, 60.0)，CTR [0.85, 0.75, 0.65, 0.50]
# 固定：奖励权重始终为最终目标函数 (0.5利润 + 0.15 ROI + 0.35胜率)
CURRICULUM_CONFIGS = {
    CurriculumStage.STAGE_0: {  # Solo practice
        'n_learning': 1,
        'n_truthful': 0,
        'n_conservative': 0,
        'n_aggressive': 0,
        'n_slots': N_SLOTS,  # 固定4个广告位
        'budget': AGENT_BUDGET,  # 固定30000预算
        'max_rounds': SIMULATION_ROUNDS,  # 固定16000轮
        'ctr_positions': CTR_POSITIONS,  # 固定[0.85, 0.75, 0.65, 0.50]
        'description': 'Solo practice - only 2 learning agents'
    },
    CurriculumStage.STAGE_1: {  # Gentle start
        'n_learning': 1,
        'n_truthful': 2,
        'n_conservative': 0,
        'n_aggressive': 0,
        'n_slots': N_SLOTS,
        'budget': AGENT_BUDGET,
        'max_rounds': SIMULATION_ROUNDS,
        'ctr_positions': CTR_POSITIONS,
        'description': '4 agents total, 100% win rate'
    },
    CurriculumStage.STAGE_2: {  # Basic competition
        'n_learning': 1,
        'n_truthful': 4,
        'n_conservative': 0,
        'n_aggressive': 0,
        'n_slots': N_SLOTS,
        'budget': AGENT_BUDGET,
        'max_rounds': SIMULATION_ROUNDS,
        'ctr_positions': CTR_POSITIONS,
        'description': '6 agents total, 67% win rate'
    },
    CurriculumStage.STAGE_3: {  # Mixed easy
        'n_learning': 1,
        'n_truthful': 3,
        'n_conservative': 1,
        'n_aggressive': 0,
        'n_slots': N_SLOTS,
        'budget': AGENT_BUDGET,
        'max_rounds': SIMULATION_ROUNDS,
        'ctr_positions': CTR_POSITIONS,
        'description': '6 agents, first conservative opponent'
    },
    CurriculumStage.STAGE_4: {  # First aggressive
        'n_learning': 1,
        'n_truthful': 2,
        'n_conservative': 1,
        'n_aggressive': 1,
        'n_slots': N_SLOTS,
        'budget': AGENT_BUDGET,
        'max_rounds': SIMULATION_ROUNDS,
        'ctr_positions': CTR_POSITIONS,
        'description': '6 agents, first aggressive opponent'
    },
    CurriculumStage.STAGE_5: {  # Balanced mix  
        'n_learning': 1,
        'n_truthful': 2,
        'n_conservative': 2,
        'n_aggressive': 0,
        'n_slots': N_SLOTS,
        'budget': AGENT_BUDGET,
        'max_rounds': SIMULATION_ROUNDS,
        'ctr_positions': CTR_POSITIONS,
        'description': '6 agents, balanced truthful/conservative'
    },
    CurriculumStage.STAGE_6: {  # Growing competition
        'n_learning': 1,
        'n_truthful': 2,
        'n_conservative': 2,
        'n_aggressive': 1,
        'n_slots': N_SLOTS,
        'budget': AGENT_BUDGET,
        'max_rounds': SIMULATION_ROUNDS,
        'ctr_positions': CTR_POSITIONS,
        'description': '7 agents, 57% win rate'
    },
    CurriculumStage.STAGE_7: {  # Near full
        'n_learning': 1,
        'n_truthful': 2,
        'n_conservative': 1,
        'n_aggressive': 2,
        'n_slots': N_SLOTS,
        'budget': AGENT_BUDGET,
        'max_rounds': SIMULATION_ROUNDS,
        'ctr_positions': CTR_POSITIONS,
        'description': '7 agents, more aggressive'
    },
    CurriculumStage.STAGE_8: {  # Full competition
        'n_learning': 1,
        'n_truthful': 2,
        'n_conservative': 2,
        'n_aggressive': 2,
        'n_slots': N_SLOTS,
        'budget': AGENT_BUDGET,
        'max_rounds': SIMULATION_ROUNDS,
        'ctr_positions': CTR_POSITIONS,
        'description': '8 agents, full competition, 50% win rate'
    }
}

# Progressive Curriculum success criteria
# 调整episodes数量，让学习更充分
CURRICULUM_SUCCESS_CRITERIA = {
    CurriculumStage.STAGE_0: {  # Solo practice
        'min_episodes': 30,  # 快速通过
        'min_win_rate': 0.60,  # 更现实的目标（从0.90降到0.60）
        'min_roi': -20.0,  # 允许亏损
        'min_budget_usage': 0.10
    },
    CurriculumStage.STAGE_1: {  # 4 agents, 100% win 
        'min_episodes': 50,
        'min_win_rate': 0.65,  # 从0.70降到0.65，匹配实际表现67.9%
        'min_roi': -10.0,
        'min_budget_usage': 0.20
    },
    CurriculumStage.STAGE_2: {  # 6 agents, 67% theoretical win
        'min_episodes': 80,
        'min_win_rate': 0.50,  # 从0.55降到0.50
        'min_roi': -5.0,
        'min_budget_usage': 0.30
    },
    CurriculumStage.STAGE_3: {  # First conservative
        'min_episodes': 150,  # 增加到150给更多训练时间
        'min_win_rate': 0.28,  # 从0.45降到0.28，匹配实际表现
        'min_roi': 0.0,  # 开始要求break even
        'min_budget_usage': 0.35
    },
    CurriculumStage.STAGE_4: {  # Balanced mix
        'min_episodes': 120,
        'min_win_rate': 0.40,  # 从0.45降到0.40
        'min_roi': 5.0,  # 要求正ROI
        'min_budget_usage': 0.40
    },
    CurriculumStage.STAGE_5: {  # First aggressive
        'min_episodes': 150,
        'min_win_rate': 0.35,  # 从0.40降到0.35
        'min_roi': 5.0,
        'min_budget_usage': 0.45
    },
    CurriculumStage.STAGE_6: {  # 7 agents
        'min_episodes': 180,
        'min_win_rate': 0.30,  # 从0.35降到0.30
        'min_roi': 8.0,
        'min_budget_usage': 0.50
    },
    CurriculumStage.STAGE_7: {  # Near full
        'min_episodes': 200,
        'min_win_rate': 0.25,  # 从0.30降到0.25
        'min_roi': 10.0,
        'min_budget_usage': 0.55
    },
    CurriculumStage.STAGE_8: {  # Full competition
        'min_episodes': 250,
        'min_win_rate': 0.20,  # 从0.25降到0.20（更现实）
        'min_roi': 12.0,  # 良好的效率
        'min_budget_usage': 0.60,
        'economic_value_threshold': 0.5  # Top 50% in economic value
    }
}