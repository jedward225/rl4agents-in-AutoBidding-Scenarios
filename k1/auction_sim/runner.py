# /auction_sim/runner.py
import numpy as np
import random
from tqdm import tqdm
from . import config
from .auction import GSPAuction
from .agents import Agent, TruthfulAgent, ConservativeAgent, AggressiveAgent, LearningAgent, MultiAgentLearningAgent
from .utils import generate_all_visualizations

def create_agents_from_config():
    """根据config中的设置创建智能体列表"""
    agents = []
    agent_id_counter = 0
    total_agents = sum(spec['count'] for spec in config.EXPERIMENT_SETUP['agents'])

    for spec in config.EXPERIMENT_SETUP['agents']:
        for _ in range(spec['count']):
            agent_id = f"{spec['type']}_{agent_id_counter}"
            agent_id_counter += 1
            
            if spec['type'] == 'Truthful':
                agents.append(TruthfulAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD))
            elif spec['type'] == 'Conservative':
                agents.append(ConservativeAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD, config.SIMULATION_ROUNDS))
            elif spec['type'] == 'Aggressive':
                agents.append(AggressiveAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD, total_agents))
            elif spec['type'] == 'Learning':
                # k > 0 时会创建学习智能体
                # 这里创建基础的LearningAgent作为占位符
                # 对于k=2，请使用 ma_runner.py 进行多智能体训练
                agents.append(LearningAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD))
            else:
                raise ValueError(f"Unknown agent type: {spec['type']}")
    
    print(f"Created {len(agents)} agents for the experiment.")
    return agents


def main():
    """主模拟函数"""
    # 1. 初始化
    random.seed(42)
    np.random.seed(42)
    
    auction = GSPAuction(config.N_SLOTS, config.CTR_POSITIONS, config.CTR_NOISE_STD)
    agents = create_agents_from_config()
    
    all_history = {agent.id: [] for agent in agents}

    # 2. 运行模拟
    print("Starting simulation...")
    for round_num in tqdm(range(1, config.SIMULATION_ROUNDS + 1), desc="Auction Rounds"):
        # 1. 生成真实价值
        true_value = random.uniform(*config.TRUE_VALUE_RANGE)
        
        # 2. 所有智能体出价
        bids = {}
        perceived_values = {}
        for agent in agents:
            perceived_value = agent.perceive(true_value)
            bid_price = agent.bid(perceived_value)
            
            # 只有当智能体有足够预算时才参与竞价
            if agent.can_afford_bid(bid_price):
                perceived_values[agent.id] = perceived_value
                bids[agent.id] = bid_price

        # 3. 运行拍卖
        if bids:
            auction_results = auction.run_auction(bids)
        else:
            auction_results = {}  # 本轮无人出价

        # 4. 更新所有智能体状态并计算利润
        for agent in agents:
            result = auction_results.get(agent.id)
            
            # 计算利润：Profit_t = TrueValue_t × CTR_pos - Cost_t
            if result and result['won']:
                true_value_profit = true_value * result['slot_ctr']
                expected_cost = result['cost_per_click'] * result['slot_ctr']
                profit = true_value_profit - expected_cost
            else:
                profit = 0.0
            
            # 更新智能体状态，传递真实价值和利润信息
            agent.update(result, round_num, true_value=true_value, profit=profit)

    # 3. 结果分析
    print("\n--- Simulation Finished ---")
    print("Final Results:")
    print(f"{'Agent ID':<20} | {'Budget Left':<12} | {'Total Cost':<12} | {'Win Count':<10} | {'Cumulative Profit':<18} | {'ROI (%)':<10}")
    print("-" * 95)
    
    for agent in agents:
        total_cost = agent.get_total_cost()
        win_count = sum(1 for record in agent.history if record['result'] and record['result']['won'])
        cumulative_profit = agent.get_cumulative_profit()
        roi = agent.get_roi()
        
        print(
            f"{agent.id:<20} | "
            f"{agent.budget:8.2f}     | "
            f"{total_cost:8.2f}     | "
            f"{win_count:4d}       | "
            f"{cumulative_profit:8.2f}          | "
            f"{roi:8.2f}"
        )

    # 4. 生成所有可视化图表
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS...")
    print("="*50)
    
    try:
        generate_all_visualizations(agents)
        print("\n✅ All visualizations generated successfully!")
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        print("You may need to install matplotlib, pandas, and scipy:")
        print("pip install matplotlib pandas scipy")

if __name__ == '__main__':
    main()