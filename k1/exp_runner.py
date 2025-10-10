# /auction_sim/experiment_runner.py
import numpy as np
import random
from tqdm import tqdm
from typing import List, Dict, Any
from auction_sim import config
from auction_sim.auction import GSPAuction
from auction_sim.agents import Agent, TruthfulAgent, ConservativeAgent, AggressiveAgent, MultiAgentLearningAgent
from auction_sim.utils import generate_all_visualizations
from auction_sim.ippo_trainer import IPPOTrainer, IPPOActorCriticNetwork # 导入 IPPO 相关模块
from bc_maddpg_trainer import MADDPGTrainer # 导入 DDPG 相关模块
from auction_sim.ma_trainer import MAPPOTrainer # 导入 Curriculum 相关模块
from auction_logger import AuctionLogger # 导入日志记录模块
from auction_visualizer import AuctionVisualizer, visualize_from_log # 导入可视化模块

def detect_model_type(model_path: str) -> str:
    """检测模型类型（IPPO、DDPG或Curriculum）"""
    if not os.path.exists(model_path):
        return "unknown"
    
    # 检查目录中的文件来判断模型类型
    files = os.listdir(model_path)
    
    # DDPG模型文件格式: Learning_0_actor.pth, Learning_0_critic.pth
    ddpg_files = [f for f in files if f.endswith('_actor.pth') or f.endswith('_critic.pth')]
    
    # IPPO模型文件格式: Learning_0_ippo_model.pth
    ippo_files = [f for f in files if f.endswith('_ippo_model.pth')]
    
    # Curriculum模型文件格式: shared_model.pth 或 Learning_0_model.pth (MAPPO)
    curriculum_files = [f for f in files if f == 'shared_model.pth' or f.endswith('_model.pth')]
    
    if ddpg_files:
        return "ddpg"
    elif ippo_files:
        return "ippo"
    elif curriculum_files:
        return "curriculum"
    else:
        return "unknown"

def create_agents_from_config(total_rounds: int, model_paths: Dict[str, str] = None) -> List[Agent]:
    """根据config中的设置创建智能体列表，并加载预训练模型（如果提供了路径）"""
    agents = []
    agent_id_counter = 0
    learning_idx = 0  # 只对学习体计数，用于正确对齐模型
    total_agents = sum(spec['count'] for spec in config.EXPERIMENT_SETUP['agents'])
    learning_agents_spec = next((spec for spec in config.EXPERIMENT_SETUP['agents'] if spec['type'] == 'Learning'), None)
    
    # 初始化trainer，根据模型类型选择IPPO或DDPG
    trainer = None
    model_type = "unknown"
    if learning_agents_spec and model_paths:
        num_learning_agents = learning_agents_spec['count']
        model_path = model_paths['learning_agent']
        model_type = detect_model_type(model_path)
        
        print(f"检测到模型类型: {model_type}")
        
        if model_type == "ippo":
            # 修复：obs_dim从7更新到9（匹配训练时的观测维度）
            trainer = IPPOTrainer(n_agents=num_learning_agents, obs_dim=9, action_dim=1)
            trainer.load_models(model_path)
            print(f"已加载IPPO模型: {model_path}")
        elif model_type == "ddpg":
            # DDPG模式要求单智能体
            if num_learning_agents != 1:
                print(f"警告: DDPG模式要求单智能体，但配置了{num_learning_agents}个学习智能体")
                print("将只使用第一个学习智能体")
            # 修复：obs_dim从7更新到9（匹配训练时的观测维度）
            trainer = MADDPGTrainer(n_agents=1, obs_dim=9, action_dim=1)
            trainer.load_models(model_path)
            print(f"已加载DDPG模型: {model_path}")
        elif model_type == "curriculum":
            # 修复：obs_dim从7更新到9（匹配训练时的观测维度）
            trainer = MAPPOTrainer(n_agents=num_learning_agents, obs_dim=9, action_dim=1)
            trainer.load_models(model_path)
            print(f"已加载Curriculum模型: {model_path}")
        else:
            print(f"未知的模型类型或模型路径不存在: {model_path}")
            print("将使用未训练的智能体")

    for spec in config.EXPERIMENT_SETUP['agents']:
        for _ in range(spec['count']):
            agent_id_prefix = spec['type']
            agent_id = f"{agent_id_prefix}_{agent_id_counter}"
            
            if agent_id_prefix == 'Truthful':
                agent = TruthfulAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD)
            elif agent_id_prefix == 'Conservative':
                agent = ConservativeAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD, total_rounds)
            elif agent_id_prefix == 'Aggressive':
                agent = AggressiveAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD, total_agents)
            elif agent_id_prefix == 'Learning':
                agent = MultiAgentLearningAgent(
                    agent_id=agent_id,
                    budget=spec['budget'],
                    perception_noise_std=config.AGENT_PERCEPTION_NOISE_STD,
                    is_training=False # 模拟时设置为非训练模式
                )
                # 将加载的模型与智能体关联，使用learning_idx正确对齐
                if trainer:
                    trainer_key = f"Learning_{learning_idx}"
                    
                    if model_type == "ippo" and trainer_key in trainer.networks:
                        # IPPO模型包装器
                        class IPPOModelWrapper:
                            def __init__(self, network, trainer, key):
                                self.network = network
                                self.trainer = trainer
                                self.key = key
                            
                            def predict(self, obs, deterministic=True):
                                # P1修复：与训练时保持一致，直接调用network.get_action()
                                # 训练时在bc_ippo_trainer.py中也是直接调用network，跳过了trainer的观测归一化
                                # 如果这里用trainer.get_action()会引入训练时没有的归一化，导致分布偏移
                                import torch
                                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                                with torch.no_grad():
                                    # network.get_action()返回3个值: (action, log_prob, value)
                                    action, _, _ = self.network.get_action(obs_tensor, deterministic)
                                return action.cpu().numpy(), None
                        
                        model_wrapper = IPPOModelWrapper(trainer.networks[trainer_key], trainer, trainer_key)
                        agent.set_model(model_wrapper)
                        print(f"已为 {agent.id} 加载IPPO模型 <- {trainer_key}")
                        
                    elif model_type == "ddpg" and trainer_key in trainer.agents:
                        # DDPG模型包装器
                        class DDPGModelWrapper:
                            def __init__(self, ddpg_agent, trainer, key):
                                self.ddpg_agent = ddpg_agent
                                self.trainer = trainer
                                self.key = key
                            
                            def predict(self, obs, deterministic=True):
                                # DDPG的get_action方法返回numpy数组，不需要log_prob和value
                                action = self.trainer.get_action(self.key, obs, add_noise=not deterministic)
                                return np.array([action[0]]) if isinstance(action, np.ndarray) else np.array([action]), None
                        
                        model_wrapper = DDPGModelWrapper(trainer.agents[trainer_key], trainer, trainer_key)
                        agent.set_model(model_wrapper)
                        print(f"已为 {agent.id} 加载DDPG模型 <- {trainer_key}")
                        
                    elif model_type == "curriculum" and trainer_key in trainer.networks:
                        # Curriculum模型包装器 (MAPPO)
                        class CurriculumModelWrapper:
                            def __init__(self, network, trainer, key):
                                self.network = network
                                self.trainer = trainer
                                self.key = key
                            
                            def predict(self, obs, deterministic=True):
                                # MAPPOTrainer的get_action只返回action和value，没有log_prob
                                action, value = self.trainer.get_action(self.key, obs, deterministic)
                                return np.array([action]), None
                        
                        model_wrapper = CurriculumModelWrapper(trainer.networks[trainer_key], trainer, trainer_key)
                        agent.set_model(model_wrapper)
                        print(f"已为 {agent.id} 加载Curriculum模型 <- {trainer_key}")
                    
                    elif trainer:
                        print(f"警告: 无法为 {agent.id} 找到对应的模型 {trainer_key}")
                learning_idx += 1  # 只对学习体递增
            else:
                raise ValueError(f"Unknown agent type: {agent_id_prefix}")
            
            agents.append(agent)
            agent_id_counter += 1
    
    print(f"Created {len(agents)} agents for the experiment.")
    return agents

def run_experiment(model_paths: Dict[str, str] = None, show_visuals: bool = True, eval_rounds: int = None, 
                   enable_logging: bool = True, log_dir: str = "auction_logs"):
    """
    运行完整的拍卖模拟实验，并记录所有数据。

    Args:
        model_paths: 可选的字典，指定要加载的预训练模型路径。
                     例如：{'learning_agent': 'auction_sim/models/bc_ippo/final'}
        show_visuals: 是否生成并显示可视化图表。
        eval_rounds: 评估轮数，默认5000（避免规则agents预算耗尽导致不公平竞争）
        enable_logging: 是否启用详细日志记录（记录每轮数据）
        log_dir: 日志文件保存目录
    """
    # P1修复：默认使用5000轮评估，避免规则agents预算耗尽后Learning agent零成本垄断
    if eval_rounds is None:
        eval_rounds = 5000  # 公平竞争的轮数
    
    # 1. 初始化
    random.seed(42)
    np.random.seed(42)
    
    auction = GSPAuction(config.N_SLOTS, config.CTR_POSITIONS, config.CTR_NOISE_STD)
    agents = create_agents_from_config(eval_rounds, model_paths)
    
    # 初始化日志记录器
    logger = None
    if enable_logging:
        logger = AuctionLogger(log_dir=log_dir)
        config_info = {
            'eval_rounds': eval_rounds,
            'n_agents': len(agents),
            'agent_types': [{'id': str(a.id), 'type': a.__class__.__name__} for a in agents],
            'model_paths': model_paths
        }
        logger.start_session(config_info=config_info)
        print(f"📝 日志记录已启用，保存目录: {log_dir}")
    
    # 存储所有回合数据
    all_round_data = []

    # 2. 运行模拟
    print("Starting simulation...")
    
    for round_num in tqdm(range(1, eval_rounds + 1), desc="Auction Rounds"):
        true_value = random.uniform(*config.TRUE_VALUE_RANGE)
        
        bids = {}
        perceived_values = {}
        opponent_win_rates = {}
        
        # 收集对手信息，用于学习智能体的观测
        learning_agent_ids = [a.id for a in agents if isinstance(a, MultiAgentLearningAgent)]
        for agent in agents:
            if agent.id not in learning_agent_ids:
                if hasattr(agent, 'win_history') and agent.win_history:
                    opponent_win_rates[agent.id] = sum(agent.win_history) / len(agent.win_history)

        for agent in agents:
            perceived_value = agent.perceive(true_value)
            
            # 学习智能体需要额外的环境信息
            if isinstance(agent, MultiAgentLearningAgent):
                bid_price = agent.bid(
                    perceived_value,
                    current_round=round_num,
                    max_rounds=eval_rounds,
                    opponent_win_rates=opponent_win_rates
                )
            else:
                bid_price = agent.bid(perceived_value)
            
            # 获取最低保底价（防止出价过低导致意外高成本）
            min_reserve = min(auction.reserve_per_slot) if len(auction.reserve_per_slot) > 0 else 0.0
            
            # P1修复：检查预算和保底价（与训练环境ma_environment保持一致）
            # 1) 有足够预算支付出价 2) 出价不低于保底价
            # 这确保智能体不会出价过低而在拍卖时被强制支付更高的保底价
            if (agent.can_afford_bid(bid_price) and 
                bid_price >= min_reserve):
                perceived_values[agent.id] = perceived_value
                bids[agent.id] = bid_price
            # 否则拒绝出价（不参与本轮拍卖）

        auction_results = auction.run_auction(bids) if bids else {}

        # 记录本回合数据
        round_data = {
            'round_num': round_num,
            'true_value': true_value,
            'bids': bids,
            'perceived_values': perceived_values,
            'results': auction_results,
            'profits': {},
            'costs': {}
        }
        
        for agent in agents:
            result = auction_results.get(agent.id)
            profit = 0.0
            
            if result and result['won']:
                true_value_profit = true_value * result['slot_ctr']
                expected_cost = result['cost_per_click'] * result['slot_ctr']
                cost = min(expected_cost, agent.budget)
                profit = true_value_profit - cost
                
            agent.update(result, round_num, true_value=true_value, profit=profit)
            
            round_data['profits'][agent.id] = profit
            round_data['costs'][agent.id] = agent.history[-1]['cost']
            
        all_round_data.append(round_data)
        
        # 记录到日志
        if logger:
            logger.log_round(round_data)

    # 3. 结果分析与展示
    print("\n--- Simulation Finished ---")
    print(f"Evaluation Rounds: {eval_rounds} (Fair competition period)")
    print("Final Results:")
    
    # 计算全局参数
    total_rounds = eval_rounds
    num_agents = len(agents)
    target_wins = (total_rounds / num_agents) * 0.8
    
    # 收集所有智能体的评测数据
    agent_results = []
    
    for agent in agents:
        # Convert numpy arrays to scalars using .item() method to avoid deprecation warnings
        initial_budget = agent.initial_budget.item() if hasattr(agent.initial_budget, 'item') else float(agent.initial_budget)
        total_cost = agent.get_total_cost().item() if hasattr(agent.get_total_cost(), 'item') else float(agent.get_total_cost())
        win_count = sum(1 for record in agent.history if record['result'] and record['result']['won'])
        cumulative_profit = agent.get_cumulative_profit().item() if hasattr(agent.get_cumulative_profit(), 'item') else float(agent.get_cumulative_profit())
        roi = agent.get_roi().item() if hasattr(agent.get_roi(), 'item') else float(agent.get_roi())
        current_budget = agent.budget.item() if hasattr(agent.budget, 'item') else float(agent.budget)
        budget_usage_raw = (initial_budget - current_budget) / initial_budget * 100
        budget_usage = float(budget_usage_raw) if hasattr(budget_usage_raw, 'item') else float(budget_usage_raw)
        
        # 计算新增的评测指标
        win_rate = win_count / total_rounds if total_rounds > 0 else 0.0  # 胜率比例
        
        # ROI转换为小数形式用于目标函数计算（避免量纲问题）
        roi_frac = roi / 100.0
        
        # 综合经济价值计算
        economic_value = (
            0.5 * cumulative_profit +
            0.15 * roi_frac * total_cost +
            0.35 * win_rate * target_wins
        )
        
        agent_id_str = str(agent.id)
        # Ensure agent_id_str is a proper string for formatting
        if hasattr(agent.id, '__iter__') and not isinstance(agent.id, str):
            agent_id_str = str(agent.id)
        else:
            agent_id_str = agent.id
        
        agent_results.append({
            'agent_id': agent_id_str,
            'current_budget': current_budget,
            'total_cost': total_cost,
            'win_count': win_count,
            'cumulative_profit': cumulative_profit,
            'roi': roi,
            'budget_usage': budget_usage,
            'win_rate': win_rate,
            'target_wins': target_wins,
            'economic_value': economic_value
        })
    
    # 按综合经济价值排序（从高到低）
    agent_results.sort(key=lambda x: x['economic_value'], reverse=True)
    
    # 打印表格头部
    print(f"{'Agent ID':<20} | {'Budget Left':<12} | {'Win Count':<10} | {'Cumulative Profit':<18} | {'ROI (%)':<10} | {'Budget Usage (%)':<15} | {'Win Rate':<10} | {'Economic Value':<15}")
    print("-" * 140)
    
    # 打印排序后的结果
    for result in agent_results:
        print(
            f"{result['agent_id']:<20} | "
            f"{result['current_budget']:8.2f}     | "
            f"{result['win_count']:4d}       | "
            f"{result['cumulative_profit']:8.2f}          | "
            f"{result['roi']:8.2f}       | "
            f"{result['budget_usage']:8.2f}        | "
            f"{result['win_rate']:8.3f}   | "
            f"{result['economic_value']:8.2f}"
        )
    
    # 打印排名总结
    print("\n" + "="*60)
    print("PERFORMANCE RANKING (by Economic Value):")
    print("="*60)
    for i, result in enumerate(agent_results, 1):
        print(f"{i:2d}. {result['agent_id']:<20} | Economic Value: {result['economic_value']:8.2f} | Win Rate: {result['win_rate']:6.3f} | ROI: {result['roi']:6.2f}%")
    
    print(f"\n📊 Evaluation Metrics Summary:")
    print(f"   • Total Rounds: {total_rounds}")
    print(f"   • Number of Agents: {num_agents}")
    print(f"   • Economic Value Formula: 0.5×Profit + 0.15×ROI×Cost + 0.35×WinRate×TargetWins")

    # 4. 保存日志并生成新的可视化图表
    log_file_path = None
    if logger:
        print("\n" + "="*60)
        print("SAVING LOGS AND GENERATING DETAILED VISUALIZATIONS...")
        print("="*60)
        
        try:
            # 记录智能体汇总信息
            logger.log_agent_summary(agents)
            
            # 保存日志文件
            log_file_path = logger.save_log()
            
            # 生成新的详细可视化（4张图：cost, 出价位次, 经济价值, ROI）
            if show_visuals:
                print("\n🎨 正在生成详细性能分析图表...")
                visualize_from_log(
                    log_file=log_file_path,
                    save_dir="auction_visualizations",
                    show=True,
                    combined=True  # 生成2x2组合图
                )
                print("✅ 详细可视化图表生成成功！")
        except Exception as e:
            print(f"\n❌ 日志或可视化生成错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 5. 生成原有的可视化图表
    if show_visuals:
        print("\n" + "="*50)
        print("GENERATING ADDITIONAL VISUALIZATIONS...")
        print("="*50)
        
        try:
            # 修复：传入eval_rounds（整数）而不是all_round_data（列表）
            generate_all_visualizations(agents, eval_rounds)
            print("\n✅ All additional visualizations generated successfully!")
        except Exception as e:
            print(f"\n❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()

    return agents, all_round_data, log_file_path

if __name__ == '__main__':
    import os
    
    # 示例：运行一个包含预训练学习智能体的实验
    # 支持自动检测IPPO、DDPG和Curriculum模型类型
    
    # IPPO模型路径示例
    trained_model_path = 'auction_sim/models/bc_ippo/best'
    # trained_model_path = 'auction_sim/models/bc_ppo/best'
    # trained_model_path = 'auction_sim/models/bc_ippo1/best'
    
    # DDPG模型路径示例（取消注释以使用）
    # trained_model_path = 'auction_sim/models/bc_maddpg/final'
    
    # Curriculum模型路径示例（取消注释以使用）
    # trained_model_path = 'auction_sim/models/pure_curriculum/STAGE_8_final'
    
    # 其他模型路径示例
    # trained_model_path = ''  # 不使用预训练模型


    # try:
    # 尝试加载模型
    if not os.path.exists(trained_model_path):
        print(f"警告: 找不到模型路径 '{trained_model_path}'。将运行没有预训练模型的模拟。")
        final_agents, final_data, log_file = run_experiment(model_paths=None, enable_logging=True)
    else:
        final_agents, final_data, log_file = run_experiment(
            model_paths={'learning_agent': trained_model_path}, 
            eval_rounds=config.SIMULATION_ROUNDS,
            enable_logging=True,
            log_dir="auction_logs"
        )
    
    print(f"\n🎉 实验完成！")
    if log_file:
        print(f"📁 日志文件: {log_file}")
        print(f"📊 可视化图表已保存到: auction_visualizations/")
            
    # except Exception as e:
    #     print(f"运行失败: {e}")
    #     print("请检查你的配置和文件路径。")