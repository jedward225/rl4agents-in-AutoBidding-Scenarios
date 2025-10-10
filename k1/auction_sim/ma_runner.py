# /auction_sim/ma_runner.py
"""
Multi-Agent Training Runner for k=2 scenario
Fixed agent configuration and reward function
"""
import numpy as np
import random
import torch
from tqdm import tqdm
import os
from . import config
from .config import CurriculumStage, CURRICULUM_CONFIGS
from .curriculum_controller import CurriculumController
from .ma_environment import MultiAgentAuctionEnv
from .ma_trainer import MAPPOTrainer
from .agents import TruthfulAgent, ConservativeAgent, AggressiveAgent, MultiAgentLearningAgent
def create_rule_agents(curriculum_stage: CurriculumStage = None):
    """Create rule-based opponents for training (curriculum-aware)"""
    rule_agents = []
    agent_id_counter = 0
    
    if curriculum_stage is not None:
        # Use curriculum configuration
        stage_config = CURRICULUM_CONFIGS[curriculum_stage]
        budget = stage_config['budget']
        max_rounds = stage_config['max_rounds']
        
        # Create Truthful agents
        for i in range(stage_config['n_truthful']):
            agent_id = f"Truthful_{agent_id_counter}"
            agent_id_counter += 1
            agent = TruthfulAgent(agent_id, budget, config.AGENT_PERCEPTION_NOISE_STD)
            rule_agents.append(agent)
        
        # Create Conservative agents
        for i in range(stage_config['n_conservative']):
            agent_id = f"Conservative_{agent_id_counter}"
            agent_id_counter += 1
            agent = ConservativeAgent(agent_id, budget, config.AGENT_PERCEPTION_NOISE_STD, max_rounds)
            rule_agents.append(agent)
        
        # Create Aggressive agents with stage-specific parameters
        for i in range(stage_config['n_aggressive']):
            agent_id = f"Aggressive_{agent_id_counter}"
            agent_id_counter += 1
            
            # 获取阶段特定的 Aggressive 配置
            if 'aggressive_config' in stage_config:
                agg_config = stage_config['aggressive_config']
                agent = AggressiveAgent(
                    agent_id, 
                    budget, 
                    config.AGENT_PERCEPTION_NOISE_STD,
                    n_agents=None,  # 使用新的参数化模式
                    lookback=agg_config['lookback'],
                    lambda_param=agg_config['lambda'],
                    beta_max=agg_config['beta_max']
                )
            else:
                # 使用默认配置（向后兼容）
                total_agents = stage_config['n_learning'] + stage_config['n_truthful'] + \
                              stage_config['n_conservative'] + stage_config['n_aggressive']
                agent = AggressiveAgent(agent_id, budget, config.AGENT_PERCEPTION_NOISE_STD, total_agents)
            
            rule_agents.append(agent)
    else:
        # Use default configuration
        for spec in config.EXPERIMENT_SETUP['agents']:
            if spec['type'] != 'Learning':
                for _ in range(spec['count']):
                    agent_id = f"{spec['type']}_{agent_id_counter}"
                    agent_id_counter += 1
                    
                    if spec['type'] == 'Truthful':
                        agent = TruthfulAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD)
                    elif spec['type'] == 'Conservative':
                        agent = ConservativeAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD, config.SIMULATION_ROUNDS)
                    elif spec['type'] == 'Aggressive':
                        total_agents = sum(s['count'] for s in config.EXPERIMENT_SETUP['agents'])
                        agent = AggressiveAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD, total_agents)
                    
                    rule_agents.append(agent)
    
    return rule_agents

def create_learning_agents(curriculum_stage: CurriculumStage = None):
    """Create learning agents with proper integration (curriculum-aware)"""
    learning_agents = []
    learning_agent_ids = []
    
    if curriculum_stage is not None:
        # Use curriculum configuration
        stage_config = CURRICULUM_CONFIGS[curriculum_stage]
        learning_count = stage_config['n_learning']
        budget = stage_config['budget']
    else:
        # Use default configuration
        learning_count = 0
        budget = config.AGENT_BUDGET
        for spec in config.EXPERIMENT_SETUP['agents']:
            if spec['type'] == 'Learning':
                learning_count = spec['count']
                budget = spec.get('budget', config.AGENT_BUDGET)
                break
    
    for i in range(learning_count):
        agent_id = f"Learning_{i}"
        agent = MultiAgentLearningAgent(
            agent_id, 
            budget, 
            config.AGENT_PERCEPTION_NOISE_STD,
            model=None,
            is_training=True
        )
        learning_agents.append(agent)
        learning_agent_ids.append(agent_id)
    
    return learning_agents, learning_agent_ids

def train_multi_agent(n_episodes: int = 100, save_models: bool = True, use_curriculum: bool = True):
    """
    Main training function for multi-agent scenario with curriculum learning
    """
    print("="*60)
    if use_curriculum:
        print("MULTI-AGENT TRAINING WITH CURRICULUM LEARNING")
    else:
        print("MULTI-AGENT TRAINING (k=2) - FIXED VERSION")
    print("="*60)
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize curriculum controller if needed
    curriculum_controller = None
    if use_curriculum:
        curriculum_controller = CurriculumController(CurriculumStage.SOLO)
        current_stage = curriculum_controller.current_stage
    else:
        current_stage = None
    
    # Create agents based on curriculum stage
    rule_agents = create_rule_agents(current_stage)
    learning_agents, learning_agent_ids = create_learning_agents(current_stage)
    
    print(f"Created {len(rule_agents)} rule-based agents")
    print(f"Created {len(learning_agents)} learning agents")
    
    # Create environment with curriculum stage
    env = SingleAgentAuctionEnv(
        learning_agent_ids, 
        rule_agents,
        curriculum_stage=current_stage if current_stage else CurriculumStage.FULL
    )
    
    # Create trainer with improved hyperparameters
    trainer = MAPPOTrainer(
        obs_dim=9,  # 修复：从7更新到9（增强观测特征）
        action_dim=1, 
        n_agents=len(learning_agents),  # Start with current number of agents
        lr=1e-4,
        gamma=0.95,
        gae_lambda=0.9,
        clip_ratio=0.1,
        vf_coef=0.5,
        ent_coef=0.02,
        max_grad_norm=0.3,
        use_curriculum=use_curriculum
    )
    
    # Connect models to agents
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
    
    # Training loop
    print(f"Starting training for {n_episodes} episodes...")
    if use_curriculum:
        print(f"Starting with curriculum stage: {current_stage.name}")
    
    best_avg_reward = float('-inf')
    episode_rewards_history = []
    
    for episode in tqdm(range(n_episodes), desc="Training Episodes"):
        # Use curriculum-specific or full simulation rounds
        if use_curriculum and current_stage:
            stage_config = CURRICULUM_CONFIGS[current_stage]
            max_steps = stage_config['max_rounds']
        else:
            max_steps = config.SIMULATION_ROUNDS
            
        episode_rewards, episode_wins = trainer.train_episode(env, max_steps=max_steps)
        
        # Calculate episode statistics for curriculum
        if use_curriculum and curriculum_controller:
            # Calculate statistics
            total_wins = sum(episode_wins.values())
            total_agents = len(learning_agents) + len(rule_agents)
            avg_win_rate = total_wins / (len(learning_agents) * max_steps) if len(learning_agents) > 0 else 0.0
            
            # Calculate ROI for learning agents
            avg_roi = 0.0
            budget_usage_ratio = 0.0
            avg_bid_ratio = 1.0  # Default
            
            for i, agent in enumerate(learning_agents):
                if hasattr(agent, 'get_roi'):
                    avg_roi += agent.get_roi()
                if hasattr(agent, 'initial_budget') and hasattr(agent, 'budget'):
                    budget_used = agent.initial_budget - agent.budget
                    budget_usage_ratio += budget_used / agent.initial_budget
            
            if len(learning_agents) > 0:
                avg_roi /= len(learning_agents)
                budget_usage_ratio /= len(learning_agents)
            
            # Update curriculum controller
            episode_stats = {
                'avg_win_rate': avg_win_rate,
                'avg_roi': avg_roi,
                'budget_usage_ratio': budget_usage_ratio,
                'avg_bid_ratio': avg_bid_ratio,
                'total_reward': sum(episode_rewards.values())
            }
            
            curriculum_controller.update_metrics(episode_stats)
            
            # Check if should advance to next stage
            if curriculum_controller.should_advance():
                print(f"\n{'='*60}")
                print(f"ADVANCING TO NEXT CURRICULUM STAGE!")
                print(f"Completed {current_stage.name} after {curriculum_controller.stage_episodes} episodes")
                print(f"Stage summary: {curriculum_controller.get_stage_summary()}")
                print(f"{'='*60}\n")
                
                if curriculum_controller.advance_stage():
                    current_stage = curriculum_controller.current_stage
                    
                    # Recreate environment with new stage
                    rule_agents = create_rule_agents(current_stage)
                    learning_agents, learning_agent_ids = create_learning_agents(current_stage)
                    
                    env = SingleAgentAuctionEnv(
                        learning_agent_ids,
                        rule_agents,
                        curriculum_stage=current_stage
                    )
                    
                    # Add new agents to trainer if needed
                    for i, agent in enumerate(learning_agents):
                        agent_id = f"Learning_{i}"
                        trainer.add_agent(agent_id)  # This will only add if agent doesn't exist
                    
                    # Reconnect models to new agents
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
                    
                    print(f"Now training on stage: {current_stage.name}")
                    print(f"Agents: {len(learning_agents)} learning, {len(rule_agents)} rule-based")
        
        # Track progress
        avg_reward = np.mean(list(episode_rewards.values()))
        episode_rewards_history.append(avg_reward)
        
        # Save best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            if save_models:
                trainer.save_models("auction_sim/models/best")
        
        # Logging
        if episode % 10 == 0:
            recent_avg = np.mean(episode_rewards_history[-10:]) if len(episode_rewards_history) >= 10 else avg_reward
            win_rates = {aid: (episode_wins[aid] / max_steps) for aid in learning_agent_ids}
            
            print(f"\nEpisode {episode} ({max_steps} rounds):")
            if use_curriculum and curriculum_controller:
                print(f"  {curriculum_controller.get_progress_string()}")
            print(f"  Average reward: {recent_avg:.2f}")
            print(f"  Episode rewards: {episode_rewards}")
            print(f"  Episode wins: {episode_wins}")
            print(f"  Win rates: {win_rates}")
            print(f"  Best avg reward: {best_avg_reward:.2f}")
    
    # Save final models
    if save_models:
        trainer.save_models("auction_sim/models/final")
        trainer.plot_training_curves("auction_sim/results/ma_training_curves.png")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    
    return trainer, learning_agents, rule_agents

def evaluate_trained_agents(trainer: MAPPOTrainer, n_eval_episodes: int = 3):
    """
    Evaluate trained agents against rule-based opponents
    """
    print("\n" + "="*50)
    print("EVALUATING TRAINED AGENTS")
    print("="*50)
    
    # Create evaluation environment
    rule_agents = create_rule_agents()
    learning_agents, learning_agent_ids = create_learning_agents()
    
    # Set models to evaluation mode
    for i, agent in enumerate(learning_agents):
        agent_id = f"Learning_{i}"
        agent.is_training = False
        
        class EvalModelWrapper:
            def __init__(self, network, trainer, agent_id):
                self.network = network
                self.trainer = trainer
                self.agent_id = agent_id
            
            def predict(self, obs, deterministic=True):
                action, _ = self.trainer.get_action(self.agent_id, obs, deterministic=True)
                return np.array([action]), None
        
        eval_wrapper = EvalModelWrapper(trainer.networks[agent_id], trainer, agent_id)
        agent.set_model(eval_wrapper)
    
    env = SingleAgentAuctionEnv(learning_agent_ids, rule_agents)
    
    # Run evaluation episodes
    all_rewards = []
    all_wins = []
    
    for episode in range(n_eval_episodes):
        obs = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in learning_agent_ids}
        episode_wins = {agent_id: 0 for agent_id in learning_agent_ids}
        
        for step in range(config.SIMULATION_ROUNDS):
            actions = {}
            
            for agent_id in learning_agent_ids:
                action, _ = trainer.get_action(agent_id, obs[agent_id], deterministic=True)
                actions[agent_id] = np.array([action])
            
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            for agent_id in learning_agent_ids:
                episode_rewards[agent_id] += rewards[agent_id]
                if step < len(env.agent_histories[agent_id]) and env.agent_histories[agent_id][-1]['won']:
                    episode_wins[agent_id] += 1
            
            obs = next_obs
            
            if any(terminated.values()) or any(truncated.values()):
                break
        
        all_rewards.append(episode_rewards)
        all_wins.append(episode_wins)
        
        print(f"Eval Episode {episode + 1}: Rewards = {episode_rewards}, Wins = {episode_wins}")
    
    # Calculate statistics
    avg_rewards = {}
    avg_win_rates = {}
    
    for agent_id in learning_agent_ids:
        rewards = [ep_rewards[agent_id] for ep_rewards in all_rewards]
        wins = [ep_wins[agent_id] for ep_wins in all_wins]
        
        avg_rewards[agent_id] = np.mean(rewards)
        avg_win_rates[agent_id] = np.mean(wins) / config.SIMULATION_ROUNDS
    
    print(f"\nEvaluation Results (averaged over {n_eval_episodes} episodes):")
    print(f"Average Rewards: {avg_rewards}")
    print(f"Average Win Rates: {avg_win_rates}")
    
    return avg_rewards, avg_win_rates

def run_full_experiment():
    """
    Run complete multi-agent experiment: train, evaluate, and visualize
    """
    # Ensure results directory exists
    os.makedirs("auction_sim/results", exist_ok=True)
    os.makedirs("auction_sim/models", exist_ok=True)
    
    # Training phase
    trainer, learning_agents, rule_agents = train_multi_agent(n_episodes=config.TRAINING_EPISODES)
    
    # Evaluation phase
    avg_rewards, avg_win_rates = evaluate_trained_agents(trainer, n_eval_episodes=3)
    
    # Create agents for final simulation with trained models
    print("\n" + "="*50)
    print("RUNNING FINAL SIMULATION")
    print("="*50)
    
    learning_agent_ids = [agent.id for agent in learning_agents]
    
    env = SingleAgentAuctionEnv(learning_agent_ids, rule_agents)
    obs = env.reset()
    
    # Set models to evaluation mode for the simulation
    for i, agent in enumerate(learning_agents):
        agent_id = f"Learning_{i}"
        agent.is_training = False
    
    print("Running final simulation to generate agent histories...")
    for step in tqdm(range(config.SIMULATION_ROUNDS), desc="Final Simulation"):
        actions = {}
        
        # Get actions for all learning agents
        for agent_id in learning_agent_ids:
            action, _ = trainer.get_action(agent_id, obs[agent_id], deterministic=True)
            actions[agent_id] = np.array([action])
        
        # Environment step
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        obs = next_obs
        
        if any(terminated.values()) or any(truncated.values()):
            break
    
    # Create final agents with populated histories
    final_learning_agents = []
    for i, agent_id in enumerate(learning_agent_ids):
        agent = SingleAgentLearningAgent(
            agent_id,
            config.AGENT_BUDGET,
            config.AGENT_PERCEPTION_NOISE_STD,
            is_training=False
        )
        
        # Copy history from environment
        agent.history = []
        for record in env.agent_histories[agent_id]:
            agent.history.append({
                'round': record['round'],
                'result': {'won': record['won']} if record['won'] else None,
                'cost': record['cost'],
                'budget': record['budget']
            })
        agent.budget = env.agent_budgets[agent_id]
        
        final_learning_agents.append(agent)
    
    # Combine all agents for visualization
    all_agents = final_learning_agents + rule_agents
    
    # Import and use updated utils with economic value
    import utils
    
    # Generate comprehensive results with new economic value metrics
    utils.generate_all_visualizations(all_agents, total_rounds=config.SIMULATION_ROUNDS)
    
    # Also show economic value ranking
    print("\n" + "="*80)
    print("ECONOMIC VALUE RANKING (Based on New Multi-Objective Target)")
    print("="*80)
    
    # Calculate economic values for ranking
    econ_results = []
    for agent in all_agents:
        econ_metrics = utils.calculate_economic_value(agent, config.SIMULATION_ROUNDS, len(all_agents))
        econ_results.append({
            'agent': agent,
            'economic_value': econ_metrics['economic_value'],
            'profit_term': econ_metrics['profit_term'],
            'efficiency_term': econ_metrics['efficiency_term'],
            'competitive_term': econ_metrics['competitive_term']
        })
    
    # Sort by economic value
    econ_results.sort(key=lambda x: x['economic_value'], reverse=True)
    
    print(f"{'Rank':<4} | {'Agent_ID':<12} | {'Economic_Value':<14} | {'Profit':<10} | {'Efficiency':<10} | {'Competitive':<12}")
    print("-" * 80)
    
    for i, result in enumerate(econ_results, 1):
        agent = result['agent']
        print(
            f"{i:<4} | "
            f"{agent.id:<12} | "
            f"{result['economic_value']:<14.2f} | "
            f"{result['profit_term']:<10.2f} | "
            f"{result['efficiency_term']:<10.2f} | "
            f"{result['competitive_term']:<12.2f}"
        )
    
    print("\n✅ Multi-agent experiment completed successfully!")
    print("Check auction_sim/results/ for models and training curves")

if __name__ == "__main__":
    run_full_experiment()