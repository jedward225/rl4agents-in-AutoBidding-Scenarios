# /auction_sim/bc_data_collector.py
"""
Behavioral Cloning Data Collection Module
Collects expert demonstrations from AggressiveAgent for pre-training
"""
import numpy as np
import pickle
import os
from tqdm import tqdm
from typing import List, Dict, Tuple
from . import config
from .auction import GSPAuction
from .agents import TruthfulAgent, ConservativeAgent, AggressiveAgent

class BCDataCollector:
    """Collects behavioral cloning data from expert agents"""
    
    def __init__(self, expert_agent_type='Aggressive'):
        self.expert_agent_type = expert_agent_type
        self.data = []  # List of (state, action, reward, info) tuples
        
    def create_expert_environment(self) -> Tuple[List, GSPAuction]:
        """Create environment with expert agent and rule-based opponents"""
        agents = []
        agent_id_counter = 0
        
        # Create diverse opponent mix for realistic training
        # Configuration: 2 Conservative, 2 Truthful, 1 Aggressive (expert), 1 extra Truthful
        opponent_configs = [
            ('Conservative', 2),
            ('Truthful', 2), 
            ('Aggressive', 1),  # This will be our expert
            ('Truthful', 1)     # Total 6 agents
        ]
        
        expert_agent = None
        
        for agent_type, count in opponent_configs:
            for i in range(count):
                agent_id = f"{agent_type}_{agent_id_counter}"
                agent_id_counter += 1
                
                if agent_type == 'Truthful':
                    agent = TruthfulAgent(
                        agent_id, 
                        config.AGENT_BUDGET, 
                        config.AGENT_PERCEPTION_NOISE_STD
                    )
                elif agent_type == 'Conservative':
                    agent = ConservativeAgent(
                        agent_id, 
                        config.AGENT_BUDGET, 
                        config.AGENT_PERCEPTION_NOISE_STD,
                        config.SIMULATION_ROUNDS
                    )
                elif agent_type == 'Aggressive':
                    total_agents = sum(c for _, c in opponent_configs)
                    agent = AggressiveAgent(
                        agent_id, 
                        config.AGENT_BUDGET, 
                        config.AGENT_PERCEPTION_NOISE_STD,
                        total_agents
                    )
                    if expert_agent is None:  # First aggressive agent is our expert
                        expert_agent = agent
                
                agents.append(agent)
        
        # Create auction environment
        auction = GSPAuction(config.N_SLOTS, config.CTR_POSITIONS, config.CTR_NOISE_STD)
        
        return agents, auction, expert_agent
    
    def get_learning_agent_state(self, expert_agent, perceived_value: float, 
                               current_round: int, max_rounds: int) -> np.ndarray:
        """Build state representation that matches LearningAgent's observation space"""
        
        # 1. Normalized perceived value
        norm_perceived_value = (perceived_value - config.TRUE_VALUE_RANGE[0]) / \
                              (config.TRUE_VALUE_RANGE[1] - config.TRUE_VALUE_RANGE[0])
        norm_perceived_value = np.clip(norm_perceived_value, 0.0, 1.0)
        
        # 2. Remaining budget ratio
        budget_ratio = expert_agent.budget / expert_agent.initial_budget
        
        # 3. Time ratio
        time_ratio = (max_rounds - current_round) / max_rounds
        
        # 4. Recent win rate (from expert's history)
        recent_wins = 0
        recent_rounds = min(len(expert_agent.history), 100)
        if recent_rounds > 0:
            recent_history = expert_agent.history[-recent_rounds:]
            recent_wins = sum(1 for record in recent_history 
                            if record.get('result') and record['result'].get('won', False))
            recent_win_rate = recent_wins / recent_rounds
        else:
            recent_win_rate = 0.0
        
        # 5. Recent average profit (normalized)
        recent_profit = 0.0
        if recent_rounds > 0:
            recent_history = expert_agent.history[-recent_rounds:]
            profits = [record.get('profit', 0.0) for record in recent_history]
            profits = [float(p) if p is not None else 0.0 for p in profits]
            recent_profit = np.mean(profits) if profits else 0.0
            recent_profit = np.clip(recent_profit / 10.0, -1.0, 1.0)
        
        # 6. Mock opponent win rate (simplified)
        opponent_win_rate = 0.3  # Reasonable default
        
        # 7. Market competition level (simplified)
        market_competition = 0.5  # Default moderate competition
        
        state = np.array([
            norm_perceived_value,
            budget_ratio,
            time_ratio,
            recent_win_rate,
            recent_profit,
            opponent_win_rate,
            market_competition
        ], dtype=np.float32)
        
        return state
    
    def collect_episode_data(self, n_rounds: int = 16000) -> List[Dict]:
        """Collect data from one episode of expert play"""
        agents, auction, expert_agent = self.create_expert_environment()
        episode_data = []
        
        print(f"Collecting data from {expert_agent.id} against {len(agents)-1} opponents")
        
        for round_num in range(n_rounds):
            # Generate true value
            true_value = np.random.uniform(*config.TRUE_VALUE_RANGE)
            
            # Get perceived values and bids for all agents
            bids = {}
            perceived_values = {}
            
            for agent in agents:
                perceived_value = agent.perceive(true_value)
                perceived_values[agent.id] = perceived_value
                
                # Check if agent can afford to bid
                if agent.budget > 0:
                    bid = agent.bid(perceived_value)
                    if agent.can_afford_bid(bid):
                        bids[agent.id] = bid
            
            # Record expert's decision
            if expert_agent.id in bids:
                expert_perceived_value = perceived_values[expert_agent.id]
                expert_bid = bids[expert_agent.id]
                
                # Get state representation
                state = self.get_learning_agent_state(
                    expert_agent, expert_perceived_value, round_num, n_rounds
                )
                
                # Normalize action (bid) to [0.5, 1.5] range (matching action space)
                if expert_perceived_value > 0:
                    bid_ratio = expert_bid / expert_perceived_value
                    # Clip to action space bounds
                    normalized_action = np.clip(bid_ratio, 0.5, 1.5)
                else:
                    normalized_action = 1.0  # Default reasonable bid
                
                # Run auction
                if bids:
                    auction_results = auction.run_auction(bids)
                    expert_result = auction_results.get(expert_agent.id)
                    
                    # Calculate reward (using stage 1 reward function)
                    if expert_result and expert_result['won']:
                        true_profit = true_value * expert_result['slot_ctr']
                        expected_cost = expert_result['cost_per_click'] * expert_result['slot_ctr']
                        current_profit = float(true_profit - expected_cost)
                        reward = current_profit / 2.0 + 0.1  # Win bonus
                    else:
                        current_profit = 0.0
                        reward = 0.0
                        
                    # Store data point
                    episode_data.append({
                        'state': state.copy(),
                        'action': normalized_action,
                        'reward': reward,
                        'true_value': true_value,
                        'perceived_value': expert_perceived_value,
                        'bid': expert_bid,
                        'won': expert_result['won'] if expert_result else False,
                        'profit': current_profit
                    })
                else:
                    auction_results = {}
                
                # Update agents
                for agent in agents:
                    result = auction_results.get(agent.id)
                    if result and result['won']:
                        true_value_profit = true_value * result['slot_ctr']
                        expected_cost = result['cost_per_click'] * result['slot_ctr']
                        profit = true_value_profit - expected_cost
                    else:
                        profit = 0.0
                    
                    agent.update(result, round_num, true_value, profit)
            
            # Check if expert ran out of budget
            if expert_agent.budget <= 0:
                print(f"Expert agent ran out of budget at round {round_num}")
                break
        
        return episode_data
    
    def collect_dataset(self, n_episodes: int = 20, save_path: str = "bc_dataset.pkl") -> Dict:
        """Collect complete dataset from multiple episodes"""
        print(f"Collecting BC dataset from {n_episodes} episodes...")
        
        all_data = []
        episode_stats = []
        
        for episode in tqdm(range(n_episodes), desc="Collecting Episodes"):
            episode_data = self.collect_episode_data()
            
            if episode_data:
                all_data.extend(episode_data)
                
                # Calculate episode statistics
                total_rounds = len(episode_data)
                wins = sum(1 for d in episode_data if d['won'])
                total_profit = sum(d['profit'] for d in episode_data)
                avg_reward = np.mean([d['reward'] for d in episode_data])
                
                episode_stats.append({
                    'episode': episode,
                    'rounds': total_rounds,
                    'wins': wins,
                    'win_rate': wins / total_rounds if total_rounds > 0 else 0,
                    'total_profit': total_profit,
                    'avg_reward': avg_reward
                })
                
                if episode % 5 == 0:
                    print(f"Episode {episode}: {wins}/{total_rounds} wins ({wins/total_rounds:.1%}), "
                          f"Profit: {total_profit:.0f}, Avg Reward: {avg_reward:.2f}")
        
        # Prepare dataset
        dataset = {
            'data': all_data,
            'stats': episode_stats,
            'config': {
                'expert_type': self.expert_agent_type,
                'n_episodes': n_episodes,
                'total_samples': len(all_data)
            }
        }
        
        # Save dataset
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\nDataset collected successfully!")
        print(f"Total samples: {len(all_data)}")
        print(f"Average win rate: {np.mean([s['win_rate'] for s in episode_stats]):.1%}")
        print(f"Saved to: {save_path}")
        
        return dataset

def main():
    """Test data collection"""
    collector = BCDataCollector()
    dataset = collector.collect_dataset(
        n_episodes=5,  # Small test
        save_path="auction_sim/bc_dataset_test.pkl"
    )
    
    # Print sample data
    print("\nSample data points:")
    for i in range(min(3, len(dataset['data']))):
        sample = dataset['data'][i]
        print(f"State: {sample['state']}")
        print(f"Action: {sample['action']:.3f}, Bid: {sample['bid']:.2f}, Won: {sample['won']}")
        print(f"Reward: {sample['reward']:.2f}, Profit: {sample['profit']:.2f}")
        print()

if __name__ == "__main__":
    main()