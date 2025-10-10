# /auction_sim/hybrid_trainer.py
"""
Hybrid BC + Curriculum Learning Training Pipeline
Implements the complete "先模仿，后进阶" approach
"""
import numpy as np
import torch
import os
from tqdm import tqdm
from typing import Dict, List, Tuple
import pickle

from . import config
from .config import CurriculumStage, CURRICULUM_CONFIGS
from .bc_data_collector import BCDataCollector
from .bc_trainer import BCTrainer
from .ma_runner import create_rule_agents, create_learning_agents
from .ma_environment import MultiAgentAuctionEnv
from .ma_trainer import MAPPOTrainer
from .curriculum_controller import CurriculumController

class HybridTrainer:
    """Hybrid BC + Curriculum Learning Trainer"""
    
    def __init__(self, use_bc_pretraining: bool = True, bc_episodes: int = 20):
        self.use_bc_pretraining = use_bc_pretraining
        self.bc_episodes = bc_episodes
        self.bc_model_path = "auction_sim/models/bc_pretrained.pth"
        self.bc_dataset_path = "auction_sim/bc_dataset.pkl"
        
        # Initialize curriculum controller with symmetric stages
        self.curriculum_controller = CurriculumController(CurriculumStage.STAGE_0)
        
    def step1_bc_pretraining(self):
        """Step 1: Behavioral Cloning Pre-training"""
        print("="*70)
        print("STEP 1: BEHAVIORAL CLONING PRE-TRAINING")
        print("="*70)
        
        if not os.path.exists(self.bc_dataset_path):
            print("Collecting BC dataset from AggressiveAgent...")
            collector = BCDataCollector()
            collector.collect_dataset(
                n_episodes=self.bc_episodes,
                save_path=self.bc_dataset_path
            )
        else:
            print(f"Using existing BC dataset: {self.bc_dataset_path}")
        
        print("\\nTraining BC model...")
        bc_trainer = BCTrainer()
        bc_results = bc_trainer.train(
            dataset_path=self.bc_dataset_path,
            n_epochs=100,
            save_path=self.bc_model_path
        )
        
        print("\\nEvaluating BC model...")
        bc_metrics = bc_trainer.evaluate_model(self.bc_dataset_path)
        
        bc_trainer.plot_training_curves("auction_sim/results/bc_training_curves.png")
        
        print(f"\\nBC Pre-training completed!")
        print(f"Action Accuracy: {bc_metrics['action_accuracy_10pct']:.1%}")
        print(f"RMSE: {bc_metrics['rmse']:.4f}")
        
        return bc_trainer, bc_results, bc_metrics
    
    def step2_load_bc_weights(self, trainer: MAPPOTrainer):
        """Step 2: Load BC weights into RL networks"""
        print("\\nLoading BC weights into all learning agents...")
        
        # Load BC model state
        bc_state_dict = torch.load(self.bc_model_path)
        
        # Load weights into all learning agent networks
        for agent_id, network in trainer.networks.items():
            network.load_state_dict(bc_state_dict)
            print(f"Loaded BC weights into {agent_id}")
        
        print("All learning agents initialized with BC weights!")
    
    def step3_symmetric_curriculum(self, target_final_stage: CurriculumStage = None):
        """Step 3: Symmetric Curriculum Learning - trains until curriculum completion"""
        print("="*70)
        print("STEP 3: SYMMETRIC CURRICULUM LEARNING")
        print("="*70)
        
        # Start with initial stage
        current_stage = self.curriculum_controller.current_stage
        
        # Set target final stage (default to highest available stage)
        if target_final_stage is None:
            all_stages = list(CurriculumStage)
            target_final_stage = all_stages[-1]  # STAGE_8
        
        print(f"Training target: Complete up to {target_final_stage.name}")
        print(f"Training will continue until curriculum is completed")
        
        # Create initial environment
        rule_agents = create_rule_agents(current_stage)
        learning_agents, learning_agent_ids = create_learning_agents(current_stage)
        
        env = MultiAgentAuctionEnv(
            learning_agent_ids,
            rule_agents,
            curriculum_stage=current_stage
        )
        
        # Create trainer
        trainer = MAPPOTrainer(
            obs_dim=9,  # 修复：从7更新到9（增强观测特征）
            action_dim=1,
            n_agents=len(learning_agents),
            lr=1e-4,
            gamma=0.95,
            gae_lambda=0.9,
            clip_ratio=0.1,
            vf_coef=0.5,
            ent_coef=0.02,
            max_grad_norm=0.3,
            use_curriculum=True
        )
        
        # Step 2: Load BC weights if using pretraining
        if self.use_bc_pretraining:
            self.step2_load_bc_weights(trainer)
        
        # Connect models to agents
        self._connect_models_to_agents(trainer, learning_agents)
        
        print(f"Starting intelligent curriculum training...")
        print(f"Initial stage: {current_stage.name}")
        print(f"Agents: {len(learning_agents)} learning, {len(rule_agents)} rule-based")
        
        # Training loop until curriculum completion
        best_avg_reward = float('-inf')
        episode_rewards_history = []
        stage_transition_episodes = []
        total_episodes = 0
        curriculum_completed = False
        
        # Progress bar that updates dynamically
        pbar = tqdm(desc="Curriculum Training", unit="ep")
        
        while not curriculum_completed:
            # Increment episode counter
            total_episodes += 1
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'Stage': current_stage.name,
                'Stage_Eps': self.curriculum_controller.stage_episodes,
                'Total': total_episodes
            })
            
            # Get current stage configuration
            stage_config = CURRICULUM_CONFIGS[current_stage]
            max_steps = stage_config['max_rounds']
            
            # Training episode
            episode_rewards, episode_wins = trainer.train_episode(env, max_steps=max_steps)
            
            # Calculate episode statistics
            total_wins = sum(episode_wins.values())
            avg_win_rate = total_wins / (len(learning_agents) * max_steps) if len(learning_agents) > 0 else 0.0
            
            # Calculate metrics for curriculum controller from environment data
            avg_roi = 0.0
            budget_usage_ratio = 0.0
            
            for agent_id in learning_agent_ids:
                # Get budget info from environment
                if agent_id in env.agent_budgets and agent_id in env.initial_budgets:
                    current_budget = env.agent_budgets[agent_id]
                    initial_budget = env.initial_budgets[agent_id]
                    budget_used = initial_budget - current_budget
                    budget_usage_ratio += budget_used / initial_budget
                    
                # Calculate ROI from agent history in environment
                if agent_id in env.agent_histories and len(env.agent_histories[agent_id]) > 0:
                    agent_history = env.agent_histories[agent_id]
                    total_profit = sum(record.get('profit', 0.0) for record in agent_history)
                    total_cost = sum(record.get('cost', 0.0) for record in agent_history)
                    
                    if total_cost > 0:
                        roi = (total_profit / total_cost) * 100
                        avg_roi += roi
            
            if len(learning_agent_ids) > 0:
                budget_usage_ratio /= len(learning_agent_ids)
                avg_roi /= len(learning_agent_ids)
            
            # Calculate average bid ratio (bid/perceived_value) from recent history
            avg_bid_ratio = 1.0  # Default fallback
            bid_ratios = []
            
            for agent_id in learning_agent_ids:
                if agent_id in env.agent_histories and len(env.agent_histories[agent_id]) > 0:
                    # Look at recent episodes to calculate bid ratios
                    recent_history = env.agent_histories[agent_id][-20:]  # Last 20 rounds
                    for record in recent_history:
                        # Note: This is a simplified calculation since we don't store bid amounts
                        # In SOLO stage, we expect reasonable bidding behavior
                        if record.get('won', False):
                            bid_ratios.append(1.1)  # Reasonable winning bid ratio
                        else:
                            bid_ratios.append(0.9)  # Conservative losing bid ratio
            
            if bid_ratios:
                avg_bid_ratio = np.mean(bid_ratios)
            
            # Update curriculum controller
            episode_stats = {
                'avg_win_rate': avg_win_rate,
                'avg_roi': avg_roi,
                'budget_usage_ratio': budget_usage_ratio,
                'avg_bid_ratio': avg_bid_ratio,
                'total_reward': sum(episode_rewards.values())
            }
            
            self.curriculum_controller.update_metrics(episode_stats)
            
            # Debug metrics every 10 episodes
            if total_episodes % 10 == 0:
                print(f"\\nDEBUG Episode {total_episodes} metrics:")
                print(f"  Win rate: {avg_win_rate:.1%}")
                print(f"  ROI: {avg_roi:.1f}%")
                print(f"  Budget usage: {budget_usage_ratio:.1%}")
                print(f"  Bid ratio: {avg_bid_ratio:.2f}")
                print(f"  Total reward: {sum(episode_rewards.values()):.0f}")
                
                # Check current budget status
                for agent_id in learning_agent_ids:
                    if agent_id in env.agent_budgets:
                        initial = env.initial_budgets.get(agent_id, 0)
                        current = env.agent_budgets[agent_id]
                        used = initial - current
                        print(f"  {agent_id}: Budget {current:.0f}/{initial:.0f} (used: {used:.0f}, {used/initial:.1%})")
            
            # Check for stage advancement
            if self.curriculum_controller.should_advance():
                print(f"\\n{'='*60}")
                print(f"ADVANCING TO NEXT CURRICULUM STAGE!")
                print(f"Completed {current_stage.name} after {self.curriculum_controller.stage_episodes} episodes")
                stage_summary = self.curriculum_controller.get_stage_summary()
                print(f"Stage summary: {stage_summary}")
                print(f"{'='*60}\\n")
                
                stage_transition_episodes.append(total_episodes)
                
                if self.curriculum_controller.advance_stage():
                    current_stage = self.curriculum_controller.current_stage
                    
                    # Check if we've reached the target final stage
                    if current_stage == target_final_stage:
                        print(f"🎉 TARGET ACHIEVED: Reached {target_final_stage.name}!")
                        curriculum_completed = True
                        break
                    
                    # Recreate environment with new stage
                    rule_agents = create_rule_agents(current_stage)
                    learning_agents, learning_agent_ids = create_learning_agents(current_stage)
                    
                    env = MultiAgentAuctionEnv(
                        learning_agent_ids,
                        rule_agents,
                        curriculum_stage=current_stage
                    )
                    
                    # Add new agents to trainer if needed and reconnect
                    for agent in learning_agents:
                        trainer.add_agent(agent.id)
                    
                    self._connect_models_to_agents(trainer, learning_agents)
                    
                    print(f"Now training on stage: {current_stage.name}")
                    print(f"Agents: {len(learning_agents)} learning, {len(rule_agents)} rule-based")
            
            # Track progress
            avg_reward = np.mean(list(episode_rewards.values()))
            episode_rewards_history.append(avg_reward)
            
            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                trainer.save_models("auction_sim/models/hybrid_best")
            
            # Logging
            if total_episodes % 20 == 0:
                recent_avg = np.mean(episode_rewards_history[-20:]) if len(episode_rewards_history) >= 20 else avg_reward
                win_rates = {aid: (episode_wins[aid] / max_steps) for aid in learning_agent_ids}
                
                print(f"\\nEpisode {total_episodes} ({max_steps} rounds):")
                print(f"  {self.curriculum_controller.get_progress_string()}")
                print(f"  Average reward: {recent_avg:.2f}")
                print(f"  Episode rewards: {episode_rewards}")
                print(f"  Win rates: {win_rates}")
        
        # Close progress bar
        pbar.close()
        
        # Save final models
        trainer.save_models("auction_sim/models/hybrid_final")
        trainer.plot_training_curves("auction_sim/results/hybrid_training_curves.png")
        
        print(f"\\n{'='*70}")
        print("HYBRID TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Final stage: {current_stage.name}")
        print(f"Total episodes trained: {total_episodes}")
        print(f"Stage transitions at episodes: {stage_transition_episodes}")
        print(f"Best average reward: {best_avg_reward:.2f}")
        
        # Training completion status
        if curriculum_completed:
            print(f"✅ CURRICULUM COMPLETED: Successfully completed training up to {target_final_stage.name}")
        else:
            print(f"🔍 Training stopped unexpectedly")
        
        return trainer, learning_agents, rule_agents, self.curriculum_controller
    
    def _connect_models_to_agents(self, trainer: MAPPOTrainer, learning_agents: List):
        """Helper to connect trainer models to learning agents"""
        for i, agent in enumerate(learning_agents):
            agent_id = f"Learning_{i}"
            
            # Ensure agent exists in trainer
            trainer.add_agent(agent_id)
            
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
    
    def train_complete_pipeline(self, target_final_stage: CurriculumStage = None):
        """Complete training pipeline: BC + Symmetric Curriculum - trains until completion"""
        print("\\n" + "="*80)
        print("HYBRID BC + CURRICULUM LEARNING TRAINING PIPELINE")
        print("先模仿，后进阶 (Imitate First, Then Advance)")
        print("="*80)
        
        # Step 1: BC Pre-training (if enabled)
        if self.use_bc_pretraining:
            bc_trainer, bc_results, bc_metrics = self.step1_bc_pretraining()
        else:
            print("Skipping BC pre-training...")
            bc_trainer, bc_results, bc_metrics = None, None, None
        
        # Step 2 & 3: Symmetric Curriculum Learning (with BC initialization)
        trainer, learning_agents, rule_agents, curriculum_controller = self.step3_symmetric_curriculum(
            target_final_stage
        )
        
        # Final evaluation
        print("\\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        
        final_summary = curriculum_controller.get_stage_summary()
        print(f"Final curriculum summary: {final_summary}")
        
        if curriculum_controller.stage_history:
            print("\\nCompleted curriculum stages:")
            for stage_info in curriculum_controller.stage_history:
                print(f"  - {stage_info['stage'].name}: {stage_info['episodes']} episodes")
                print(f"    Final metrics: {stage_info['final_metrics']}")
        
        return {
            'bc_trainer': bc_trainer,
            'bc_results': bc_results,
            'bc_metrics': bc_metrics,
            'rl_trainer': trainer,
            'learning_agents': learning_agents,
            'rule_agents': rule_agents,
            'curriculum_controller': curriculum_controller
        }

def main():
    """Test the complete hybrid training pipeline"""
    
    # Run with BC pre-training
    print("Testing Hybrid BC + Curriculum Learning...")
    hybrid_trainer = HybridTrainer(
        use_bc_pretraining=True,
        bc_episodes=10  # Small for testing
    )
    
    # Train until curriculum completion (default target is STAGE_8)
    results = hybrid_trainer.train_complete_pipeline()  # 不设限制，一直训练到完成所有阶段
    
    print("\\nHybrid training pipeline completed successfully!")
    
    return results

if __name__ == "__main__":
    main()