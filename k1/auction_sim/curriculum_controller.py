# /auction_sim/curriculum_controller.py
"""
Curriculum Learning Controller for Multi-Agent Auction Environment
"""
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Optional
from .config import CurriculumStage, CURRICULUM_SUCCESS_CRITERIA

class CurriculumController:
    """Controls the progression through curriculum learning stages"""
    
    def __init__(self, start_stage: CurriculumStage = CurriculumStage.STAGE_0):
        self.current_stage = start_stage
        self.stage_episodes = 0
        self.success_episodes = 0
        self.stage_start_episode = 0
        
        # Metrics tracking
        self.stage_metrics = {
            'win_rates': deque(maxlen=100),
            'roi_values': deque(maxlen=100),
            'budget_usage': deque(maxlen=100),
            'avg_bid_ratios': deque(maxlen=100),
            'episode_rewards': deque(maxlen=100)
        }
        
        # History for analysis
        self.stage_history = []
        
    def reset_stage_metrics(self):
        """Reset metrics when entering a new stage"""
        self.stage_episodes = 0
        self.success_episodes = 0
        for key in self.stage_metrics:
            self.stage_metrics[key].clear()
    
    def update_metrics(self, episode_stats: Dict):
        """Update metrics based on episode statistics"""
        self.stage_episodes += 1
        
        # Extract and store metrics
        self.stage_metrics['win_rates'].append(episode_stats.get('avg_win_rate', 0.0))
        self.stage_metrics['roi_values'].append(episode_stats.get('avg_roi', 0.0))
        self.stage_metrics['budget_usage'].append(episode_stats.get('budget_usage_ratio', 0.0))
        self.stage_metrics['avg_bid_ratios'].append(episode_stats.get('avg_bid_ratio', 1.0))
        self.stage_metrics['episode_rewards'].append(episode_stats.get('total_reward', 0.0))
        
        # Check if this episode meets success criteria
        if self.check_episode_success(episode_stats):
            self.success_episodes += 1
    
    def check_episode_success(self, episode_stats: Dict) -> bool:
        """Check if a single episode meets the success criteria for current stage"""
        criteria = CURRICULUM_SUCCESS_CRITERIA[self.current_stage]
        
        # Check each criterion
        checks = []
        
        if 'min_win_rate' in criteria:
            checks.append(episode_stats.get('avg_win_rate', 0.0) >= criteria['min_win_rate'])
        
        if 'min_roi' in criteria:
            checks.append(episode_stats.get('avg_roi', -100.0) >= criteria['min_roi'])
        
        if 'min_budget_usage' in criteria:
            checks.append(episode_stats.get('budget_usage_ratio', 0.0) >= criteria['min_budget_usage'])
        
        if 'max_avg_bid_ratio' in criteria:
            checks.append(episode_stats.get('avg_bid_ratio', 2.0) <= criteria['max_avg_bid_ratio'])
        
        # All criteria must be met
        return all(checks) if checks else False
    
    def should_advance(self) -> bool:
        """Determine if agent should advance to next stage"""
        criteria = CURRICULUM_SUCCESS_CRITERIA[self.current_stage]
        min_episodes = criteria.get('min_episodes', 100)
        
        # Need minimum episodes
        if self.stage_episodes < min_episodes:
            return False
        
        # Check recent performance (last 20 episodes)
        if len(self.stage_metrics['win_rates']) < 20:
            return False
        
        recent_metrics = {
            'win_rate': np.mean(list(self.stage_metrics['win_rates'])[-20:]),
            'roi': np.mean(list(self.stage_metrics['roi_values'])[-20:]),
            'budget_usage': np.mean(list(self.stage_metrics['budget_usage'])[-20:]),
            'bid_ratio': np.mean(list(self.stage_metrics['avg_bid_ratios'])[-20:])
        }
        
        # Check if recent performance meets criteria
        checks = []
        
        if 'min_win_rate' in criteria:
            checks.append(recent_metrics['win_rate'] >= criteria['min_win_rate'])
        
        if 'min_roi' in criteria:
            checks.append(recent_metrics['roi'] >= criteria['min_roi'])
        
        if 'min_budget_usage' in criteria:
            checks.append(recent_metrics['budget_usage'] >= criteria['min_budget_usage'])
        
        if 'max_avg_bid_ratio' in criteria:
            checks.append(recent_metrics['bid_ratio'] <= criteria['max_avg_bid_ratio'])
        
        success = all(checks) if checks else False
        
        # Additional stability check - performance should be consistent
        if success and len(self.stage_metrics['win_rates']) >= 50:
            # Check if performance is stable (low variance in recent episodes)
            recent_win_rates = list(self.stage_metrics['win_rates'])[-20:]
            win_rate_std = np.std(recent_win_rates)
            if win_rate_std > 0.2:  # Too much variance
                success = False
        
        return success
    
    def advance_stage(self) -> bool:
        """Advance to the next curriculum stage"""
        # Save current stage history
        self.stage_history.append({
            'stage': self.current_stage,
            'episodes': self.stage_episodes,
            'success_episodes': self.success_episodes,
            'final_metrics': self.get_stage_summary()
        })
        
        # Find next stage
        stages = list(CurriculumStage)
        current_idx = stages.index(self.current_stage)
        
        if current_idx < len(stages) - 1:
            self.current_stage = stages[current_idx + 1]
            self.stage_start_episode = self.stage_episodes
            self.reset_stage_metrics()
            return True
        
        return False
    
    def get_stage_summary(self) -> Dict:
        """Get summary statistics for current stage"""
        if not self.stage_metrics['win_rates']:
            return {}
        
        return {
            'stage': self.current_stage.name,
            'episodes': self.stage_episodes,
            'success_episodes': self.success_episodes,
            'avg_win_rate': np.mean(self.stage_metrics['win_rates']),
            'avg_roi': np.mean(self.stage_metrics['roi_values']),
            'avg_budget_usage': np.mean(self.stage_metrics['budget_usage']),
            'avg_bid_ratio': np.mean(self.stage_metrics['avg_bid_ratios']),
            'avg_reward': np.mean(self.stage_metrics['episode_rewards']),
            'final_win_rate': np.mean(list(self.stage_metrics['win_rates'])[-20:]) if len(self.stage_metrics['win_rates']) >= 20 else 0.0
        }
    
    def get_progress_string(self) -> str:
        """Get a formatted string showing curriculum progress"""
        summary = self.get_stage_summary()
        
        progress = f"[Stage: {self.current_stage.name}] "
        progress += f"Episodes: {self.stage_episodes} | "
        progress += f"Success: {self.success_episodes} | "
        
        if self.stage_metrics['win_rates']:
            progress += f"Win Rate: {summary['avg_win_rate']:.2%} | "
            progress += f"ROI: {summary['avg_roi']:.1f}% | "
            progress += f"Budget Use: {summary['avg_budget_usage']:.1%}"
        
        return progress