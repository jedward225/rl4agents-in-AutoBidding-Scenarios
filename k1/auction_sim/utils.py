# /auction_sim/utils.py
"""
Utility functions for visualization and analysis
"""
import os
import pandas as pd
import numpy as np

def calculate_economic_value(agent, total_rounds, total_agents):
    """
    Calculate the weighted multi-objective economic value as defined in README
    Target: 0.5 * Profit + 0.15 * ROI * TotalCost + 0.35 * WinRate * TargetWins
    """
    total_cost = agent.get_total_cost()
    win_count = sum(1 for record in agent.history if record.get('result') and record.get('result', {}).get('won', False))
    cumulative_profit = agent.get_cumulative_profit()
    roi = agent.get_roi() / 100.0  # Convert percentage to decimal
    win_rate = win_count / total_rounds if total_rounds > 0 else 0
    target_wins = (total_rounds / total_agents) * 0.8  # Expected competitive wins
    
    # Weighted multi-objective economic value
    profit_term = 0.5 * cumulative_profit
    efficiency_term = 0.15 * roi * total_cost
    competitive_term = 0.35 * win_rate * target_wins
    
    economic_value = profit_term + efficiency_term + competitive_term
    
    return {
        'economic_value': economic_value,
        'profit_term': profit_term,
        'efficiency_term': efficiency_term,
        'competitive_term': competitive_term,
        'win_rate': win_rate,
        'target_wins': target_wins
    }

def generate_all_visualizations(agents, total_rounds=16000):
    """
    Generate visualizations for experiment results with new economic value metrics
    """
    # Ensure results directory exists
    os.makedirs("auction_sim/results", exist_ok=True)
    
    total_agents = len(agents)
    
    # Calculate metrics for all agents
    data = []
    for agent in agents:
        total_cost = agent.get_total_cost()
        win_count = sum(1 for record in agent.history if record.get('result') and record.get('result', {}).get('won', False))
        cumulative_profit = agent.get_cumulative_profit()
        roi = agent.get_roi()
        win_rate = (win_count / total_rounds) * 100 if total_rounds > 0 else 0
        
        # Calculate economic value components
        econ_metrics = calculate_economic_value(agent, total_rounds, total_agents)
        
        data.append({
            'Agent_ID': agent.id,
            'Agent_Type': agent.__class__.__name__.replace('Agent', ''),
            'Budget_Left': agent.budget,
            'Total_Cost': total_cost,
            'Win_Count': win_count,
            'Win_Rate(%)': win_rate,
            'Cumulative_Profit': cumulative_profit,
            'ROI(%)': roi,
            'Economic_Value': econ_metrics['economic_value'],
            'Profit_Term': econ_metrics['profit_term'],
            'Efficiency_Term': econ_metrics['efficiency_term'],
            'Competitive_Term': econ_metrics['competitive_term'],
            'Avg_Cost_Per_Win': total_cost / win_count if win_count > 0 else 0
        })
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv("auction_sim/results/experiment_summary.csv", index=False)
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    print("\nExperiment summary saved to: auction_sim/results/experiment_summary.csv")
    
    return df