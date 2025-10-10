"""
拍卖数据可视化模块
根据日志数据生成可视化图表
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
import os
from matplotlib import rcParams

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号


class AuctionVisualizer:
    """拍卖数据可视化器"""
    
    def __init__(self, log_data: Dict[str, Any]):
        """
        初始化可视化器
        
        Args:
            log_data: 从日志文件加载的数据
        """
        self.log_data = log_data
        self.rounds_data = log_data['rounds']
        self.agent_summaries = log_data.get('agent_summaries', [])
        
        # 获取所有智能体ID
        self.agent_ids = self._extract_agent_ids()
        
        # 为每个智能体分配颜色
        self.colors = self._assign_colors()
        
        # 预处理数据
        self.processed_data = self._preprocess_data()
    
    def _extract_agent_ids(self) -> List[str]:
        """提取所有智能体ID"""
        agent_ids = set()
        for round_data in self.rounds_data:
            if 'bids' in round_data:
                agent_ids.update(round_data['bids'].keys())
        return sorted(list(agent_ids))
    
    def _assign_colors(self) -> Dict[str, str]:
        """为每个智能体分配不同的颜色"""
        # 使用matplotlib的颜色循环
        color_cycle = plt.cm.tab20(np.linspace(0, 1, max(20, len(self.agent_ids))))
        colors = {}
        for i, agent_id in enumerate(self.agent_ids):
            colors[agent_id] = color_cycle[i % len(color_cycle)]
        return colors
    
    def _preprocess_data(self) -> Dict[str, Any]:
        """预处理数据，计算累积指标"""
        data = {agent_id: {
            'rounds': [],
            'costs': [],
            'cumulative_costs': [],
            'bid_ranks': [],
            'economic_values': [],
            'roi_values': [],
            'cumulative_profits': [],
            'win_counts': []
        } for agent_id in self.agent_ids}
        
        # 累积变量
        cumulative = {agent_id: {
            'cost': 0.0,
            'profit': 0.0,
            'wins': 0
        } for agent_id in self.agent_ids}
        
        for round_data in self.rounds_data:
            round_num = round_data['round_num']
            bids = round_data.get('bids', {})
            costs = round_data.get('costs', {})
            profits = round_data.get('profits', {})
            
            # 修复：确保bids中所有值都是标量（处理历史日志中的列表格式）
            bids = {agent_id: (float(bid[0]) if isinstance(bid, (list, np.ndarray)) else float(bid)) 
                    for agent_id, bid in bids.items()}
            
            # 修复：确保costs和profits也是标量
            costs = {agent_id: (float(c[0]) if isinstance(c, (list, np.ndarray)) else float(c))
                     for agent_id, c in costs.items()}
            profits = {agent_id: (float(p[0]) if isinstance(p, (list, np.ndarray)) else float(p))
                       for agent_id, p in profits.items()}
            
            # 计算出价排名
            sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
            bid_ranks = {agent_id: rank + 1 for rank, (agent_id, _) in enumerate(sorted_bids)}
            
            for agent_id in self.agent_ids:
                # 更新累积值
                cost = costs.get(agent_id, 0.0)
                profit = profits.get(agent_id, 0.0)
                
                cumulative[agent_id]['cost'] += cost
                cumulative[agent_id]['profit'] += profit
                
                if agent_id in round_data.get('results', {}):
                    result = round_data['results'][agent_id]
                    if result.get('won', False):
                        cumulative[agent_id]['wins'] += 1
                
                # 计算ROI
                total_cost = cumulative[agent_id]['cost']
                total_profit = cumulative[agent_id]['profit']
                roi = (total_profit / total_cost * 100) if total_cost > 0 else 0.0
                
                # 计算经济价值（使用与exp_runner相同的公式）
                win_rate = cumulative[agent_id]['wins'] / round_num if round_num > 0 else 0.0
                target_wins = (round_num / len(self.agent_ids)) * 0.8
                economic_value = (
                    0.5 * total_profit +
                    0.15 * (roi / 100.0) * total_cost +
                    0.35 * win_rate * target_wins
                )
                
                # 记录数据
                data[agent_id]['rounds'].append(round_num)
                data[agent_id]['costs'].append(cost)
                data[agent_id]['cumulative_costs'].append(cumulative[agent_id]['cost'])
                data[agent_id]['bid_ranks'].append(bid_ranks.get(agent_id, len(self.agent_ids) + 1))
                data[agent_id]['economic_values'].append(economic_value)
                data[agent_id]['roi_values'].append(roi)
                data[agent_id]['cumulative_profits'].append(cumulative[agent_id]['profit'])
                data[agent_id]['win_counts'].append(cumulative[agent_id]['wins'])
        
        return data
    
    def plot_all_metrics(self, save_dir: str = "auction_visualizations", show: bool = True):
        """
        生成所有4张图表
        
        Args:
            save_dir: 保存图表的目录
            show: 是否显示图表
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('拍卖智能体性能分析', fontsize=16, fontweight='bold')
        
        # 1. 累积成本变化
        self._plot_cumulative_cost(axes[0, 0])
        
        # 2. 出价位次变化
        self._plot_bid_rank(axes[0, 1])
        
        # 3. 经济价值变化
        self._plot_economic_value(axes[1, 0])
        
        # 4. ROI变化
        self._plot_roi(axes[1, 1])
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = self.log_data['metadata'].get('timestamp', 'unknown')
        save_path = os.path.join(save_dir, f"auction_metrics_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[图表] 可视化图表已保存到: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def _plot_cumulative_cost(self, ax):
        """绘制累积成本变化图"""
        for agent_id in self.agent_ids:
            data = self.processed_data[agent_id]
            ax.plot(data['rounds'], data['cumulative_costs'], 
                   label=agent_id, color=self.colors[agent_id], linewidth=2, alpha=0.8)
        
        ax.set_xlabel('拍卖轮次', fontsize=12)
        ax.set_ylabel('累积成本', fontsize=12)
        ax.set_title('累积成本变化', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    
    def _plot_bid_rank(self, ax):
        """绘制出价位次变化图（使用移动平均平滑）"""
        window_size = min(50, len(self.rounds_data) // 10)  # 动态窗口大小
        
        for agent_id in self.agent_ids:
            data = self.processed_data[agent_id]
            ranks = data['bid_ranks']
            
            # 计算移动平均
            if len(ranks) >= window_size:
                smoothed_ranks = np.convolve(ranks, np.ones(window_size)/window_size, mode='valid')
                smoothed_rounds = data['rounds'][window_size-1:]
            else:
                smoothed_ranks = ranks
                smoothed_rounds = data['rounds']
            
            ax.plot(smoothed_rounds, smoothed_ranks, 
                   label=agent_id, color=self.colors[agent_id], linewidth=2, alpha=0.8)
        
        ax.set_xlabel('拍卖轮次', fontsize=12)
        ax.set_ylabel('平均出价排名', fontsize=12)
        ax.set_title(f'出价位次变化 (移动平均, 窗口={window_size})', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # 排名1在上方
    
    def _plot_economic_value(self, ax):
        """绘制经济价值变化图"""
        for agent_id in self.agent_ids:
            data = self.processed_data[agent_id]
            ax.plot(data['rounds'], data['economic_values'], 
                   label=agent_id, color=self.colors[agent_id], linewidth=2, alpha=0.8)
        
        ax.set_xlabel('拍卖轮次', fontsize=12)
        ax.set_ylabel('经济价值', fontsize=12)
        ax.set_title('经济价值变化 (0.5×Profit + 0.15×ROI×Cost + 0.35×WinRate×Target)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    
    def _plot_roi(self, ax):
        """绘制ROI变化图"""
        for agent_id in self.agent_ids:
            data = self.processed_data[agent_id]
            ax.plot(data['rounds'], data['roi_values'], 
                   label=agent_id, color=self.colors[agent_id], linewidth=2, alpha=0.8)
        
        ax.set_xlabel('拍卖轮次', fontsize=12)
        ax.set_ylabel('ROI (%)', fontsize=12)
        ax.set_title('ROI变化', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)  # 添加0线参考
    
    def plot_individual_metrics(self, save_dir: str = "auction_visualizations", show: bool = True):
        """
        生成单独的4张图表（每张图独立保存）
        
        Args:
            save_dir: 保存图表的目录
            show: 是否显示图表
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = self.log_data['metadata'].get('timestamp', 'unknown')
        
        metrics = [
            ('cumulative_cost', '累积成本变化', self._plot_cumulative_cost),
            ('bid_rank', '出价位次变化', self._plot_bid_rank),
            ('economic_value', '经济价值变化', self._plot_economic_value),
            ('roi', 'ROI变化', self._plot_roi)
        ]
        
        saved_paths = []
        for metric_name, title, plot_func in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_func(ax)
            
            save_path = os.path.join(save_dir, f"auction_{metric_name}_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            saved_paths.append(save_path)
            print(f"[图表] {title}图表已保存到: {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
        
        return saved_paths


def visualize_from_log(log_file: str, save_dir: str = "auction_visualizations", 
                      show: bool = True, combined: bool = True):
    """
    从日志文件生成可视化图表
    
    Args:
        log_file: 日志文件路径
        save_dir: 保存图表的目录
        show: 是否显示图表
        combined: True-生成2x2组合图，False-生成4张独立图
        
    Returns:
        保存的图表路径列表
    """
    # 加载日志
    with open(log_file, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    
    # 创建可视化器
    visualizer = AuctionVisualizer(log_data)
    
    # 生成图表
    if combined:
        return [visualizer.plot_all_metrics(save_dir, show)]
    else:
        return visualizer.plot_individual_metrics(save_dir, show)

