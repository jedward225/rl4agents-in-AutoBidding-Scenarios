"""
拍卖日志记录模块
用于记录所有拍卖轮次和智能体的详细数据
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import numpy as np


class AuctionLogger:
    """拍卖日志记录器"""
    
    def __init__(self, log_dir: str = "auction_logs"):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志文件保存目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_log_file = None
        self.session_data = {
            'metadata': {},
            'rounds': [],
            'agent_summaries': []
        }
    
    def start_session(self, config_info: Dict[str, Any] = None):
        """
        开始新的拍卖会话
        
        Args:
            config_info: 配置信息
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = os.path.join(self.log_dir, f"auction_log_{timestamp}.json")
        
        self.session_data = {
            'metadata': {
                'timestamp': timestamp,
                'start_time': datetime.now().isoformat(),
                'config': config_info or {}
            },
            'rounds': [],
            'agent_summaries': []
        }
    
    def log_round(self, round_data: Dict[str, Any]):
        """
        记录单轮拍卖数据
        
        Args:
            round_data: 单轮数据字典，包含round_num, true_value, bids, results, profits, costs等
        """
        # 转换numpy类型为Python原生类型
        serializable_data = self._make_serializable(round_data)
        self.session_data['rounds'].append(serializable_data)
    
    def log_agent_summary(self, agents: List[Any]):
        """
        记录所有智能体的汇总信息
        
        Args:
            agents: 智能体列表
        """
        summaries = []
        for agent in agents:
            summary = {
                'agent_id': str(agent.id),
                'type': agent.__class__.__name__,
                'initial_budget': self._to_python_type(agent.initial_budget),
                'final_budget': self._to_python_type(agent.budget),
                'total_cost': self._to_python_type(agent.get_total_cost()),
                'cumulative_profit': self._to_python_type(agent.get_cumulative_profit()),
                'roi': self._to_python_type(agent.get_roi()),
                'win_count': sum(1 for r in agent.history if r['result'] and r['result']['won']),
                'total_rounds': len(agent.history)
            }
            summaries.append(summary)
        
        self.session_data['agent_summaries'] = summaries
    
    def save_log(self):
        """保存日志到文件"""
        if self.current_log_file is None:
            raise ValueError("未开始会话，请先调用 start_session()")
        
        self.session_data['metadata']['end_time'] = datetime.now().isoformat()
        
        with open(self.current_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📝 日志已保存到: {self.current_log_file}")
        return self.current_log_file
    
    def _make_serializable(self, obj):
        """递归转换对象为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return self._to_python_type(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return self._to_python_type(obj)
    
    def _to_python_type(self, value):
        """转换numpy类型为Python原生类型"""
        if hasattr(value, 'item'):
            return value.item()
        elif isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return float(value) if isinstance(value, (int, float)) else value


def load_auction_log(log_file: str) -> Dict[str, Any]:
    """
    加载拍卖日志文件
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        日志数据字典
    """
    with open(log_file, 'r', encoding='utf-8') as f:
        return json.load(f)


