# /auction_sim/auction.py
import numpy as np
from typing import Dict, List
from . import config

class GSPAuction:
    """
    实现广义第二价格拍卖 (Generalized Second-Price Auction, GSP)
    
    修复：添加保底价（reserve price）机制，防止免费点击漏洞
    """
    def __init__(self, n_slots: int, ctr_positions: np.ndarray, ctr_noise_std: float, 
                 reserve_per_slot: np.ndarray = None, qualities: Dict[str, float] = None):
        """
        Args:
            n_slots: 广告位数量
            ctr_positions: 每个位置的基础CTR
            ctr_noise_std: CTR噪声标准差
            reserve_per_slot: 每个位置的保底价（防止免费点击）。如果为None，使用默认值
            qualities: agent的质量分 {agent_id: quality}，默认为1.0
        """
        self.n_slots = n_slots
        
        # 确保CTR_POSITIONS数组长度与N_SLOTS匹配
        if len(ctr_positions) != n_slots:
            if len(ctr_positions) < n_slots:
                # 如果CTR数组太短，用递减值填充
                print(f"Warning: CTR_POSITIONS length ({len(ctr_positions)}) < N_SLOTS ({n_slots})")
                additional_ctrs = []
                last_ctr = ctr_positions[-1]
                step = last_ctr / (n_slots - len(ctr_positions) + 1)
                for i in range(n_slots - len(ctr_positions)):
                    last_ctr -= step
                    additional_ctrs.append(max(0.1, last_ctr))  # 最小CTR为0.1
                self.ctr_positions = np.concatenate([ctr_positions, additional_ctrs])
            else:
                # 如果CTR数组太长，截取前n_slots个
                print(f"Warning: CTR_POSITIONS length ({len(ctr_positions)}) > N_SLOTS ({n_slots}), truncating")
                self.ctr_positions = ctr_positions[:n_slots]
        else:
            self.ctr_positions = ctr_positions
        
        # 设置保底价（每个位置的最低CPC）
        if reserve_per_slot is None:
            # 默认保底价：按位置递减，确保即使只有1个参与者也要付费
            # 使用TRUE_VALUE的10-20%作为保底价
            base_reserve = config.TRUE_VALUE_RANGE[0] * 0.15  # 约1.5
            self.reserve_per_slot = np.array([
                base_reserve * (1.0 - 0.1 * i) for i in range(n_slots)
            ])
        else:
            self.reserve_per_slot = np.array(reserve_per_slot)
        
        # 质量分（默认所有agent质量为1.0）
        self.qualities = qualities if qualities is not None else {}
        
        self.ctr_noise_std = ctr_noise_std
        print(f"GSP Auction initialized with {n_slots} slots")
        print(f"  CTR: {self.ctr_positions}")
        print(f"  Reserve prices: {self.reserve_per_slot}")

    def run_auction(self, bids: Dict[str, float]) -> Dict[str, Dict]:
        """
        运行一轮GSP拍卖。

        Args:
            bids (Dict[str, float]): 一个字典，key是agent_id，value是出价。
                                     {'agent_A': 1.2, 'agent_B': 1.5}

        Returns:
            Dict[str, Dict]: 一个结果字典，key是agent_id，value包含其排名、支付成本和赢得的槽位CTR。
                             {
                                'agent_B': {'rank': 1, 'cost_per_click': 1.2, 'slot_ctr': 0.7...},
                                'agent_A': {'rank': 2, 'cost_per_click': 0, 'slot_ctr': 0.3...},
                                'agent_C': {'rank': 3, 'cost_per_click': 0, 'slot_ctr': 0},
                             }
        """
        if not bids:
            return {}

        # 1. 对出价进行降序排序
        sorted_bidders = sorted(bids.items(), key=lambda item: item[1], reverse=True)
        
        # 2. 确定赢家和支付价格
        winners = sorted_bidders[:self.n_slots]
        losers = sorted_bidders[self.n_slots:]
        
        results = {}
        
        # 为本轮拍卖生成带噪声的CTR
        noisy_ctrs = self.ctr_positions * (1 + np.random.uniform(-self.ctr_noise_std, self.ctr_noise_std, size=self.n_slots))
        noisy_ctrs = np.clip(noisy_ctrs, 0, 1) # 确保CTR在[0,1]范围内

        # 3. 计算赢家的成本和信息（带保底价机制）
        for i in range(len(winners)):
            agent_id, bid_price = winners[i]
            quality_i = self.qualities.get(agent_id, 1.0)
            
            # 计算下一名的AdRank（可能来自下一个赢家或第一个输家）
            if i + 1 < len(winners):
                # 下一个赢家的出价
                next_agent, next_bid = winners[i+1]
                next_quality = self.qualities.get(next_agent, 1.0)
                next_rank = next_bid * next_quality
            elif losers:
                # 第一个输家的出价
                next_agent, next_bid = losers[0]
                next_quality = self.qualities.get(next_agent, 1.0)
                next_rank = next_bid * next_quality
            else:
                # 没有后继者，只使用保底价
                next_rank = 0.0
            
            # 本位保底价（防止免费点击）
            reserve_price = self.reserve_per_slot[i] if i < len(self.reserve_per_slot) else 0.0
            
            # GSP定价：max(保底价, 下一名出价) / 自己的质量分
            # 关键修复：即使没有竞争对手，也要支付保底价
            price_rank = max(reserve_price, next_rank)
            cost_per_click = price_rank / max(quality_i, 1e-8)
            
            # 确保CPC不超过自己的出价（越界保护）
            cost_per_click = min(bid_price, cost_per_click)

            results[agent_id] = {
                'rank': i + 1,
                'won': True,
                'bid': bid_price,
                'cost_per_click': cost_per_click,
                'slot_ctr': noisy_ctrs[i]
            }
            
        # 4. 记录输家的信息
        for i in range(len(losers)):
            agent_id, bid_price = losers[i]
            results[agent_id] = {
                'rank': len(winners) + i + 1,
                'won': False,
                'bid': bid_price,
                'cost_per_click': 0.0,
                'slot_ctr': 0.0
            }
            
        return results