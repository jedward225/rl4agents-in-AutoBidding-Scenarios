# /auction_sim/bc_trainer.py
"""
Behavioral Cloning Pre-training Module
Trains learning agents to imitate AggressiveAgent behavior
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os

class BCActor(nn.Module):
    """BC Actor网络 - 与DDPG的Actor完全一致"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        # 完整Sequential结构，与DDPG的Actor一致
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # 输出范围[0,1]，后续需要缩放
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化 - 与DDPG一致"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 最后一层用小权重初始化
        with torch.no_grad():
            self.network[-2].weight.uniform_(-3e-3, 3e-3)
            self.network[-2].bias.uniform_(-3e-3, 3e-3)
    
    def forward(self, obs):
        out = self.network(obs)
        # 缩放到拍卖环境合理范围 [0.0, 1.5]，与DDPG完全一致
        return out * 1.5

class BCDataset(Dataset):
    """Dataset for behavioral cloning"""
    
    def __init__(self, data: List[Dict]):
        self.states = []
        self.actions = []
        self.rewards = []
        
        for sample in data:
            self.states.append(sample['state'])
            self.actions.append(sample['action'])
            self.rewards.append(sample['reward'])
        
        self.states = torch.FloatTensor(self.states)
        self.actions = torch.FloatTensor(self.actions).unsqueeze(1)  # Shape: (N, 1)
        self.rewards = torch.FloatTensor(self.rewards)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx]

class BCTrainer:
    """Behavioral Cloning Trainer - 使用与DDPG完全一致的网络结构"""
    
    def __init__(self, obs_dim: int = 9, action_dim: int = 1, lr: float = 1e-3):
        # obs_dim=9（与DDPG一致）
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr
        
        # 🔧 使用与DDPG完全一致的Actor网络结构
        self.network = BCActor(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # For BC, we only need the actor part
        self.criterion = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'epoch_losses': [],
            'val_losses': [],
            'learning_rates': []
        }
        
        print(f"✨ BC Trainer初始化（与DDPG网络结构一致）:")
        print(f"   Obs dim: {obs_dim}")
        print(f"   Action dim: {action_dim}")
        print(f"   网络结构: 3层hidden(128) + Sigmoid → ×1.5")
    
    def _check_data_dimension(self, dataset_path: str) -> Tuple[bool, int]:
        """
        检查BC数据的观测维度是否匹配
        
        Returns:
            (is_valid, current_dim): 是否有效，当前维度
        """
        if not os.path.exists(dataset_path):
            return False, 0
        
        try:
            with open(dataset_path, 'rb') as f:
                dataset_dict = pickle.load(f)
            
            data = dataset_dict.get('data', [])
            if not data:
                return False, 0
            
            current_dim = len(data[0]['state'])
            is_valid = (current_dim == self.obs_dim)
            
            return is_valid, current_dim
        except Exception as e:
            print(f"⚠️ 无法读取数据文件: {e}")
            return False, 0
    
    def prepare_bc_data(self, dataset_path: str, n_episodes: int = 50, 
                       force_recollect: bool = False) -> bool:
        """
        准备BC训练数据：检查维度并在需要时自动收集
        
        Args:
            dataset_path: 数据集保存路径
            n_episodes: 收集的episode数量
            force_recollect: 是否强制重新收集
            
        Returns:
            bool: 是否准备成功
        """
        print("\n" + "="*70)
        print("📊 BC数据准备")
        print("="*70)
        
        # 检查现有数据
        if not force_recollect:
            is_valid, current_dim = self._check_data_dimension(dataset_path)
            
            if is_valid:
                print(f"✅ 发现有效的BC数据:")
                print(f"   路径: {dataset_path}")
                print(f"   观测维度: {current_dim} (匹配)")
                
                # 显示数据统计
                try:
                    with open(dataset_path, 'rb') as f:
                        dataset_dict = pickle.load(f)
                    data = dataset_dict.get('data', [])
                    actions = [d['action'] for d in data]
                    
                    print(f"   样本数: {len(data)}")
                    print(f"   动作范围: [{min(actions):.4f}, {max(actions):.4f}]")
                    print(f"   动作均值: {sum(actions)/len(actions):.4f}")
                except:
                    pass
                
                return True
            elif current_dim > 0:
                print(f"⚠️  现有BC数据维度不匹配:")
                print(f"   期望维度: {self.obs_dim}")
                print(f"   当前维度: {current_dim}")
                print(f"   将重新收集...")
        else:
            print(f"🔄 强制重新收集BC数据...")
        
        # 收集新数据
        print(f"\n开始收集BC数据:")
        print(f"  Episodes: {n_episodes}")
        print(f"  Rounds per episode: 16000")
        print(f"  Expert agent: Conservative (适应性出价)")
        print(f"  观测维度: {self.obs_dim}")
        print(f"  保底价机制: 已启用")
        print("")
        
        try:
            from .bc_data_collector import BCDataCollector
            
            collector = BCDataCollector()
            dataset = collector.collect_dataset(
                n_episodes=n_episodes,
                save_path=dataset_path
            )
            
            # 验证新收集的数据
            is_valid, current_dim = self._check_data_dimension(dataset_path)
            
            if not is_valid:
                print(f"❌ 数据收集后验证失败:")
                print(f"   期望维度: {self.obs_dim}")
                print(f"   实际维度: {current_dim}")
                return False
            
            print(f"\n✅ BC数据收集成功！")
            print(f"   观测维度: {current_dim} (正确)")
            print(f"   样本数: {len(dataset['data'])}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ BC数据收集失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.network.train()
        total_loss = 0.0
        n_batches = 0
        
        for states, actions, rewards in dataloader:
            self.optimizer.zero_grad()
            
            # Get actor output (与DDPG一致的forward)
            predicted_actions = self.network(states)
            
            # BC loss: minimize difference between predicted and expert actions
            loss = self.criterion(predicted_actions, actions)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validation step"""
        self.network.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for states, actions, rewards in dataloader:
                predicted_actions = self.network(states)
                loss = self.criterion(predicted_actions, actions)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def train(self, dataset_path: str, n_epochs: int = 100, batch_size: int = 64,
              val_split: float = 0.2, save_path: str = "bc_model.pth",
              n_collect_episodes: int = 50, auto_prepare_data: bool = True) -> Dict:
        """
        Complete training procedure
        
        Args:
            dataset_path: BC数据集路径
            n_epochs: 训练轮数
            batch_size: 批大小
            val_split: 验证集比例
            save_path: 模型保存路径
            n_collect_episodes: 如需收集数据，收集的episode数量
            auto_prepare_data: 是否自动准备数据（检查维度并在需要时收集）
        """
        
        # 自动准备BC数据（检查维度，必要时重新收集）
        if auto_prepare_data:
            if not self.prepare_bc_data(dataset_path, n_episodes=n_collect_episodes):
                raise RuntimeError("BC数据准备失败，无法开始训练")
        
        print(f"\nLoading dataset from {dataset_path}")
        with open(dataset_path, 'rb') as f:
            dataset_dict = pickle.load(f)
        
        data = dataset_dict['data']
        print(f"Loaded {len(data)} samples")
        
        # 验证数据维度
        if len(data[0]['state']) != self.obs_dim:
            raise ValueError(f"数据维度不匹配: 期望{self.obs_dim}维, 实际{len(data[0]['state'])}维")
        
        # Create dataset
        full_dataset = BCDataset(data)
        
        # Train/validation split
        val_size = int(val_split * len(full_dataset))
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")
        print(f"Training for {n_epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(n_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Record history
            self.training_history['epoch_losses'].append(train_loss)
            self.training_history['val_losses'].append(val_loss)
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.network.state_dict(), save_path)
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"Epoch {epoch:3d}/{n_epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Best Val: {best_val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience: {patience})")
                break
        
        # Load best model
        self.network.load_state_dict(torch.load(save_path))
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to: {save_path}")
        
        return {
            'best_val_loss': best_val_loss,
            'n_epochs_trained': epoch + 1,
            'training_history': self.training_history
        }
    
    def plot_training_curves(self, save_path: str = "bc_training_curves.png"):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        epochs = range(len(self.training_history['epoch_losses']))
        ax1.plot(epochs, self.training_history['epoch_losses'], label='Train Loss', alpha=0.7)
        ax1.plot(epochs, self.training_history['val_losses'], label='Val Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('BC Training Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2.plot(epochs, self.training_history['learning_rates'], color='orange', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to: {save_path}")
    
    def evaluate_model(self, dataset_path: str) -> Dict:
        """Evaluate trained model performance"""
        print("Evaluating BC model...")
        
        with open(dataset_path, 'rb') as f:
            dataset_dict = pickle.load(f)
        
        data = dataset_dict['data']
        dataset = BCDataset(data)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        self.network.eval()
        all_predictions = []
        all_targets = []
        all_errors = []
        
        with torch.no_grad():
            for states, actions, rewards in dataloader:
                predicted_actions = self.network(states)
                
                predictions = predicted_actions.cpu().numpy().flatten()
                targets = actions.cpu().numpy().flatten()
                errors = np.abs(predictions - targets)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                all_errors.extend(errors)
        
        # Calculate metrics
        mae = np.mean(all_errors)
        mse = np.mean(np.square(all_errors))
        rmse = np.sqrt(mse)
        
        # Calculate action accuracy (within 10% of target)
        action_accuracy = np.mean(np.array(all_errors) < 0.1)
        
        metrics = {
            'mae': mae,
            'mse': mse, 
            'rmse': rmse,
            'action_accuracy_10pct': action_accuracy,
            'mean_prediction': np.mean(all_predictions),
            'mean_target': np.mean(all_targets),
            'prediction_std': np.std(all_predictions),
            'target_std': np.std(all_targets)
        }
        
        print("BC Model Evaluation Results:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Action Accuracy (±10%): {action_accuracy:.1%}")
        print(f"  Mean Prediction: {metrics['mean_prediction']:.3f}")
        print(f"  Mean Target: {metrics['mean_target']:.3f}")
        
        return metrics

def main():
    """Test BC training"""
    # Collect data first if needed
    from .bc_data_collector import BCDataCollector
    
    print("Step 1: Collecting BC data...")
    collector = BCDataCollector()
    collector.collect_dataset(
        n_episodes=10,
        save_path="auction_sim/bc_dataset.pkl"
    )
    
    print("\nStep 2: Training BC model...")
    trainer = BCTrainer()
    results = trainer.train(
        dataset_path="auction_sim/bc_dataset.pkl",
        n_epochs=50,
        save_path="auction_sim/bc_model.pth"
    )
    
    print("\nStep 3: Plotting training curves...")
    trainer.plot_training_curves("auction_sim/bc_training_curves.png")
    
    print("\nStep 4: Evaluating model...")
    metrics = trainer.evaluate_model("auction_sim/bc_dataset.pkl")
    
    return trainer, results, metrics

if __name__ == "__main__":
    main()