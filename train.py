import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import argparse
from models.spr_rcc_model import KGModel
from tqdm import tqdm
import wandb
from typing import Dict, Tuple, List
import numpy as np
import torch.nn.functional as F

class KGDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.entity2id = self._create_mappings(self.data['head'].tolist() + self.data['tail'].tolist())
        self.relation2id = self._create_mappings(self.data['relation'].tolist())
        
    def _create_mappings(self, items):
        unique_items = sorted(set(items))
        return {item: idx for idx, item in enumerate(unique_items)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'head': torch.tensor(self.entity2id[row['head']]),
            'relation': torch.tensor(self.relation2id[row['relation']]),
            'tail': torch.tensor(self.entity2id[row['tail']])
        }

class RLTrainer:
    def __init__(self, model: KGModel, args: argparse.Namespace):
        self.model = model
        self.args = args
        self.optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Initialize wandb for experiment tracking
        if not args.disable_wandb:
            wandb.init(
                project="kg-trainer",
                config=vars(args),
                name=f"run_{args.run_name}"
            )
    
    def compute_returns(self, rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch using GRPO"""
        self.model.train()
        total_loss = 0
        total_reward = 0
        num_aha_moments = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            heads = batch['head']
            relations = batch['relation']
            tails = batch['tail']
            
            if torch.cuda.is_available():
                heads = heads.cuda()
                relations = relations.cuda()
                tails = tails.cuda()
            
            # Forward pass with RL and metacognition
            actions, spatial_rels, metacog_outputs = self.model(heads, relations, tails)
            
            # Calculate rewards
            rewards = []
            for i in range(len(heads)):
                correct = (actions[i] == tails[i]).item()
                reward = self.model.reward_calculator.calculate_reward(
                    correct=correct,
                    reasoning_quality=spatial_rels[i].mean().item(),
                    metacog_outputs={k: v[i] for k, v in metacog_outputs.items()},
                    actual_mistake=not correct
                )
                rewards.append(reward)
            
            # Calculate returns
            returns = self.compute_returns(rewards)
            if torch.cuda.is_available():
                returns = returns.cuda()
            
            # Calculate loss using GRPO
            policy_loss = -torch.mean(returns * torch.stack(
                [self.model.policy.get_action(state)[1] for state in self.model.reasoning_history[-len(heads):]]))
            
            # Add metacognitive loss
            confidence_loss = F.mse_loss(
                metacog_outputs['confidence'],
                (actions == tails).float()
            )
            
            # Total loss
            loss = policy_loss + 0.1 * confidence_loss
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_reward += sum(rewards)
            
            # Count 'a-ha' moments
            for i in range(len(heads)):
                if len(self.model.reasoning_history) > 1:
                    is_aha, _ = self.model.aha_detector.detect_aha_moment(
                        metacog_outputs['confidence'][i].item(),
                        rewards[i],
                        [str(h['action'].item()) for h in self.model.reasoning_history[-2:]]
                    )
                    if is_aha:
                        num_aha_moments += 1
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        avg_reward = total_reward / len(dataloader.dataset)
        aha_rate = num_aha_moments / len(dataloader.dataset)
        
        metrics = {
            'loss': avg_loss,
            'reward': avg_reward,
            'aha_rate': aha_rate
        }
        
        if not self.args.disable_wandb:
            wandb.log(metrics)
        
        return metrics

def train(args):
    # Load dataset
    dataset = KGDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    model = KGModel(
        num_entities=len(dataset.entity2id),
        num_relations=len(dataset.relation2id),
        embedding_dim=args.embedding_dim
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Initialize trainer
    trainer = RLTrainer(model, args)
    
    # Training loop
    best_reward = float('-inf')
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(dataloader)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Reward: {metrics['reward']:.4f}")
        print(f"A-ha Rate: {metrics['aha_rate']:.4f}")
        
        # Save best model
        if metrics['reward'] > best_reward:
            best_reward = metrics['reward']
            torch.save({
                'model_state_dict': model.state_dict(),
                'entity2id': dataset.entity2id,
                'relation2id': dataset.relation2id,
                'metrics': metrics
            }, args.output_path)
            print(f"New best model saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to knowledge graph data CSV')
    parser.add_argument('--output_path', type=str, default='model.pt', help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--run_name', type=str, default='default', help='Name for the training run')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    train(args)
