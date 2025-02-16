import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

class GRPOPolicy(nn.Module):
    """Group Relative Policy Optimization (GRPO) policy network"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.policy_network(state), dim=-1)
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

class MetacognitiveModule(nn.Module):
    """Handles self-reflection and adaptive reasoning time"""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.reflection_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # confidence, mistake_detection, time_needed
        )
        
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.reflection_network(state)
        return {
            'confidence': torch.sigmoid(outputs[:, 0]),
            'mistake_detection': torch.sigmoid(outputs[:, 1]),
            'time_needed': F.softplus(outputs[:, 2])  # Always positive
        }

class RewardCalculator:
    """Calculates rewards based on accuracy, reasoning quality, and metacognition"""
    def __init__(self, base_reward: float = 1.0):
        self.base_reward = base_reward
        
    def calculate_reward(
        self,
        correct: bool,
        reasoning_quality: float,
        metacog_outputs: Dict[str, torch.Tensor],
        actual_mistake: bool
    ) -> float:
        reward = 0.0
        
        # Base reward for correct answers
        if correct:
            reward += self.base_reward
        
        # Reward for good reasoning
        reward += reasoning_quality * 0.5
        
        # Metacognitive rewards
        confidence = metacog_outputs['confidence'].item()
        mistake_detection = metacog_outputs['mistake_detection'].item()
        
        # Reward for accurate confidence
        if correct and confidence > 0.5:
            reward += 0.2
        elif not correct and confidence < 0.5:
            reward += 0.2
            
        # Reward for accurate mistake detection
        if actual_mistake and mistake_detection > 0.5:
            reward += 0.3
        elif not actual_mistake and mistake_detection < 0.5:
            reward += 0.3
            
        return reward

class AhaDetector:
    """Detects and processes 'a-ha' moments during reasoning"""
    def __init__(self, confidence_threshold: float = 0.8, improvement_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.improvement_threshold = improvement_threshold
        self.previous_performance = None
        
    def detect_aha_moment(
        self,
        current_confidence: float,
        current_performance: float,
        reasoning_path: List[str]
    ) -> Tuple[bool, float]:
        """
        Detects if an 'a-ha' moment occurred based on sudden improvements
        in confidence and performance, along with changes in reasoning path.
        """
        if self.previous_performance is None:
            self.previous_performance = current_performance
            return False, 0.0
        
        performance_improvement = current_performance - self.previous_performance
        
        is_aha_moment = (
            current_confidence > self.confidence_threshold and
            performance_improvement > self.improvement_threshold
        )
        
        self.previous_performance = current_performance
        
        return is_aha_moment, performance_improvement
