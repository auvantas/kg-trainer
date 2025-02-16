import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from .rl_components import GRPOPolicy, MetacognitiveModule, RewardCalculator, AhaDetector

class SPREncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class RCCModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.spatial_relations = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 8)  # 8 basic RCC relations
        )
    
    def forward(self, x):
        return self.spatial_relations(x)

class KGModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=256, hidden_dim=512):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        self.spr_encoder = SPREncoder(embedding_dim, hidden_dim, embedding_dim)
        self.rcc_module = RCCModule(embedding_dim)
        
        # Add RL and metacognitive components
        state_dim = embedding_dim * 3  # Combined embeddings dimension
        action_dim = num_entities  # Number of possible tail entities
        self.policy = GRPOPolicy(state_dim, action_dim)
        self.metacog = MetacognitiveModule(state_dim)
        
        self.reward_calculator = RewardCalculator()
        self.aha_detector = AhaDetector()
        
        self.reasoning_history = []
        
    def encode_triple(self, head, relation, tail) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        head_emb = self.entity_embeddings(head)
        rel_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        
        # Apply SPR encoding
        head_spr = self.spr_encoder(head_emb)
        tail_spr = self.spr_encoder(tail_emb)
        
        # Apply RCC for spatial reasoning
        spatial_relations = self.rcc_module(torch.cat([head_spr, tail_spr], dim=-1))
        
        # Combine embeddings for state representation
        state = torch.cat([head_spr, rel_emb, tail_spr], dim=-1)
        
        # Get policy action and metacognitive outputs
        action, log_prob = self.policy.get_action(state)
        metacog_outputs = self.metacog(state)
        
        # Track reasoning steps for 'a-ha' moment detection
        self.reasoning_history.append({
            'state': state.detach(),
            'action': action.detach(),
            'metacog': metacog_outputs
        })
        
        # Detect potential 'a-ha' moments
        if len(self.reasoning_history) > 1:
            is_aha, improvement = self.aha_detector.detect_aha_moment(
                metacog_outputs['confidence'].item(),
                log_prob.exp().item(),
                [str(h['action'].item()) for h in self.reasoning_history]
            )
            if is_aha:
                # Adjust reasoning based on 'a-ha' moment
                self._adjust_reasoning(improvement)
        
        return action, spatial_relations, metacog_outputs
    
    def _adjust_reasoning(self, improvement: float):
        """Adjust reasoning strategy based on detected 'a-ha' moment"""
        # Increase attention to successful reasoning patterns
        with torch.no_grad():
            last_state = self.reasoning_history[-1]['state']
            last_action = self.reasoning_history[-1]['action']
            
            # Strengthen the policy's preference for successful actions
            action_probs = self.policy(last_state)
            action_probs[0, last_action] += improvement * 0.1
            action_probs = F.softmax(action_probs, dim=-1)
    
    def forward(self, heads, relations, tails):
        actions = []
        spatial_rels = []
        metacog_outputs_list = []
        
        for head, relation, tail in zip(heads, relations, tails):
            action, spatial_rel, metacog = self.encode_triple(
                head.unsqueeze(0),
                relation.unsqueeze(0),
                tail.unsqueeze(0)
            )
            actions.append(action)
            spatial_rels.append(spatial_rel)
            metacog_outputs_list.append(metacog)
        
        return (
            torch.cat(actions),
            torch.cat(spatial_rels),
            {k: torch.cat([d[k] for d in metacog_outputs_list]) 
             for k in metacog_outputs_list[0].keys()}
        )
