import torch
import argparse
from models.spr_rcc_model import KGModel
import json
from typing import List, Tuple, Dict

def load_model(model_path):
    checkpoint = torch.load(model_path)
    
    model = KGModel(
        num_entities=len(checkpoint['entity2id']),
        num_relations=len(checkpoint['relation2id']),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    return model, checkpoint['entity2id'], checkpoint['relation2id']

def predict_relations(
    model: KGModel,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    head_entity: str,
    relation: str,
    top_k: int = 10,
    reasoning_threshold: float = 0.7
) -> List[Tuple[str, float, Dict]]:
    """
    Predict relations with reasoning information and confidence scores.
    Returns list of (entity, score, reasoning_info) tuples.
    """
    # Convert inputs to indices
    head_idx = entity2id.get(head_entity)
    relation_idx = relation2id.get(relation)
    
    if head_idx is None or relation_idx is None:
        return "Entity or relation not found in training data"
    
    # Convert to tensors
    head = torch.tensor([head_idx])
    relation = torch.tensor([relation_idx])
    
    if torch.cuda.is_available():
        head = head.cuda()
        relation = relation.cuda()
    
    # Score all possible tail entities
    predictions = []
    id2entity = {v: k for k, v in entity2id.items()}
    
    with torch.no_grad():
        for tail_idx in range(len(entity2id)):
            tail = torch.tensor([tail_idx])
            if torch.cuda.is_available():
                tail = tail.cuda()
            
            # Get prediction and metacognitive outputs
            action, spatial_rel, metacog = model.encode_triple(head, relation, tail)
            
            confidence = metacog['confidence'].item()
            mistake_detection = metacog['mistake_detection'].item()
            time_needed = metacog['time_needed'].item()
            
            # Only include predictions with high confidence
            if confidence > reasoning_threshold:
                predictions.append({
                    'entity': id2entity[tail_idx],
                    'confidence': confidence,
                    'reasoning_info': {
                        'spatial_relations': spatial_rel.tolist(),
                        'mistake_likelihood': mistake_detection,
                        'reasoning_time': time_needed
                    }
                })
    
    # Sort by confidence
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Check for 'a-ha' moments in top predictions
    if len(model.reasoning_history) > 1:
        for pred in predictions[:top_k]:
            is_aha, improvement = model.aha_detector.detect_aha_moment(
                pred['confidence'],
                pred['confidence'],  # Using confidence as performance metric
                [str(h['action'].item()) for h in model.reasoning_history[-2:]]
            )
            if is_aha:
                pred['reasoning_info']['had_aha_moment'] = True
                pred['reasoning_info']['improvement'] = improvement
    
    return predictions[:top_k]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--head', type=str, required=True, help='Head entity')
    parser.add_argument('--relation', type=str, required=True, help='Relation')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top predictions to show')
    parser.add_argument('--reasoning_threshold', type=float, default=0.7,
                       help='Minimum confidence threshold for including predictions')
    
    args = parser.parse_args()
    
    model, entity2id, relation2id = load_model(args.model_path)
    predictions = predict_relations(
        model, entity2id, relation2id,
        args.head, args.relation,
        args.top_k, args.reasoning_threshold
    )
    
    print(f"\nPredictions for {args.head} {args.relation}:")
    for pred in predictions:
        print(f"\nEntity: {pred['entity']}")
        print(f"Confidence: {pred['confidence']:.4f}")
        print("Reasoning Information:")
        print(f"- Spatial Relations: {pred['reasoning_info']['spatial_relations']}")
        print(f"- Mistake Likelihood: {pred['reasoning_info']['mistake_likelihood']:.4f}")
        print(f"- Reasoning Time: {pred['reasoning_info']['reasoning_time']:.4f}")
        if pred['reasoning_info'].get('had_aha_moment'):
            print(f"- Had 'a-ha' moment! Improvement: {pred['reasoning_info']['improvement']:.4f}")
