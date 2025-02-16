"""
Example script demonstrating basic usage of the Knowledge Graph Training system.
This example shows how to:
1. Prepare data
2. Train a model
3. Make predictions
4. Visualize results
"""

import pandas as pd
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.spr_rcc_model import KGModel
from train import train, RLTrainer
from visualize import create_knowledge_graph, visualize_graph

# Example knowledge graph data
def create_sample_data():
    """Create a sample knowledge graph dataset"""
    data = {
        'head': [
            'Albert_Einstein', 'Physics', 'Quantum_Mechanics',
            'Max_Planck', 'Relativity_Theory', 'Werner_Heisenberg',
            'Copenhagen', 'Germany', 'ETH_Zurich'
        ],
        'relation': [
            'field', 'subfield', 'discoveredBy',
            'workedOn', 'proposedBy', 'bornIn',
            'locatedIn', 'hasUniversity', 'employedAt'
        ],
        'tail': [
            'Physics', 'Quantum_Mechanics', 'Max_Planck',
            'Quantum_Mechanics', 'Albert_Einstein', 'Germany',
            'Denmark', 'ETH_Zurich', 'Albert_Einstein'
        ]
    }
    df = pd.DataFrame(data)
    return df

def main():
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Prepare data
    print("Creating sample dataset...")
    df = create_sample_data()
    data_path = output_dir / 'sample_kg.csv'
    df.to_csv(data_path, index=False)
    
    # 2. Train model
    print("\nTraining model...")
    class Args:
        data_path = str(data_path)
        output_path = str(output_dir / 'model.pt')
        epochs = 10
        batch_size = 4
        embedding_dim = 128
        learning_rate = 0.001
        run_name = "example_run"
        disable_wandb = True  # Disable wandb for example
    
    args = Args()
    train(args)
    
    # 3. Make predictions
    print("\nMaking predictions...")
    model = torch.load(args.output_path)
    
    # Example prediction
    head = "Albert_Einstein"
    relation = "field"
    
    from predict import predict_relations
    predictions = predict_relations(
        model['model_state_dict'],
        model['entity2id'],
        model['relation2id'],
        head,
        relation,
        top_k=3,
        reasoning_threshold=0.5
    )
    
    print(f"\nPredictions for {head} {relation}:")
    for pred in predictions:
        print(f"\nEntity: {pred['entity']}")
        print(f"Confidence: {pred['confidence']:.4f}")
        print("Reasoning Information:")
        print(f"- Mistake Likelihood: {pred['reasoning_info']['mistake_likelihood']:.4f}")
        print(f"- Reasoning Time: {pred['reasoning_info']['reasoning_time']:.4f}")
        if pred['reasoning_info'].get('had_aha_moment'):
            print(f"- Had 'a-ha' moment! Improvement: {pred['reasoning_info']['improvement']:.4f}")
    
    # 4. Visualize knowledge graph
    print("\nCreating visualization...")
    G = create_knowledge_graph(str(data_path))
    visualize_graph(G, str(output_dir / 'knowledge_graph.png'))
    print("Visualization saved to output/knowledge_graph.png")

if __name__ == "__main__":
    main()
