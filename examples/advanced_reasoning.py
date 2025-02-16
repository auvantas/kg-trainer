"""
Example script demonstrating advanced reasoning capabilities of the Knowledge Graph system.
This example shows:
1. Complex multi-hop reasoning
2. Metacognitive analysis
3. 'A-ha' moment detection
4. Confidence-based filtering
"""

import torch
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.spr_rcc_model import KGModel
from models.rl_components import MetacognitiveModule, AhaDetector
from predict import predict_relations

def create_complex_kg():
    """Create a more complex knowledge graph for testing reasoning capabilities"""
    data = {
        'head': [
            # Physics concepts
            'Quantum_Mechanics', 'Wave_Function', 'Particle_Physics',
            'Quantum_Entanglement', 'Quantum_Superposition', 'Quantum_Field_Theory',
            
            # Scientists and their work
            'Schrodinger', 'Einstein', 'Bohr',
            'Heisenberg', 'Dirac', 'von_Neumann',
            
            # Institutions
            'Cambridge', 'Princeton', 'Copenhagen_University',
            'ETH_Zurich', 'Berlin_University', 'Vienna_University'
        ],
        'relation': [
            # Scientific relations
            'describes', 'partOf', 'relatesTo',
            'foundationalTo', 'explainsEffect', 'theoreticalBasisFor',
            
            # Personal relations
            'discoveredBy', 'collaboratedWith', 'studiedAt',
            'taughtAt', 'supervisedBy', 'influencedBy',
            
            # Institutional relations
            'locatedIn', 'affiliatedWith', 'researchFocusOf',
            'establishedBy', 'contemporaryWith', 'historicallyLinkedTo'
        ],
        'tail': [
            # Completing the triples
            'Wave_Function', 'Quantum_Mechanics', 'Quantum_Entanglement',
            'Quantum_Field_Theory', 'Particle_Physics', 'Quantum_Mechanics',
            
            'Wave_Function', 'Bohr', 'Cambridge',
            'Quantum_Mechanics', 'Einstein', 'Schrodinger',
            
            'UK', 'USA', 'Denmark',
            'Switzerland', 'Germany', 'Austria'
        ]
    }
    return pd.DataFrame(data)

def analyze_reasoning_path(model, predictions, head_entity):
    """Analyze the reasoning path and metacognitive behavior"""
    print(f"\nAnalyzing reasoning for queries related to {head_entity}:")
    
    # Track metacognitive patterns
    confidence_pattern = []
    reasoning_times = []
    aha_moments = []
    
    for pred in predictions:
        confidence_pattern.append(pred['confidence'])
        reasoning_times.append(pred['reasoning_info']['reasoning_time'])
        if pred['reasoning_info'].get('had_aha_moment'):
            aha_moments.append({
                'entity': pred['entity'],
                'improvement': pred['reasoning_info']['improvement']
            })
    
    # Analyze patterns
    print("\nMetacognitive Analysis:")
    print(f"- Average confidence: {sum(confidence_pattern) / len(confidence_pattern):.4f}")
    print(f"- Confidence trend: {'Increasing' if confidence_pattern[-1] > confidence_pattern[0] else 'Decreasing'}")
    print(f"- Average reasoning time: {sum(reasoning_times) / len(reasoning_times):.4f}s")
    print(f"- Number of 'a-ha' moments: {len(aha_moments)}")
    
    if aha_moments:
        print("\n'A-ha' Moments Analysis:")
        for moment in aha_moments:
            print(f"- Entity: {moment['entity']}, Improvement: {moment['improvement']:.4f}")
    
    return {
        'confidence_pattern': confidence_pattern,
        'reasoning_times': reasoning_times,
        'aha_moments': aha_moments
    }

def main():
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Create and save complex knowledge graph
    print("Creating complex knowledge graph...")
    df = create_complex_kg()
    data_path = output_dir / 'complex_kg.csv'
    df.to_csv(data_path, index=False)
    
    # Load trained model (assuming it's already trained)
    model_path = output_dir / 'model.pt'
    if not model_path.exists():
        print(f"Please train the model first using basic_usage.py")
        return
    
    model = torch.load(str(model_path))
    
    # Perform multi-hop reasoning
    print("\nPerforming multi-hop reasoning...")
    
    # Example: Find connections between Quantum Mechanics and Scientists
    queries = [
        ('Quantum_Mechanics', 'discoveredBy'),
        ('Wave_Function', 'discoveredBy'),
        ('Schrodinger', 'studiedAt'),
        ('Einstein', 'collaboratedWith')
    ]
    
    all_predictions = {}
    for head, relation in queries:
        predictions = predict_relations(
            model['model_state_dict'],
            model['entity2id'],
            model['relation2id'],
            head,
            relation,
            top_k=5,
            reasoning_threshold=0.6
        )
        all_predictions[f"{head}_{relation}"] = predictions
        
        print(f"\nQuery: {head} {relation}")
        for pred in predictions:
            print(f"\nPredicted: {pred['entity']}")
            print(f"Confidence: {pred['confidence']:.4f}")
            if pred['reasoning_info'].get('had_aha_moment'):
                print("*** 'A-ha' moment detected! ***")
    
    # Analyze reasoning patterns
    print("\nAnalyzing reasoning patterns across queries...")
    for query, preds in all_predictions.items():
        head = query.split('_')[0]
        analysis = analyze_reasoning_path(model, preds, head)
        
        # Save analysis results
        analysis_path = output_dir / f'reasoning_analysis_{head}.pt'
        torch.save(analysis, str(analysis_path))
        print(f"Analysis saved to {analysis_path}")

if __name__ == "__main__":
    main()
