import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def create_knowledge_graph(data_path):
    # Read the knowledge graph data
    df = pd.read_csv(data_path)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges to the graph
    for _, row in df.iterrows():
        G.add_edge(row['head'], row['tail'], relation=row['relation'])
    
    return G

def visualize_graph(G, output_path=None, figsize=(15, 10)):
    plt.figure(figsize=figsize)
    
    # Create layout
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos)
    
    # Draw edge labels (relations)
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("Knowledge Graph Visualization")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to knowledge graph data CSV')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save visualization (optional)')
    parser.add_argument('--figsize_width', type=int, default=15,
                        help='Figure width')
    parser.add_argument('--figsize_height', type=int, default=10,
                        help='Figure height')
    
    args = parser.parse_args()
    
    G = create_knowledge_graph(args.data_path)
    visualize_graph(G, args.output_path, 
                   figsize=(args.figsize_width, args.figsize_height))
