from itertools import combinations
from collections import Counter
import random
import math
import os

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import eigsh


def load_slashdot_graph(file_path, sample_size=None):
    G = nx.DiGraph()
    with open(file_path, 'r') as f:
        edges = [line.strip().split() for line in f if not line.startswith('#')]
    
    if sample_size:
        edges = random.sample(edges, min(sample_size, len(edges)))
    
    for source, target, sign in edges:
        G.add_edge(int(source), int(target), sign=int(sign))
    return G


def transform_to_undirected(G):
    G_undirected = nx.Graph()
    for u, v, data in G.edges(data=True):
        if G_undirected.has_edge(u, v):
            G_undirected[u][v]['sign'] += data['sign']
        else:
            G_undirected.add_edge(u, v, sign=data['sign'])
    
    for u, v, data in G_undirected.edges(data=True):
        data['sign'] = 1 if data['sign'] > 0 else -1
    
    return G_undirected


def enumerate_triads(G):
    for node in G:
        neighbors = list(G.neighbors(node))
        for u, v in combinations(neighbors, 2):
            if G.has_edge(u, v):
                yield (node, u, v)


def count_and_classify_triads(G):
    total_triads = stable_triads = unstable_triads = 0
    for triad in enumerate_triads(G):
        if G.has_edge(triad[0], triad[1]) and G.has_edge(triad[0], triad[2]) and G.has_edge(triad[1], triad[2]):
            total_triads += 1
            signs = [G[u][v]['sign'] for u, v in combinations(triad, 2)]
            if signs.count(-1) % 2 == 0:
                stable_triads += 1
            else:
                unstable_triads += 1
    return total_triads, stable_triads, unstable_triads


def calculate_tss(original_unstable, modified_unstable):
    """Calculate TSS (Triadic Stability Score) - the fraction of unstable triads"""
    if original_unstable == 0:
        return 1  # If no unstable triads were in the original, it's fully balanced
    tss = (original_unstable - modified_unstable) / original_unstable
    return tss


def calculate_tts(original_G, modified_G):
    """Calculate Total Triad Shifts (TTS) - the number of edge sign changes."""
    tts = 0
    for u, v, data in original_G.edges(data=True):
        if modified_G.has_edge(u, v):
            original_sign = data['sign']
            modified_sign = modified_G[u][v]['sign']
            if original_sign != modified_sign:
                tts += 1
    return tts


def modify_unstable_triads(G):
    for triad in enumerate_triads(G):
        # Ensure it's a closed triad
        if G.has_edge(triad[0], triad[1]) and G.has_edge(triad[0], triad[2]) and G.has_edge(triad[1], triad[2]):
            signs = [G[u][v]['sign'] for u, v in combinations(triad, 2)]
            if signs.count(-1) % 2 != 0:  # Unstable triad
                # Find the first negative edge and flip its sign
                for u, v in combinations(triad, 2):
                    if G[u][v]['sign'] == -1:
                        G[u][v]['sign'] = 1
                        break


def spectral_balance_with_fiedler(G):
    # Create a mapping of node labels to contiguous integers
    node_map = {node: i for i, node in enumerate(G.nodes())}
    reverse_map = {i: node for node, i in node_map.items()}

    # Create the Laplacian matrix
    A = nx.adjacency_matrix(G, nodelist=list(node_map.keys()))
    D = sparse.diags([deg for _, deg in G.degree()])
    L = D - A  # Laplacian matrix

    # Compute the second smallest eigenvector (Fiedler vector)
    _, eigenvectors = eigsh(L, k=2, which='SM')
    fiedler_vector = eigenvectors[:, 1]  # Fiedler vector

    # Partition nodes based on the Fiedler vector
    partition = {reverse_map[i]: 0 if v >= 0 else 1 for i, v in enumerate(fiedler_vector)}

    return partition


def characterize_groups(G, partition):
    groups = [[], []]
    for node, group in partition.items():
        groups[group].append(node)
    
    sizes = [len(group) for group in groups]
    densities = [nx.density(G.subgraph(group)) for group in groups]
    inter_edges = sum(1 for u, v in G.edges() if partition[u] != partition[v])
    inter_density = inter_edges / (sizes[0] * sizes[1]) if sizes[0] * sizes[1] > 0 else 0
    
    return sizes, densities, inter_density


def calculate_balance_distance(original_partition, modified_partition, G):
    """Calculate Balance Distance between the partitions of the original and modified graphs."""
    original_groups = {group: [] for group in set(original_partition.values())}
    modified_groups = {group: [] for group in set(modified_partition.values())}
    
    # Separate nodes into groups for original and modified partitions
    for node, group in original_partition.items():
        original_groups[group].append(node)
    
    for node, group in modified_partition.items():
        modified_groups[group].append(node)
    
    # Calculate group densities and inter-group densities for both
    original_sizes, original_densities, original_inter_density = characterize_groups(G, original_partition)
    modified_sizes, modified_densities, modified_inter_density = characterize_groups(G, modified_partition)
    
    # Use a simple L2 distance between the densities as a proxy for balance distance
    balance_distance = math.sqrt(
        sum((orig_d - mod_d) ** 2 for orig_d, mod_d in zip(original_densities, modified_densities))
    ) + abs(original_inter_density - modified_inter_density)
    
    return balance_distance


def visualize_degree_distribution(G):
    degrees = [d for n, d in G.degree()]
    plt.figure(figsize=(10, 6))
    sns.histplot(degrees, stat="count")
    plt.xlim(0, 50)
    plt.title("Degree Distribution")
    plt.xlabel("Edge Degree")
    plt.ylabel("Number of edges")
    plt.savefig("plots/degree_distribution.svg")
    # plt.show()


def visualize_sign_distribution(G):
    signs = [data['sign'] for _, _, data in G.edges(data=True)]
    sign_counts = Counter(signs)
    plt.figure(figsize=(8, 6))
    plt.bar(sign_counts.keys(), sign_counts.values())
    plt.title("Edge Sign Distribution")
    plt.xlabel("Sign")
    plt.ylabel("Count")
    plt.xticks([-1, 1], ['Negative', 'Positive'])
    plt.savefig("plots/sign_distribution.svg")
    # plt.show()


def visualize_partition_groups(G, partition, num_nodes=5000):
    # Select a random sample of nodes from each group
    group_0 = [node for node, group in partition.items() if group == 0]
    group_1 = [node for node, group in partition.items() if group == 1]
    
    sample_size_0 = math.ceil(len(group_0)/len(partition) * num_nodes) * 10
    sample_size_1 = math.floor(len(group_1)/len(partition) * num_nodes)

    sample_nodes_0 = random.sample(group_0, sample_size_0)
    sample_nodes_1 = random.sample(group_1, sample_size_1)
    sample_nodes = sample_nodes_0 + sample_nodes_1
    
    subgraph = G.subgraph(sample_nodes)
    
    # Create a layout with group 0 on the left and group 1 on the right
    pos = {}
    columns = 20
    adjustment = 1/columns
    for i, node in enumerate(sample_nodes_0):
        pos[node] = (- 0.5 - i % columns * adjustment, i // columns * adjustment)
    for i, node in enumerate(sample_nodes_1):
        pos[node] = (0.5 + i % columns * adjustment, i // columns * adjustment)
    
    plt.figure(figsize=(20, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos, 
                           nodelist=sample_nodes_0, 
                           node_color='skyblue', 
                           node_size=10, 
                           label='Group 0')
    nx.draw_networkx_nodes(subgraph, pos, 
                           nodelist=sample_nodes_1, 
                           node_color='lightgreen', 
                           node_size=10, 
                           label='Group 1')
    
    # Draw edges within groups
    within_edges_0 = [(u, v) for (u, v) in subgraph.edges() if partition[u] == partition[v] == 0]
    within_edges_1 = [(u, v) for (u, v) in subgraph.edges() if partition[u] == partition[v] == 1]
    nx.draw_networkx_edges(subgraph, pos, edgelist=within_edges_0, edge_color='blue', alpha=0.3)
    nx.draw_networkx_edges(subgraph, pos, edgelist=within_edges_1, edge_color='green', alpha=0.3)
    
    # Draw edges between groups
    between_edges = [(u, v) for (u, v) in subgraph.edges() if partition[u] != partition[v]]
    nx.draw_networkx_edges(subgraph, pos, edgelist=between_edges, edge_color='red', style='dashed', alpha=0.3)
    
    plt.title(f"Spectral Balance Partition Visualization\n(Sample of {len(sample_nodes)} nodes)")
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("plots/sample_nodes_partition.svg")
    # plt.show()
    
    # Some statistics
    print(f"Group 0 size: {len(group_0)}, Sampled: {sample_size_0}")
    print(f"Group 1 size: {len(group_1)}, Sampled: {sample_size_1}")
    print(f"Edges within Group 0: {len(within_edges_0)}")
    print(f"Edges within Group 1: {len(within_edges_1)}")
    print(f"Edges between groups: {len(between_edges)}")


if __name__ == "__main__":
    file_path = "soc-sign-Slashdot081106.txt"  # Update this with the actual file path

    sample_size = None  # If None< consider entire dataset, otherwise take sample_size samples of it

    os.makedirs("plots")
    with open("results_output.txt", "w") as output_file:

        def log_and_print(message):
            print(message)
            output_file.write(message + "\n")

        log_and_print("Loading and transforming graph...")
        G_directed = load_slashdot_graph(file_path, sample_size)
        G_undirected = transform_to_undirected(G_directed)

        # Make a copy of the original graph and original partition
        original_G_undirected = G_undirected.copy()

        log_and_print("Analyzing triads before modification...")
        total_triads, stable_triads, unstable_triads = count_and_classify_triads(G_undirected)
        log_and_print(f"Total triads before modification: {total_triads}")
        log_and_print(f"Stable triads before modification: {stable_triads}")
        log_and_print(f"Unstable triads before modification: {unstable_triads}")

        log_and_print("Modifying unstable triads...")
        modify_unstable_triads(G_undirected)

        log_and_print("Analyzing triads after modification...")
        total_triads, stable_triads, unstable_triads = count_and_classify_triads(G_undirected)
        log_and_print(f"Total triads after modification: {total_triads}")
        log_and_print(f"Stable triads after modification: {stable_triads}")
        log_and_print(f"Unstable triads after modification: {unstable_triads}")

        log_and_print("Calculating Total Triad Shifts (TTS)...")
        tts = calculate_tts(original_G_undirected, G_undirected)
        log_and_print(f"Total Triad Shifts (TTS): {tts}")

        # Perform spectral balance on the original and modified graphs
        log_and_print("Balancing the original graph...")
        original_partition = spectral_balance_with_fiedler(original_G_undirected)

        log_and_print("Balancing the modified graph...")
        modified_partition = spectral_balance_with_fiedler(G_undirected)

        log_and_print("Calculating Balance Distance...")
        balance_distance = calculate_balance_distance(original_partition, modified_partition, G_undirected)
        log_and_print(f"Balance Distance: {balance_distance}")

        log_and_print("Visualizing partition groups...")
        visualize_partition_groups(G_undirected, modified_partition)

        log_and_print("Characterizing groups...")
        group_sizes, group_densities, inter_group_density = characterize_groups(G_undirected, modified_partition)

        log_and_print(f"Group sizes: {group_sizes}")
        log_and_print(f"Group densities: {group_densities}")
        log_and_print(f"Inter-group density: {inter_group_density}")