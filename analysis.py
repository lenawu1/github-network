# SNAP Github dataset overview:
# 37700 developers, 289003 mutual follower relationships, undirected
# binary-labeled: 0 for web developer, 1 for ML developer
# goal: visualize the social network to identify key players

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load the datasets
edges_df = pd.read_csv('musae_git_edges.csv')
target_df = pd.read_csv('musae_git_target.csv')

# Create a mapping from node ID to developer name
id_name_dict = pd.Series(target_df.name.values, index=target_df.id).to_dict()

# Create the full graph
G_full = nx.Graph()
edges_list = list(zip(edges_df['id_1'], edges_df['id_2']))
G_full.add_edges_from(edges_list)

# Relabel nodes with their corresponding developer names
G_full = nx.relabel_nodes(G_full, id_name_dict)

# Calculate PageRank centrality for the full graph
pagerank_full = nx.pagerank(G_full)

# Get the top 300 nodes by PageRank
top_nodes_by_pagerank = sorted(pagerank_full, key=pagerank_full.get, reverse=True)[:300]

# Create a subgraph with these top nodes by PageRank
G = G_full.subgraph(top_nodes_by_pagerank)

# Normalize PageRank for the subgraph for visualization purposes
pagerank = {node: pagerank_full[node] for node in G.nodes()}
max_rank = max(pagerank.values())
min_rank = min(pagerank.values())
normalized_pagerank = {node: (rank - min_rank) / (max_rank - min_rank) for node, rank in pagerank.items()}

# Define node colors and sizes based on PageRank
# Nodes with the highest PageRank will have a distinct color and larger size
base_node_size = 100
base_color = 'blue'
highlight_color = 'red'
node_color = [highlight_color if rank == max_rank else base_color for node, rank in normalized_pagerank.items()]
node_size = [rank * 2000 + base_node_size for rank in normalized_pagerank.values()]

# Visualize the network
plt.figure(figsize=(15, 10))
# Use the Kamada-Kawai layout to spread nodes out for better visibility
pos = nx.kamada_kawai_layout(G)

# Drawing nodes
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=0.8)

# Drawing edges
nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)

# Drawing labels for the top PageRank nodes
top_rank_nodes = set(top_nodes_by_pagerank[:10])  # Adjust number for labels as needed
node_labels = {node: node for node in top_rank_nodes}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color='black')

plt.title('Top GitHub Developers by PageRank')
plt.axis('off')  # Turn off the axis
plt.show()
