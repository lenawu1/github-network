import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load the datasets
edges_df = pd.read_csv('musae_git_edges.csv')
target_df = pd.read_csv('musae_git_target.csv')

# Create the graph
G = nx.Graph()
edges_list = list(zip(edges_df['id_1'], edges_df['id_2']))
G.add_edges_from(edges_list)

# Calculate centrality measures
pagerank = nx.pagerank(G)
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
degree = nx.degree_centrality(G)

# Extract the top nodes for each centrality measure
top_n = 50
top_nodes_pagerank = sorted(pagerank, key=pagerank.get, reverse=True)[:top_n]
top_nodes_betweenness = sorted(betweenness, key=betweenness.get, reverse=True)[:top_n]
top_nodes_closeness = sorted(closeness, key=closeness.get, reverse=True)[:top_n]
top_nodes_degree = sorted(degree, key=degree.get, reverse=True)[:top_n]

# Union of all top nodes
top_nodes_union = set(top_nodes_pagerank) | set(top_nodes_betweenness) | set(top_nodes_closeness) | set(top_nodes_degree)

# Create subgraph of top nodes
G_top = G.subgraph(top_nodes_union)

# Visualization
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G_top, k=0.1, iterations=20)  # Adjust layout parameters as needed

# Draw nodes with different color and size based on centrality measure
nx.draw_networkx_nodes(G_top, pos, nodelist=top_nodes_pagerank, node_size=300, node_color='red', label='PageRank')
nx.draw_networkx_nodes(G_top, pos, nodelist=top_nodes_betweenness, node_size=300, node_color='blue', label='Betweenness')
nx.draw_networkx_nodes(G_top, pos, nodelist=top_nodes_closeness, node_size=300, node_color='green', label='Closeness')
nx.draw_networkx_nodes(G_top, pos, nodelist=top_nodes_degree, node_size=300, node_color='yellow', label='Degree')

# Draw edges
nx.draw_networkx_edges(G_top, pos, alpha=0.5)

# Draw labels
nx.draw_networkx_labels(G_top, pos, font_size=8)

# Legend
plt.legend(scatterpoints=1)

plt.title('Top GitHub Developers by Centrality Measures')
plt.axis('off')
plt.show()
