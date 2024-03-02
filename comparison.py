
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from community import community_louvain

edges_df = pd.read_csv('musae_git_edges.csv')
target_df = pd.read_csv('musae_git_target.csv')
id_name_dict = pd.Series(target_df.name.values, index=target_df.id).to_dict()

G_full = nx.Graph()
edges_list = list(zip(edges_df['id_1'], edges_df['id_2']))
G_full.add_edges_from(edges_list)
G_full = nx.relabel_nodes(G_full, id_name_dict)

pagerank = nx.pagerank(G_full)
degree_centrality = nx.degree_centrality(G_full)

top_nodes_pagerank = sorted(pagerank, key=pagerank.get, reverse=True)[:30]
top_nodes_degree = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:30]
overlap_nodes = set(top_nodes_pagerank) & set(top_nodes_degree)

# community detection
partition = community_louvain.best_partition(G_full)
unique_communities = len(set(partition.values()))

with open('github_network_analysis_results.txt', 'w') as f:
    f.write(f"Number of developers in both top 30 PageRank and Degree Centrality lists: {len(overlap_nodes)}\n")
    f.write(f"Unique communities detected: {unique_communities}\n")
    if overlap_nodes:
        f.write("Overlap nodes representing particularly influential developers:\n")
        for node in overlap_nodes:
            f.write(f"- {node}\n")
    else:
        f.write("No overlap found between top PageRank and Degree Centrality nodes.\n")

def visualize(G_sub, metric_values, ax, title, cmap='viridis'):
    pos = nx.spring_layout(G_sub, seed=42)
    node_sizes = [metric_values[node] * 5000 + 100 for node in G_sub.nodes()]  # Adjusted size scaling
    nx.draw_networkx_nodes(G_sub, pos, ax=ax, node_size=node_sizes, alpha=0.8, edgecolors='black',
                           node_color=[metric_values[node] for node in G_sub.nodes()], cmap=cmap)
    nx.draw_networkx_edges(G_sub, pos, ax=ax, alpha=0.5, edge_color="gray")
    labels = {node: node for node in G_sub.nodes()}
    nx.draw_networkx_labels(G_sub, pos, labels=labels, font_size=8,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7), ax=ax)
    ax.set_title(title)
    ax.axis('off')


fig, axs = plt.subplots(1, 3, figsize=(24, 8))
G_pagerank = G_full.subgraph(top_nodes_pagerank)
visualize(G_pagerank, pagerank, axs[0], "Top Github Developers By PageRank", cmap="YlGnBu")
G_degree = G_full.subgraph(top_nodes_degree)
visualize(G_degree, degree_centrality, axs[1], "Top Github Developers by Degree Centrality", cmap="YlOrRd")

G_overlap = G_full.subgraph(overlap_nodes)
visualize(G_overlap, pagerank, axs[2], "Overlap of Top Developers by PageRank and Degree Centrality", cmap="PuRd")
plt.tight_layout()
plt.show()

fig_community, ax_community = plt.subplots(figsize=(12, 8))
community_map = [partition[node] for node in G_full.nodes()]
cmap = mcolors.LinearSegmentedColormap.from_list("community_cmap", plt.cm.tab20.colors, N=unique_communities)
nx.draw_networkx(G_full, pos=nx.spring_layout(G_full, seed=42), node_color=community_map, with_labels=False,
                 node_size=20, cmap=cmap, ax=ax_community)
ax_community.set_title("Community Structure in GitHub Developer Network")
plt.tight_layout()
plt.show()


