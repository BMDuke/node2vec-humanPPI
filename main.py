from ast import walk
import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd

from Models.Node2Vec import Node2Vec

SEED = 3

graph = nx.barabasi_albert_graph(10, 2, seed=SEED)
nx.draw(graph)
# plt.show()

# print(graph.nodes)
# print(graph.edges)
# print(graph.adj)

# G = Node2Vec(graph, p=.5, q=.7)
# print(G.graph)
# print(G.d_graph)
# print(G.d_graph.is_directed())

# iters = range(10)
# nodes = G.d_graph.nodes
# for i in iters:
#     print(G.d_graph[i])
# print('\n')

# G.preprocess_weights()

# for i in iters:
#     print(G.d_graph[i])
#     for j in G.d_graph[i]:
#         print(sum(G.d_graph[i][j]['second_order_probs'].values()))
#     print()
# print('\n')

# print(G.dims)
# print(G.num_walks)

# print()

# # print(G.generate_walks(num_walks=10, walk_length=10))

# for w in G.generate_walks(num_walks=10, walk_length=10):
#     print(w)

path = 'data/node_classification_edge_list.csv'

human_ppi = pd.read_csv(path)

edge_list = [(x, y) for x, y in human_ppi.to_numpy()]

human_ppi_graph = nx.Graph(edge_list) # 92 nodes shy of total 

print(human_ppi.head())
print(human_ppi.to_numpy()[:, 0])
print(edge_list[:10])
print(human_ppi_graph)



# print(human_ppi_graph)

path_nx_graph = 'cache/nx'
path_tf_dataset = 'cache/tf'

G = Node2Vec(human_ppi_graph, p=.5, q=.7)
G.preprocess_weights(use_cache=True, path=path_nx_graph)    
walks = G.generate_walks(num_walks=15, walk_length=40)
# print(G.stringify(walks))
G.fit(walks, 20, use_cache=False, path=path_tf_dataset)