import networkx as nx
import matplotlib.pyplot as plt

from Models.Node2Vec import Node2Vec

SEED = 3

graph = nx.barabasi_albert_graph(10, 2, seed=SEED)
nx.draw(graph)
# plt.show()

# print(graph.nodes)
# print(graph.edges)
# print(graph.adj)

G = Node2Vec(graph, p=.5, q=.7)
print(G.graph)
print(G.d_graph)
print(G.d_graph.is_directed())

iters = range(10)
nodes = G.d_graph.nodes
for i in iters:
    print(G.d_graph[i])
print('\n')

G.preprocess_weights()

for i in iters:
    print(G.d_graph[i])
    for j in G.d_graph[i]:
        print(sum(G.d_graph[i][j]['second_order_probs'].values()))
    print()
print('\n')

print(G.dims)
print(G.num_walks)