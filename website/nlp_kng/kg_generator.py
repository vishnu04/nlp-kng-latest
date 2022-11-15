import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from . import config

# @config.timer
# def plot_kg(svo_df):
#     # plt.figure(figsize=(12,12))
#     edges = []
#     edge_labels = {}
#     for index, svo_tuple in svo_df.iterrows():
#         edges.append([svo_tuple.subject, svo_tuple.object])
#         edge_labels[(svo_tuple.subject, svo_tuple.object)] = svo_tuple.verb

#     #print(edges)
#     #print(edge_labels)
#     G=nx.Graph()
#     G=nx.from_pandas_edgelist(svo_df, "subject", "object", 
#                             edge_attr=True, create_using=nx.MultiDiGraph())
#     pos = nx.spring_layout(G)
#     # G.add_edges_from(svo_df.verb)
#     # nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
#     # nx.draw_networkx_edge_labels(G, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos, edge_labels = svo_df.verb)
#     # nx.draw_networkx_edge_labels(G, pos = pos,edge_labels = edge_labels)
#     return G, pos, edge_labels


@config.timer
def plot_kg(svo_df):
    svo_edges = []
    svo_edge_labels = {}
    for index, svo_tuple in svo_df.iterrows():
        svo_edges.append([svo_tuple.subject.strip(' '), svo_tuple.object.strip(' ')])
        svo_edge_labels[(svo_tuple.subject.strip(' '), svo_tuple.object.strip(' '))] = svo_tuple.verb
    # print(svo_edges)
    # print(svo_edge_labels)
    G=nx.DiGraph()
    svo_nodes = list(set(list(svo_df.subject) + list(svo_df.object)))
    svo_nodes = [x.strip(' ') for x in svo_nodes]
    print(svo_nodes)
    G.add_nodes_from(svo_nodes)
    G.add_edges_from(svo_edges)
    pos = nx.spring_layout(G)
    # nx.draw_networkx(G,pos, node_color='skyblue', edge_cmap=plt.cm.Blues)
    # nx.draw_networkx_edge_labels(G, pos = pos,edge_labels = svo_edge_labels)
    return G, pos, svo_edge_labels


# if __name__ == "__main__":
#     plt.figure(figsize=(12,12))
#     edges = []
#     edge_labels = {}
#     svo_df = 
#     for index, svo_tuple in svo_df.iterrows():
#         edges.append([svo_tuple.subject, svo_tuple.object])
#         edge_labels[(svo_tuple.subject, svo_tuple.object)] = svo_tuple.verb

#     #print(edges)
#     #print(edge_labels)
#     G=nx.Graph()
#     G=nx.from_pandas_edgelist(svo_df, "subject", "object", 
#                             edge_attr=True, create_using=nx.MultiDiGraph())
#     pos = nx.spring_layout(G)
#     #G.add_edges_from(svo_df.verb)
#     nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
#     #nx.draw_networkx_edge_labels(G, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos, edge_labels = svo_df.verb)
#     nx.draw_networkx_edge_labels(G, pos = pos,edge_labels = edge_labels)

#     plt.show()