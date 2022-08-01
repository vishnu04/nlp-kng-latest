import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from . import config

@config.timer
def plot_kg(svo_df):
    # plt.figure(figsize=(12,12))
    edges = []
    edge_labels = {}
    for index, svo_tuple in svo_df.iterrows():
        edges.append([svo_tuple.subject, svo_tuple.object])
        edge_labels[(svo_tuple.subject, svo_tuple.object)] = svo_tuple.verb

    #print(edges)
    #print(edge_labels)
    G=nx.Graph()
    G=nx.from_pandas_edgelist(svo_df, "subject", "object", 
                            edge_attr=True, create_using=nx.MultiDiGraph())
    pos = nx.spring_layout(G)
    # G.add_edges_from(svo_df.verb)
    # nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
    # nx.draw_networkx_edge_labels(G, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos, edge_labels = svo_df.verb)
    # nx.draw_networkx_edge_labels(G, pos = pos,edge_labels = edge_labels)
    return G, pos, edge_labels


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