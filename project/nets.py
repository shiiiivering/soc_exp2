import networkx as nx

import config
import data_preprocess

def compute_member_similarity(u, v):
    return len([topic for topic in u['topics'] if topic in v['topics']])

def compute_member_similarity_influence(u, v):
    u_t = len(u['topics'])
    inter = len([topic for topic in u['topics'] if topic in v['topics']]) # topic 交集长度
    if inter == 0:
        return 0
    return inter * 1.0 / u_t

def create_member_similarity_network(member_list, topic_dict):
    G = nx.Graph()
    # adding nodes
    node_list = [(member['id'], member) for member in member_list]
    G.add_nodes_from(node_list)
    # adding weighted edges
    for topic in topic_dict.values():
        members = list(topic['members'])
        for u in members:
            for v in members:
                if not G.has_edge(u, v):
                    weight = compute_member_similarity(G.nodes[u], G.nodes[v])
                    if weight != 0:
                        G.add_edge(u, v, weight=weight)

    return G

def create_member_similarity_dinetwork(member_list, topic_dict):
    # create a directed graph for members
    # for edge e: u-->v, e.weight = num of u's topics / num topics u and v have in common
    print("begin construct directed network")
    G = nx.DiGraph()
    # adding nodes
    node_list = [(member['id'], member) for member in member_list]
    G.add_nodes_from(node_list)
    print("set nodes complete, begin set edges")

    # adding weighted edges
    counter = 0
    found_edge = 0
    for topic in topic_dict.values():
        print(f"proceeding topic {topic['id']} with {len(topic['members'])} members")
        members = list(topic['members'])
        for u in members:
            for v in members:
                if u == v:
                    continue
                counter += 1
                if not G.has_edge(u, v):
                    weight = compute_member_similarity_influence(G.nodes[u], G.nodes[v])
                    if weight != 0:
                        G.add_edge(u, v, weight=weight)
                        found_edge += 1
                        if found_edge % 1000000 == 0:
                            print(f"##########found {found_edge} edges (out of {counter} tests)")
    return G