import networkx as nx

import config
import data_preprocess
import numpy as np

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


def create_member_similarity_array(member_list):
    member_num = max(len(member_list), member_list[-1]['id'] + 1)
    net = np.zeros((member_num, member_num), dtype=int)
    for index1, member1 in enumerate(member_list):
        print(f"processing {index1} / {len(member_list)}")
        for index2, member2 in enumerate(member_list):
            net[index1][index2] = compute_member_similarity(member1, member2)

    return net

def create_group_net(member_list, group_list, event_list, member_sim_mode = 'topic'):
    # 用团体之间的联系建图， 成员间的联系权重为成员相似度和共同组之间的加权

    def member_sim(u, v):
        if member_sim_mode == 'topic':
            if len(u['topics']) == 0 or len(v['topics']) == 0:
                return 0.0
            return len([topic for topic in u['topics'] if topic in v['topics']]) ** 2 / (len(u['topics'])) / len(v['topics'])
        else:
            u_invited_set = set(u['yes'] + u['no'] + u['maybe'])
            v_invited_set = set(v['yes'] + v['no'] + v['maybe'])
            common_invited = len(u_invited_set & v_invited_set)
            u_set = set(u["yes"])
            v_set = set(v['yes'])
            yes = len(u_set & v_set) / common_invited
            u_set = set(u["no"])
            v_set = set(v["no"])
            no = len(u_set & v_set) / common_invited
            u_set = set(u['maybe'])
            v_set = set(v['maybe'])
            maybe = len(u_set & v_set) / common_invited
            return yes + no + maybe

    group_list = data_preprocess.group_member_extend(group_list, event_list)
    print("begin construct directed network")
    G = nx.Graph()
    # adding nodes
    node_list = [member['id'] for member in member_list]
    G.add_nodes_from(node_list)
    # adding edges
    for group in group_list:
        for idx1 in range(len(group['members']) - 1):
            for idx2 in range(idx1 + 1, len(group['members'])):
                g_id1 = group['members'][idx1]
                g_id2 = group['members'][idx2]
                if not G.has_edge(g_id1, g_id2):
                    G.add_edge(g_id1, g_id2, common_group_num = 1, sim = member_sim(member_list[g_id1], member_list[g_id2]))
                else:
                    G.edges[g_id1, g_id2]['common_group_num'] += 1
        print(f'processed group {group["id"]}(with {len(group["members"])} members)')
    return  G