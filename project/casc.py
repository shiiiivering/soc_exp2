import os
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import data_preprocess
import json


def read_one_event(path):
    event_list = list()
    group_id = 481
    # while True:
    ge_path = os.path.join(path, 'GroupEvent', 'G' + str(group_id) + '.txt')
    # if not os.path.exists(ge_path):
    #     break
    with open(ge_path, 'r') as f:
        while True:
            line = f.readline().strip('\n')
            if not line:
                break
            event = data_preprocess.new_event()
            event_info = line.split(" ")
            event['id'] = int(event_info[0][1:])
            event['limit'] = int(event_info[1])
            event['timestamp'] = int(event_info[2])
            members = f.readline().strip('\n').split(' ')[0:-1]
            members = filter(lambda x: x != 'null', members)
            event['organizers'] = [int(member[1:]) for member in members]
            members = f.readline().strip('\n').split(' ')[0:-1]
            members = filter(lambda x: x != 'null', members)
            event['yes'] = [int(member[1:]) for member in members]
            members = f.readline().strip('\n').split(' ')[0:-1]
            members = filter(lambda x: x != 'null', members)
            event['no'] = [int(member[1:]) for member in members]
            members = f.readline().strip('\n').split(' ')[0:-1]
            members = filter(lambda x: x != 'null', members)
            event['maybe'] = [int(member[1:]) for member in members]
            event['group'] = group_id
            event_list.append(event)
    # group_id = group_id + 1
    return event_list


def read_single_group_event(path, rebuild=False):
    # Group Event
    ge_json_path = os.path.join(path, 'one_event_list.json')
    # if event_list.json
    if os.path.exists(ge_json_path) and (not rebuild):
        with open(ge_json_path) as f:
            event_one_list = json.load(f)
    # read from source file and save file
    else:
        event_one_list = read_one_event(path)
        with open(ge_json_path, 'w') as f:
            json.dump(event_one_list, f)
    print("load event one info complete")
    return event_one_list


def set_No(G, list1):
    for i in list1:
        G.nodes[i]['action'] = 'No'
    return G


def get_colors(G):
    color = []
    for i in G.nodes():
        if G.nodes[i]['action'] == 'No':
            color.append('red')
        else:
            color.append('green')
    return color


def recalculate(G):
    dict1 = {}
    a = 0.7
    b = 0.1

    for i in G.nodes():
        neigh = G.neighbors(i)
        count_Yes = 0
        count_No = 0

        for j in neigh:
            if G.nodes[j]['action'] == 'Yes':
                count_Yes += 1
            else:
                count_No += 1
        payoff_Yes = a * count_Yes
        payoff_No = b * count_No

        if payoff_Yes >= payoff_No:
            dict1[i] = 'Yes'
        else:
            dict1[i] = 'No'
    return dict1


def Calculate_key_people(G):
    continuee = True
    count = 0
    c = 0

    while continuee and count < 100:
        count += 1

        # action_dict will hold a dictionary
        action_dict = recalculate(G)
        G = reset_node_attributes(G, action_dict)
        colors = get_colors(G)

        if colors.count('red') == len(colors) or colors.count('green') == len(colors):
            continuee = False
            if colors.count('green') == len(colors):
                c = 1

    if c == 1:
        print('key people')
    else:
        print('not key people')


def key_people(G):
    for i in G.nodes():
        for j in G.nodes():
            if i < j:
                list1 = []
                list1.append(i)
                list1.append(j)
                print(list1, ':', end="")
                # colors = get_colors(G)
                Calculate_key_people(G)


def Calculate_Cluster(G):
    terminate = True
    count = 0
    c = 0
    while terminate and count < 100:
        count += 1

        # action_dict will hold a dictionary
        action_dict = recalculate(G)
        G = reset_node_attributes(G, action_dict)
        colors = get_colors(G)

        if colors.count('red') == len(colors) or colors.count('green') == len(colors):
            terminate = False
            if colors.count('green') == len(colors):
                c = 1

    if c == 1:
        print('cascade complete')
    else:
        print('cascade incomplete')
    # nx.draw(G, with_labels=1, node_color=colors, node_size=800)
    # plt.show()


def cascading_on_cluster(G):
    list2 = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 12],
             [2, 3, 4, 12], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 12]]

    for list1 in list2:
        print(list1)

        G = set_No(G, list1)
        colors = get_colors(G)
        # nx.draw(G, with_labels=1, node_color=colors, node_size=800)
        # plt.show()
        Calculate_Cluster(G)


def reset_node_attributes(G, action_dict):
    for i in action_dict:
        G.nodes[i]['action'] = action_dict[i]
    return G


def Calculate_Payoff(G):
    terminate = True
    count = 0
    c = 0

    while terminate and count < 10:
        count += 1

        # action_dict will hold a dictionary
        action_dict = recalculate(G)
        G = reset_node_attributes(G, action_dict)
        colors = get_colors(G)

        if colors.count('red') == len(colors) or colors.count('green') == len(colors):
            terminate = False
            if colors.count('green') == len(colors):
                c = 1
        # nx.draw(G, with_labels=1, node_color=colors, node_size=800)
        # plt.show()
    if c == 1:
        print('cascade complete for payoff')
    else:
        print('cascade incomplete for payoff')


def get_events(G_event):
    colors = []
    y = []
    n = []
    m = []
    i = 1
    for event in G_event:
        for e in event['yes']:
            if e in y:
                continue
            else:
                y.append(i)
                i += 1
        for e in event['no']:
            if e in n:
                continue
            else:
                n.append(i)
                i += 1
        for e in event['maybe']:
            if e in m:
                continue
            else:
                m.append(i)
                i += 1
    return y, n, m


def get_community(G, event, color, action):
    f = event[0]
    l = event[len(event) - 1]
    colors = []
    for i in range(f, l):
        G.add_node(i)
        colors.append(color)
        G.nodes[i]['action'] = action
    for i in range(f, l):
        for j in range(f, l):
            if i < j:
                r = random.random()
                if r < 0.5:
                    G.add_edge(i, j)
    return G, colors


def community_impact(G_event):
    G = nx.Graph()
    colors = []
    yes, no, maybe = get_events(G_event)
    # if yes:
    #     G, green = get_community(G, yes[0:10], 'green', 'Yes')
    # if no:
    #     G, red = get_community(G, no[len(no) - 8:len(no)], 'red', 'No')
    #     colors = green + red
    if yes:
        G, green = get_community(G, yes, 'green', 'Yes')
    if no:
        G, red = get_community(G, no, 'red', 'No')
        colors = green + red
    # nx.draw(G, node_color=colors, with_labels=True, node_size=500)
    # plt.show()
    G.add_edge(1, no[len(no) - 2])
    return G
