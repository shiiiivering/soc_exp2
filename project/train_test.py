import random
import data_preprocess
import numpy as np
import config
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import scale

# 仅划分事件，
def process_k_fold(member_list, event_list, topic_dict, group_list, k):
    # k折划分测试
    random.shuffle(event_list)
    total_size = len(event_list)
    grid_size = total_size // k
    for i in range(k):
        train_set = list()
        for idx in range(k):
            if idx == i:
                test_set = event_list[grid_size * i : grid_size * (i + 1)]
            else:
                train_set = train_set + event_list[grid_size * i : grid_size * (i + 1)]
        correct_rate = test_similarity(member_list, topic_dict, group_list, train_set, test_set, 'topic')
        print(f"the {i}th fold, correct rate: {correct_rate}")


def test_similarity(member_list, topic_dict, group_list, train_set, test_set, sim_type='topic'):
    '''
    仅使用用户间相似度的网络
    流程：
    1. 将训练集中的event信息加入member_list （member_event_extend）
    2. 对每个事件中的有关用户进行预测
    2.2 计算要预测的用户和所有其他用户的相似度
        sim_type:
        - 'topic': 基于用户topic计算相似度，(共同topic数 / ((用户1 topic数) * (用户2 topic数)))
        - 'event': 基于用户共同参与的事件计算相似度，使用评分矩阵，行：用户，列：事件，评分： yes:2, no:-1, maybe:1, organizer:3, other: 0
    2.3 找出config.num_neighbour个最相似的邻居，并找出他们有关的事件集
    2.4 有关事件集与要预测的事件计算相似度(共同topic数, 见compute_event_sim)
    2.5 事件相似度 * 成员相似度，即为这个事件对某个决定(yes, no, maybe)可能性的权重加成
    2.6 找出可能性权重最大的一个决定即为预测值
    :param member_list: 原始的memberlist
    :param topic_dict:
    :param group_list:
    :param train_set: 划分出来的事件的训练集
    :param test_set: 划分出来的事件的测试集
    :param sim_type: 相似度计算方法
    :return:
    '''
    member_list = data_preprocess.member_event_extend(member_list, train_set)
    compared_event = dict()  # keys are event_ids; values are event similarity
    correct_num = 0
    total_test = 0

    if sim_type == 'topic':
        member_topic_num = np.array([len(member['topics']) for member in member_list])
    elif sim_type == 'event':
        #注意：需要 100GB 内存！！！
        sim_array = np.zeros((len(member_list), len(train_set)+len(test_set)), dtype = float)
        for member in member_list:
            sim_array[member['id']][[e['id']-1 for e in member['yes']]] = 2
            sim_array[member['id']][[e['id']-1 for e in member['no']]] = -1
            sim_array[member['id']][[e['id']-1 for e in member['maybe']]] = 1
            sim_array[member['id']][[e['id']-1 for e in member['organizer']]] = 3
        sim_array = scale(sim_array, axis=1, with_std=False)
        sim_array = cosine_similarity(sim_array)    # len(member_list) * len(member_list)

    def compute_event_sim(event1, event2):
        # 事件之间的相似度按照（共有topic数量 ** 2 / 两者各自topic数量和）  来计算
        related_topic1 = set()
        related_topic2 = set()
        for o in (event1['yes'] + event1['no'] + event1["maybe"]):
            related_topic1.update(member_list[o]['topics'])
        for o in (event2['yes'] + event2['no'] + event2["maybe"]):
            related_topic2.update(member_list[o]['topics'])
        base = (len(related_topic1) + len(related_topic2))
        if base == 0:
            return 0.0
        return len(related_topic1 & related_topic2) ** 2 / (len(related_topic1) + len(related_topic2))

        # # 事件相似度按照向量乘积平方根算 效果不好
        # related_topic = set()
        # for o in (event1['yes'] + event1['no'] + event1["maybe"]):
        #     related_topic.update(member_list[o]['topics'])
        # for o in (event2['yes'] + event2['no'] + event2["maybe"]):
        #     related_topic.update(member_list[o]['topics'])
        # related_topic = list(related_topic)
        # topic_hash = dict()
        # for idx, i in enumerate(related_topic):
        #     if i not in topic_hash.keys():
        #         topic_hash[i] = idx
        # related_topic1 = np.zeros(len(topic_hash.keys()))
        # related_topic2 = np.zeros(len(topic_hash.keys()))
        # for o in (event1['yes'] + event1['no'] + event1["maybe"]):
        #     a = [topic_hash[p] for p in member_list[o]['topics']]
        #     related_topic1[a] += 1
        # for o in (event2['yes'] + event2['no'] + event2["maybe"]):
        #     a = [topic_hash[p] for p in member_list[o]['topics']]
        #     related_topic2[a] += 1
        # return np.dot(related_topic2, related_topic1) ** 0.5

    def predict(member_id, event):
        # predict member_id's choice of event
        if sim_type == 'topic':
            member = member_list[member_id]
            sim_v = np.zeros(len(member_list), dtype = float)
            for topic_id in member['topics']:
                related_member_list = topic_dict[str(topic_id)]['members']
                sim_v[related_member_list] += 1
            with np.errstate(divide='ignore', invalid='ignore'):
                sim_v = sim_v ** 2
                sim_v /= member_topic_num
                np.nan_to_num(sim_v, 0.0)
            k_ns = np.argpartition(sim_v, -config.num_neighbours)[-config.num_neighbours:]
        elif sim_type == 'event':
            sim_v = sim_array[member_id]
            k_ns = np.argpartition(sim_v, -config.num_neighbours)[-config.num_neighbours:]

        total_yes = 0
        total_maybe = 0
        total_no = 0

        for related_member_id in k_ns:
            if sim_v[related_member_id] <= 0.0:
                continue
            related_member = member_list[related_member_id]
            yes = 0
            maybe = 0
            no = 0
            for e in related_member['yes']:
                if e['id'] in compared_event.keys():
                    event_sim = compared_event[e['id']]
                else:
                    event_sim = compute_event_sim(e, event)
                    compared_event[e['id']] = event_sim
                yes += event_sim
            for e in related_member['no']:
                if e['id'] in compared_event.keys():
                    event_sim = compared_event[e['id']]
                else:
                    event_sim = compute_event_sim(e, event)
                    compared_event[e['id']] = event_sim
                no += event_sim
            for e in related_member['maybe']:
                if e['id'] in compared_event.keys():
                    event_sim = compared_event[e['id']]
                else:
                    event_sim = compute_event_sim(e, event)
                    compared_event[e['id']] = event_sim
                maybe += event_sim

            c = sim_v[related_member["id"]]
            total_no += no * c
            total_yes += yes * c
            total_maybe += maybe * c
        max_ = max(total_no, total_yes, total_maybe)
        if max_ == total_yes:
            return 1
        elif max_ == total_no:
            return -1
        else:
            return 0


    for idx, event in enumerate(test_set):
        compared_event = dict()  # keys are event_ids; values are event similarity
        for member in event['yes']:
            total_test += 1
            if predict(member, event) == 1:
                correct_num += 1
        for member in event['no']:
            total_test += 1
            if predict(member, event) == -1:
                correct_num += 1
        for member in event['maybe']:
            total_test += 1
            if predict(member, event) == 0:
                correct_num += 1
        print(f"predicted {idx} / {len(test_set)} event, correct rate: {correct_num / total_test}")
    return correct_num / total_test
