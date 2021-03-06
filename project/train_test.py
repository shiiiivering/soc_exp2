import random
import data_preprocess
import numpy as np
import config
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import scipy

# 仅划分事件，
def process_k_fold(member_list, event_list, topic_dict, group_list, k, test='similarity', user_sim='topic', event_sim='all', social_constraint='event'):
    # k折划分测试
    random.shuffle(event_list)
    total_size = len(event_list)
    grid_size = total_size // k
    test_set = list()
    all_correct_rates = []
    for i in range(k):
        train_set = list()
        for idx in range(k):
            if idx == i:
                test_set = event_list[grid_size * idx : grid_size * (idx + 1)] if idx != k-1 else event_list[grid_size * idx : ]
            else:
                train_set += event_list[grid_size * idx : grid_size * (idx + 1)] if idx != k-1 else event_list[grid_size * idx : ]
        assert(len(test_set)+len(train_set) == total_size)
        if test == 'similarity':
            correct_rate = test_similarity(member_list, topic_dict, group_list, train_set, test_set, user_sim, event_sim, social_constraint)
        elif test == 'cascade':
            correct_rate = test_cascade(member_list, topic_dict, group_list, train_set, test_set, user_sim, social_constraint)
        all_correct_rates.append(correct_rate)
        print(f"the {i}th fold, correct rate: {correct_rate}", flush=True)
    print(f"mean correct rate: {np.mean(all_correct_rates)}", flush=True)


def test_similarity(member_list, topic_dict, group_list, train_set, test_set, user_sim='topic', event_sim='organizers', social_constraint='group', print_correct=False):
    '''
    仅使用用户间相似度的网络
    流程：
    1. 将训练集中的event信息加入member_list （member_event_extend）
    2. 对每个事件中的有关用户进行预测
    2.2 计算要预测的用户和所有其他用户的相似度
        user_sim:
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
    :param user_sim: 用户相似度计算方法
    :param event_sim: 事件相似度计算方法
    :param social_constraint: 使用什么社交约束
    :return:
    '''
    member_list = data_preprocess.member_event_extend(member_list, train_set)
    group_list = data_preprocess.group_member_extend(group_list, train_set)
    compared_event = dict()  # keys are event_ids; values are event similarity
    correct_num = 0
    total_test = 0

    if user_sim == 'topic':
        member_topic_num = np.array([len(member['topics']) for member in member_list])
    elif user_sim == 'event':
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
        if event_sim == 'organizers':
            for o in event1['organizers']:
                related_topic1.update(member_list[o]['topics'])
            for o in event2['organizers']:
                related_topic2.update(member_list[o]['topics'])
        else:
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
        if user_sim == 'topic':
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
        elif user_sim == 'event':
            sim_v = sim_array[member_id]
            k_ns = np.argpartition(sim_v, -config.num_neighbours)[-config.num_neighbours:]

        total_yes = 0
        total_maybe = 0
        total_no = 0

        group_member = None
        if social_constraint == 'group':
            group_member = group_list[event['group']]['members']
        elif social_constraint == 'event':
            group_member = set([member_id for member_id in event['yes'] + event['no'] + event['maybe']])
        
        # else social_constraint == 'None'

        for related_member_id in k_ns:
            if sim_v[related_member_id] <= 0.0:
                continue
            if (social_constraint!='None') and (related_member_id not in group_member):
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
        if print_correct:
            print(f"predicted {idx} / {len(test_set)} event, correct rate: {correct_num / total_test}")
    return correct_num / total_test


def test_cascade(member_list, topic_dict, group_list, train_set, test_set, user_sim='choice', social_constraint='group', print_correct=False):
    # 社交级联，用迭代的方式使用户间互相影响， 最大迭代次数为10
    # 用户相似性为三维，对应不同决定, 计算方式见choice_dim函数，注意这个相似性不是对称的，而是表示当另一个用户做出某个选择时对此用户的影响强度
    # 函数最后注释部分为绘制级联迭代次数统计图
    group_list = data_preprocess.group_member_extend(group_list, train_set)
    correct = 0
    predicted = 0
    cascading_step_counter = np.zeros(10, dtype=int)
    # 将事件信息加入member list
    member_list = data_preprocess.member_event_extend(member_list, train_set)
    # # 创建成员和团体的相关矩阵 暂时没用到
    # member_group = np.zeros((len(member_list), len(group_list)), dtype=float)
    # for event in train_set:
    #     member_group[event['yes'], event['group']] += 2
    #     member_group[event['no'], event['group']] += 1
    #     member_group[event['maybe'], event['group']] -= 1
    #     member_group[event['organizers'], event['group']] += 4
    # 测试

    def compute_member_sim(related_member):
        member_sim = None
        if user_sim == 'choice':
            member_sim = np.zeros((len(related_member), len(related_member), 3), dtype = float)
            member_event = np.zeros((len(related_member), len(train_set) + len(test_set) + 1))
            # 创建用户事件关系矩阵
            # yes & organizer: 1
            # no: 2
            # maybe: 3
            for idx1, member_id in enumerate(related_member):
                member = member_list[member_id]
                member_event[idx1, [e['id'] for e in member['yes'] + member['organizer']]] = 1
                member_event[idx1, [e['id'] for e in member['no']]] = 2
                member_event[idx1, [e['id'] for e in member['maybe']]] = 3
            # 计算用户相似性
            for idx1, member_id1 in enumerate(related_member):
                for idx2, member_id2 in enumerate(related_member):
                    if member_id1 == member_id2:
                        continue
                    else:
                        def choice_sim(m1, m2, choice):
                            c1 = member_event[m1] == choice
                            c2 = member_event[m2] == choice
                            c2 = c2 & (member_event[m1] != 0)
                            if c2.sum() == 0:
                                return 0.0
                            result =  (c1 & c2).sum() / (c2).sum()
                            return result
                        member_sim[idx1][idx2][0] = choice_sim(idx1, idx2, 1)
                        member_sim[idx1][idx2][1] = choice_sim(idx1, idx2, 2)
                        member_sim[idx1][idx2][2] = choice_sim(idx1, idx2, 3)
        elif user_sim == 'event':
            member_sim = np.zeros((len(related_member), len(train_set) + len(test_set) + 2))
            # 创建用户事件关系矩阵
            # yes: 2
            # no: -1
            # maybe: 1
            # organizer: 3
            for idx1, member_id in enumerate(related_member):
                member = member_list[member_id]
                member_sim[idx1, [e['id'] for e in member['yes']]] = 2
                member_sim[idx1, [e['id'] for e in member['no']]] = -1
                member_sim[idx1, [e['id'] for e in member['maybe']]] = 1
                member_sim[idx1, [e['id'] for e in member['organizer']]] = 3
            member_sim = scale(member_sim, axis=1, with_std=False)
            member_sim = cosine_similarity(member_sim)    # len(related_member) * len(related_member)
        return member_sim
            

    for idx, event in enumerate(test_set):
        related_member = set([member_id for member_id in event['yes'] + event['no'] + event['maybe']])
        if social_constraint == 'group':
            related_member.update(group_list[event['group']]['members'])
        related_member = list(related_member)        
        if len(related_member) == 0:
            continue
        choice = np.zeros(len(related_member), dtype = int)
        member_sim = compute_member_sim(related_member)
        # 初始化, 使用用户在其他事件中做出最多次数的回应作为回应
        for idx1, member_id in enumerate(related_member):
            member = member_list[member_id]
            weight = np.array([0, 0, 0])
            for event1 in member['yes']:
                if event1['group'] == event['group']:
                    weight[0] += 2
                else:
                    weight[0] += 1
            for event1 in member['no']:
                if event1['group'] == event['group']:
                    weight[1] += 2
                else:
                    weight[1] += 1
            for event1 in member['maybe']:
                if event1['group'] == event['group']:
                    weight[2] += 2
                else:
                    weight[2] += 1
            choice[idx1] = np.argmax(weight)
        # 级联, 并迭代
        for i in range(config.cascading_max_step):
            temp_choice = choice.copy()
            for idx1, member in enumerate(related_member):
                weight = np.array([0.0, 0.0, 0.0])
                for idx2, member_choice in enumerate(choice):
                    if user_sim == 'choice':
                        weight[member_choice] += member_sim[idx1][idx2][member_choice]
                    elif user_sim == 'event':
                        weight[member_choice] += member_sim[idx1][idx2] if idx1 != idx2 else 0.0
                temp_choice[idx1] = np.argmax(weight)
            if np.all(temp_choice == choice):
                break
            choice = temp_choice
        cascading_step_counter[i] += 1
        # 检测:
        event_member = None
        if social_constraint != 'event':
            event_member = set([member_id for member_id in event['yes'] + event['no'] + event['maybe']])
        for idx1, pre in enumerate(choice):
            member_id = related_member[idx1]
            if pre == 0 and (member_id in event['yes']):
                correct += 1
            elif pre == 1 and (member_id in event['no']):
                correct += 1
            elif pre == 2 and (member_id in event['maybe']):
                correct += 1
            if social_constraint == 'event' or member_id in event_member:
                predicted += 1
            if predicted % 100 == 0 and print_correct:
                print(f"predicted {predicted} times in {idx + 1} events, {correct} correct, correct rate is {correct / predicted}")
            # # Count the number of cascading iterations and draw plots
            # if predicted % 300 == 0:
            #     plt.bar(range(config.cascading_max_step), cascading_step_counter)
            #     plt.show()
    return correct / predicted


def evolution_over_time(member_list, topic_dict, group_list, event_list, start_point=0):
    # 声明
    update_frequency = 100
    correct = 0
    predicted = 0
    part_correct = 0
    part_predicted = 0
    correct_rate_list = list()
    cascading_max_step = 10
    cascading_step_counter = np.zeros(10, dtype=int)

    # 向member添加事件信息初始化
    for member in member_list:
        member['yes'] = list()
        member['no'] = list()
        member['maybe'] = list()
        member['organizer'] = list()
    # 向group添加成员信息初始化
    for group in group_list:
        group['members'] = set()

    event_list.sort(key=lambda x : x['timestamp'])
    event_time_weight = np.array([0.0] + [event['timestamp'] for event in event_list], dtype=float)
    event_time_weight -= event_time_weight[0]

    for current_event_idx, current_event in enumerate(event_list):
        if current_event_idx <= start_point:
            event = event_list[current_event_idx]
            for member in event['yes']:
                member_list[member]['yes'].append(event)
            for member in event['no']:
                member_list[member]['no'].append(event)
            for member in event['maybe']:
                member_list[member]['maybe'].append(event)
            continue

        related_member = [member for member in current_event['yes'] + current_event['no'] + current_event['maybe']]
        choice = np.zeros(len(related_member), dtype=int)
        member_sim = np.zeros((len(related_member), len(related_member), 3), dtype=float)
        member_event = np.zeros((len(related_member), len(event_list) + 1))

        # 计算事件随时间变化的权重
        # current_event_time_weight = event_time_weight - (event_time_weight[max(0, current_event_idx - 30000)] - 1)
        # current_event_time_weight /= current_event_time_weight[current_event_idx]
        # current_event_time_weight[current_event_time_weight < 0] = 0.0
        # current_event_time_weight += current_event_time_weight[current_event_idx] / 2.0
        current_event_time_weight = event_time_weight - event_time_weight[0]
        current_event_time_weight[current_event_idx + 1 : ] = 0
        current_event_time_weight /= current_event_time_weight[current_event_idx]
        # current_event_time_weight = scipy.special.softmax(current_event_time_weight)
        current_event_time_weight = current_event_time_weight ** 0.5
        # 创建用户事件关系矩阵
        # yes: 1
        # no: 2
        # maybe: 3
        # organizer: 4
        for idx1, member_id in enumerate(related_member):
            member = member_list[member_id]
            member_event[idx1, [e['id'] for e in member['yes'] + member['organizer']]] = 1
            member_event[idx1, [e['id'] for e in member['no']]] = 2
            member_event[idx1, [e['id'] for e in member['maybe']]] = 3
        # 计算用户相似性
        for idx1, member_id1 in enumerate(related_member):
            for idx2, member_id2 in enumerate(related_member):
                if member_id1 == member_id2:
                    continue
                else:
                    def choice_sim(m1, m2, choice):
                        c1 = member_event[m1] == choice
                        c2 = member_event[m2] == choice
                        c2 = c2 & (member_event[m1] != 0)
                        if c2.sum() == 0:
                            return 0.0
                        result =  ((c1 & c2) * current_event_time_weight).sum() / (c2 * current_event_time_weight).sum()
                        return result
                    member_sim[idx1][idx2][0] = choice_sim(idx1, idx2, 1)
                    member_sim[idx1][idx2][1] = choice_sim(idx1, idx2, 2)
                    member_sim[idx1][idx2][2] = choice_sim(idx1, idx2, 3)
        # 初始化, 使用用户在其他事件中做出最多次数的回应作为回应
        for idx1, member_id in enumerate(related_member):
            member = member_list[member_id]
            weight = np.array([0, 0, 0])
            for event1 in member['yes']:
                if event1['group'] == current_event['group']:
                    weight[0] += 2 * current_event_time_weight[event1["id"]]
                else:
                    weight[0] += 1 * current_event_time_weight[event1["id"]]
            for event1 in member['no']:
                if event1['group'] == current_event['group']:
                    weight[1] += 2 * current_event_time_weight[event1["id"]]
                else:
                    weight[1] += 1 * current_event_time_weight[event1["id"]]
            for event1 in member['maybe']:
                if event1['group'] == current_event['group']:
                    weight[2] += 2 * current_event_time_weight[event1["id"]]
                else:
                    weight[2] += 1 * current_event_time_weight[event1["id"]]
            choice[idx1] = np.argmax(weight)
        # 级联, 并迭代
        for i in range(cascading_max_step):
            temp_choice = choice.copy()
            for idx1, member in enumerate(related_member):
                weight = np.array([0.0, 0.0, 0.0])
                for idx2, member_choice in enumerate(choice):
                    weight[member_choice] += member_sim[idx1][idx2][member_choice]
                temp_choice[idx1] = np.argmax(weight)
            if np.all(temp_choice == choice):
                break
            choice = temp_choice
        cascading_step_counter[i] += 1
        # 检测:
        for idx1, pre in enumerate(choice):
            member_id = related_member[idx1]
            if pre == 0 and (member_id in current_event['yes']):
                correct += 1
                part_correct += 1
            elif pre == 1 and (member_id in current_event['no']):
                correct += 1
                part_correct += 1
            elif pre == 2 and (member_id in current_event['maybe']):
                correct += 1
                part_correct += 1
            predicted += 1
            part_predicted += 1
            if predicted % 100 == 0:
                print(
                    f"predicted {predicted} times in {current_event_idx + 1} events, {correct} correct, correct rate is {correct / predicted}")
        # 每update_frequency个事件更新一下原始数据
        if (current_event_idx + 1) % update_frequency == 0:
            for idx in range(current_event_idx - update_frequency, current_event_idx):
                event = event_list[idx]
                for member in event['yes']:
                    member_list[member]['yes'].append(event)
                for member in event['no']:
                    member_list[member]['no'].append(event)
                for member in event['maybe']:
                    member_list[member]['maybe'].append(event)
            correct_rate_list.append(part_correct / part_predicted)
            part_correct = 0
            part_predicted = 0
            plt.plot(np.array(range(0, len(correct_rate_list))) * 100 + start_point, correct_rate_list)
            plt.show()

