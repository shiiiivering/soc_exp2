import config
import data_preprocess
import train_test
import nets
import casc
import networkx as nx
import argparse
import matplotlib.pyplot as plt
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="data", help='dataset path')
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        raise NotADirectoryError(args.path)
    group_list, member_list, event_list, topic_dict = data_preprocess.read_data(args.path)
    assert data_preprocess.check_data(group_list)
    assert data_preprocess.check_data(member_list)

    # 使用团体的信息创建网络，只有有共同团体的成员之间有边，边权为成员topic相似度  --胡煜霄
    # G = nets.create_group_net(member_list, group_list, event_list)
    # nx.write_gml(G, os.path.join(args.path, 'group_network.gml'))

    # # 使用成员间事件的相似度作为成员相似度。创建完整矩阵，需要100gb内存，请在服务器上运行
    # strange_org = set()
    # normal_org = set()
    # for event in event_list:
    #     for o in event['organizers']:
    #         if len(member_list[o]['topics']) == 0:
    #             strange_org.add(member_list[o]['id'])
    #         else:
    #             normal_org.add(member_list[o]['id'])
    # train_test.evolution_over_time(member_list, topic_dict, group_list, event_list, start_point=50000)
    train_test.process_k_fold(member_list, event_list, topic_dict, group_list, 5)
    # G = nets.create_member_similarity_array(member_list)
    # nx.write_gml(G, path + 's_dinetwork.gml')

    # all tests
    # for user_sim in ['choice', 'event']:
    #     for event_sim in ['all']:
    #         for social_constraint in ['event', 'None']:
    #             print(f'similarity: user_sim={user_sim}, event_sim={event_sim}, social_constraint={social_constraint}', flush=True)
    #             train_test.process_k_fold(member_list, event_list, topic_dict, group_list, 5, test='similarity', user_sim=user_sim, event_sim=event_sim, social_constraint=social_constraint)

    # for user_sim in ['event']:
    #     for social_constraint in ['event']:
    #         print(f'cascade: user_sim={user_sim}, social_constraint={social_constraint}', flush=True)
    #         train_test.process_k_fold(member_list, event_list, topic_dict, group_list, 5, test='cascade', user_sim=user_sim, event_sim=None, social_constraint=social_constraint)

    # effects of increasing payoff, finding key people, generating communities and cluster cascading (Waleed)
    # on the basis of Group event data "Yes", "No"
    # G_event = casc.read_single_group_event(config.data_path)
    # G = casc.community_impact(G_event)
    # casc.Calculate_Payoff(G)
    # casc.key_people(G)
    # casc.cascading_on_cluster(G)
    print(0)


if __name__ == "__main__":
    main()
