import data_preprocess
import train_test
import nets
import networkx as nx

def main():
    path = "D:\\learning\\社会计算\\exp2\\data"
    group_list, member_list, event_list, topic_dict = data_preprocess.read_data(path)
    assert data_preprocess.check_data(group_list)
    assert data_preprocess.check_data(member_list)

    strange_org = set()
    normal_org = set()
    for event in event_list:
        for o in event['organizers']:
            if len(member_list[o]['topics']) == 0:
                strange_org.add(member_list[o]['id'])
            else:
                normal_org.add(member_list[o]['id'])

    train_test.process_k_fold(member_list, event_list, topic_dict, group_list, 5)
    # G = nets.create_member_similarity_array(member_list)
    # nx.write_gml(G, path + 's_dinetwork.gml')
    print(0)

if __name__ == "__main__":
    main()