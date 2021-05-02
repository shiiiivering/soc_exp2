import data_preprocess
import train_test
import nets
import networkx as nx

def main():
    path = "D:\\learning\\社会计算\\exp2\\data\\"
    group_list, member_list, event_list, topic_dict = data_preprocess.read_data(path)
    assert data_preprocess.check_data(group_list)
    assert data_preprocess.check_data(member_list)
    # assert data_preprocess.check_data(event_list)
    G = nets.create_member_similarity_dinetwork(member_list, topic_dict)
    nx.write_gml(G, path + 's_dinetwork.gml')
    print(0)

if __name__ == "__main__":
    main()