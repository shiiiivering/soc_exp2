import os
import numpy as np
import random
import json

def new_group():
    return {
        "id": 0,
        "topics": list(),
        "leader": 0
    }


def new_member():
    return {
        "id": 0,
        "topics": list()
    }

def new_event():
    return {
        "id": 0,
        "group": 0,
        'limit': 0,
        "timestamp": 0,
        "organizers": list(),
        "yes": list(),
        "no": list(),
        "maybe": list()
    }

def new_topic(id = 0):
    return {
        'id': id,
        'groups': list(),
        'members': list(),
    }

def read_group_topic(gt_path):
    # gt_path: path of file GroupTopic.txt
    group_list = list()
    with open (gt_path, 'r') as f:
        while 1:
            line = f.readline().strip('\n')
            if not line:
                break
            group = new_group()
            group_info = line.split(' ')
            group['id'] = int(group_info[0][1:])
            if group_info[1] == "null":
                group['leader'] = -1
            else:
                group['leader'] = int(group_info[1][1:])
            line = f.readline().strip('\n')
            topics = line.split(' ')[0:-1]
            group['topics'] = [int(topic[1:]) for topic in topics]
            group_list.append(group)
    return group_list


def read_member_topic(mt_path):
    # mt_path: path of file MemberTopic.txt
    member_list = list()
    with open(mt_path, 'r') as f:
        while 1:
            line = f.readline().strip('\n')
            if not line:
                break
            member = new_member()
            member["id"] = int(line[1:])
            line = f.readline().strip('\n')
            topics = line.split(' ')[0:-1]
            member['topics'] = [int(topic[1:]) for topic in topics]
            member_list.append(member)
    return member_list

def read_group_event(path):
    event_list = list()
    group_id = 0
    while 1:
        ge_path = path + 'GroupEvent\\G' + str(group_id) + '.txt'
        if not os.path.exists(ge_path):
            break
        with open(ge_path, 'r') as f:
            while 1:
                line = f.readline().strip('\n')
                if not line:
                    break
                event = new_event()
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
        group_id = group_id + 1
    return event_list


def read_data(path, rebuild=False):
    # read the raw data from datafiles
    # Group Topic
    gt_path = path + 'GroupTopic.txt'
    gt_json_path = path + "group_list.json"
    # if group_list.json
    if os.path.exists(gt_json_path) and (not rebuild):
        with open (gt_json_path) as f:
            group_list = json.load(f)
    # read from source file and save file.
    else:
        group_list = read_group_topic(gt_path)
        with open (gt_json_path, 'w') as f:
            json.dump(group_list, f)
    print("load group info complete")

    # MemberTopic
    mt_path = path + 'MemberTopic.txt'
    mt_json_path = path + 'member_topic.json'
    # if member_topic.json
    if os.path.exists(mt_json_path) and (not rebuild):
        with open(mt_json_path) as f:
            member_list = json.load(f)
    # read from source file and save json file
    else:
        member_list = read_member_topic(mt_path)
        with open(mt_json_path, 'w') as f:
            json.dump(member_list, f)
    print("load member info complete")

    # Group Event
    ge_json_path = path + 'event_list.json'
    # if event_list.json
    if os.path.exists(ge_json_path) and (not rebuild):
        with open (ge_json_path) as f:
            event_list = json.load(f)
    # read from source file and save file
    else:
        event_list = read_group_event(path)
        with open(ge_json_path, 'w') as f:
            json.dump(event_list, f)
    print("load event info complete")

    # topics
    topic_json_path = path + "topic_dict.json"
    topic_dict = dict()
    if os.path.exists(topic_json_path) and (not rebuild):
        with open(topic_json_path) as f:
            topic_dict = json.load(f)
    else:
        for member in member_list:
            for topic in member['topics']:
                if not (topic in topic_dict.keys()):
                    topic_dict[topic] = new_topic(topic)
                if not(member['id'] in topic_dict[topic]["members"]):
                    topic_dict[topic]["members"].append(member['id'])
        for group in group_list:
            for topic in group['topics']:
                if not (topic in topic_dict.keys()):
                    topic_dict[topic] = new_topic(topic)
                if not (group['id'] in topic_dict[topic]['groups']):
                    topic_dict[topic]["groups"].append(group['id'])
        with open(topic_json_path, 'w') as f:
            json.dump(topic_dict, f)
    print("load topic info complete")

    return group_list, member_list, event_list, topic_dict


def member_event_extend(member_list, event_list):
    event_list = sorted(event_list, key=lambda x : x['id'])
    for member in member_list:
        member['yes'] = list()
        member['no'] = list()
        member['maybe'] = list()
        member['organizer'] = list()
    for event in event_list:
        for user_id in event['yes']:
            member_list[user_id]['yes'].append(event)
        for user_id in event['no']:
            member_list[user_id]['no'].append(event)
        for user_id in event['maybe']:
            member_list[user_id]['maybe'].append(event)
        for user_id in event['organizers']:
            member_list[user_id]['organizer'].append(event)
    return member_list


def check_data(data):
    for id, i in enumerate(data):
        if id != i["id"]:
            print (id)
            return False
    return True

















