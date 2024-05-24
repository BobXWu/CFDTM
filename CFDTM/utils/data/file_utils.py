import os
import argparse
import yaml
import numpy as np
from collections import defaultdict


def print_topic_words(beta, vocab, num_top_word=15):
    topic_str_list = []
    for i, topic_dist in enumerate(beta):
        topic_words = np.asarray(vocab)[np.argsort(topic_dist)][:-(num_top_word + 1):-1]
        topic_str = ' '.join(topic_words)
        topic_str_list.append(topic_str)

    return topic_str_list


def read_yaml(path):
    with open(path) as file:
        config = yaml.safe_load(file)
    return config


def update_args(args, path, key=None):
    config = read_yaml(path)
    if config:
        args = vars(args)
        if key:
            args[key] = config
        else:
            args.update(config)

        args = restructure_as_namespace(args)
    return args


def restructure_as_namespace(args):
    if not isinstance(args, dict):
        return args

    for key in args:
        args[key] = restructure_as_namespace(args[key])

    args = argparse.Namespace(**args)

    return args


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def read_text(path):
    texts = list()
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            texts.append(line.strip())
    return texts


def save_text(texts, path):
    with open(path, 'w', encoding='utf-8') as file:
        for text in texts:
            file.write(text.strip() + '\n')


def split_text_word(texts):
    texts = [text.split() for text in texts]
    return texts


def read_topic_words(path):
    topic_str_list = read_text(path)
    time_topic_dict = convert_topicStr_to_dict(topic_str_list)

    return time_topic_dict


def convert_topicStr_to_dict(topic_str_list):
    time_topic_dict = defaultdict(list)
    # topic_str:  Time-0_K-0 w1 w2 w3 ...
    for topic_str in topic_str_list:
        item_info = topic_str.split()[0]
        time, k = (int(item.split('-')[1]) for item in item_info.split('_'))

        time_topic_dict[time].append(' '.join(topic_str.split()[1:]))

    return time_topic_dict
