import argparse
import numpy as np
import scipy.io
from collections import Counter
from topmost.evaluations import dynamic_TD, dynamic_TC

import sys
sys.path.append('./')
from utils.data import file_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--topic_path', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    train_times = np.loadtxt(f'{args.data_dir}/train_times.txt')
    train_texts = file_utils.read_text(f'{args.data_dir}/train_texts.txt')
    train_bow = scipy.sparse.load_npz(f'{args.data_dir}/train_bow.npz').toarray().astype('float32')
    train_times = np.loadtxt(f'{args.data_dir}/train_times.txt').astype('int32')
    vocab = file_utils.read_text(f'{args.data_dir}/vocab.txt')

    time_topic_dict = file_utils.read_topic_words(args.topic_path)

    time_idx = np.sort(np.unique(train_times))

    TC = dynamic_TC(train_texts, train_times, vocab, list(time_topic_dict.values()))
    print(f"===>dynamic_TC: {TC:.5f}")

    TD = dynamic_TD(time_idx, time_topic_dict, train_bow, train_times, vocab)
    print(f"===>dynamic_TD: {TD:.5f}")
