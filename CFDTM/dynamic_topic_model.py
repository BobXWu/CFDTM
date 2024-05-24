import os
import torch
import scipy.io
import argparse

from utils.data import file_utils
from utils.data.DatasetHandler import DatasetHandler
from runners.Runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config')
    parser.add_argument('--dataset')
    parser.add_argument('--num_topic', type=int, default=50)
    parser.add_argument('--num_top_word', type=int, default=15)
    parser.add_argument('--test_index', type=int, default=0)
    args = parser.parse_args()
    return args


def export_beta(beta, vocab, num_top_word=15, time=None):
    topic_str_list = file_utils.print_topic_words(beta, vocab, num_top_word=num_top_word)
    for k, topic_str in enumerate(topic_str_list):
        topic_str_list[k] = f"Time-{time}_K-{k} " + topic_str

    return topic_str_list


def main():
    DATASET_DIR = '../data'

    args = parse_args()

    args = file_utils.update_args(args, f'./configs/model/{args.model_config}.yaml')

    output_prefix = f'output/{args.dataset}/{args.model_config}_K{args.num_topic}_{args.test_index}th'
    file_utils.make_dir(os.path.dirname(output_prefix))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_handler = DatasetHandler(DATASET_DIR, args.dataset, batch_size=args.training.batch_size, device=device)

    args.device = device
    args.vocab_size = dataset_handler.vocab_size
    args.num_times = dataset_handler.num_times
    args.train_size = dataset_handler.train_size
    args.word_embeddings = dataset_handler.word_embeddings
    args.train_time_wordcount = dataset_handler.train_time_wordcount

    runner = Runner(args)

    # beta: TxKxV
    beta = runner.train(dataset_handler)

    topic_str_list = list()
    for time in range(args.num_times):
        topic_str_list.extend(export_beta(beta[time], dataset_handler.vocab, args.num_top_word, time))

    for i, x in enumerate(topic_str_list[-args.num_topic:]):
        print(x)

    file_utils.save_text(topic_str_list, f'{output_prefix}_T{args.num_top_word}')

    rst_dict = {
        'beta': beta,
    }

    rst_dict['train_theta'] = runner.test(dataset_handler.train_dataset)
    rst_dict['test_theta'] = runner.test(dataset_handler.test_dataset)

    scipy.io.savemat(f'{output_prefix}_rst.mat', rst_dict)


if __name__ == '__main__':
    main()
