import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset, DataLoader
from utils.data import file_utils


class TimeSeriesDataset(Dataset):
    def __init__(self, bow, times):
        super().__init__()
        self.bow = bow
        self.times = times

    def __len__(self):
        return len(self.bow)

    def __getitem__(self, index):
        return_dict = {
            'bow': self.bow[index],
            'times': self.times[index],
        }

        return return_dict


class DatasetHandler:
    def __init__(self, DATASET_DIR, dataset, batch_size, device):
        dataset_path = f'{DATASET_DIR}/{dataset}'
        self.load_data(dataset_path)

        self.vocab_size = len(self.vocab)
        self.train_size = len(self.train_bow)
        self.num_times = len(np.unique(self.train_times))
        self.train_time_wordcount = self.get_train_time_wordcount(self.train_bow, self.train_times)

        self.train_bow = torch.from_numpy(self.train_bow).float().to(device)
        self.test_bow = torch.from_numpy(self.test_bow).float().to(device)
        self.train_times = torch.from_numpy(self.train_times).long().to(device)
        self.test_times = torch.from_numpy(self.test_times).long().to(device)
        self.train_time_wordcount = torch.from_numpy(self.train_time_wordcount).float().to(device)

        self.train_dataset = TimeSeriesDataset(self.train_bow, self.train_times)
        self.test_dataset = TimeSeriesDataset(self.test_bow, self.test_times)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def load_data(self, path):
        self.train_bow = scipy.sparse.load_npz(f'{path}/train_bow.npz').toarray().astype('float32')
        self.test_bow = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')
        self.word_embeddings = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')

        self.train_times = np.loadtxt(f'{path}/train_times.txt').astype('int32')
        self.test_times = np.loadtxt(f'{path}/test_times.txt').astype('int32')

        self.vocab = file_utils.read_text(f'{path}/vocab.txt')

        self.train_texts = file_utils.read_text(f'{path}/train_texts.txt')
        self.test_texts = file_utils.read_text(f'{path}/test_texts.txt')

    def get_train_time_wordcount(self, bow, times):
        train_time_wordcount = np.zeros((self.num_times, self.vocab_size))
        for time in range(self.num_times):
            idx = np.where(times == time)[0]
            train_time_wordcount[time] += bow[idx].sum(0)
        cnt_times = np.bincount(times)

        train_time_wordcount = train_time_wordcount / cnt_times[:, np.newaxis]
        return train_time_wordcount
