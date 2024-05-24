import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from models.CFDTM import CFDTM


class Runner:
    def __init__(self, args):
        self.args = args
        self.model = CFDTM(args)
        self.model = self.model.to(args.device)

    def make_optimizer(self):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.args.training.learning_rate
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def train(self, dataset_handler):
        optimizer = self.make_optimizer()

        for epoch in tqdm(range(1, self.args.training.num_epoch + 1), leave=False):
            self.model.train()
            for batch_data in dataset_handler.train_dataloader:
                batch_bow = batch_data['bow']
                batch_times = batch_data['times']

                rst_dict = self.model(batch_bow, batch_times)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            output_log = f'Epoch: {epoch:03d}'

            print(output_log)

        self.model.eval()
        beta = self.model.get_beta().detach().cpu().numpy()

        return beta

    def test(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.args.training.batch_size, shuffle=False)
        theta_list = list()

        self.model.eval()
        with torch.no_grad():
            for batch_data in dataloader:
                batch_bow = batch_data['bow']
                theta = self.model.get_theta(batch_bow)
                theta_list.extend(theta.detach().cpu().numpy().tolist())

        return np.asarray(theta_list)
