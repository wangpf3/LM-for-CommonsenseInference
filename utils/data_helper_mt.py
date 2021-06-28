import os
import pickle
import json
import math
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset

from .data_processor import DataProcessor

class DataHelper(object):
    """docstring for DataHelper"""
    def __init__(self, task_list, rel2token, tokenizer, max_seq_length, sample_ratio=0, seed=0, seen_tails=None):

        self.trainset = {}
        self.devset = {}

        for task in task_list:
            cache_features_path = os.path.join('./data/', task, 'features_{}_len{}_sample{}_seed{}.pkl'.format(tokenizer.name_or_path.replace('/', '-'), max_seq_length, sample_ratio, seed))
            data_dir = os.path.join('./data/', task)
            if not os.path.exists(cache_features_path):
                processor = DataProcessor(data_dir, rel2token[task], tokenizer, max_seq_length, sample_ratio, seen_tails)
                trainset = processor.get_tensors('train')
                devset = processor.get_tensors('dev')
                # testset = processor.get_tensors('test')

                with open(cache_features_path, 'wb') as handle:
                    pickle.dump([trainset, devset], handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(cache_features_path, 'rb') as handle:
                    trainset, devset = pickle.load(handle)

            self.trainset[task] = TensorDataset(trainset['input'], trainset['label'])
            self.devset[task] = TensorDataset(devset['input'], devset['label'])

class DataHelper_Test(object):
    """docstring for DataHelper"""
    def __init__(self, task_list, rel2token, tokenizer, max_seq_length, seen_tails=None, include_seen=False):

        self.testset = {}
        for task in task_list:
            data_dir = os.path.join('./data/', task)
            processor = DataProcessor(data_dir, rel2token[task], tokenizer, max_seq_length, 0, seen_tails, include_seen)
            testset = processor.get_tensors('test')
            self.testset[task] = TensorDataset(testset['input'], testset['label'])

class DataLoader(object):
    """docstring for SampleFromKB"""
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset
        self.data_size = len(self.dataset)
        self.randperm_index = torch.randperm(self.data_size)
        self.start_index = 0

    def reset(self,):
        self.randperm_index = torch.randperm(self.data_size)
        self.start_index = 0

    def get_batch(self, batch_size, device):

        batch = self.dataset[self.randperm_index[self.start_index:(self.start_index+batch_size)]]
        self.start_index += batch_size

        if self.start_index >= self.data_size:
            self.randperm_index = torch.randperm(self.data_size)
            self.start_index = 0

        return [feature.to(device) for feature in batch] 

    def sequential_iterate(self, batch_size, device):
        batch_num = math.ceil(self.data_size / batch_size)
                 
        for batch_id in range(batch_num):
            start_index = batch_id * batch_size
            end_index = min((batch_id+1) * batch_size, self.data_size)
            batch = self.dataset[start_index:end_index]
            yield [feature.to(device) for feature in batch]

    def random_iterate(self, batch_size, device):
        batch_num = math.ceil(self.data_size / batch_size)
        randperm_index = torch.randperm(self.data_size)

        for batch_id in range(batch_num):
            start_index = batch_id * batch_size
            end_index = min((batch_id+1) * batch_size, self.data_size)
            batch = self.dataset[self.randperm_index[start_index:end_index]]
            yield [feature.to(device) for feature in batch]
