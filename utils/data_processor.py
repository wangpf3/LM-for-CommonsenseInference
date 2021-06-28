import json
import os 
import tqdm
import torch

from collections import defaultdict
import random

class DataProcessor:

    def __init__(self, data_dir, rel2token, tokenizer, max_length, sample_ratio=0, seen_tails=None, include_seen=False):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rel2token = rel2token
        # self.num_sample_per_class = num_sample_per_class
        self.sample_ratio =sample_ratio
        self.seen_tails = seen_tails
        self.include_seen = include_seen

    def get_tensors(self, split):
        print("LOOKING AT {} {}".format(self.data_dir, split))
        input_path = os.path.join(self.data_dir, '{}.txt'.format(split))
        if split == 'test':
            return self._create_tensors_test(self._read_txt(input_path))
        else:
            return self._create_tensors(self._read_txt(input_path))

    def _read_txt(self, input_file):
        with open(input_file, "r") as fr:
            lines = fr.readlines()
            random.shuffle(lines)
            return lines

    def _create_tensors(self, lines):

        input_tensor = []
        label_tensor = []
        # class_count = defaultdict(set)
        num_seen = 0
        for line in lines:
            data_raw = line.strip().split('\t')
            if not data_raw[1] in self.rel2token:
                continue
            if self.seen_tails is not None:
                if data_raw[2] in self.seen_tails:
                    num_seen += 1
        count_seen = 0
        sample_size = int(num_seen * self.sample_ratio)

        for line in tqdm.tqdm(lines, desc='read data'):
            data_raw = line.strip().split('\t')

            if not data_raw[1] in self.rel2token:
                continue

            if self.seen_tails is not None and data_raw[2] in self.seen_tails: 
                if count_seen == sample_size:
                    continue 
                else:
                    count_seen += 1

            if len(data_raw) == 4:
                context_id = self.tokenizer.encode('{} {}{}'.format(data_raw[3], data_raw[0], self.rel2token[data_raw[1]]))
            else:
                context_id = self.tokenizer.encode('{}{}'.format(data_raw[0], self.rel2token[data_raw[1]]))
            ending_id = self.tokenizer.encode(' ' + data_raw[2])
            input_id = (context_id + ending_id)[:self.max_length] + [self.tokenizer.eos_token_id]
            input_id += [self.tokenizer.eos_token_id] * (self.max_length + 1 - len(input_id))
            label = ([-100] * len(context_id) + ending_id)[:self.max_length] + [self.tokenizer.eos_token_id]
            label += [-100] * (self.max_length + 1 - len(label))

            input_tensor.append(input_id)
            label_tensor.append(label)

        print("len examples:", str(len(input_tensor)))
        input_tensor = torch.tensor(input_tensor, dtype=torch.long)
        label_tensor = torch.tensor(label_tensor, dtype=torch.long)

        for f1, f2 in zip(input_tensor[:2], label_tensor[:2]):
            print("*** Example ***")
            print("feature: %s" % f1)
            print("label: %s" % f2)

        return {'input': input_tensor, 'label': label_tensor}

    def _create_tensors_test(self, lines):

        input_tensor = []
        label_tensor = []

        for line in tqdm.tqdm(lines, desc='read data'):
            data_raw = line.strip().split('\t')

            if not data_raw[1] in self.rel2token:
                continue

            tail = data_raw[2]
            if self.seen_tails is not None:
                if tail in self.seen_tails and (not self.include_seen):
                    continue
                if (not tail in self.seen_tails) and self.include_seen:
                    continue

            if len(data_raw) == 4:
                context_id = self.tokenizer.encode('{} {}{}'.format(data_raw[3], data_raw[0], self.rel2token[data_raw[1]]))
            else:
                context_id = self.tokenizer.encode('{}{}'.format(data_raw[0], self.rel2token[data_raw[1]]))
            ending_id = self.tokenizer.encode(' ' + data_raw[2])
            input_id = (context_id + ending_id)[:self.max_length] + [self.tokenizer.eos_token_id]
            input_id += [self.tokenizer.eos_token_id] * (self.max_length + 1 - len(input_id))
            label = ([-100] * len(context_id) + ending_id)[:self.max_length] + [self.tokenizer.eos_token_id]
            label += [-100] * (self.max_length + 1 - len(label))

            input_tensor.append(input_id)
            label_tensor.append(label)

        print("len examples:", str(len(input_tensor)))
        input_tensor = torch.tensor(input_tensor, dtype=torch.long)
        label_tensor = torch.tensor(label_tensor, dtype=torch.long)

        for f1, f2 in zip(input_tensor[:2], label_tensor[:2]):
            print("*** Example ***")
            print("feature: %s" % f1)
            print("label: %s" % f2)

        return {'input': input_tensor, 'label': label_tensor}
