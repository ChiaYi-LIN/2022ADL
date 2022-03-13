from typing import List, Dict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame = None,
        vocab: Vocab = None,
        label_mapping: Dict[str, int] = None,
        max_len: int = 128
        ):
        
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.num_class = len(label_mapping)
        self.max_len = max_len

        self.x = x.iloc[:, 0].apply(self.trim_string).apply(vocab.encode).values
        # print(x.values[0])
        # print(self.x[0])
        
        # Onehot is wrong
        # self.y = torch.from_numpy(np.array(
        #     y.iloc[:, 0].apply(self.label2idx).apply(self.idx2vec).to_list()
        #     ))
        if y is None:
            self.y = y
        else:
            self.y = y.iloc[:, 0].apply(self.label2idx).values

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index):
        if self.y is None:
            return torch.LongTensor(self.x[index])
        else:
            return self.x[index], self.y[index]

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, batch):
    # TODO: implement collate_fn
        seq_vec, seq_label, lengths = zip(*[
            (torch.LongTensor(vec), label, len(vec))
            for (vec, label) in sorted(batch, key=lambda x: len(x[0]), reverse=True)
        ])

        padded_seq_vec = pad_sequence(seq_vec, batch_first=True, padding_value=0)
        # print(padded_seq_vec.shape)
        # print(torch.LongTensor(seq_label).shape)

        return padded_seq_vec, torch.LongTensor(seq_label), torch.LongTensor(lengths)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

    def trim_string(self, x):
        x = x.lower().split(maxsplit=self.max_len)
        # x = ' '.join(x[:self.max_len])
        return x

    def idx2vec(self, x):
        v = [0 for _ in range(self.num_class)]
        v[x] = 1
        return v

class SeqIOBDataset(Dataset):
    def __init__(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame = None,
        vocab: Vocab = None,
        label_mapping: Dict[str, int] = None,
        max_len: int = 128
        ):
        
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: tag for tag, idx in self.label_mapping.items()}
        self.num_class = len(label_mapping)
        self.max_len = max_len

        self.x = x.iloc[:, 0].apply(vocab.encode).values
        # print(x.values[0])
        # print(self.x[0])
        
        if y is None:
            self.y = y
        else:
            self.y = y.iloc[:, 0].apply(self.labels2idxs).values 
        # print(y.values[0])
        # print(self.y[0])

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index):
        if self.y is None:
            return torch.LongTensor(self.x[index])
        else:
            return self.x[index], self.y[index]

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, batch):
    # TODO: implement collate_fn
        seq_vec, seq_label, lengths = zip(*[
            (torch.LongTensor(vec), torch.LongTensor(labels), len(vec))
            for (vec, labels) in sorted(batch, key=lambda x: len(x[0]), reverse=True)
        ])

        padded_seq_vec = pad_sequence(seq_vec, batch_first=True, padding_value=0)
        # print('padded_seq_vec', padded_seq_vec.shape)
        
        padded_sequ_label = pad_sequence(seq_label, batch_first=True, padding_value=-100)
        # print('padded_sequ_label', padded_sequ_label.shape)

        return padded_seq_vec, padded_sequ_label, torch.LongTensor(lengths)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def labels2idxs(self, labels):
        return [self.label_mapping[label] for label in labels]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
