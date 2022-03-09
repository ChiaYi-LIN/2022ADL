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

        self.x = torch.from_numpy(np.array(
            self.vocab.encode_batch(x.iloc[:, 0].apply(self.trim_string).values, self.max_len)
            ))
        
        # Onehot is wrong
        # self.y = torch.from_numpy(np.array(
        #     y.iloc[:, 0].apply(self.label2idx).apply(self.idx2vec).to_list()
        #     ))
        if y is None:
            self.y = y
        else:
            self.y = torch.from_numpy(y.iloc[:, 0].apply(self.label2idx).values)
    

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    # def collate_fn(self, samples):
    #     # TODO: implement collate_fn
    #     # raise NotImplementedError
    #     data = torch.from_numpy(samples[0])
    #     label = torch.from_numpy(samples[1])
    #     return data, label

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