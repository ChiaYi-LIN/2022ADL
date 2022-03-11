#%%
from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.gru = nn.GRU(input_size=embeddings.shape[1],
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)

        self.lstm = nn.LSTM(input_size=embeddings.shape[1],
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        
        self.fc = nn.Linear(2 * hidden_size, num_class)
        self.act = nn.Softmax(dim=1)

    # @property
    # def encoder_output_size(self) -> int:
    #     # TODO: calculate the output dimension of rnn
    #     raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # raise NotImplementedError
        # print('batch =', batch.shape)

        embeds = self.embed(batch) # embeds = torch.Size([batch size, seq length, embeddings])
        # print('embeds =', embeds.shape)

        # out, h = self.gru(embeds) # out = torch.Size([batch size, seq length, 2 * hid dim]), h = torch.Size([2 * num layers, batch size, hid dim])
        out, (h, c) = self.lstm(embeds)
        # print('out =', out.shape)
        # print('h =', h.shape)

        # concat the final forward and backward hidden state
        hidden = torch.cat((h[-2,:,:], h[-1,:,:]), dim = 1) # hidden = torch.Size([batch size, 2 * hid dim])
        # print('hidden =', hidden.shape)

        dense_outputs = self.fc(hidden) # dense_outputs = torch.Size([batch size, num class])
        # print('dense_outputs =', dense_outputs.shape)

        #Final activation function
        outputs = self.act(dense_outputs) # outputs = torch.Size([batch size, num class])
        # print('outputs =', outputs.shape)

        return outputs

class SeqIOBTagger(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqIOBTagger, self).__init__()
        self.num_class = num_class
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.gru = nn.GRU(input_size=embeddings.shape[1],
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)

        self.lstm = nn.LSTM(input_size=embeddings.shape[1],
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        
        self.fc = nn.Linear(2 * hidden_size, num_class)
        self.act = nn.Softmax(dim=2)

    # @property
    # def encoder_output_size(self) -> int:
    #     # TODO: calculate the output dimension of rnn
    #     raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # raise NotImplementedError
        
        embeds = self.embed(batch) # embeds = torch.Size([batch size, seq length, embeddings])

        # out, h = self.gru(embeds) # out = torch.Size([batch size, seq length, 2 * hid dim]), h = torch.Size([2 * num layers, batch size, hid dim])
        out, (h, c) = self.lstm(embeds)

        tag_space = self.fc(out) # tag_space = torch.Size([batch size, seq length, num_class])
        # print('tag_space = ', tag_space.shape)

        outputs = self.act(tag_space) # outputs = torch.Size([batch size, seq length, num class])
        # print('outputs = ', outputs.shape)
        # print(outputs)
        
        outputs = outputs.transpose(-1, 1) # outputs = torch.Size([batch size, num_class, seq length])
        
        return outputs