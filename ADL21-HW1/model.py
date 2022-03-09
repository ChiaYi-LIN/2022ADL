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
        
        self.fc = nn.Linear(2 * hidden_size, num_class)
        self.act = nn.Softmax(dim=1)

    # @property
    # def encoder_output_size(self) -> int:
    #     # TODO: calculate the output dimension of rnn
    #     raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # raise NotImplementedError
        embeds = self.embed(batch)
        out, h = self.gru(embeds)

        #concat the final forward and backward hidden state
        hidden = torch.cat((h[-2,:,:], h[-1,:,:]), dim = 1)
        #hidden = [batch size, hid dim * num directions]
        
        dense_outputs = self.fc(hidden)
        
        #Final activation function
        outputs = self.act(dense_outputs)
        
        return outputs
