# Andreas Goulas <goulasand@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class BiLstm(nn.Module):
    def __init__(self, embed_dim, hidden_dim, fc_dim, dropout,
            lstm_layers, num_labels, vectors):
        super(BiLstm, self).__init__()
        self.encode = nn.Embedding.from_pretrained(vectors, freeze=True)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=lstm_layers,
            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sents, seq_lens):
        sorted_lens, idxs = torch.sort(seq_lens, descending=True)
        _, back_idxs = torch.sort(idxs)
        x_embed = self.encode(sents[idxs])

        packed = pack_padded_sequence(x_embed, sorted_lens.cpu(), batch_first=True)
        _, (out, _) = self.rnn(packed)
        out = out[:, back_idxs, :]

        x = out[-2:].permute(1, 0, 2).contiguous().view(sents.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return x
