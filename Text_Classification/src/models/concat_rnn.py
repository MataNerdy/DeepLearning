import torch
import torch.nn as nn


class RNNConcatClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim,
        vocab_size,
        num_classes=4,
        dropout=0.1,
        aggregation_type="max",
        pad_idx=0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        self.aggregation_type = aggregation_type

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)

        self.linear = nn.Linear(2 * hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, num_classes)

        self.non_lin = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, _ = x.shape

        emb = self.embedding(x)
        rnn_out, _ = self.rnn(emb)

        mask = (x != self.pad_idx).unsqueeze(-1)
        lengths = mask.squeeze(-1).sum(dim=1)

        if self.aggregation_type == "max":
            masked = rnn_out.masked_fill(~mask, float("-inf"))
            pooled = torch.amax(masked, dim=1)
            pooled[torch.isinf(pooled)] = 0.0

        elif self.aggregation_type == "mean":
            masked = rnn_out * mask
            pooled = masked.sum(dim=1) / lengths.clamp_min(1).unsqueeze(-1)

        else:
            raise ValueError("aggregation_type must be 'max' or 'mean'")

        last_idx = (lengths - 1).clamp_min(0)
        batch_idx = torch.arange(batch_size, device=x.device)
        last = rnn_out[batch_idx, last_idx, :]

        features = torch.cat([pooled, last], dim=-1)

        h = self.dropout(self.non_lin(self.linear(features)))
        return self.projection(self.non_lin(h))