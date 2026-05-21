import torch.nn as nn


class GRUClassifier(nn.Module):

    def __init__(
        self,
        hidden_dim,
        vocab_size,
        num_classes=4,
        dropout=0.1,
        aggregation_type='max'
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.rnn = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_dim, hidden_dim)

        self.projection = nn.Linear(
            hidden_dim,
            num_classes
        )

        self.dropout = nn.Dropout(dropout)

        self.non_lin = nn.Tanh()

        self.aggregation_type = aggregation_type

    def forward(self, x):

        emb = self.embedding(x)

        output, _ = self.rnn(emb)

        if self.aggregation_type == 'max':
            output = output.max(dim=1)[0]

        else:
            output = output.mean(dim=1)

        output = self.dropout(
            self.linear(self.non_lin(output))
        )

        logits = self.projection(
            self.non_lin(output)
        )

        return logits