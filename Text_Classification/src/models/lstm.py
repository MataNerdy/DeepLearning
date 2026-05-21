import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, hidden_dim, vocab_size, num_classes=4, dropout=0.1, aggregation_type="max"):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, num_classes)

        self.non_lin = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.aggregation_type = aggregation_type

    def forward(self, x):
        emb = self.embedding(x)
        output, _ = self.rnn(emb)

        if self.aggregation_type == "max":
            output = output.max(dim=1)[0]
        elif self.aggregation_type == "mean":
            output = output.mean(dim=1)
        else:
            raise ValueError("aggregation_type must be 'max' or 'mean'")

        output = self.dropout(self.linear(self.non_lin(output)))
        return self.projection(self.non_lin(output))