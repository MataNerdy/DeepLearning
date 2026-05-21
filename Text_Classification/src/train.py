import torch
from torch.utils.data import DataLoader

from data import load_data, build_vocab
from dataset import WordDataset, collate_fn
from models.gru import GRUClassifier
from evaluate import evaluate


device = 'cuda' if torch.cuda.is_available() else 'cpu'


dataset = load_data()

vocab, word2ind, ind2word = build_vocab(dataset)

train_dataset = WordDataset(
    dataset['train'],
    word2ind
)

test_dataset = WordDataset(
    dataset['test'],
    word2ind
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda x: collate_fn(
        x,
        pad_id=word2ind['<pad>']
    )
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=lambda x: collate_fn(
        x,
        pad_id=word2ind['<pad>']
    )
)

model = GRUClassifier(
    hidden_dim=256,
    vocab_size=len(vocab),
    aggregation_type='max'
).to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

num_epochs = 10

for epoch in range(num_epochs):

    model.train()

    losses = []

    for batch in train_loader:

        x = batch['input_ids'].to(device)
        y = batch['label'].to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        losses.append(loss.item())

    acc = evaluate(
        model,
        test_loader,
        device
    )

    print(
        f"Epoch {epoch} | "
        f"loss={sum(losses)/len(losses):.4f} | "
        f"acc={acc:.4f}"
    )