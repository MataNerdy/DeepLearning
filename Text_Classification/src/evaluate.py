import torch


@torch.no_grad()
def evaluate(model, dataloader, device):

    model.eval()

    predictions = []
    targets = []

    for batch in dataloader:

        x = batch['input_ids'].to(device)
        y = batch['label'].to(device)

        logits = model(x)

        predictions.append(
            logits.argmax(dim=1)
        )

        targets.append(y)

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    accuracy = (
        predictions == targets
    ).float().mean().item()

    return accuracy