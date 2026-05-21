import torch
from torch.utils.data import Dataset

from preprocessing import preprocess_text


class WordDataset(Dataset):

    def __init__(self, dataset_split, word2ind):
        self.data = dataset_split
        self.word2ind = word2ind

        self.unk_id = word2ind['<unk>']
        self.bos_id = word2ind['<bos>']
        self.eos_id = word2ind['<eos>']
        self.pad_id = word2ind['<pad>']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        tokens = preprocess_text(self.data[idx]['text'])

        ids = [self.bos_id]

        ids += [
            self.word2ind.get(token, self.unk_id)
            for token in tokens
        ]

        ids += [self.eos_id]

        return {
            "text": ids,
            "label": self.data[idx]['label']
        }


def collate_fn(batch, pad_id, max_len=256):

    lengths = [len(x['text']) for x in batch]
    max_seq_len = min(max(lengths), max_len)

    padded = []

    for sample in batch:
        seq = sample['text'][:max_seq_len]

        seq += [pad_id] * (max_seq_len - len(seq))

        padded.append(seq)

    return {
        "input_ids": torch.LongTensor(padded),
        "label": torch.LongTensor(
            [x['label'] for x in batch]
        )
    }