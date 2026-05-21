from datasets import load_dataset
from collections import Counter

from preprocessing import preprocess_text


SPECIAL_TOKENS = ['<unk>', '<bos>', '<eos>', '<pad>']


def build_vocab(dataset, min_freq=25):
    counter = Counter()

    for text in dataset['train']['text']:
        tokens = preprocess_text(text)
        counter.update(tokens)

    vocab = set(SPECIAL_TOKENS)

    for token, freq in counter.items():
        if freq > min_freq:
            vocab.add(token)

    word2ind = {w: i for i, w in enumerate(vocab)}
    ind2word = {i: w for w, i in word2ind.items()}

    return vocab, word2ind, ind2word


def load_data():
    return load_dataset("ag_news")