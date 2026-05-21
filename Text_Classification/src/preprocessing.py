import string
from nltk.tokenize import word_tokenize


def preprocess_text(text: str):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(text)