"""Download pretrained StackOverflow Word2Vec vectors from Zenodo."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/record/1199620/files/SO_vectors_200.bin?download=1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/SO_vectors_200.bin")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, output)
    print(f"Saved pretrained vectors to {output}")


if __name__ == "__main__":
    main()
