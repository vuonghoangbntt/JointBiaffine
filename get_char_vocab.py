from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json


def get_char_vocab(input_filenames, output_filename):
    vocab = {
        "UNK": 1, "PAD": 0
    }
    for filename in input_filenames:
        with open(filename,'r',encoding='utf-8') as f:
            for line in f.readlines():
                for sentence in json.loads(line)["sentence"]:
                    for word in sentence:
                        if word.strip()!='' and word not in vocab:
                            vocab[word] = len(vocab)
    with open(output_filename, "w") as f:
        json.dump(vocab, f)
    print("Wrote {} characters to {}".format(len(vocab), output_filename))


if __name__ == "__main__":
    get_char_vocab(["./data/snips/train.json"], "./data/snips/char_index.json")
