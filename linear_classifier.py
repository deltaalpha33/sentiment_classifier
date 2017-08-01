from mygrad.nnet.layers import dense
from mygrad.nnet.activations import softmax, relu

from mygrad import Tensor


import nltk

import numpy as np

from collections import defaultdict, Counter

import io
with io.open('imdb.vocab','r',encoding='utf8') as f:
    text = f.read()

    corpus = text.encode('ascii', 'ignore')
    bag_of_words = str(corpus)


bag_of_words = bag_of_words.split('\\n')[100:]


def get_descriptor(text):
    word_counts = Counter()
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        if token in bag_of_words:
            word_counts[token] += 1

    return word_counts


with io.open('train/unsup/0_0.txt','r',encoding='utf8') as f:
    text = f.read()

    corpus = text.encode('ascii', 'ignore')
    test_doc = str(corpus)

print(get_descriptor(test_doc))




