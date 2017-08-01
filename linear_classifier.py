from mygrad.nnet.layers import dense
from mygrad.nnet.activations import softmax, relu

from mygrad import Tensor


import nltk

import numpy as np

from collections import Counter

import glob

import io
with io.open('imdb.vocab','r',encoding='utf8') as f:
    text = f.read()

    corpus = text.encode('ascii', 'ignore')
    bag_of_words = str(corpus)


bag_of_words = bag_of_words.split('\\n')[100:]


def get_descriptor(text):
    word_counts = Counter()
    base_desciptor = np.zeros(len(bag_of_words))
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        if token in bag_of_words:
            word_counts[tokens.index(token)] += 1


    for k in word_counts:
        base_desciptor[k] = word_counts[k]
    return base_desciptor

def load_text_files(dirt):
    file_paths = glob.glob(dirt +'*.txt')
    documents = list()
    for file_path in file_paths:
        with io.open('train/unsup/0_0.txt','r',encoding='utf8') as f:
            unicode_data = f.read()

            document = str(unicode_data.encode('ascii', 'ignore'))
            documents.append(document)
    return documents


print(len(load_text_files("train/unsup/")))



