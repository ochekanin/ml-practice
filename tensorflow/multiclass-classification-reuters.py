#!/usr/bin/python3

import tensorflow
from tensorflow import keras
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print("\nTrain data length: %d" % len(train_data))
print("\nTest data length: %d" % len(test_data))

word_index = reuters.get_word_index()
reverse_word_index = dict([(v, k) for k, v in word_index.items()])


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros(len(sequences), dimension)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)