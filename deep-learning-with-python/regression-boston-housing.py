#!/usr/bin/env python

import time

import keras
from keras import models
from keras import layers

import numpy as np

print("Keras version: %s" % keras.__version__)

from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()

print("\nTrain data shape: ", train_data.shape)
print("Test data shape: ", test_data.shape)


print("\nNormalizing data")
mean = train_data.mean(axis=0)
train_data -= mean

std = train_data.std(axis=0)
train_data /= std

def build_model():
    model = models.Sequential()

    model.add(layers.Dense(64, activation='relu',
              input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

print("\nDoing K-fold cross-validation")

k = 4

num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print("Processing fold #", i)

    # Prepare the validation data. Data from partition # k

    val_start = i * num_val_samples
    val_end = (i + 1) * num_val_samples

    val_data = train_data[val_start : val_end]
    val_targets = train_targets[val_start : val_end]

    # Prepare the training data. Data from all other partitions

    train_first_range_end = val_start
    train_second_range_start = val_end

    partial_train_data = np.concatenate(
        [train_data[:train_first_range_end],
        train_data[train_second_range_start:]], axis=0)

    partial_train_targets = np.concatenate(
        [train_targets[:train_first_range_end],
        train_targets[train_second_range_start:]], axis=0)

    
    model = build_model()

    start_time = time.time()

    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=8, verbose=0)

    print("Execution time %s seconds" % str(time.time() - start_time))

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print("\nAll folds processed")
for i in range(k):
    print("Score #{fold}: {mae}".format(fold=i, mae=all_scores[i]))

print("Mean absolure error for %d folds: %f" % (k, np.mean(all_scores)))
