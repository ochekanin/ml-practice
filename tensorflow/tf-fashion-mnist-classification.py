#!/usr/bin/env python

import random
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version = %s" % tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Training images shape = ", train_images.shape)
print("Testing images shape = ", test_images.shape)

# Scale images to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
], "My first neural network")

print("\n")
print(">>> Configuring the model")
print(">>> Optimizer = Adam; Loss function = crossentropy; Metric = accuracy")
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\n")
print(">>> Training the model")
model.fit(train_images, train_labels, epochs=8)

print("\n")
print(">>> Running the model with test dataset")
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("\n")
print(">>> Test accuracy = ", test_acc)

print("\n")
print(">>> Making predictions")
predictions = model.predict(test_images, verbose=1)

# Get 10 different numbers in range [0 - test_images]
prediction_numbers = 10
prediction_indices = random.sample(range(0, len(test_labels)), prediction_numbers)

for i in range(prediction_numbers):
    print("\n")
    print(">>> Prediction ", i)
    print(predictions[i])
    print("Predicted label {predicted}, Actual label {actual}",
          np.argmax(predictions[i]), test_labels[i])