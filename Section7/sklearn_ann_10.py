# Section 7, Lecture 54
# Compare it with ann_train_6.py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

from process_2 import get_data

X, Y = get_data()
X, Y = shuffle(X, Y)

# split train and test
X_train = X[:-100]
Y_train = Y[:-100]
X_test = X[-100:]
Y_test = Y[-100:]

# create the neural network with two hidden layers each of size 20. The max no of iterations of backpropagation are 2000
model = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=2000)

# train the neural network
model.fit(X_train, Y_train)

# print the train and test accuracy
train_score = model.score(X_train, Y_train)
test_score = model.score(X_test, Y_test)

print("train score:", train_score, "test score:", test_score)