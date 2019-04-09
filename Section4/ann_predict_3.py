# Section 4, Lecture 24

import numpy as np
import pandas as pd

from process_2 import get_data

X, Y = get_data()

# Determine the dimensionality of the input matrix X
D = len(X[0])

M = 5 # hidden layer size
K = len(set(Y)) # number of unique values in Y or number of classes
w = np.random.randn(D, M)
b = np.zeros(M)
v = np.random.randn(M, K)
c = np.zeros(K)

def softmax(a):
	expA = np.exp(a)
	return expA / expA.sum(axis = 1, keepdims = True)
	
def forward(X, w, b, v, c):
	z = np.tanh(np.dot(X, w) + b)
	A = np.dot(z, v) + c
	return softmax(A)

P_Y_given_X = forward(X, w, b, v, c)
predictions = np.argmax(P_Y_given_X, axis = 1)

def classification_rate(Y, P):
	return np.mean(Y == P)					# Alternatively we can also use (np.sum(predictions == y_test) / N) as given in Classification notes
	
print("Score:", classification_rate(Y, predictions))