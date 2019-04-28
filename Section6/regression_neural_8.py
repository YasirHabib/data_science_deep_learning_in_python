# Section 6, Lecture 43

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# generate and plot the data
N = 500
X = np.random.random((N, 2))*4 - 2 # in between (-2, +2)
Y = X[:,0]*X[:,1] # makes a saddle shape

# make a neural network and train it
D = len(X[0]) # dimensionality of input
M = 100 # hidden layer size or number of hidden units

# randomly initialize weights
W1 = np.random.randn(D, M) / np.sqrt(D)
b1 = np.zeros(M)
W2 = np.random.randn(M) / np.sqrt(M)
b2 = 0

def forward(X, W1, b1, W2, b2):
	Z = X.dot(W1) + b1
	Z = Z * (Z > 0) # relu
	
	A = Z.dot(W2) + b2
	
	return A, Z
	
def cost(T, pY):
	return np.mean((T - pY) ** 2)
	#return ((T - pY)**2).mean()

learning_rate = 1e-4
epochs = 200
costs = []

for t in range(epochs):
	pY, Z = forward(X, W1, b1, W2, b2)
	c = cost(Y, pY)
	costs.append(c)
	
	W2 -= learning_rate * Z.T.dot(pY - Y)
	b2 -= learning_rate * (pY - Y).sum()
	dZ = np.outer(pY - Y, W2) * (Z > 0)  # relu
	W1 -= learning_rate * X.T.dot(dZ)
	b1 -= learning_rate * dZ.sum(axis = 0)
	
	if t % 25 == 0:
		print(t, c)

plt.plot(costs)
plt.show()