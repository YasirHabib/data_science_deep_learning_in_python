# Section 6, Lecture 42
# Binary classification -> outer layer always sigmoid
# Binary classification -> inner layers can be sigmoid/tanh/relu

# Multi-class classification -> outer layer always softmax
# Multi-class classification -> inner layers can be sigmoid/tanh/relu

# dZ = np.outer(T-Y, W2) * Z * (1 - Z) # this is for sigmoid activation
# dZ = np.outer(T-Y, W2) * (1 - Z * Z) # this is for tanh activation
# dZ = np.outer(T-Y, W2) * (Z > 0) # this is for relu activation

import numpy as np
import matplotlib.pyplot as plt

# for binary classification! no softmax here

def forward(X, W1, b1, W2, b2):
    # sigmoid
    Z = 1 / (1 + np.exp( -(X.dot(W1) + b1) ))

    # tanh
    # Z = np.tanh(X.dot(W1) + b1)

    # relu
    # Z = X.dot(W1) + b1
    # Z = Z * (Z > 0)

    activation = Z.dot(W2) + b2
    Y = 1 / (1 + np.exp(-activation))
    return Y, Z
	
def predict(P_Y_given_X):
	return np.round(P_Y_given_X)
	
def cost(T, Y):
    return -np.sum(T*np.log(Y) + (1-T)*np.log(1-Y))

def test_xor():
	X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	Y = np.array([0, 1, 1, 0])
	W1 = np.random.randn(2, 5)
	b1 = np.zeros(5)
	W2 = np.random.randn(5)
	b2 = 0
	
	learning_rate = 1e-2
	epochs = 30000
	costs = []
	regularization = 0.
	
	for t in range(epochs):
		pY, Z = forward(X, W1, b1, W2, b2)
		c = cost(Y, pY)
		costs.append(c)
		
		W2 -= learning_rate * (Z.T.dot(pY - Y) + regularization * W2)
		b2 -= learning_rate * ((pY - Y).sum() + regularization * b2)
		dZ = np.outer(pY - Y, W2) * Z * (1 - Z) 
		W1 -= learning_rate * (X.T.dot(dZ) + regularization * W1)
		b1 -= learning_rate * (dZ.sum(axis = 0) + regularization * b1)

		if t % 1000 == 0:
			print(t, c)

	print("final classification rate:", np.mean(predict(pY) == Y))
	plt.plot(costs)
	plt.show()


def test_donut():
	N = 1000
	D = 2

	R_inner = 5
	R_outer = 10

	# distance from origin is radius + random normal
	# angle theta is uniformly distributed between (0, 2pi)
	R1 = np.random.randn(N//2) + R_inner
	theta = 2*np.pi * np.random.random(N//2)
	x = R1 * np.cos(theta)
	y = R1 * np.sin(theta)
	X_inner = np.column_stack([x, y])                         # We can alternatively also use X_inner = np.concatenate([[x], [y]]).T

	R2 = np.random.randn(N//2) + R_outer
	theta = 2*np.pi * np.random.random(N//2)
	x = R2 * np.cos(theta)
	y = R2 * np.sin(theta)
	X_outer = np.column_stack([x, y])

	X = np.vstack([X_inner, X_outer])    					  # We can alternatively also use X = np.concatenate([X_inner, X_outer])
	Y = np.array([0]*500 + [1]*500)
	
	n_hidden = 8
	W1 = np.random.randn(2, n_hidden)
	b1 = np.zeros(n_hidden)
	W2 = np.random.randn(n_hidden)
	b2 = np.random.randn(1)
	
	learning_rate = 5e-5
	epochs = 30000
	costs = []
	regularization = 0.2
	
	for t in range(epochs):
		pY, Z = forward(X, W1, b1, W2, b2)
		c = cost(Y, pY)
		costs.append(c)
		
		W2 -= learning_rate * (Z.T.dot(pY - Y) + regularization * W2)
		b2 -= learning_rate * ((pY - Y).sum() + regularization * b2)
		dZ = np.outer(pY - Y, W2) * Z * (1 - Z) 
		W1 -= learning_rate * (X.T.dot(dZ) + regularization * W1)
		b1 -= learning_rate * (dZ.sum(axis = 0) + regularization * b1)

		if t % 300 == 0:
			print(t, c)
			
	print("final classification rate:", np.mean(predict(pY) == Y))
	plt.plot(costs)
	plt.show()
			
if __name__ == '__main__':
    test_xor()
    # test_donut()