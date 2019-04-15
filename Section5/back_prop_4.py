# Section 5, Lecture 34

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
	
def forward(X, w, b, v, c):
	# use sigmoid nonlinearity in the hidden layer
	z = sigmoid(np.dot(X, w) + b)
	
	# calculate softmax of the next layer
	A = np.dot(z, v) + c
	expA = np.exp(A)
	Y = expA / expA.sum(axis = 1, keepdims = True)
	return Y, z

# determine the classification rate
# num correct / num total
def classification_rate(Target, predictions):
	n_correct = 0
	n_total = 0
	for x in range(len(Target)):
		n_total += 1
		if Target[x] == predictions[x]:
			n_correct += 1
			
	return (n_correct / n_total)

def derivative_w2(z, T, Y):
	zT = np.transpose(z)
	return np.dot(zT, T - Y)
	
def derivative_b2(T, Y):
	return (T - Y).sum(axis=0)

def derivative_w1(X, z, T, Y, w2):
	w2T = np.transpose(w2)
	xT = np.transpose(X)
	dZ = np.dot(T - Y, w2T) * z * (1 - z)
	ret2 = np.dot(xT, dZ)
	return ret2

def derivative_b1(T, Y, w2, z):
	w2T = np.transpose(w2)
	return (np.dot(T - Y, w2T) * z * (1 - z)).sum(axis=0)

def cost(T, Y):
	return (T*np.log(Y)).sum()
	#return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()
	
def main():
	Nclass = 500
	D = 2 # dimensionality of input
	M = 3 # hidden layer size
	K = 3 # number of classes
	
	# Gaussian cloud centered at (0, -2)
	X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
	# Can also be done as follows
	# X1[:, 0] = X1[:, 0] + 0
	# X1[:, 1] = X1[:, 1] - 2

	# Gaussian cloud centered at (2, 2)
	X2 = np.random.randn(Nclass, 2) + np.array([2, 2])

	# Gaussian cloud centered at (-2, 2)
	X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
	
	X = np.vstack([X1, X2, X3])
	
	Target = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
	N = len(Target)
	
	# turn Target into an indicator matrix for training
	T = np.zeros((N, K))
	for x in range(N):
		T[x, Target[x]] = 1
	
	# let's see what it looks like
	plt.scatter(X[:,0], X[:,1], c=Target, s=100, alpha=0.5)
	plt.show()
	
	# randomly initialize weights
	w1 = np.random.randn(D, M)
	b1 = np.random.randn(M)
	w2 = np.random.randn(M, K)
	b2 = np.random.randn(K)
	
	learning_rate = 1e-3
	epochs = 1000
	costs = []
	
	for t in range(epochs):
		# forward propagation and cost calculation
		output, hidden = forward(X, w1, b1, w2, b2)
		
		if t % 100 == 0:
			c = cost(T, output)
			P = np.argmax(output, axis = 1)
			r = classification_rate(Target, P)
			print("cost:", c, "classification_rate:", r)
			costs.append(c)

		# this is gradient ASCENT, not DESCENT
		w2 += learning_rate * derivative_w2(hidden, T, output)
		b2 += learning_rate * derivative_b2(T, output)
		w1 += learning_rate * derivative_w1(X, hidden, T, output, w2)
		b1 += learning_rate * derivative_b1(T, output, w2, hidden)
	
	plt.plot(costs)
	plt.show()

if __name__ == '__main__':
    main()