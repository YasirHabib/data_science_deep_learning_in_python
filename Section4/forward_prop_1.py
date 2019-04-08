# Section 4, Lecture 22

import numpy as np
import matplotlib.pyplot as plt

Nclass = 500

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

# let's see what it looks like
plt.scatter(X[:,0], X[:,1], c=Target, s=100, alpha=0.5)
plt.show()

# randomly initialize weights
D = 2 # dimensionality of input
M = 3 # hidden layer size
K = 3 # number of classes
w = np.random.randn(D, M)
b = np.random.randn(M)
v = np.random.randn(M, K)
c = np.random.randn(K)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
	
def forward(X, w, b, v, c):
	# use sigmoid nonlinearity in the hidden layer
	z = sigmoid(np.dot(X, w) + b)
	
	# calculate softmax of the next layer
	A = np.dot(z, v) + c
	expA = np.exp(A)
	predictions = expA / expA.sum(axis = 1, keepdims = True)
	return predictions

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
		
predictions = forward(X, w, b, v, c)
predictions = np.argmax(predictions, axis = 1)

# verify we chose the correct axis
assert(len(predictions) == len(Target))

print("Classification rate for randomly chosen weights:", classification_rate(Target, predictions))