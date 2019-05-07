# Section 8, Lecture 59
# We use neural network with a sigmoid on facial expression recognition problem before we move on to softmax.
# This is to compare binary classification using logistic regression which we did in previous course to binary classification using neural network
# which we are going to do below.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import getBinaryData, sigmoid, sigmoid_cost, error_rate, relu

class ANN(object):
	def __init__(self, M):
		self.M = M
	
	# this function trains our model
	def fit (self, X, Y, learning_rate=5*10e-7, reg=1.0, epochs=10000, show_fig=False):
		X, Y = shuffle(X, Y)
		
		# Below we are splitting X & Y into training & validation sets
		Xvalid, Yvalid = X[-1000:], Y[-1000:]	# Retain last 1000 rows (all columns are also retained)
		X, Y = X[:-1000], Y[:-1000]				# Retain all except last 1000 rows (all columns are also retained)
												# Same as X, Y = X[:-1000, :], Y[:-1000]
		
		N = len(X)								# Same as N, D = X.shape
		D = len(X[0])
		
		# randomly initialize w
		self.W1 = np.random.randn(D, self.M) / np.sqrt(D + self.M)
		self.b1 = np.zeros(self.M)
		self.W2 = np.random.randn(self.M) / np.sqrt(self.M)
		self.b2 = 0
		
		cost = []
		best_validation_error = 1
		
		for t in range(epochs):
			# forward propagation
			pY, Z = self.forward(X)

			# gradient descent step
			self.W2 -= learning_rate * (Z.T.dot(pY - Y) + reg*self.W2)
			self.b2 -= learning_rate * ((pY - Y).sum() + reg*self.b2)
			dZ = np.outer(pY - Y, self.W2) * (Z > 0)  # relu
			self.W1 -= learning_rate * (X.T.dot(dZ) + reg*self.W1)
			self.b1 -= learning_rate * (dZ.sum(axis = 0) + reg*self.b1)
			
			if t % 20 == 0:
				pYvalid, _ = self.forward(Xvalid)
				c = sigmoid_cost(Yvalid, pYvalid)
				cost.append(c)
				e = error_rate(Yvalid, np.round(pYvalid))
				print("t:", t, "cost:", c, "error:", e)
				if e < best_validation_error:
					best_validation_error = e
		
		print("best_validation_error:", best_validation_error)
		
		if show_fig:
			plt.plot(cost)
			plt.show()
				
	def forward(self, X):
		Z = X.dot(self.W1) + self.b1
		Z = relu(Z)
		
		A = Z.dot(self.W2) + self.b2
		A = sigmoid(A)
		
		return A, Z
		
	def predict(self, X):
		pY, _ = self.forward(X)
		return np.round(pY)
		
	def score(self, X, Y):
		predictions = self.predict(X)
		return 1 - error_rate(Y, predictions)


def main():
	X, Y = getBinaryData()
	
	X0 = X[Y==0, :]				# This means everytime the value of Y == 0, then select the entire corresponding row in X. Done due to binary classification
	X1 = X[Y==1, :]				# This means everytime the value of Y == 1, then select the entire corresponding row in X. Done due to binary classification
	X1 = np.repeat(X1, 9, axis=0)  # To address label imbalance, we lengthen label 1 by repeating it 9 times
	X = np.vstack([X0, X1])
	Y = np.array([0]*len(X0) + [1]*len(X1))
	
	model = ANN(100)
	model.fit(X, Y, show_fig=True)
	model.score(X, Y)

if __name__ == "__main__":
	main()