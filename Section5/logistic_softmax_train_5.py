# Section 5, Lecture 36

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process_2 import get_data

def target2indicator(Target, K):
	N = len(Target)
	T = np.zeros((N,K))
	for x in range(N):
		T[x, Target[x]] = 1
	return T
	
X, Target = get_data()
X, Target = shuffle(X, Target)

D = len(X[0]) # dimensionality of input
K = len(set(Target)) # number of unique values in Y or number of classes

X_train = X[:-100]
Target_train = Target[:-100]
Target_train_ind = target2indicator(Target_train, K)

X_test = X[-100:]
Target_test = Target[-100:]
Target_test_ind = target2indicator(Target_test, K)

w = np.random.randn(D, K)	
b = np.zeros(K)

def softmax(a):
	expA = np.exp(a)
	return expA / expA.sum(axis = 1, keepdims = True)
	
def forward(X, w, b):
	A = np.dot(X, w) + b
	return softmax(A)
	
def predict(P_Y_given_X):
	return np.argmax(P_Y_given_X, axis = 1)
	
def classification_rate(Y, P):
	return np.mean(Y == P)
	
def cost(T, pY):
	return -np.mean(T*np.log(pY))

learning_rate = 1e-3
epochs = 10000
train_costs = []
test_costs = []

for t in range(epochs):
	pY_train = forward(X_train, w, b)
	pY_test = forward(X_test, w, b)
	
	c_train = cost(Target_train_ind, pY_train)
	c_test = cost(Target_test_ind, pY_test)
	
	train_costs.append(c_train)
	test_costs.append(c_test)
	
	xT = np.transpose(X_train)
	w = w - learning_rate * (np.dot(xT, pY_train - Target_train_ind))
	b = b - learning_rate * (pY_train - Target_train_ind).sum(axis = 0)
	
	if t % 1000 == 0:
		print(t, c_train, c_test)
		
print("Final train classification_rate:", classification_rate(Target_train, predict(pY_train)))
print("Final test classification_rate:", classification_rate(Target_test, predict(pY_test)))

plt.plot(train_costs, label='train cost')
plt.plot(test_costs, label='test cost')
plt.legend()
plt.show()