# Section 7, Lecture 49

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
N = len(Y)

# turn Y into an indicator matrix for training
T = np.zeros((N, K))
for x in range(N):
	T[x, Y[x]] = 1

# let's see what it looks like
plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()

# tensor flow variables are not the same as regular Python variables
def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01)) # size shape with a standard deviation of 0.01

def forward(X, W1, b1, W2, b2):
	Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
	# We do not return the softmax but just the activation. That's one thing different between np and tf is that when we do the cost later,
	# we want the logits which is this as the input & not the output of the softmax.
	return tf.matmul(Z, W2) + b2

# tf placeholders for X & Y data	
tfX = tf.placeholder(tf.float32, [None, D])	# type float32 of an arbitrary shape of none by D
tfY = tf.placeholder(tf.float32, [None, K])

W1 = init_weights([D, M])
b1 = init_weights([M])
W2 = init_weights([M, K])
b2 = init_weights([K])

logits = forward(tfX, W1, b1, W2, b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tfY, logits=logits))

# tf is going to calculate the gradients & do gradient descent automatically so we don't have to specify the derivative in tf. So we will create
# a train function
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # learning rate = 0.05

predict_op = tf.argmax(logits, 1)  # axis = 1

# just stuff that has to be done
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for t in range(1000):
	sess.run(train_op, feed_dict={tfX: X, tfY: T})
	pred = sess.run(predict_op, feed_dict={tfX: X, tfY: T})
	
	if t % 10 == 0:
		print("Accuracy:", np.mean(Y == pred))