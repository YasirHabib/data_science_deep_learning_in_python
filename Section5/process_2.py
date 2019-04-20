# Section 4, Lecture 23
# Exact copy of logistic_process_2.py

import numpy as np
import pandas as pd

def get_data():
	# Load the data
	df = pd.read_csv("ecommerce_data.csv")
	
	# Split the data into data & target
	data = df[['is_mobile', 'n_products_viewed', 'visit_duration', 'is_returning_visitor', 'time_of_day', ]].values
	target = df['user_action'].values

	# Normalization
	mean_1 = np.mean(data[:, 1])
	s_d_1 = np.std(data[:, 1])
	data[:, 1] = (data[:, 1] - mean_1) / s_d_1

	mean_2 = np.mean(data[:, 2])
	s_d_2 = np.std(data[:, 2])
	data[:, 2] = (data[:, 2] - mean_2) / s_d_2

	column_of_zeros = np.zeros((len(data), 4))
	data_new = np.column_stack([data[:, 0:(4)], column_of_zeros])
	# one-hot encode to accommodate 0, 1, 2, 3. If 0 -> 5th column, 1 -> 6th column, 2 -> 7th column, 3 -> 8th column.

	for x in range((len(data))):
		t = int(data[x, 4])
		data_new[x, (t+4)] = 1

	return data_new, target

def get_binary_data():
	X, Y = get_data()
								# print(X.shape) -> (500, 8), print(Y.shape) -> (500,)

	X2 = X[Y <= 1]				# This means everytime the value of Y <= 1, then select the entire corresponding row in X.
	Y2 = Y[Y <= 1]				# This means everytime the value of Y <= 1, then select that value.
								# print(X2.shape) -> (398, 8), print(Y2.shape) -> (398,)
	return X2, Y2



