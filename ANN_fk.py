"""
Foward kinematic model using joint angles to find enf effector x,y position and angular displacement 

Insired by : 
- 'Building ANN to solve Inverse Kinematics of a 3 DOF Robot Arm'
https://medium.com/@kasunvimukthijayalath/building-ann-to-solve-inverse-kinematics-of-a-3-dof-robot-arm-2b1c3655a303

- 'How to use Data Scaling Improve Deep Learning Model Stability and Performance'
https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
"""

# mlp with scaled outputs on the regression problem
import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import tensorflow as tf
import toy_data


def scaling(trainX, testX, trainy, testy):

	# create scalers for x and y data shapes 
	scaler_x = MinMaxScaler(feature_range=(-1,1))
	scaler_y = MinMaxScaler(feature_range=(-1,1))

	# fit scaler on training data
	scaler_x.fit(trainX)
	scaler_y.fit(trainy)

	# apply transform
	trainX_ = scaler_x.transform(trainX)
	testX_ = scaler_x.transform(testX)
	trainy_ = scaler_y.transform(trainy)
	testy_ = scaler_y.transform(testy)

	return scaler_x, scaler_y, trainX_, testX_, trainy_, testy_ 

def import_data(filename=toy_data.filename):
	with open(filename, 'r') as f:
		data = list(csv.reader(f))[1:]       # exclude first row (heading)  
		data = [d[1:] for d in data]         # exclude first column (link length)
		data = np.array(data).astype(float)  
	return data


def build_model():
	"""
	ANN model
	Input layer : 3 neurons to input the joint angles (a,b,c)
	Hidden layer: Contains 100 fully connected neurons
	Output Layer: 3 neurons to output end effector x,y coordinates and angle (theta)
	"""
	model = Sequential()
	model.add(Dense(3, use_bias=True, activation='linear'))
	model.add(Dense(100, use_bias=True, activation='tanh'))
	model.add(Dense(3, use_bias=True, activation='linear'))


	# compile model
	# model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01, momentum=0.9))
	model.compile(optimizer=tf.optimizers.Adam(0.05),
		          loss="mse",       # mean squared error
		          metrics=['accuracy']
		          )
	return model

def plot_loss(history):
	plt.clf()
	plt.title('Loss (Mean Squared Error)')
	plt.plot(history.history['loss'], label='train')
	#plt.plot(history.history['val_loss'], label='test')
	plt.legend()
	plt.show()


def single_prediction(point):
	# data_point = np.array([-2, -1.2, 1]).reshape(1, trainX.shape[1])
	point = np.array(point).reshape(1, -1)
	point = scaler_x.transform(point)
	prediction = model.predict(point)
	real_prediction = scaler_y.inverse_transform(prediction)
	real_prediction = np.around(np.around(real_prediction, 2), 2)

	return real_prediction


def analytical_solution(point):
	point = np.array(point).reshape(1, -1)
	a = point[0,0]
	b = point[0,1]
	c = point[0,2]

	sol = []
	sol.append(toy_data.Xe(a, b, c, toy_data.L))
	sol.append(toy_data.Ye(a, b, c, toy_data.L))
	sol.append(toy_data.theta(a, b, c))
	sol = np.array(sol)
	sol = np.around(sol, 2)

	return sol

def predict_and_check(point):
	real_prediction = single_prediction(point)

	# check prediction against analytical solution
	sol = analytical_solution(point)

	print(f'data point = {point} \nprediction (real units) = {real_prediction} \nanalytical solution = {sol}')



if __name__ == "__main__":

	# import data 
	data = import_data()


	# segmentation 
	features = data[:, :3] # features are joint angles 
	labels = data[:, 3:]    # labels are end effector x,y position and angle (theta) 


	# split into train and test
	trainX, testX, trainy, testy = train_test_split(features, labels, test_size=0.3)
	# reshape 1d array y data to 2d array if needed
	try: 
		trainy = trainy.reshape(len(trainy), 1)
		testy = testy.reshape(len(testy), 1)
	except:	
		pass


	# scaling using tanh
	scaler_x, scaler_y, trainX, testX, trainy, testy = scaling(trainX, testX, trainy, testy)


	# ANN model
	model = build_model()

	# Training the NN with the Data Set
	history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)

	# evaluate the model
	train_mse = model.evaluate(trainX, trainy, verbose=0)
	test_mse = model.evaluate(testX, testy, verbose=0)
	print(f'Train: {train_mse}, Test: {test_mse}')


	# plot loss during training
	plot_loss(history)
	

	# single prediction
	point = [-2.8, -1, 1]
	# check a single prediction
	predict_and_check(point)
