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
from matplotlib import pyplot
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

	# # inverse transform
	# inverse = scaler.inverse_transform(normalized)

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


if __name__ == "__main__":

	# Generate toy data set
	# toy_data(1)

	# Plot toy data
	# plot_toy_data()

	# import data 
	data = import_data()
	# with open('robot_data.csv', 'r') as f:
	# 	data = list(csv.reader(f))[1:]       # exclude first row (heading)  
	# 	data = [d[1:] for d in data]         # exclude first column (link length)
	# 	data = np.array(data).astype(float)  


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


	# scaling
	scaler_x, scaler_y, trainX, testX, trainy, testy = scaling(trainX, testX, trainy, testy)


	# ANN model
	model = build_model()

	# fit model
	history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)

	# evaluate the model
	train_mse = model.evaluate(trainX, trainy, verbose=0)
	test_mse = model.evaluate(testX, testy, verbose=0)
	print(f'Train: {train_mse}, Test: {test_mse}')


	# plot loss during training
	pyplot.title('Loss (Mean Squared Error)')
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	pyplot.show()


	# single prediction
	# data_point = np.array([-2, -1.2, 1]).reshape(1, trainX.shape[1])
	data_point = np.array([-2, -1.2, 1]).reshape(1, -1)
	data_point = scaler_x.transform(data_point)
	prediction = model.predict(data_point)
	real_prediction = scaler_y.inverse_transform(prediction)
	print(real_prediction)