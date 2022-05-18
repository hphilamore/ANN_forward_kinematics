"""
Foward kinematic model using joint angles to find enf effector x,y position and angular displacement 

Insired by : 
- 'Building ANN to solve Inverse Kinematics of a 3 DOF Robot Arm'
https://medium.com/@kasunvimukthijayalath/building-ann-to-solve-inverse-kinematics-of-a-3-dof-robot-arm-2b1c3655a303

- 'How to use Data Scaling Improve Deep Learning Model Stability and Performance'
https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
"""


import random
import math
import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import keras.backend as K




def Xe(a,b,c,L):
	"X coordinate of end effector"
	return (L * np.cos(a) + 
			L * np.cos(a+b) +
			L * np.cos(a+b+c))

def Ye(a,b,c,L):
	"Y coordinate of end effector"
	return (L * np.sin(a) + 
			L * np.sin(a+b) +
			L * np.sin(a+b+c))

def theta(a,b,c):
	"Total angular displacement of end effector"
	return(a + b + c)



def toy_data(L):
	"""
	Saves set of end effector x,y coordinates and angular displacements (theta)
	for fixed link lengths (L) and randomly selected input angles (a-c) 
	"""
	with open('robot_data.csv', 'w') as f:
			w = csv.writer(f)

			w.writerow(['L', 'a', 'b', 'c', 'xe', 'ye', 'th'])

			for i in range(100):
				a = random.uniform(-np.pi, 0) # angle 1
				b = random.uniform(-np.pi, 0) # angle 2
				c = random.uniform(-np.pi/2, np.pi/2) # angle 3

				xe = Xe(a,b,c,L)
				ye = Ye(a,b,c,L)
				th = theta(a,b,c)

				w.writerow([L, a, b, c, xe, ye, th])

def plot_toy_data():
	""" Plots toy data """
	with open('robot_data.csv', 'r') as f:
		r = csv.reader(f)


		for line in list(r)[1:]:
			a, b, c, xe, ye, th  = float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6])

			# plot angle of end effector
			plt.plot([xe, xe+0.2*np.cos(th)],
					 [ye, ye+0.2*np.sin(th)], 'k-')	

			# plot position of end effector
			plt.plot(xe, ye, 'ro')

		plt.title('Data set of end effector positions and orientations')
		plt.savefig('robot_data_plotted.png')
		#plt.show()


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



def customloss(Ytrue, Ypred):
	""" Custom loss function for optimizer """
	return(K.sum((Ytrue - Ypred)**2))/len(trainX_)


def build_model():
	"""
	ANN model
	Input layer : 3 neurons to input the joint angles (a,b,c)
	Hidden layer: Contains 100 fully connected neurons
	Output Layer: 3 neurons to output end effector x,y coordinates and angle (theta)
	"""
	model = keras.Sequential()
	model.add(keras.layers.Dense(3, use_bias=True, activation='linear'))
	model.add(keras.layers.Dense(100, use_bias=True, activation='tanh'))
	model.add(keras.layers.Dense(3, use_bias=True, activation='linear'))
	model.compile(optimizer=tf.optimizers.Adam(0.05),
		          #loss="mse",       # mean squared error
		          loss=customloss, 
		          metrics=['accuracy']
		          )
	return model




if __name__ == "__main__":

	# Generate toy data set
	toy_data(1)

	# Plot toy data
	plot_toy_data()

	# import data 
	with open('robot_data.csv', 'r') as f:
		data = list(csv.reader(f))[1:]       # exclude first row (heading)
		data = [d[1:] for d in data]         # exclude first column (link length) 
		data = np.array(data).astype(float)

	# segmentation 
	features = data[:, :3] # features are joint angles 
	labels = data[:, 3:]    # labels are end effector x,y position and angle (theta) 

	trainX, testX, trainy, testy = train_test_split(features, labels, 
													test_size=0.3)  # split test and train
	

	# Scaling using tanh
	scaler_x, scaler_y, trainX_, testX_, trainy_, testy_ = scaling(trainX, testX, trainy, testy)

	
	# ANN model
	model = build_model()


	# Training the NN with the Data Set
	history = model.fit(trainX_, trainy_, epochs=100, 
						verbose = 0, validation_split=0.2
						)


	# evaluate the model
	train_mse = model.evaluate(trainX, trainy, verbose=0)
	test_mse = model.evaluate(testX, testy, verbose=0)
	print(f'Train: {train_mse}, Test: {test_mse}')


	# plot loss during training
	plt.clf()
	plt.title('Loss (Mean Squared Error)')
	plt.plot(history.history['loss'], label='train')
	#pyplot.plot(history.history['val_loss'], label='test')
	plt.legend()
	plt.show()

	# single prediction
	# data_point = np.array([-2, -1.2, 1]).reshape(1, trainX.shape[1])
	data_point = np.array([-2, -1.2, 1]).reshape(1, -1)
	data_point = scaler_x.transform(data_point)
	prediction = model.predict(data_point)
	real_prediction = scaler_y.inverse_transform(prediction)
	print(real_prediction)




















		






