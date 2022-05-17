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


"""
Insired by : 'Building ANN to solve Inverse Kinematics of a 3 DOF Robot Arm'
https://medium.com/@kasunvimukthijayalath/building-ann-to-solve-inverse-kinematics-of-a-3-dof-robot-arm-2b1c3655a303
"""

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


def scaling():
	with open('robot_data.csv', 'r') as f:
		data = list(csv.reader(f))[1:]
		data = np.array(data).astype(float)
		# scaler = MinMaxScaler(feature_range=(-1,1))
		# data_scaled = scaler.fit_transform(data)

		# create scaler
		scaler = MinMaxScaler(feature_range=(-1,1))
		# # fit scaler on data
		# scaler.fit(data)
		# # apply transform
		# normalized = scaler.transform(data)

		# fit and transform in one step
		normalized = scaler.fit_transform(data)
		# inverse transform
		inverse = scaler.inverse_transform(normalized)

		return scaler, inverse, normalized 


def customloss(Ytrue, Ypred):
	""" Custom loss function for optimizer """
	return(K.sum((Ytrue - Ypred)**2))/len(Xtrain)


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
		          loss=customloss, 
		          metrics=['accuracy']) # mean squared error
	return model




if __name__ == "__main__":

	# Generate toy data set
	toy_data(1)

	# Plot toy data
	plot_toy_data()

	# Scaling using tanh
	scaler, inverse, data_scaled = scaling()


	# Segmentation
	features = data_scaled[:, 1:4] # features are joint angles 
	labels = data_scaled[:, 4:]    # labels are end effector x,y position and angle (theta) 

	Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, labels, 
													test_size=0.3)  # split test and train
	
	Xval, Xtest, Yval, Ytest = train_test_split(Xtest, Ytest, 
												test_size=0.5) # split test and validation (0.5 x 0.3 = 0.15)


	
	# ANN model
	model = build_model()


	# Training the NN with the Data Set
	history = model.fit(Xtrain, Ytrain, epochs=100)
	# TODO: plot history - https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

	# Evalute using validation data
	loss, mae = model.evaluate(Xval, Yval, verbose=0) 
	print(f"Mean average error, validation data = {mae}")

	# Test using test data
	prediction = model.predict(Xtest) 
	#prediction = scaler.inverse_transform(prediction) # transform back to real units 




















		






