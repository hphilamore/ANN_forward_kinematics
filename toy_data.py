"""
Generates toy data for foward kinematic model using joint angles to find enf effector x,y position and angular displacement 

"""
import random
import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches

filename = 'robot_data.csv'
n_data_points = 1000
L = 1 # link length 

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
	with open(filename, 'w') as f:
			w = csv.writer(f)

			w.writerow(['L', 'a', 'b', 'c', 'xe', 'ye', 'th'])

			for i in range(n_data_points):
				a = random.uniform(-np.pi, 0) # angle 1
				b = random.uniform(-np.pi, 0) # angle 2
				c = random.uniform(-np.pi/2, np.pi/2) # angle 3

				xe = Xe(a,b,c,L)
				ye = Ye(a,b,c,L)
				th = theta(a,b,c)

				w.writerow([L, a, b, c, xe, ye, th])

def plot_toy_data():
	""" Plots toy data """
	with open(filename, 'r') as f:
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
		plt.show()


if __name__ == "__main__":

	# Generate toy data set
	toy_data(L)

	# Plot toy data
	plot_toy_data()

else:
	pass