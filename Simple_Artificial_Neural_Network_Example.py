


########################################################################### Imports ############################################################################
import sys
import numpy as np                                                                     # For mathematical calculations 
import pandas as pd
import cryptocompare                                                                     # To Obtain Crypto Data From CryptoCompare API
import csv                                                                               # To Be Able To Wite in CSV Format
import json                                                                              # To Work On Object Type Data
from matplotlib import pyplot as plot                                                    # To Plot Graphs
from matplotlib import dates as axis_time                                                # To Plot Graphs
from datetime import datetime, timedelta                                                 # To Access Datetime
from scipy.signal import savgol_filter                                                   # To Smooth BTC Data Chart
from pandas import Series                                                                # To work on series
import copy
import warnings                                                                          # To ignore the warnings
warnings.filterwarnings("ignore")
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
########################################################### Activation Functions And Their Derivatives #########################################################
def tanh(x):
    # Tanh(x) = (e^(x) - e^(-x))/(e^(x) + e^(-x))
    return np.tanh(x);

def tanh_prime(x): 
    # Tanh'(x) = 1 - Tanh(x)^2
    return 1-np.tanh(x)**2;

def sigmoid(self, x):
  # Sigmoid(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def sigmoid_prime(self, x):
  # Sigmoid'(x) =  f'(x) = Sigmoid(x) * (1 - Sigmoid(x))
  fx = sigmoid(x)
  return fx * (1 - fx)
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
################################################################# Loss Function And Its Derivative #############################################################
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;
################################################################################################################################################################
################################################################################################################################################################
# We Have Two Different Kinds Of Layers, One Is Activation Layer And The Other Is FCLayer
################################################################################################################################################################
############################################################################# FCLayer ##########################################################################
class FCLayer():
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
######################################################################### Activation Layer #####################################################################
class ActivationLayer():
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # Returns The Activated Input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error = dE/dX For A Given output_error = dE/dY.
    # learning_rate Is Not Used Because There Is No "Learnable" Parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    
    # Add Layer To Network
    def add(self, layer):
        self.layers.append(layer)

    # Set Loss To Use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # Predict Output For Given Input
    def predict(self, input_data):
        # Sample Dimension First
        samples = len(input_data)
        result = []

        # Run Network Over All Samples
        for i in range(samples):
            # Forward Propagation
            output = input_data[i]
            for layer in self.layers:                                                    # This Computes The Layer Output With Respect To The Layer Type
                output = layer.forward_propagation(output)                               # Calculation Differs For Activation/FC Layers
            result.append(output)

        return result

    # Train The Network
    def train(self, x_train, y_train, epochs, learning_rate):
        # Sample Dimension First
        samples = len(x_train)

        # Training Loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # Forward Propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # Compute Loss (For Display Purpose Only)
                err += self.loss(y_train[j], output)

                # Backward Propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # Calculate Average Error On All Samples
            err /= samples
            if i % 100 == 0 :
                print('epoch %d/%d   error=%f' % (i+1, epochs, err))
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################


# training data

x = np.linspace(0, 25, 251)
y = np.sinc(x)#*(x**(3/2))

# Remember If delays = np.array([0]), Just X Will Be Inputed To The Neural Network
# Delays Should be Inserted Like delays = np.array([0, d0, d1, ...]) 
delays = np.array([0, 10, 30])        
n_delays = delays.size
max_delay = delays.max()

x_train = np.zeros((len(y) - max_delay)*n_delays).reshape(*((len(y) - max_delay), 1, n_delays))
for i in range(n_delays):
    x_train[:,0,i] = y[delays[i]:(len(y) - max_delay + delays[i])]

######################

#np.random.shuffle(x_train)
y_train = x_train[:,0,0]       # Selecting First Column Of x_train[][][] 3D Array


######################

# network
net = Network()
net.add(FCLayer(3, 2))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(2, 2))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(2, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.train(x_train, y_train, epochs=1000, learning_rate=0.05)

# test
out = net.predict(x_train[:,0,0])
out = np.squeeze(np.array(out))

plot.plot(x,y)
plot.show(block=False)

if n_delays > 1 :
    plot.plot(x[0:len(y_train)],out[:,0])
else:
    plot.plot(x[0:len(y_train)],out)

plot.show(block=False)

