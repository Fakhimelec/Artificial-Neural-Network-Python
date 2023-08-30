
# Version 20230119
# In This Version The 'High', 'Low', 'Open', 'Close', 'VolumeFrom' And 'VolumeTo' Values Will Be Given As Input
# Also The 'High', 'Low', 'Open', 'Close', 'VolumeFrom' And 'VolumeTo' Of Next h Steps Values Will Be Required As Output
# x Has Change Into t
# y Has Changed Into x

################################################################################################################################################################
######################################################################### Definition ###########################################################################
# Consider We Want To Engage 3 Delays (0, 1, 5) Of A Time Series To Calculate The Value Of The 3(h_step = 3)
# Steps Ahead. We Have 1000 Samples.  Here Is We Want To Actually Do :
# Each x(t) Equals A Set Of ['High', 'Low', 'Open', 'Close', 'VolumeFrom', 'VolumeTo'] Array At Time (t)
#
#
#                                          Inputs And Targets Array Order In The Following Code
#                          /=========================================^=========================================\
#           y_train                                       x_train                                       y_train
#          ,---^---,       ,----------------------------------^-----------------------------------,    ,---^---,
#   t       x(t+3)          x(t)        x(t-1)      x(t-2)	x(t-7)      x(t-9)      x(t-26)         x(t+3)
#   0       /////           /////       /////       /////       /////       /////       /////           /////
#   1       /////           /////       /////       /////       /////       /////       /////           /////
#   2       /////           /////       /////       /////       /////       /////       /////           /////
#   3       /////           /////       /////       /////       /////       /////       /////           /////
#   .       .               .           .           .           .           .           .               .
#   .       .               .           .           .           .           .           .               .
#   .       .               .           .           .           .           .           .               .
#   25      /////           /////       /////       /////       /////       /////       /////           /////
#   26      x(29)           x(26)       x(25)       x(24)       x(19)       x(17)       x(0)            x(29)          
#   27      x(30)           x(27)       x(26)       x(25)       x(20)       x(18)       x(1)            x(30)          
#   28      x(31)           x(28)       x(27)       x(26)       x(21)       x(19)       x(2)            x(31)          
#   29      x(32)           x(29)       x(28)       x(27)       x(22)       x(20)       x(3)            x(32)          
#   .       .               .           .           .           .           .           .               .
#   .       .               .           .           .           .           .           .               .
#   .       .               .           .           .           .           .           .               .
#  1437     x(1440)         x(1437)     x(1436)     x(1435)     x(1430)     x(1428)     x(1411)         x(1440)
#  1438	    x(1441)?        x(1438)     x(1437)     x(1436)     x(1431)     x(1429)     x(1412)         x(1441)?
#  1439     x(1442)?        x(1439)     x(1438)     x(1437)     x(1432)     x(1430)     x(1413)         x(1442)?
#  1440     x(1443) ---->   x(1440)     x(1439)     x(1438)     x(1433)     x(1431)     x(1414)  ---->  x(1443)
#
# As You See We Can Only Use The Data From Maximum Delay To h_steps Less Than The Samples Count
# So Our Array Of Inputs Will Be : (nSamples - max_delay - h_steps) Rows And (nDelays * nInputs) Columns
# And The Array Of Outputs Will Also Be : (nSamples - max_delay - h_steps) Rows And (nOutputs) Columns
#
# Remember That The h_step Doesn't Affect The calculating Process And All Of The Codes Remains Same But Due To The Change Of Inices,
# Neural Network Weights Change And The Network Will Be Trained Different.
################################################################################################################################################################
################################################################################################################################################################

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
from sklearn import preprocessing                                                        # To Normalize Data And Map Them To The [0 1] Period
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
    fx = tanh(x)
    return 1-fx**2;

def sigmoid(x):
  # Sigmoid(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
  # Sigmoid'(x) =  f'(x) = Sigmoid(x) * (1 - Sigmoid(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def gauss(x):
  # Gauss(x) = e^(-x^2))
  return np.exp(-(0.1*x)**2)

def gauss_prime(x):
  # Sigmoid'(x) =  f'(x) = Sigmoid(x) * (1 - Sigmoid(x))
  fx = gauss(x)
  return -2*x*fx

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
#################################################################### Data Normalizing Function #################################################################

def normalize(x, min_x, max_x):
    normalized_x = x
    for i in range(x[0,].size):
        normalized_x[:,i] = (x[:,i] - min_x[i])/(max_x[i] - min_x[i]) * 2 - 1        # Normalizing Data To The Scale Of -1 To 1
    return normalized_x

def denormalize(x_normalized, min_x, max_x):
    denormalized_x = x_normalized  
    for i in range(x_normalized[0,].size):
        denormalized_x[:,i] = ((x_normalized[:,i] + 1) *
                                 (max_x[i] - min_x[i])) / 2 + min_x[i]                   # Denormalizing Data To The Privious Scale
    return denormalized_x

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
    def train(self, x_train, y_train, epochs, learning_rate, goal_error):
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
                print('epoch %05d/%05d   error=%f' % (i+1, epochs, err))
            if i == epochs :
                print('Iterations Reached Till End!')
            if err < goal_error:
                print('Minimum Error Condition Has Been Met!')
                break;
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################


##########################################################################################################################
################################################## Importing BTC Data ####################################################
# BTC.csv Has 1441 Samples Of 14 Different Elements Whish We Only Need Some Of Them
filename = 'BTC.csv'
btc_data = np.array(pd.read_csv('BTC.csv'))                            # BTC Data Frame
# Pay Attention That In This Dataset The Last Data Sample Is The Data Which Is On Going right Now And We Don't Need That!
# We Need To Guess It !
btc_data = np.delete(btc_data, 1440, axis = 0)
# t_btc Is TimeStep Of The Data And Is Used For Plotting Charts
t_btc = btc_data[:,0]

# Select 6 Columns Of BTC Data Which Corresponds To The Values Of
# 'High', 'Low', 'Open', 'Close', 'VolumeFrom', 'VolumeTo',
# x_btc Is A 2D Array
x_btc = btc_data[:, 8:14]

# Output of Our Network Will Be The Low And High Value Of Next Step
targets = btc_data[:, 8:14]                                                               # Targets

##########################################################################################################################
########################################## Considering Some Example Function #############################################
##x_ex = np.linspace(0, 25, 251)
##y_ex = np.sinc(x_ex)*np.cos(x_ex)

##########################################################################################################################
############################################## Assigning x And y Matrices ################################################

nSamples, nInputs = x_btc.shape                             
h_step = 1                                                                               # The TimeSteps That We Want To Predict
delays = np.array([0, 1, 2, 7, 9, 26])        
nDelays = delays.size
max_delay = delays.max()
nOutputs = targets.shape[1]

# Reshape Inputs And Targets(Output) Arrays To Have a Correct Dimensions
x_btc = x_btc.reshape(nSamples, nInputs)
targets = targets.reshape(nSamples, nOutputs)

# Save A Copy Of Original Inputs
t = copy.deepcopy(t_btc)                                                                 # TimeStamp
x = copy.deepcopy(x_btc)                                                                 # Inputs

##########################################################################################################################
############################################# Calculating Delay Parameters ###############################################
# Here We Are Trying To Give The Multiple Delays Of Output As An Input Of Our Neural Network And See The Results
# Remember If delays = np.array([0]), Just Y Will Be Inputed To The Neural Network
# Delays Should be Inserted Like delays = np.array([0, d0, d1, ...])
# delays = np.array([0]) Will Only Engage x(t) For Calculating x(t+h)

# Building A [nSamples, nInputs * nDelays] Array As Input And Naming It x_train
x_train = np.zeros((nSamples - max_delay - h_step) * nDelays * nInputs
                   ).reshape(*((nSamples - max_delay - h_step), 1, nDelays * nInputs))

column_counter = 0
for i in range(nDelays):
    for j in range(nInputs):
        x_train[:, 0, column_counter] = x[max_delay - delays[i]: (nSamples - delays[i] - h_step), j]
        column_counter += 1

# Building A [nSamples, nOutputs] Array As Output And Naming It y_train
y_train = np.zeros((nSamples - max_delay - h_step) * nOutputs
                   ).reshape(*((nSamples - max_delay - h_step), nOutputs))

for i in range(nOutputs):
    y_train[:,i] = targets[max_delay + h_step :, i]

# Saving Original x_train And y_train
x_train_orig = copy.deepcopy(x_train)
y_train_orig = copy.deepcopy(y_train)

##########################################################################################################################
###################################################### Saving Data #######################################################
save = np.zeros((nSamples - max_delay - h_step) * (nInputs * nDelays + nOutputs)
                ).reshape((nSamples - max_delay - h_step), (nInputs * nDelays + nOutputs))
save[:,0:(nInputs * nDelays)] = x_train[:,0,:]
save[:,(nInputs * nDelays):] = y_train[:,:]
pd.DataFrame(save).to_csv('Data_Raw.csv', index=False, header=False)

##########################################################################################################################
#################################################### Shuffling Data ######################################################
# Here In Order To Prevent Data From Getting Mixed, First We Put The Data Together Horizontally
# Then Generate A Random Permutaation
temp_io = np.zeros((nSamples - max_delay - h_step) * (nInputs * nDelays + nOutputs)
                ).reshape((nSamples - max_delay - h_step), (nInputs * nDelays + nOutputs))
temp_io[:,0:(nInputs * nDelays)] = x_train[:,0,:]
temp_io[:,(nInputs * nDelays):] = y_train[:,:]
temp_io = np.random.permutation(temp_io)
x_train[:,0,:] = temp_io[:,0: (nInputs * nDelays)]
y_train = temp_io[:,(nInputs * nDelays):]

##########################################################################################################################
###################################################### Saving Data #######################################################
save = np.zeros((nSamples - max_delay - h_step) * (nInputs * nDelays + nOutputs)
                ).reshape((nSamples - max_delay - h_step), (nInputs * nDelays + nOutputs))
save[:,0:(nInputs * nDelays)] = x_train[:,0,:]
save[:,(nInputs * nDelays):] = y_train[:,:]
pd.DataFrame(save).to_csv('Data_Permute.csv', index=False, header=False)

##########################################################################################################################
################################################### Normalizing Data #####################################################
min_x_train = x_train[:,0,:].min(axis=0)
max_x_train = x_train[:,0,:].max(axis=0)
min_y_train = y_train.min(axis=0)
max_y_train = y_train.max(axis=0)
normalized_x_train = x_train
normalized_x_train[:,0,:] = normalize(x_train[:,0,:], min_x_train, max_x_train)          # Normalizing Data To The Period 
normalized_y_train = y_train
normalized_y_train = normalize(normalized_y_train, min_y_train, max_y_train)             # Of -1 To 1


##########################################################################################################################
############################################# Defining x_train And y_train ###############################################
x_train = normalized_x_train
y_train = normalized_y_train

##########################################################################################################################
###################################################### Saving Data #######################################################
save = np.zeros((nSamples - max_delay - h_step) * (nInputs * nDelays + nOutputs)
                ).reshape((nSamples - max_delay - h_step), (nInputs * nDelays + nOutputs))
save[:,0:(nInputs * nDelays)] = x_train[:,0,:]
save[:,(nInputs * nDelays):] = y_train[:,:]
pd.DataFrame(save).to_csv('Data_Normal.csv', index=False, header=False)

################################################################################################################################################################
################################################################################################################################################################
#
######################
#exit()
# Network
net = Network()
net.add(FCLayer(nDelays * nInputs, 9))
net.add(ActivationLayer(gauss, gauss_prime))
##net.add(FCLayer(9, 2))
##net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(9, nOutputs))
net.add(ActivationLayer(tanh, tanh_prime))

# Trainin

net.use(mse, mse_prime)

epochs = 2000
learning_rate = 0.2
goal_error = 0#1e-6
net.train(x_train, y_train, epochs, learning_rate, goal_error)


# Testing
x_test = copy.deepcopy(x_train_orig)
x_test = normalize(x_test[:,0,:], min_x_train, max_x_train)
out = net.predict(x_test)
out = np.squeeze(np.array(out))


# Here We Should Notice That The Index:1272 Is Our Latest Time And Index:1273 Will Be The Next Step
x_stepahead = np.zeros(nDelays * nInputs).reshape(1, 1, nDelays * nInputs)
x_stepahead_temp = copy.deepcopy(x)
x_stepahead_temp = normalize(x_stepahead_temp, x_stepahead_temp.min(axis = 0), x_stepahead_temp.max(axis = 0))

for i in range(nDelays):
    x_stepahead[0,0,i * nInputs : (i+1) * nInputs] = x_stepahead_temp[nSamples - delays[i] - 1,:]

out_stepahead = net.predict(x_stepahead)
out_stepahead_denormalized = denormalize(np.squeeze(out_stepahead).reshape(1,nOutputs), min_y_train, max_y_train)
print('The High Price Of The Next Step Will Be ' + str(out_stepahead_denormalized[0,0]) + '$')
print('The Low Price Of The Next Step Will Be ' + str(out_stepahead_denormalized[0,1]) + '$')
print('The Open Price Of The Next Step Will Be ' + str(out_stepahead_denormalized[0,2]) + '$')
print('The Close Price Of The Next Step Will Be ' + str(out_stepahead_denormalized[0,3]) + '$')
print('The VolumeFrom Of The Next Step Will Be ' + str(out_stepahead_denormalized[0,4]))
print('The Volume To Of The Next Step Will Be ' + str(out_stepahead_denormalized[0,5]))
# Denormalization
# It Should Be Noted That The Denormalization Must Have Been Applied After All Calculations
# In Other Words, Neural Network Should Always Be Feed With The Normalized Data

out_denormalized = denormalize(out, min_y_train, max_y_train)

#exit(0)

# Demonstration And Plots
for i in range(nOutputs):
    plot.figure(i)
    plot.plot(t_btc[max_delay : nSamples - h_step],x_btc[max_delay : nSamples - h_step, i], 'k', label = 'Actual Data')
    plot.plot(t_btc[max_delay : nSamples - h_step],out_denormalized[:,i], 'r', label = 'ANN  Output Data')
plot.legend(loc = 'best')
plot.show(block=False)

#exit(0)


# Notice The Normalization And Denormalization Process In Order Not To Have A Miscalculation
