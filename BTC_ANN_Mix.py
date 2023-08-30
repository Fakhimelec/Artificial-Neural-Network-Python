
###################################################################### Imports #################################################################################
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

#===============================================================================================================================================================

#######         ##      ##        ##     ##             ####        ##########      ####            ##          ##     ##           ###
##              ##      ##        ## #   ##           ##                ##           ##           ##  ##        ## #   ##         ##   #
##              ##      ##        ##  #  ##          ##                 ##           ##          ##    ##       ##  #  ##         ##
######          ##      ##        ##   # ##          ##                 ##           ##          ##    ##       ##   # ##           ###
##               ##    ##         ##    ###           ##                ##           ##           ##  ##        ##    ###         #   ##
##                 ####           ##     ##             ####            ##          ####            ##          ##     ##          ###

#===============================================================================================================================================================

################################################## This Function will Return The Extermums Of Series f #########################################################
# f Is Input Curve And count Is The Number Of Extermums Before end_time
def privious_extermum(f, end_time, count):
    dif_f = func.gradient(f)
    output = [0] * count
    j = 0
    
    for i in range(0, end_time)[::-1]:
        if (dif_f[i] * dif_f[i-1] < 0):
            output[j] = i-1
            j = j + 1
        if (j == count):
            return output #[::-1]
            break
    return output #[::-1]
################################################################################################################################################################
################################################################################################################################################################

############################################### This Function will Return The Possible Direction Of Series f ###################################################
# f Is Input Curve And x Is The Number Of Privious Extermums
def direction(f):
    xi = privious_extermum(f, 1440, 5)
    if (df[len(f)-1]>0):
        #print('Upward')
        if (f[xi[0]]>f[xi[2]]) and (f[xi[1]]>f[xi[3]]):
            print('↑↑↑')
        else:
            print('Can Not Define!')
    elif (df[len(f)-1]<0):
        #print('Downward')
        if (f[xi[0]]<f[xi[2]]) and (f[xi[1]]<f[xi[3]]):
            print('↓↓↓')
        else:
            print('Can Not Define!')
    else:
        print('Extermum')
        if (f[xi[0]]<f[xi[2]]<f[xi[1]]):
            if (ddf(len(f)-1)>0):
                print('↑↑↑')
            elif(ddf(len(f)-1)<0):
                print('↓↓↓')
################################################################################################################################################################
################################################################################################################################################################
                
######################################################## Activation Functions And Their Derivatives ############################################################
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

################################################################# Data Normalizing Function ####################################################################
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

################################################################# Loss Function And Its Derivative #############################################################
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;
################################################################################################################################################################
################################################################################################################################################################

# We Have Two Different Kinds Of Layers, One Is Activation Layer And The Other Is FCLayer

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

########################################################################## Network Class #######################################################################
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
            
#===============================================================================================================================================================

##        ##            ##             ####         ##     ##
## #    # ##           ####             ##          ## #   ##
##  #  #  ##          ##  ##            ##          ##  #  ##
##   ##   ##         ########           ##          ##   # ##
##        ##        ##      ##          ##          ##    ###
##        ##       ##        ##        ####         ##     ##

#===============================================================================================================================================================
            
#################################################################### Get Data From CryptoCompare.com ###########################################################

# In This Piece Of Code The User Is Out To Insert A Timestamp To Calculate The Timing Of The Table By That Sale
# It's Importnt To Know That In The Following 'cur_input' Variable Which Is Obtained From CryptoCompare's API, 
# Index[0] Is The Oldest Price And Index[1440] Is The Latest Price (Now's Price) And 'time' Index Is The Last 19271 Day's Data
# The Data Corresponds To 60 Days In Time Steps Of Seconds But In Free Version We Can't Accuire More Acurate Data Than Minute TimeStep

currency = 'BTC'

print('TimeStamps Are Like These: Min, Hou or Day')
while True :
    timestamp = str(input('Please Insert TimeStamp: '))
    if (timestamp == 'Min') or (timestamp == 'min') or (timestamp == 'm'):
        cur_input = cryptocompare.get_historical_price_minute(currency, currency='USD')  # Get Minutely Data From CryptoCompare API (Currency Input)
        m = 1                                                                            # These Variables Are Used To Formulate cur_data Table
        h = 0                                                                            # When Minute, Hour Or Day Is Selected By The User
        d = 0
        timestamp == 'Minutly'                                                           # This Variable Will Be Used Later In Naming The Output File
        break;
    elif (timestamp == 'Hou') or (timestamp == 'hou') or (timestamp == 'h'):
        cur_input = cryptocompare.get_historical_price_hour(currency, currency='USD')    # Get Hourly Data From CryptoCompare API (Currency Input)
        m = 0
        h = 1
        d = 0
        timestamp == 'Hourly'
        break;
    elif (timestamp == 'Day') or (timestamp == 'day') or (timestamp == 'd'):
        cur_input = cryptocompare.get_historical_price_day(currency, currency='USD')     # Get Daily Data From CryptoCompare API (Currency Input)
        m = 0
        h = 0
        d = 1
        timestamp == 'Daily'
        break;
    else :
        print('Please Insert One Of The Three TimeStamps...')

cur_price = cryptocompare.get_price(currency, currency='USD')                            # This Is The Live Price (Now's Price)                                                                                    
################################################################################################################################################################
################################################################################################################################################################

###################################################################### Calculating Time Zone Difference ########################################################
time_zone = '+03:30'                                                                     # Consider TimeZone Difference ('Asia/Tehran')
#tzm = []
#tzm[:] = time_zone
tz = list(time_zone)                                                                     # tz Variable Contains TimeZone Difference
if tz[0] == '+':
    tz_hour = int(tz[1]+tz[2])                                                           # tz_hour Is Time Zone Hour
    tz_min  = int(tz[4]+tz[5])                                                           # tz_min Is Time Zone Minute
else:
    tz_hour = -1*int(tz[1]+tz[2])                                                        # tz Variable Sign Should Be Considered
    tz_min  = -1*int(tz[4]+tz[5])

sp_time = list('08:30')                                                                  # To Calculate The Situation in Specific Time 
sp_time_hour = int(sp_time[0] + sp_time[1])
sp_time_min = int(sp_time[3] + sp_time[4])
#specific_time_index =                                                                   # Change Time Of 'HH:MM' Format To Minutes
dif_time = (datetime.now() - timedelta(hours = sp_time_hour + tz_hour,                   # Calculating Time Difference Between Now
                                            minutes = sp_time_min + tz_min)              # And The Specific Time
            ).strftime('%H:%M')
sp_time = list(dif_time)
sp_time_index = len(cur_input) - (int(sp_time[0] + sp_time[1]) * 60 +                    # sp_time_index Of cur_input Is Related To 
                                  int(sp_time[3] + sp_time[4])) - 1                      # Specific Time

original_stdout = sys.stdout                                                             # Save a reference to the original standard output
################################################################################################################################################################
################################################################################################################################################################

################################################### Processing Data And Arranging Table Columns to Look Comprehensible #########################################

cur_data = pd.DataFrame(cur_input)                                                       # Turn Data In To Table As A DataFrame (Currency Data)
cur_data.drop('conversionType', axis = 1, inplace = True)                                # Delete Unwanted Columns
cur_data.drop('conversionSymbol', axis = 1, inplace = True)                              # axis = 1 Refers To Column, axis = 0 Refers To Row
cur_data.insert(0, 'Weekend', datetime.now().weekday())
cur_data.insert(0, 'Minute', datetime.now().strftime('%M'))                              # To Access A Specific Cell One Can Use _
cur_data.insert(0, 'Hour', datetime.now().strftime('%H'))                                # cur_data['Column']['Row']   Syntax
cur_data.insert(0, 'Day', datetime.now().strftime('%d'))                                 
cur_data.insert(0, 'Month', datetime.now().strftime('%m'))                               
cur_data.insert(0, 'Year', datetime.now().strftime('%Y'))
cur_data.insert(0, 'TimeInMinute', 0)
cur_data.insert(11, 'close', cur_data.pop('close'))                                      # Move The 'close' Column Next To 'open' Column (Column 11)

cur_data = cur_data.rename(columns={"time": "LocalTime",                                 # Rename DataFrame Keys()
                                    "high": "High",
                                    "low": "Low",
                                    "open": "Open",
                                    "close": "Close",
                                    "volumefrom": "VolumeFrom",
                                    "volumeto": "VolumeTo"})

rows, columns = cur_data.shape
for counter in range(0,rows)[::-1]:                                        # Calculating Date in Minutes
    cur_data['Minute'][rows - counter - 1] = (datetime.now() - timedelta(  # Update The 'Year', 'Month', 'Day' And 'HH:MM' Keys
        days = d*counter, hours = h*counter + tz_hour,
        minutes = m*counter + tz_min)).strftime('%M')                 # tz Variable Is Time Zone Difference Consideration
    cur_data['Hour'][rows - counter - 1] = (datetime.now() - timedelta(
        days = d*counter, hours = h*counter + tz_hour,
        minutes = m*counter + tz_min)).strftime('%H')
    cur_data['Day'][rows - counter - 1] = (datetime.now() - timedelta(
        days = d*counter, hours = h*counter + tz_hour,
        minutes = m*counter + tz_min)).strftime('%d')
    cur_data['Month'][rows - counter - 1] = (datetime.now() - timedelta(
        days = d*counter, hours = h*counter + tz_hour,
        minutes = m*counter + tz_min)).strftime('%m')
    cur_data['Year'][rows - counter - 1] = (datetime.now() - timedelta(
        days = d*counter, hours = h*counter + tz_hour,
        minutes = m*counter + tz_min)).strftime('%Y')
    cur_data['TimeInMinute'][rows - counter - 1] = rows - counter - 1
    ####################### Determine That Today Is Weekend Or Not ###############################
    cur_data['Weekend'][rows - counter - 1] = (datetime.now() - timedelta( 
        days = d*counter, hours = h*counter + tz_hour,
        minutes = m*counter + tz_min)).weekday()

    if (cur_data['Weekend'][rows - counter - 1] == 5 or                    
        cur_data['Weekend'][rows - counter - 1] == 6) :                            
        cur_data['Weekend'][rows - counter - 1] = 1
    else:
        cur_data['Weekend'][rows - counter - 1] = 0
    ##############################################################################################
    ################ Order Local Time Column According To Minute, Hour Or Day ####################
    if (timestamp == 'Min') or (timestamp == 'min') or (timestamp == 'm'):
        cur_data['LocalTime'][rows - counter - 1] = datetime.fromtimestamp(cur_data['LocalTime'][rows - counter - 1]).strftime("%H:%M")
    elif (timestamp == 'Hou') or (timestamp == 'hou') or (timestamp == 'h'):
        cur_data['LocalTime'][rows - counter - 1] = datetime.fromtimestamp(cur_data['LocalTime'][rows - counter - 1]).strftime("%m-%d-%H")
    elif (timestamp == 'Day') or (timestamp == 'day') or (timestamp == 'd'):
        cur_data['LocalTime'][rows - counter - 1] = datetime.fromtimestamp(cur_data['LocalTime'][rows - counter - 1]).strftime("%Y-%m-%d")
    ##############################################################################################
################################################################################################################################################################
################################################################################################################################################################
   
############################################## Write Data Obtained From Website to a *.txt And *.csv File ######################################################
cur_data.to_csv(currency + '.csv', index = False)                                        # Writes The Final 'cur_data' Variable In The dat File
                                                                                         # For Later Use In Matlab

cur_data.to_csv(currency + '.dat', index = False, header=False)                          # Writes The Final 'cur_data' Variable In The dat File
                                                                                         # For Later Use In Matlab ( Keys Are Eliminated)
# cur_data Variable Contains The BTC Data Table Wich Has 14 Columns And 1441 Rows Which Each Row Refers To a Specific Day

# Arrange Some Data For Plotting CandleStick Diagram, Notice That We Eliminate Last Data Sample And Try To Guess That.
cs_data_last = cur_data.iloc[1440,7:14]                                                  # Last Candle
cs_data      = cur_data.iloc[0:-1,7:14]                                                  # CandleStick Data
#cs_data    = cs_data.set_index('LocalTime')
ups_color = 'green'
downs_color = 'red'
# Setting Width Of Candlestick Elements
body_width = 0.98
shadow_width = .1



################################################################################################################################################################
################################################################################################################################################################

################################################################### Importing BTC Data #########################################################################
# BTC.csv Has 1441 Samples Of 14 Different Elements Whish We Only Need Some Of Them
filename = currency + '.csv'
btc_data = np.array(pd.read_csv(filename))                                               # BTC Data Frame
# Pay Attention That In This Dataset The Last Data Sample Is The Data Which Is On Going right Now And We Don't Need That!
# We Need To Guess It !
# t_btc Is TimeStep Of The Data And Is Used For Plotting Charts
t_btc = btc_data[0:-1,0].astype('float')
#t_btc = btc_data[:,0].astype('float')
# Select 6 Columns Of BTC Data Which Corresponds To The Values Of
# 'High', 'Low', 'Open', 'Close', 'VolumeFrom', 'VolumeTo',
# x_btc Is A 2D Array
x_btc = btc_data[0:-1, 8:14].astype('float')
#x_btc = btc_data[0:-1, 8:14].astype('float')
# Output of Our Network Will Be The Low And High Value Of Next Step
targets = btc_data[0:-1, 8:14].astype('float')
#targets = btc_data[0:-1, 8:14].astype('float') # Targets
################################################################################################################################################################
################################################################################################################################################################

################################################################# Assigning x And y Matrices ###################################################################
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
################################################################################################################################################################
################################################################################################################################################################

################################################################ Calculating Delay Parameters ##################################################################
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
################################################################################################################################################################
################################################################################################################################################################

####################################################################### Saving Raw Data ########################################################################
save = np.zeros((nSamples - max_delay - h_step) * (nInputs * nDelays + nOutputs)
                ).reshape((nSamples - max_delay - h_step), (nInputs * nDelays + nOutputs))
save[:,0:(nInputs * nDelays)] = x_train[:,0,:]
save[:,(nInputs * nDelays):] = y_train[:,:]
pd.DataFrame(save).to_csv('Data_Raw.csv', index=False, header=False)
################################################################################################################################################################
################################################################################################################################################################

######################################################################## Shuffling Data ########################################################################
# Here In Order To Prevent Data From Getting Mixed, First We Put The Data Together Horizontally
# Then Generate A Random Permutaation
temp_io = np.zeros((nSamples - max_delay - h_step) * (nInputs * nDelays + nOutputs)
                ).reshape((nSamples - max_delay - h_step), (nInputs * nDelays + nOutputs))
temp_io[:,0:(nInputs * nDelays)] = x_train[:,0,:]
temp_io[:,(nInputs * nDelays):] = y_train[:,:]
temp_io = np.random.permutation(temp_io)
x_train[:,0,:] = temp_io[:,0: (nInputs * nDelays)]
y_train = temp_io[:,(nInputs * nDelays):]
################################################################################################################################################################
################################################################################################################################################################

################################################################## Saving Permuted (Shuffled) Data #############################################################
save = np.zeros((nSamples - max_delay - h_step) * (nInputs * nDelays + nOutputs)
                ).reshape((nSamples - max_delay - h_step), (nInputs * nDelays + nOutputs))
save[:,0:(nInputs * nDelays)] = x_train[:,0,:]
save[:,(nInputs * nDelays):] = y_train[:,:]
pd.DataFrame(save).to_csv('Data_Permute.csv', index=False, header=False)
################################################################################################################################################################
################################################################################################################################################################

######################################################################### Normalizing Data #####################################################################
min_x_train = x_train[:,0,:].min(axis=0)
max_x_train = x_train[:,0,:].max(axis=0)
min_y_train = y_train.min(axis=0)
max_y_train = y_train.max(axis=0)
normalized_x_train = x_train
normalized_x_train[:,0,:] = normalize(x_train[:,0,:], min_x_train, max_x_train)          # Normalizing Data To The Period 
normalized_y_train = y_train
normalized_y_train = normalize(normalized_y_train, min_y_train, max_y_train)             # Of -1 To 1
################################################################################################################################################################
################################################################################################################################################################

################################################################### Defining x_train And y_train ###############################################################
x_train = normalized_x_train
y_train = normalized_y_train
################################################################################################################################################################
################################################################################################################################################################

################################################################### Saving Normalized Final Data ###############################################################
save = np.zeros((nSamples - max_delay - h_step) * (nInputs * nDelays + nOutputs)
                ).reshape((nSamples - max_delay - h_step), (nInputs * nDelays + nOutputs))
save[:,0:(nInputs * nDelays)] = x_train[:,0,:]
save[:,(nInputs * nDelays):] = y_train[:,:]
pd.DataFrame(save).to_csv('Data_Normal.csv', index=False, header=False)
################################################################################################################################################################
################################################################################################################################################################

############################################ Defining Network Structure And Assigning Activation Functions Of Each Layer #######################################
net = Network()
net.add(FCLayer(nDelays * nInputs, 9))
net.add(ActivationLayer(gauss, gauss_prime))
net.add(FCLayer(9, 9))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(9, nOutputs))
net.add(ActivationLayer(tanh, tanh_prime))
################################################################################################################################################################
################################################################################################################################################################

############################################################# Using Prepared Data For Training Network #########################################################
net.use(mse, mse_prime)

epochs = 2000
learning_rate = 0.1
goal_error = 0#1e-6
net.train(x_train, y_train, epochs, learning_rate, goal_error)


######################################################### Testing The Final Network With The Original Inputs ###################################################
x_test = copy.deepcopy(x_train_orig)
x_test = normalize(x_test[:,0,:], min_x_train, max_x_train)
out = net.predict(x_test)
out = np.squeeze(np.array(out))
out = denormalize(out, min_y_train, max_y_train)
################################################################################################################################################################
################################################################################################################################################################

######################################################### Evaluating h_step Ahead Value Using Trained Network ##################################################
# Here We Should Notice That The Index:1272 Is Our Latest Time And Index:1273 Will Be The Next Step
x_stepahead = np.zeros(nDelays * nInputs).reshape(1, 1, nDelays * nInputs)
x_stepahead_temp = copy.deepcopy(x)
x_stepahead_temp = normalize(x_stepahead_temp, x_stepahead_temp.min(axis = 0), x_stepahead_temp.max(axis = 0))

for i in range(nDelays):
    x_stepahead[0,0,i * nInputs : (i+1) * nInputs] = x_stepahead_temp[nSamples - delays[i] - 1,:]

out_stepahead = net.predict(x_stepahead)
out_stepahead = denormalize(np.squeeze(out_stepahead).reshape(1,nOutputs), min_y_train, max_y_train)
################################################################################################################################################################
################################################################################################################################################################

########################################################### Printing Outputs (Next h_step Ahead Parameters) ####################################################

print('The ' + currency + ' Price Of Now Is : ' + str(cur_price[currency]['USD']) + ' $')
print('The High Price Of The Next Step Will Be : ' + str(out_stepahead[0,0]) + ' $')
print('The Low Price Of The Next Step Will Be : ' + str(out_stepahead[0,1]) + ' $')
print('The Open Price Of The Next Step Will Be : ' + str(out_stepahead[0,2]) + ' $')
print('The Close Price Of The Next Step Will Be : ' + str(out_stepahead[0,3]) + ' $')
print('The VolumeFrom Of The Next Step Will Be : ' + str(out_stepahead[0,4]))
print('The Volume To Of The Next Step Will Be : ' + str(out_stepahead[0,5]))
# Denormalization
# It Should Be Noted That The Denormalization Must Have Been Applied After All Calculations
# In Other Words, Neural Network Should Always Be Feed With The Normalized Data


#exit(0)
################################################################################################################################################################
################################################################################################################################################################

####################################################################### Demonstration And Plots ################################################################
# Plot The Original And Outputs Of Trained Nueral Network 
##for i in range(nOutputs):
##    plot.figure(i)
##    plot.plot(t_btc[max_delay : nSamples - h_step],x_btc[max_delay : nSamples - h_step, i], 'k', label = 'Actual Data')
##    plot.plot(t_btc[max_delay : nSamples - h_step],out_denormalized[:,i], 'r', label = 'ANN  Output Data')
##plot.legend(loc = 'best')
##plot.show(block=False)
#exit(0)

################################################################################################################################
################################################################################################################################

# Plotting The Original Raw Data In A CandleStick Chart
# "cs_ups" DataFrame Will Store The CandleStick Data('cs_data') When The Closing Price Is Greater Than Or Equal To The Opening Prices
cs_ups       = cs_data[cs_data.Close >= cs_data.Open]
# "cs_downs" DataFrame Will Store The CandleStick Data('cs_data') When The Closing Price Is Lesser Than The Opening Prices
cs_downs   = cs_data[cs_data.Close < cs_data.Open]


plot.figure()

# Plotting Up Candles
plot.bar(cs_ups.index, cs_ups.Close - cs_ups.Open, body_width, bottom = cs_ups.Open, color = ups_color)
plot.bar(cs_ups.index, cs_ups.High - cs_ups.Close, shadow_width, bottom = cs_ups.Close, color = ups_color)
plot.bar(cs_ups.index, cs_ups.Low - cs_ups.Open, shadow_width, bottom = cs_ups.Open, color = ups_color)


# Plotting Down Candles
plot.bar(cs_downs.index, cs_downs.Close - cs_downs.Open, body_width, bottom = cs_downs.Open, color = downs_color)
plot.bar(cs_downs.index, cs_downs.High - cs_downs.Open, shadow_width, bottom = cs_downs.Close, color = downs_color)
plot.bar(cs_downs.index, cs_downs.Low - cs_downs.Close, shadow_width, bottom = cs_downs.Open, color = downs_color)


# Rotating The x-axis Tick Labels Towards Right
plot.xticks(rotation=90, ha='right')#,  fontsize = 'xx-small')
plot.show(block=False)

################################################################################################################################
################################################################################################################################

# Plotting The Output Data Of Nueral Network In A CandleStick Chart
# Arrange Output Data For Plotting CandleStick Diagram Of Network Output.
# Putting 'out' Array Into A DataFrame
out_df        = pd.DataFrame(out,columns=['High', 'Low', 'Open', 'Close', 'VolumeFrom', 'VolumeTo'], index=range(max_delay + h_step ,nSamples))

# "out_ups" DataFrame Will Store The CandleStick Data('out_df') When The Closing Price Is Greater Than Or Equal To The Opening Prices
out_ups       = out_df[out_df.Close >= out_df.Open]
# "out_downs" DataFrame Will Store The CandleStick Data('out_df') When The Closing Price Is Lesser Than The Opening Prices
out_downs     = out_df[out_df.Close < out_df.Open]

# Plot CandleStick Chart
plot.figure()

out_sa_df    = pd.DataFrame(out_stepahead, columns=['High', 'Low', 'Open', 'Close', 'VolumeFrom', 'VolumeTo'])
# Plotting Up Candles
plot.bar(out_ups.index, out_ups.Close - out_ups.Open, body_width, bottom = out_ups.Open, color = ups_color)
plot.bar(out_ups.index, out_ups.High - out_ups.Close, shadow_width, bottom = out_ups.Close, color = ups_color)
plot.bar(out_ups.index, out_ups.Low - out_ups.Open, shadow_width, bottom = out_ups.Open, color = ups_color)

# Plotting Down Candles
plot.bar(out_downs.index, out_downs.Close - out_downs.Open, body_width, bottom = out_downs.Open, color = downs_color)
plot.bar(out_downs.index, out_downs.High - out_downs.Open, shadow_width, bottom = out_downs.Close, color = downs_color)
plot.bar(out_downs.index, out_downs.Low - out_downs.Close, shadow_width, bottom = out_downs.Open, color = downs_color)


# Rotating The x-axis Tick Labels Towards Right
plot.xticks(rotation=90, ha='right')#,  fontsize = 'xx-small')



# Plotting Only The h_step Ahead In A CandleStick Chart
# Plotting Up Of The h_step Ahead (Guess) Candle
out_sa_df     = pd.DataFrame(out_stepahead, columns=['High', 'Low', 'Open', 'Close', 'VolumeFrom', 'VolumeTo'], index = [1440])
# Plotting Candles
if (out_sa_df.Close.item() >= out_sa_df.Open.item()):
    plot.bar(out_sa_df.index, out_sa_df.Close - out_sa_df.Open, body_width, bottom = out_sa_df.Open, color = 'magenta')
    plot.bar(out_sa_df.index, out_sa_df.High - out_sa_df.Close, shadow_width, bottom = out_sa_df.Close, color = 'magenta')
    plot.bar(out_sa_df.index, out_sa_df.Low - out_sa_df.Open, shadow_width, bottom = out_sa_df.Open, color = 'magenta')
else:
    plot.bar(out_sa_df.index, out_sa_df.Close - out_sa_df.Open, body_width, bottom = out_sa_df.Open, color = 'blue')
    plot.bar(out_sa_df.index, out_sa_df.High - out_sa_df.Open, shadow_width, bottom = out_sa_df.Close, color = 'blue')
    plot.bar(out_sa_df.index, out_sa_df.Low - out_sa_df.Close, shadow_width, bottom = out_sa_df.Open, color = 'blue')

plot.show()
#exit(0)
################################################################################################################################
################################################################################################################################




# Notice The Normalization And Denormalization Process In Order Not To Have A Miscalculation


#direction(smooth_price)


