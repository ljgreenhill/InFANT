from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
import tensorflow as tf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', 200)

#based off of this example: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

def superviseLearning(data, timesteps):
	newArr = []
	newRow = []
	i = 0
	x = 0
	maxSamples = int(data.shape[0]/timesteps)
	maxSamples = (maxSamples-1) * timesteps
	while i < maxSamples:
		while x < timesteps:
			newRow.extend(data[i+x])
			x = x + 1
		newArr.append(newRow)
		newRow = []
		x = 0
		i = i + 1
	newArr = np.asarray(newArr)
	return newArr
			

def plot(actual, prediction):
	plt.figure(figsize=(16,6))
	plt.plot(actual, label='Actual',color='b',linewidth=3)
	plt.plot((prediction),  label='Prediction',color='y')       
	print("Plotting")
        plt.legend()
        plt.show()

#write to tensorboard
log_dir="logs/realLSTM/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

timesteps = 3
params = 5
samples = 500000

dataset = read_csv('mergedRandom.csv', header=0, usecols = ['people','time', 'src', 'dst', 'length', 'protocol'])
values = dataset.values
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

supervised = superviseLearning(scaled,timesteps)

values = supervised[:,:(supervised.shape[1]-1)]

test = values[samples:, :]

# split into input and outputs
n_obs = timesteps * params
test_X= test[:, :n_obs]

y = values[:,5]
test_y = y[samples:]

# reshape input to be 3D [samples, timesteps, features]
test_X = test_X.reshape((test_X.shape[0], timesteps, params))

model = load_model("supervised2.h5")

#predicts
yhat = model.predict(test_X)
plot(test_y, yhat)
