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
import tensorflow as tf
import datetime
import numpy as np
import matplotlib.pyplot as plt

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

dataset = read_csv('merged.csv', header=0, usecols = ['time', 'src', 'dst', 'length', 'protocol', 'people'])
values = dataset.values

# integer encode direction
encoder = LabelEncoder()
values[:,5] = encoder.fit_transform(values[:,5])

values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

labels = scaled.copy()
scaled = np.delete(scaled, 5, axis=1)
labels = np.delete(labels, 0, axis =1)
labels = np.delete(labels, 0, axis =1)
labels = np.delete(labels, 0, axis =1)
labels = np.delete(labels, 0, axis =1)
labels = np.delete(labels, 0, axis =1)

labels = scaler.fit_transform(labels)

timesteps = 2
params = 5
samples = 500000

labels = labels[:(samples/timesteps)]

scaled = scaled[:samples]
reframed = np.reshape(scaled,(samples, params))
values = np.reshape(reframed,((samples/timesteps), timesteps,-1))

size = ((len(values))/timesteps)
sizeL = ((len(labels))/timesteps)

train_X = values[size:]
test_X = values[:size]

train_y = labels[sizeL:]
test_y = labels[:sizeL]

model = Sequential()
#epochs, (timesteps, features)
model.add(LSTM(50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(30, return_sequences=True))
model.add(LSTM(10))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

history = model.fit(train_X, train_y, epochs=5, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False, callbacks=[tensorboard_callback])

model.save("test50.h5")
