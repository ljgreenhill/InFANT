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
import matplotlib.pyplot as plt
import numpy as np

def plot(actual, prediction):
	plt.figure(figsize=(16,6))
	plt.plot(actual, label='Actual',color='b',linewidth=3)
	plt.plot((prediction),  label='Prediction',color='y')       
	print("Plotting")
        plt.legend()
        plt.show()

n_hours = 2
n_features = 5
samples = 500000

# load dataset
dataset = read_csv('merged.csv', header=0, usecols = ['time', 'src', 'dst', 'length', 'protocol', 'people'])
values = dataset.values

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

labels = labels[:(samples/n_hours)]

scaled = scaled[:samples]
reframed = np.reshape(scaled,(samples, n_features))
values = np.reshape(reframed,((samples/n_hours), n_hours,-1))

size = ((len(values))/n_hours)
sizeL = ((len(labels))/n_hours)

test_X = values[:size]
test_y = labels[:sizeL]

model = load_model("test50.h5")

#predicts
yhat = model.predict(test_X)
plot(test_y, yhat)

