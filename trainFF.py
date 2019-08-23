import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, LSTM, RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import datetime
import csv

#based off of this example: https://medium.com/datadriverninvestor/a-simple-way-to-know-how-important-your-input-is-in-neural-network-86cbae0d3689

#settings
datafile = 'trainMultiple.csv'
pcapModel = 'trainsize200000.h5'
batch_size = 10
hidden_neuron = 10
trainsize = 200000
iterasi = 5

def generatemodel(totvar):
    # create and fit the LSTM network
    model = Sequential()
    model.add(Dense(3, batch_input_shape=(batch_size, totvar), activation='sigmoid'))
    model.add(Dense(hidden_neuron, activation='sigmoid'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#write to tensorboard
log_dir="logs/new/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#read data
alldata = np.genfromtxt(datafile,delimiter=',')[1:]

#separate between training and test
trainparam = alldata[:trainsize, :-1]
trainlabel = alldata[:trainsize, -1]

testparam = alldata[trainsize:, :-1]
testlabel = alldata[trainsize:, -1]

trainparam = trainparam[len(trainparam)%batch_size:]
trainlabel = trainlabel[len(trainlabel)%batch_size:]

testparam = testparam[len(testparam)%batch_size:]
testlabel = testlabel[len(testlabel)%batch_size:]

#normalization
trainparamnorm = np.zeros(np.shape(trainparam))
trainlabelnorm = np.zeros(np.shape(trainlabel))
testparamnorm = np.zeros(np.shape(testparam))
testlabelnorm = np.zeros(np.shape(testlabel))

#for param
for i in xrange(len(trainparam[0])-2):
    trainparamnorm[:,i] = (trainparam[:,i] - np.min(trainparam[:,i])) / (np.max(trainparam[:,i]) - np.min(trainparam[:,i]))
    testparamnorm[:,i] = (testparam[:,i] - np.min(trainparam[:,i])) / (np.max(trainparam[:,i]) - np.min(trainparam[:,i]))

for i in xrange(2):
    print("in range")
    trainparamnorm[:,-2+i] = (trainparam[:,-2+i] - 0.0) / (20.0 - 0.0)
    testparamnorm[:,-2+i] = (testparam[:,-2+i] - 0.0) / (20.0 - 0.0)

#for label
trainlabelnorm = (trainlabel - np.min(trainlabel)) / (np.max(trainlabel) - np.min(trainlabel))
testlabelnorm = (testlabel - np.min(trainlabel)) / (np.max(trainlabel) - np.min(trainlabel))

mod = generatemodel(len(trainparamnorm[0]))

mod.fit(trainparamnorm, trainlabelnorm, epochs=iterasi, batch_size=batch_size, verbose=2, shuffle=True, callbacks=[tensorboard_callback])

#save trained model
mod.save(pcapModel)
