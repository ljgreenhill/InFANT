import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import random
import os
import csv
import datetime
import matplotlib.pyplot as plt

#based off of this example: https://medium.com/datadriverninvestor/a-simple-way-to-know-how-important-your-input-is-in-neural-network-86cbae0d3689

def plot(actual, original, time, source, destination, length, protocol):
	plt.figure(figsize=(16,6))
        plt.plot(actual, label='Actual',color='b',linewidth=3)
	plt.plot(time,  label='Time',color='g')
	plt.plot(source,  label='Source',color='r')
	plt.plot(destination,  label='Destination',color='c')
	plt.plot(length,  label='length',color='k')
	plt.plot(length,  label='protocol',color='m')
	plt.plot(original,  label='Original',color='y')
        plt.legend()
        plt.show()

def eliminateInput(arrParam,arrIndex):
	for i in range(len(arrParam)):
		for x in range(len(arrIndex)):
			arrParam[i,arrIndex[x]] = 9999
	return arrParam

def printPredictions(actual, prediction):
	for i in range(len(actual)):
		print("ACTUAL: " + str(actual[i]) + " --- PREDICTION: " + str(prediction[i]))

def normalizeData(trainparam, trainlabel, testparam, testlabel):
	trainparamnorm = np.zeros(np.shape(trainparam)).astype('float32')
	trainlabelnorm = np.zeros(np.shape(trainlabel)).astype('float32')
	testparamnorm = np.zeros(np.shape(testparam)).astype('float32')
	testlabelnorm = np.zeros(np.shape(testlabel)).astype('float32')

	for i in xrange(len(trainparam[0])-2):
		if (np.max(trainparam[:,i]) - np.min(trainparam[:,i])) != 0:
			trainparamnorm[:,i] = (trainparam[:,i] - np.min(trainparam[:,i])) / (np.max(trainparam[:,i]) - np.min(trainparam[:,i]))
    			testparamnorm[:,i] = (testparam[:,i] - np.min(trainparam[:,i])) / (np.max(trainparam[:,i]) - np.min(trainparam[:,i]))

	for i in xrange(2):
		trainparamnorm[:,-2+i] = (trainparam[:,-2+i] - 0.0) / (20.0 - 0.0)
		testparamnorm[:,-2+i] = (testparam[:,-2+i] - 0.0) / (20.0 - 0.0)

	trainlabelnorm = (trainlabel - np.min(trainlabel)) / (np.max(trainlabel) - np.min(trainlabel))
	testlabelnorm = (testlabel - np.min(trainlabel)) / (np.max(trainlabel) - np.min(trainlabel))

	return testparamnorm, testlabelnorm

def getError(index, batch_size, trainparam, trainlabel, testparam, testlabel,mod):
	if index != -1:
		trainparam = eliminateInput(trainparam, index)
        testparamnorm, testlabelnorm = normalizeData(trainparam, trainlabel, testparam, testlabel)
	pred = mod.predict(testparamnorm, batch_size=batch_size)
	error = mean_squared_error(testlabelnorm, pred)
	return pred, error

#block tensorflow backend messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

datafile = 'trainMultiple.csv'
model = 'pcapModelMultiple3.h5'

batch_size = 10
hidden_neuron = 10
trainsize = 160000
iterasi = 200

#read csv
alldata = np.genfromtxt(datafile,delimiter=',')[1:]

#separate between training and test data
trainparam = alldata[:trainsize, :-1]
trainlabel = alldata[:trainsize, -1]

testparam = alldata[trainsize:, :-1]
testlabel = alldata[trainsize:, -1]

trainparam = trainparam[len(trainparam)%batch_size:]
trainlabel = trainlabel[len(trainlabel)%batch_size:]

testparam = testparam[len(testparam)%batch_size:]
testlabel = testlabel[len(testlabel)%batch_size:]

#load trained model
mod = load_model(model)

#print error
poriginal, eoriginal = getError(-1, batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Original MSE: ' + str(eoriginal))

ptime, etime = getError([0], batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Without Time: ' + str(etime))

psource, esource = getError([1], batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Without Source: ' + str(esource))

pdestination, edestination = getError([2], batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Without Destination: ' + str(edestination))

plength, elength = getError([3], batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Without Length: ' + str(elength))

pprotocol, eprotocol = getError([4], batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Without Protocol: ' + str(eprotocol))

ptimesource, etimesource = getError([0,1], batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Without Time, Source: ' + str(etimesource))

ptimedestination, etimedestination = getError([0,2], batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Without Time, Destination: ' + str(etimedestination))

ptimelength, etimelength = getError([0,3], batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Without Time, Length: ' + str(etimelength))

ptimeprotocol, etimeprotocol = getError([0,4], batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Without Time, Protocol: ' + str(etimeprotocol))

psourcedestination, esourcedestination = getError([1,2], batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Without Source, Destination: ' + str(esourcedestination))

psourcelength, esourcelength = getError([1,3], batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Without Source, Length: ' + str(esourcelength))

psourceprotocol, esourceprotocol = getError([1,4], batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Without Source, Protocol: ' + str(esourceprotocol))

pdestinationlength, edestinationlength = getError([2,3], batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Without Destination, Length: ' + str(edestinationlength))

pdestinationprotocol, edestinationprotocol = getError([2,4], batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Without Destination, Protocol: ' + str(edestinationprotocol))

plengthprotocol, elengthprotocol = getError([3,4], batch_size, trainparam.copy(), trainlabel, testparam, testlabel,mod)
print('Without Length, Protocol: ' + str(elengthprotocol))

#plot data
plot(testlabel,poriginal, ptime, psource, pdestination, plength, pprotocol)

