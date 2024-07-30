# XOR Problem for a two layer neural network
# written by Tautvydas Lisas
# 30/06/2021

import numpy as np
import pandas as pd

# sigmoid activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# derivative of sigmoid activation function
def sigmoid_derivative(x):
    return (x)*(1-(x))

# initialise neural network class
class NeuralNetwork:
    def __init__(self, x, y):
        # initialise weights, input and output, the matrix size will determine the number of neurons each layer has
        self.input = x
        self.weights1 = np.random.rand(x.shape[1],4)
        self.weights2   = np.random.rand(4,1)  
        self.y = y
        self.output = np.zeros(y.shape)

    # forward propogation 
    def feedforward(self):
        self.layer1 = (sigmoid((np.dot(self.input, self.weights1))))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        #print(self.output)
        
    # backward propogation
    def backprop(self):
        # differential 
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

# XOR training data for two-layer NN
x = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# import Bow-tie data to test 2-layer network limitations
#data = pd.read_csv('bow_tie_dataset.csv', header=None)

# training data
#x = np.array(data)

# targets for a two layer NN, and three layer network 
y = np.array([[0],[1],[1],[0]])
#y = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])

nn = NeuralNetwork(x,y)

# set number of epochs
for i in range(500):
    nn.feedforward()
    nn.backprop()

print(np.round(nn.output))
