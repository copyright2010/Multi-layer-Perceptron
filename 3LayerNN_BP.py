# Bowtie problem for a three layer neural network
# written by Tautvydas Lisas
# 01/07/2021

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
        # initialise weights, input, output and target matrix. The matrix size will determine the number of neurons each layer has
        self.input = x
        print(x.shape[1])
        self.weights1 = np.random.rand(x.shape[1],15)
        self.weights2   = np.random.rand(15,5)
        self.weights3 = np.random.rand(5,1)
        self.y = y
        self.output = np.zeros(y.shape)

    # forward propogation 
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.output = sigmoid(np.dot(self.layer2, self.weights3))
        
    # backward propogation
    def backprop(self):
        d_weights3 = np.dot(self.layer2.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))

        d_weights2 = np.dot(self.layer1.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2)))

        d_weights1 =  np.dot(self.input.T,(np.dot(np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * \
            sigmoid_derivative(self.layer2), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights3 += d_weights3
        
# import Bowtie data
data = pd.read_csv('data.csv', header=None)
x = np.array(data)

# targets 
y = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])

nn = NeuralNetwork(x,y)

# set number of epochs
for i in range(50000):
    nn.feedforward()
    nn.backprop()

print(np.round(nn.output.T))

