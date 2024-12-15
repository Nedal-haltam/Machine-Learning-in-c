
'''
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
rng = np.random.default_rng()

def mse(pred, expected):
    return np.mean((pred - expected)**2)
def mse_dir(pred, expected):
    return 2 * (pred - expected)
def LossCCE(y_pred, y_true):
    samples = len(y_pred)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

    if len(y_true.shape) == 1:
        correct_confidences = y_pred_clipped[range(samples), y_true]

    elif len(y_true.shape) == 2:
        correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

    negative_log_likelihoods = -np.log(correct_confidences)
    data_loss = np.mean(negative_log_likelihoods)
    return data_loss
def sigmoid(inputs):
    return 1/(1 + np.exp(-inputs))
def sigmoid_dir(inputs):
    return sigmoid(inputs)*(1 - sigmoid(inputs))
def Activation_ReLU(inputs):
    return np.maximum(0, inputs)
def Activation_ReLU_dir(inputs):
    c = []
    for a in inputs:
        if (a > 0):
            c.append(1)
        else:
            c.append(0)
    return c
def Activation_Softmax(inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities




class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.numofin = n_inputs
        self.numofout = n_neurons
        self.weights = rng.standard_normal((n_inputs, n_neurons)) / n_inputs**0.5
        self.biases = rng.standard_normal((1, n_neurons))
        self.costgradW = np.zeros(self.weights.shape)
        self.costgradB = np.zeros(self.biases.shape)
    def forward(self, inputs, ac_index):
        self.output = np.dot(inputs, self.weights) + self.biases
        # self.acoutput = sigmoid(self.output)
        if ac_index == 0:
            self.acoutput = Activation_ReLU(self.output)
        elif ac_index == 1:
            self.acoutput = Activation_Softmax(self.output)

    def applygrad(self, LearRate):
        self.biases -= LearRate * self.costgradB
        self.weights -= LearRate * self.costgradW


class NN:
    def __init__(self, layer_struct):
        self.layers = [Layer(a,b) for a,b in zip(layer_struct[:-1], layer_struct[1:])]
    def forward(self, inputs):
        temps = inputs
        retoutput = []
        retacoutput = [inputs]
        for i in range(len(self.layers)):
            self.layers[i].forward(temps, i == len(self.layers) - 1)
            temps = self.layers[i].acoutput
            retacoutput.append(temps)
            retoutput.append(self.layers[i].output)
        return retoutput, retacoutput

    def print_accuracy(self, inputs, lbls):
        self.forward(inputs)
        pred = self.layers[-1].acoutput
        compare = [np.argmax(a) == np.argmax(b) for a,b in zip(pred, lbls)]
        print("{0}/{1} accuracy: {2}%".format(sum(compare), len(compare), sum(compare) / len(compare)*100))
    
    # def evaluate(self, test_data):
    #     test_results = [(np.argmax(self.forward(x)), y)
    #                     for (x, y) in test_data]
    #     return sum(int(x == y) for (x, y) in test_results)

    def cost(self, inputs, expected):
        pred = self.forward(inputs)
        self.lastloss = LossCCE(pred, expected)
        return self.lastloss


    def backprop(self, x, y):
        nabla_b = [np.zeros(l.biases.shape) for l in self.layers]
        nabla_w = [np.zeros(l.weights.shape) for l in self.layers]
        # feedforward
        zs, activations = self.forward(x)
        # backward pass
        delta = mse_dir(activations[-1], y) * sigmoid_dir(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta.T, activations[-2])
        for l in range(2, len(self.layers)):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, LearRate):
        nabla_b = [np.zeros(l.biases.shape) for l in self.layers]
        nabla_w = [np.zeros(l.weights.shape) for l in self.layers]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b += delta_nabla_b
            nabla_w += delta_nabla_w
        
        for i in range(len(self.layers)):
            self.layers[i].costgradW = nabla_w[i]
            self.layers[i].costgradB = nabla_b[i]
            self.layers[i].applygrad(LearRate/len(mini_batch))

    def SGD(self, training_data, epochs, mini_batch_size, LearRate,inputs,expected, test_data=False):
        if test_data: n_test = len(inputs)

        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, LearRate)

            if test_data:
                self.print_accuracy(inputs, expected)
                # print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))






with np.load("mnist.npz") as data:
    timg = np.hstack(data["training_images"]).T
    tlbl = np.hstack(data["training_labels"]).T
    testimg = np.hstack(data["test_images"]).T
    testlbl = np.hstack(data["test_labels"]).T
    valimg = np.hstack(data["validation_images"]).T
    vallbl = np.hstack(data["validation_labels"]).T

train    = [a for a in zip(timg, tlbl)]
test     = [a for a in zip(testimg, testlbl)]
validate = [a for a in zip(valimg, vallbl)]


nn_struct = (784, 15, 10)
nn = NN(nn_struct)
epochs = 10
mini_batch_size = 10
LearRate = 3

nn.SGD(train, epochs, mini_batch_size, LearRate, testimg, testlbl, True)

evaluation = s.evaluate(test)
print("test: {0}/{1}, {2}%".format(evaluation, len(test), evaluate/len(test)))
'''
# for i in range(batch_size):
#     minitimg = timg[i*16:(i+1)*16]
#     minitlbl = tlbl[i*16:(i+1)*16]
#     nn.learn(minitimg, minitlbl, LearRate)


import random
import numpy as np
import matplotlib.pyplot as plt

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        i = 0
        for b, w in zip(self.biases, self.weights):
            if i != len(self.biases) - 1:
                a = hiddenlayer_activation(np.dot(w, a)+b)
            else:
                a = outputlayer_activation(np.dot(w, a)+b)
            i += 1
        
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, RegParam,test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(3):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, RegParam, n)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
            print("loss : ", sum([sum(((self.feedforward(x)) - y)**2) for x,y in training_data])/n)

    def update_mini_batch(self, mini_batch, eta, RegParam, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(RegParam/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = hiddenlayer_activation(z)  
            activations.append(activation)
        activations.pop()
        activations.append(outputlayer_activation(zs[-1])) 
        # backward pass
        # delta = self.mse_dir(activations[-1], y) * sigmoid_prime(zs[-1]) 
        delta = self.crossentropy_dir(activations[-1], y)   # * outputlayer_activation_dir(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) 
        for l in range(2, self.num_layers):
            z = zs[-l]
            activation_prime = hiddenlayer_activation_dir(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * activation_prime
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) 
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        lbls = [y for x,y in test_data]
        pred = []
        for x,y in test_data:
            temp = self.feedforward(x)
            pred.append(temp)
        compare = []
        if (np.array(pred)).shape[1] == 1:
            for a,b in zip(pred, lbls):
                if abs(a - b) < 0.1:
                    compare.append(1)
        else:
            compare = [np.argmax(a) == np.argmax(b) for a,b in zip(pred, lbls)]
        return sum(compare)

    def test_nn(self, test_data):
        outputs = [self.feedforward(x) for x, y in test_data]
        return outputs

    def mse_dir(self, output_activations, y):
        return 2 * (output_activations-y)

    def crossentropy_dir(self, output_activations, y):
        return output_activations - y

    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def load(filename):
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        cost = getattr(sys.modules[__name__], data["cost"])
        net = Network(data["sizes"], cost=cost)
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net   

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
def Activation_ReLU(inputs):
    return np.maximum(0, inputs)
def Activation_ReLU_dir(inputs):
    return [[int(b > 0) for b in a] for a in inputs]
def Activation_LeakyReLU(inputs):
    return np.maximum(0.1*inputs, inputs)
def Activation_LeakyReLU_dir(inputs):
    return [[0 if b > 0 else 0.1 for b in a] for a in inputs]
def Activation_Softmax(inputs):
    if (np.array(inputs)).shape[0] == 1:
        return normalized_tanh(inputs)
    exp_values = np.exp(inputs - np.max(inputs))
    probabilities = exp_values / np.sum(exp_values)
    return probabilities
def normalized_tanh(inputs):
    return (((np.exp(inputs) - np.exp(-inputs))/(np.exp(inputs) + np.exp(-inputs))) + 1) / 2
def normalized_tanh_dir(inputs):
    return 1 - (normalized_tanh(inputs) ** 2)

def outputlayer_activation(inputs):
    # return Activation_Softmax(inputs)
    return normalized_tanh(inputs)
def outputlayer_activation_dir(inputs):
    return normalized_tanh_dir(inputs)
def hiddenlayer_activation(inputs):
    return Activation_ReLU(inputs)
    # return Activation_LeakyReLU(inputs)
    # return normalized_tanh(inputs)
def hiddenlayer_activation_dir(inputs):
    return Activation_ReLU_dir(inputs)
    # return Activation_LeakyReLU_dir(inputs)
    # return normalized_tanh_dir(inputs)



                








# best one is relu -> softmax / with crossentropy / with good hyper-parameter (LearRate, NN Size, mini batch size)

with np.load("mnist.npz") as data:
    timg = data["training_images"]
    tlbl = data["training_labels"]
    testimg = data["test_images"]
    testlbl = data["test_labels"]
    valimg = data["validation_images"]
    vallbl = data["validation_labels"]

training_data = [a for a in zip(timg, tlbl)]
test_data = [a for a in zip(testimg, testlbl)]

nn_struct = (784, 30, 10)
nn = Network(nn_struct)
epochs = 50
mini_batch_size = 10
LearRate = 0.05
RegParam = 5


# in_ = np.array([  [[0],[0]]  ,
#                   [[0],[1]]  ,
#                   [[1],[0]]  ,
#                   [[1],[1]]  ])
# out_ = np.array([[0], [1], [1], [0]])

# training_data = [a for a in zip(in_, out_)]
# test_data = [a for a in zip(in_, out_)]

# nn_struct = (2, 2, 1)
# nn = Network(nn_struct)
# epochs = 10*1000
# mini_batch_size = 1
# LearRate = 0.5
# RegParam = 0

nn.SGD(training_data, epochs, mini_batch_size, LearRate, RegParam, test_data=test_data)


# x = []
# y = []
# z = []
# in_ = np.array([[[0],[0]]])
# for i in range(10):
#     for j in range(10):
#         xcoord = i*0.1
#         ycoord = j*0.1
#         in_ = np.array([  [[xcoord],[ycoord]]  ])
#         out_ = np.array([[0]])
#         test_data = [a for a in zip(in_, out_)]
#         nnout = nn.test_nn(test_data)
#         x.append(xcoord)
#         y.append(ycoord)
#         z.append(nnout[0][0][0])


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(x, y, z);
# plt.show()



'''
import numpy as np 
import nnfs
import pandas as pd 
from matplotlib import pyplot as plt
from nnfs.datasets import spiral_data

rng = np.random.default_rng()

def sd(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


def Activation_ReLU(inputs):
    return np.maximum(0, inputs)
def Activation_ReLU_dir(inputs):
    c = []
    for a in inputs:
        if (a > 0):
            c.append(1)
        else:
            c.append(0)
    return c
def Activation_Softmax(inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities
def sigmoid(inputs):
    return 1/(1 + np.exp(-inputs))
def sigmoid_dir(inputs):
    return sigmoid(inputs)*(1 - sigmoid(inputs))
def LossCCE(y_pred, y_true):
    samples = len(y_pred)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

    if len(y_true.shape) == 1:
        correct_confidences = y_pred_clipped[range(samples), y_true]

    elif len(y_true.shape) == 2:
        correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

    negative_log_likelihoods = -np.log(correct_confidences)
    data_loss = np.mean(negative_log_likelihoods)
    return data_loss
def costmse(y_pred, y_true):
    c = (y_pred - y_true) ** 2
    return np.mean(c)
def costmse_dir(y_pred, y_true):
    return 2 * (y_pred - y_true)

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.numofin = n_inputs
        self.numofout = n_neurons
        self.weights = rng.standard_normal((n_inputs, n_neurons)) / n_inputs**0.5
        self.biases = np.zeros((1, n_neurons))
        self.costgradW = np.zeros(self.weights.shape)
        self.costgradB = np.zeros(self.biases.shape)
    def forward(self, inputs, ac_index):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.acoutput = sigmoid(self.output)
        if ac_index == 0:
            self.acoutput = Activation_ReLU(self.output)
        elif ac_index == 1:
            self.acoutput = Activation_Softmax(self.output)

    def applygrad(self, LearRate):
        self.biases -= LearRate * self.costgradB
        self.weights -= LearRate * self.costgradW

    def calcoutputlayernodevalues(self, expected):
        nodevalues = []
        for i in range(len(expected)):
            costdir = costmse_dir(self.acoutput[i], expected[i])
            acdir = sigmoid_dir(self.output)
            nodevalues.append(acdir * costdir)
        return nodevalues


class NN:
    def __init__(self, layer_struct):
        self.layers = [Layer(a,b) for a,b in zip(layer_struct[:-1], layer_struct[1:])]
    def forward(self, inputs):
        temps = inputs
        for i in range(len(self.layers)):
            self.layers[i].forward(temps, i == len(self.layers) - 1)  
            temps = self.layers[i].acoutput
        return temps

    def print_accuracy(self, inputs, lbls):
        pred = self.forward(inputs)
        compare = [np.argmax(a) == np.argmax(b) for a,b in zip(pred, lbls)]
        print("{0}/{1} accuracy: {2}%".format(sum(compare), len(compare), sum(compare) / len(compare)*100))

    def applyallgrad(self, inputs, expected, LearRate):
        # self.forward()
        # nodevalues = self.layers[len(self.layers)].calcoutputlayernodevalues(expected)
        for i in range(len(self.layers)):
            self.layers[i].applygrad(LearRate)

    def cost(self, inputs, expected):
        pred = self.forward(inputs)
        self.lastloss = LossCCE(pred, expected)
        return self.lastloss

    # def updateallgrad(self, input, expected):
        # self.forward(input)
        # layerslength = len(self.layers)
        # nodevalues = self.layers[layerslength - 1].calcoutputlayernodevalues(expected)
        # self.layers[layerslength - 1].updategrad(nodevalues)
        # for i in reversed(range(layerslength - 2)):



    def learn(self, inputs, expected, LearRate):
        h = 0.00001
        origcost = self.cost(inputs, expected)
        for i in range(len(self.layers)):
            for j in range(self.layers[i].numofin):
                for k in range(self.layers[i].numofout):
                    self.layers[i].weights[j][k] += h

                    delta = self.cost(inputs, expected) -  origcost
                    self.layers[i].weights[j][k] -= h
                    self.layers[i].costgradW[j][k] = delta / h

            for j in range(self.layers[i].numofout):
                self.layers[i].biases[0][j] += h
                delta = self.cost(inputs, expected) -  origcost
                self.layers[i].biases[0][j] -= h
                self.layers[i].costgradB[0][j] = delta / h
        self.applyallgrad(inputs, expected, LearRate)
        # inputslength = len(inputs)
        # for i in /rad(LearRate / inputslength)


# inputs, expected_outputs = spiral_data(samples=100, classes=3)

with np.load("mnist.npz") as data:
    timg = np.hstack(data["training_images"]).T
    tlbl = np.hstack(data["training_labels"]).T
    testimg = np.hstack(data["test_images"]).T
    testlbl = np.hstack(data["test_labels"]).T


nn_struct = (784,30,10)
nn = NN(nn_struct)
it = 50000//10
print(it)
for i in range(it):
    print("batch ", i)
    nn.learn(timg[i*10:(i+1)*10], tlbl[i*10:(i+1)*10], 0.1)
    print("Loss:", nn.lastloss)
    # print("test : ")
    # nn.print_accuracy(testimg, testlbl)





# loss = LossCCE(nn.layers[-1].acoutput, expected_outputs)
# print("Loss:", loss)


# data_path = "D:\\AIML\\NNfSiX-master\\Python\\train.csv\\train.csv"
# data = np.array(pd.read_csv(data_path))
# m, n = data.shape
# np.random.shuffle(data)

















# f = 2
# t = np.arange(0, 1, 0.001)
# y = np.sin(2*np.pi*f*t)
# plt.plot(x, y)
# plt.show()


# import matplotlib.pyplot as plt
# from scipy.fftpack import fft
# from scipy.io import wavfile # get the api
# import librosa
# import numpy as np
# import numpy.fft as npfft

# import pygame

# import math, cmath, time, cv2
# import audioread
# import numpy.fft as npfft, matplotlib.pyplot as plt
# import array, wave, playsound, scipy, pydub
# import pydub.playback, soundfile as sf, pygame
# import timer, pygame.mixer as pm, librosa, librosa.display
# import pyaudio
# import matplotlib.animation as animation
# from matplotlib import style
'''