from sk8ai.sk8 import Skate
from sk8ai.maths import *
from tensorflow.keras.datasets import mnist
from numpy import zeros

# Importing dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# One-hot prep function
def prep(tag):
    one_hot = zeros((10,1))
    one_hot[tag] = 1.0
    return one_hot

# One-hot prepping
Y_train = [prep(_tag) for _tag in Y_train]
Y_test = [prep(_tag) for _tag in Y_test]

# NN init, training and accuracy
neuralnet = Skate((784,20,20,10), linear )
neuralnet.train(1,X_train/255,Y_train)
print(neuralnet.accuracy(X_test/255,Y_test))
