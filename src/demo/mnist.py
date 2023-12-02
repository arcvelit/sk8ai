from tensorflow.keras.datasets import mnist
from sk8ai.sk8 import Skate
from PIL import Image
import numpy as np


def load_image_as_numpy_array(file_path):
    # Get image and convert as greyscale
    img = Image.open(file_path).convert('L')
    return np.array(img)/255.0

# Importing dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# One-hot prep function
def prep(tag):
    one_hot = np.zeros((10,1))
    one_hot[tag] = 1.0
    return one_hot

# One-hot prepping
Y_train = [prep(_tag) for _tag in Y_train]
Y_test = [prep(_tag) for _tag in Y_test]

# Network training and accuracy
neuralnet = Skate((784,20,20,10))
neuralnet.reveal()
neuralnet.train(1,X_train/255,Y_train)

# Test the model
accuracy = neuralnet.accuracy(X_test/255,Y_test)
print("Accuracy of the model: %.3f" % accuracy)


while (True):

    # Training was done on inversed greyscales
    # Feel free to open paint :^)
    digit_array = 1.0 - load_image_as_numpy_array('digit.png')

    prediction = neuralnet.predict(digit_array)
    digit_predicted = np.argmax(prediction) 

    print("Recognized digit: [%i]" % digit_predicted)

    # User can save new image and start over
    input()

