# What is sk8ai
Sk8ai is a project aiming to build a basic modular neural  network model (**Skate**) on **Python 3**.

# Downloading packages

Open your terminal and type:
```bash
pip install -i https://test.pypi.org/simple/ sk8ai
```

# Features
There are two main files in sk8ai: `sk8.py` and `maths.py`

1. **sk8.py** encapsulates the general NN model, its set-up and learning methods. This is from where you call the **Skate** (NN).
2. **maths.py** provides activation functions and their derivatives. It includes the *sigmoid*, *tanh*, *linear*, *ReLU* and *softplus* functions. Each of the functions featured in **maths.py** return a tuple of two lambda functions which are respectively: `(Activation,Derivative)`

# Skate neural net

The `sk8.Skate` class gives the opportunity of creating a neural network of this type:


![Image of Neural Network](https://upload.wikimedia.org/wikipedia/commons/e/e4/Artificial_neural_network.svg "Neural Network")

[en:User:Cburnett](https://www.google.com "Wikipedia"), [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/ "Creative commons"), via Wikipedia Commons

where you can **choose** the number of neurons in the input, output __AND__ hidden layers. Obviously it is possible to build a NN with as many layers and neurons as necessary. For instance, to set-up a NN identical to the one in the image one can write:

```python
from sk8ai.sk8 import Skate

layers = (3,4,2)
NeuralNetwork = Skate(layers)
```

The `NeuralNetwork` NN would therefore have an input layer of 3 neurons, x1 hidden layer of 4 neurons, and an output layer of 2 neurons. As aforementioned, a Skate NN can take any form (you can also print the layers of this network with `NeuralNetwork.reveal()`)

# Applications (classification of MNIST digits databse)

Using Sk8ai's neural network model, we can perform the "Hello World!" of artificial intelligence: recognizing 28x28 images of handwritten digits from the mnist database. In this case, the TensorFlow library was used, but other ones can provide the same datasets.

```python
from sk8ai.sk8 import Skate
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
neuralnet = Skate((784,20,20,10))
neuralnet.train(1,X_train/255,Y_train)
print(neuralnet.accuracy(X_test/255,Y_test))
```

This neural network format (<784 20 20 10>) is pretty accurate. In the worst cases, it returns an accuracy of ~91%; in the best cases, ~93%. Configurations like <784 16 10>, <784 10 10> and <784 10> can return lower accuracies (≤90%) but are significantly faster than 1-2+ layered networks.

# Pending upgrades...

Implementing epoch & batches, multiprocessing in cases of deeplearning networks, other mild optimizations.
