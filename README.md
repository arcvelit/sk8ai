# What is Sk8ai
Sk8ai is a project aiming to build a basic modular neural  network model (**Skate**) on **Python 3.X**.

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

The `NeuralNetwork` NN would therefore have an input layer of 3 neurons, x1 hidden layer of 4 neurons, and an output layer of 2 neurons. As aforementioned, a Skate NN can take any form (you can also reveal the layers of this network with `NeuralNetwork.reveal()`)
