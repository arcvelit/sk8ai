from numpy import argmax, matmul, zeros, array, ndarray
from numpy.random import standard_normal
from itertools import count
from json import dump
import maths

class Skate:

	def __init__(self, layers: tuple, act=maths.sigmoid):
		self.act, self.act_prime = act()
		self.layers = layers
		if not layers: return
		try:
			if len(layers)==1: raise AssertionError("Invalid number of layers (value: 1)")
		except:
			raise AssertionError("Invalid number of layers (value: 1)")
		assert (err:=min(layers))>0, "Invalid number of neurons in layer #{} ({} value)".format(layers.index(err)+1,"negative" if err<0 else "zero") 
		self.weights = [standard_normal((m,n))/n**0.5 for m,n in zip(layers[1:],layers[:-1])]
		self.biases = [zeros((S,1)) for S in layers[1:]]
		
	# Training neural net - Core learning
	def _feedForward(self, learning_rate: {int, float}, inputs, expected):
		assert self.layers, "Cannot train an empty neural network, load the data first"
		inputs = inputs.reshape(inputs.size,1)
		activations = [inputs]
		# Forward propagation
		for W,B in zip(self.weights, self.biases):
			inputs = self.act(matmul(W, inputs)+B)
			activations.append(inputs)

		# Calculating error of output layer
		delta = (inputs - expected)*self.act_prime(inputs)/inputs.size

		# Backpropagation (synchronic gradient descent)
		for layer, W in enumerate(self.weights[::-1], start=1):
			# Training weights & biases
			self.weights[-layer] -= learning_rate*activations[-layer-1].transpose()*delta
			self.biases[-layer] -= learning_rate*delta
			# Propagating error 
			delta = matmul(W.transpose(),delta)*self.act_prime(activations[-layer-1])

	# Train neural network on dataset
	def train(self, learning_rate: {int, float}, input_train: list, output_train: list, with_name=""):
		try:
			counter = 0
			for i, in_sample, out_sample in zip(count(), input_train, output_train):
				self._feedForward(learning_rate, in_sample, out_sample)
				if (i+1) % (len(input_train)//10) == 0 and i+1 > len(input_train)%10 and counter<9:
					print("[{}]".format((counter:=counter+1)*"="+(10-counter)*" "), i+1)
		except ZeroDivisionError:
			pass
		print(f"[==========]", len(input_train))
		print("\nTraining completed\n")

		# If name given, save weights in .json file
		if with_name:
			self.save(with_name)

	# Engage a forward propagation
	def predict(self, inputs):
		inputs = inputs.reshape(inputs.size,1)
		for W,B in zip(self.weights, self.biases):
			inputs = self.act(matmul(W, inputs)+B)
		return inputs

	# Test accuracy (float) 
	def accuracy(self, input_test: list, output_test: list):
		correct = 0
		for input_sample, output_sample in zip(input_test, output_test):
			correct += argmax(output_sample) == argmax(self.predict(input_sample))
		return correct/len(input_test)

	# Save weights, biases & info of sk8 in JSON
	def save(self, filename="Skate_data"):
		filename += (not filename.endswith(".json"))*".json"
		with open(filename, "w") as file:
			dump({
				"layers": self.layers,
				"weights": [W.tolist() for W in self.weights],
				"biases": [B.tolist() for B in self.biases]
				}, file)

	# Load saved Skate
	def load(self, data: dict):
		self.layers = data["layers"]
		self.weights, self.biases = [], []
		for W,B,s1,s2 in zip(data["weights"],data["biases"],self.layers[1:], self.layers[:-1]):
			self.weights.append(array(W,dtype=float).reshape((s1,s2)))
			self.biases.append(array(B,dtype=float).reshape((s1,1)))
		print("Skate successfully imported")

	def reveal(self):
		size_array = " ".join(str(lay) for lay in self.layers) if self.layers else "empty"
		print(f"Neural net: <{size_array}>")


	@staticmethod
	def logo():
		print(
"""    
	.-..___
   (_)	   \"\"--..__
	\\\\ < sk8		 \"\":.__
	 \\\\		ai > (_)   \'\'\'._
	 (_)\'-..___	 \\\\		 .
			   \"\"-.__\\\\		  |
					 (_).____.\'
"""
			)


# GJLMA�N
