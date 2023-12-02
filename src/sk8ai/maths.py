from numpy import tanh, maximum, log, exp

# Sigmoid activation and rel. derivative
def sigmoid():
	return lambda x: 1./(1.+ exp(-x)), lambda x: x* (1.-x)

# Hyperbolic tangent activation and rel. derivative
def tanh():
	return lambda x: tanh(x), lambda x: 1.-tanh(x)**2

# ReLU activation and derivative
def ReLU():
	return lambda x: maximum(0,x), lambda x: x>0

# Softplus or Smooth ReLU activation and rel. derivative
def softplus():
	return lambda x: log(1.+exp(x)), lambda x: 1./(1.+exp(-x))

# Linear activation and derivative
def linear():
	return lambda x: x, lambda x: 1

