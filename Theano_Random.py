import theano.tensor as T
from theano import function
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

# initializing random streams
random = RandomStreams(seed=42)

# getting a normal from random stream, normal distribution
a = random.normal((1,3)) # here, (1,3) defines the dimensionality of the random data

# defining another data matrix
b = T.dmatrix('b')

# defining the function
f1 = a * b   # elementwise multiplication between a random vector (a)
             # and the data matrix (b) which is an imput to the function
g1 = function([b], f1)

# printing out different instances of the function
print("Invocation 1:", g1(np.ones((1,3))))
print("Invocation 2:", g1(np.ones((1,3))))
print("Invocation 3:", g1(np.ones((1,3))))
