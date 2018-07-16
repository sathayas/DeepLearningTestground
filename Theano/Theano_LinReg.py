import numpy as np
import theano
import theano.tensor as T
from sklearn.metrics import mean_squared_error

# L2 function
def L2(x):
    return T.sum(x**2)

# squred error function
def squared_error(x,y):
    return (x - y) ** 2


# some parameters
examples = 1000
features = 100
training_steps = 1000

# generating random data
D = (numpy.random.randn(examples, features), 
     numpy.random.randn(examples))
