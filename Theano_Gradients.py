import theano.tensor as T
from theano import function
from theano import shared
import numpy as np

x = T.dmatrix('x')
y = shared(np.array([[4, 5, 6]]))
z = T.sum(((x * x) + y) * x)

f = function(inputs = [x], outputs = [z])

g = T.grad(z,[x])
g_f = function([x], g)

print("Original:", f([[1, 2, 3]]))  # the value of the function at x
print("Original Gradient:", g_f([[1, 2, 3]]))  # gradient at x


# updating the offset
y.set_value(np.array([[1, 1, 1]]))

# new gradients according to the new offset
print("Updated:", f([[1, 2, 3]]))
print("Updated Gradient", g_f([[1, 2, 3]]))

