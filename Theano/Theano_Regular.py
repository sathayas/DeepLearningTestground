import theano.tensor as T
from theano import function

# L1 Regularization
def L1(x):
    return T.sum(abs(x))

# L2 Regularization
def L2(x):
    return T.sum(x**2)

# Calling L1 function
a = T.dmatrix('a')
f_a = L1(a)
f_L1 = function([a], f_a)

print("L1 Regularization:", f_L1([[0,1,2,3,4,5]]))



# Calling L2 function
b = T.dmatrix('b')
f_b = L2(b)
f_L2 = function([b], f_b)

print("L2 Regularization:", f_L2([[0,1,2,3,4,5]]))
