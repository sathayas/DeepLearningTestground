import theano.tensor as T
from theano import function
import matplotlib.pyplot as plt

# binary cross entropy
a1 = T.dmatrix('a1')
a2 = T.dmatrix('a2')
f_a = T.nnet.binary_crossentropy(a1, a2)
f_sigmoid = function([a1, a2],[f_a])
print("Binary Cross Entropy [[0.01,0.01,0.01]],[[0.99,0.99,0.01]]:",
f_sigmoid([[0.01,0.01,0.01]],[[0.99,0.99,0.01]]))

