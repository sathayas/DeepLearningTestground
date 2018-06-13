import theano.tensor as T
from theano import function

# sigmoid
a = T.dmatrix('a')
f_a = T.nnet.sigmoid(a)
f_sigmoid = function([a],[f_a])
print "sigmoid:", f_sigmoid([[-1,0,1]])
