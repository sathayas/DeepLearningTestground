import theano.tensor as T
from theano import function
import matplotlib.pyplot as plt
import numpy as np

# sigmoid
a = T.dmatrix('a')
f_a = T.nnet.sigmoid(a)
f_sigmoid = function([a],[f_a])
f_sigmoid_out = f_sigmoid([[-1,0,1]])
print("sigmoid:", f_sigmoid_out)
#plt.plot(np.arange(-5,5.1,0.5), f_sigmoid_out[0][0])
#plt.show()


# tanh
b = T.dmatrix('b')
f_b = T.tanh(b)
f_tanh = function([b],[f_b])
print("tanh:", f_tanh([[-1,0,1]]))


# fast sigmoid
c = T.dmatrix('c')
f_c = T.nnet.ultra_fast_sigmoid(c)
f_fast_sigmoid = function([c],[f_c])
print("fast sigmoid:", f_fast_sigmoid([[-1,0,1]]))


# softplus
d = T.dmatrix('d')
f_d = T.nnet.softplus(d)
f_softplus = function([d],[f_d])
f_softplus_out = f_softplus([np.arange(-5,5.1,0.5)])
#print("soft plus:",f_softplus_out)
plt.plot(np.arange(-5,5.1,0.5), f_softplus_out[0][0])
plt.title('softplus')
plt.show()


# relu
e = T.dmatrix('e')
f_e = T.nnet.relu(e)
f_relu = function([e],[f_e])
f_relu_out = f_relu([np.arange(-5,5.1,0.5)])
#print("relu:",f_relu_out[0][0])
plt.plot(np.arange(-5,5.1,0.5), f_relu_out[0][0])
plt.title('relu')
plt.show()


# softmax
f = T.dmatrix('f')
f_f = T.nnet.softmax(f)
f_softmax = function([f],[f_f])
f_softmax_out = f_softmax([np.arange(-5,5.1,0.5)])
#print("soft max:",f_softmax_out[0][0])
plt.plot(np.arange(-5,5.1,0.5), f_softmax_out[0][0])
plt.title('softmax')
plt.show()
