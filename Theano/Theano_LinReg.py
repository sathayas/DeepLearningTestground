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
alpha = 0.1
lamb = 0.01  # regularization parameter for the loss function

# generating random data
D = (np.random.randn(examples, features), 
     np.random.randn(examples))

# defining variables and such for Theano
x = T.dmatrix("x")
y = T.dvector("y")
# weights, initialized with random numbers
w = theano.shared(np.random.randn(features), name="w")
# bias, starts out with 0
b = theano.shared(0., name="b")

# the output is simply a linear combination of inputs * weights + bias
p = T.dot(x, w) + b

# error - squared error
error = squared_error(p,y)
loss = error.mean() + lamb * L2(w)
gw, gb = T.grad(loss, [w, b])


# training function
train = theano.function(inputs=[x,y],
                        outputs=[p, error], 
                        updates=((w, w - alpha * gw),(b, b - alpha * gb)))
# prediction function
predict = theano.function(inputs=[x], outputs=p)

# before training
print("RMSE before training:", 
      mean_squared_error(D[1],predict(D[0])))

# training
for i in range(training_steps):
    prediction, error = train(D[0], D[1])

# after training
print("RMSE after training:", 
      mean_squared_error(D[1],predict(D[0])))
