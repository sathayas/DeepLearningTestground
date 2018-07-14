import numpy as np
import theano
import theano.tensor as T
from sklearn.metrics import accuracy_score

# L2 function
def L2(x):
    return T.sum(x**2)

# some parameters
examples = 1000
features = 100
training_steps = 2000
alpha = 0.1
lamb = 0.01

# generating random data
D = (np.random.randn(examples, features), 
     np.random.randint(size=examples,low=0, high=2))

# defining variables and such for Theano
x = T.dmatrix("x")
y = T.dvector("y")
# weights, initialized with random numbers
w = theano.shared(np.random.randn(features), name="w")
# bias, starts out with 0
b = theano.shared(0., name="b")

# logit function
p = 1 / (1 + T.exp(-T.dot(x, w) - b))

# error - binary cross entropy since binary outcome
error = T.nnet.binary_crossentropy(p,y)
loss = error.mean() + lamb * L2(w)
prediction = p > 0.5
gw, gb = T.grad(loss, [w, b])

# training function
train = theano.function(inputs=[x,y],
                        outputs=[p, error], 
                        updates=((w, w - alpha * gw),(b, b - alpha * gb)))

# prediction function
predict = theano.function(inputs=[x], outputs=prediction)


# before training
print("Accuracy before Training:",
      accuracy_score(D[1], predict(D[0])))

# training
for i in range(training_steps):
    prediction, error = train(D[0], D[1])

# after training
print("Accuracy after Training:",
      accuracy_score(D[1], predict(D[0])))
