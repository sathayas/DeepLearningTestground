import numpy as np
import theano
import theano.tensor as T
from sklearn.metrics import accuracy_score
from sklearn import datasets

# L2 function
def L2(x):
    return T.sum(x**2)

# some parameters
training_steps = 10000
alpha = 0.025
lamb = 0.01

# loading the breast cancer data from sklearn
brca = datasets.load_breast_cancer()
D = (brca.data, brca.target)
features = D[0].shape[1]
examples = D[0].shape[0]

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
      accuracy_score(D[1], predict(D[0]).astype(int)))

# training
for i in range(training_steps):
    prediction, error = train(D[0], D[1])

# after training
print("Accuracy after Training:",
      accuracy_score(D[1], predict(D[0]).astype(int)))
