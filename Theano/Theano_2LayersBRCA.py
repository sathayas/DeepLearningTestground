import numpy as np
import theano
import theano.tensor as T
from sklearn.metrics import accuracy_score
from sklearn import datasets
import matplotlib.pyplot as plt

# L2 function
def L2(x):
    return T.sum(x**2)

# loading the breast cancer data from sklearn
brca = datasets.load_breast_cancer()
D = (brca.data, brca.target)
features = D[0].shape[1]
examples = D[0].shape[0]
hidden = 10
training_steps = 10000
lamb = 0.01   # for regularization of weights in loss function
alpha = 0.05   # parameter for updating weights and biases

# generating random data
D = (np.random.randn(examples, features), 
     np.random.randint(size=examples,low=0, high=2))


# defining variables and such for Theano
x = T.dmatrix("x")
y = T.dvector("y")

# weights, initialized with random numbers
w1 = theano.shared(np.random.randn(features, hidden), name="w1")
w2 = theano.shared(np.random.randn(hidden), name="w2")

# bias, starts out with 0
b1 = theano.shared(np.zeros(hidden), name="b1")
b2 = theano.shared(0., name="b2")

# squashing functions -- tanh
p1 = T.tanh(T.dot(x, w1) + b1)
p2 = T.tanh(T.dot(p1, w2) + b2)

# error function -- binary cross entropy for binary outcome
error = T.nnet.binary_crossentropy(p2,y)

# loss function -- combination of binary cross entro and L2 regularization
loss = error.mean() + lamb * (L2(w1) + L2(w2))

# prediction is whether prob is greather than 0.5
prediction = p2 > 0.5

# gradients
gw1, gb1, gw2, gb2 = T.grad(loss, [w1, b1, w2, b2])

# training function
train = theano.function(inputs=[x,y],
                        outputs=[p2, error], 
                        updates=((w1, w1 - alpha * gw1),
                                 (b1, b1 - alpha * gb1), 
                                 (w2, w2 - alpha * gw2), 
                                 (b2, b2 - alpha * gb2)))

# Prediction function
predict = theano.function(inputs=[x], outputs=[prediction])

# before training
print("Accuracy before Training:", 
      accuracy_score(D[1], np.array(predict(D[0])).ravel()))

# training
xStep = []
yAccu = []
for i in range(training_steps):
    prediction, error = train(D[0], D[1])
    if i%100 == 0:
        xStep.append(i)
        yAccu.append(accuracy_score(D[1], np.array(predict(D[0])).ravel()))

# after training
print("Accuracy after Training:", 
      accuracy_score(D[1], np.array(predict(D[0])).ravel()))


# plotting the accuracy
plt.plot(xStep,yAccu)
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.show()

