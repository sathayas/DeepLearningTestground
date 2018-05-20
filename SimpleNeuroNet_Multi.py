import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
import sklearn.metrics
import pylab

# Generate Dataset
examples = 1000
features = 100
X = npr.randn(examples, features)   # scalar features
randY = npr.rand(examples)
Y = np.array([])
for y in randY:
    if 0<=y<0.25:
        Y = np.append(Y,1)
    elif 0.25<=y<0.5:
        Y = np.append(Y,2)
    elif 0.5<=y<0.75:
        Y = np.append(Y,3)
    else:
        Y = np.append(Y,4)
D = (X, Y)

# Specify the network
layer1_units = 10
layer2_units = 4
layer3_units = 4
w1 = npr.rand(features, layer1_units)
b1 = npr.rand(layer1_units)
w2 = npr.rand(layer1_units, layer2_units)
b2 = npr.rand(layer2_units)
w3 = npr.rand(layer2_units, layer3_units)
b3 = npr.rand(layer3_units)
theta = (w1, b1, w2, b2, w3, b3)


# Define the loss function (cross entropy)
def cross_entropy(y, y_hat):
    return np.sum(-(y * np.log(y_hat)))

def sigmoid(x):
    return 1/(1+np.exp(-x))

##### Start here

# Wraper around the Neural Network
def neural_network(x, theta):
    w1, b1, w2, b2 = theta
    return sigmoid(np.dot((sigmoid(np.dot(x,w1) + b1)), w2) + b2)

# Wrapper around the objective function to be optimised
def objective(theta, idx):
    return binary_cross_entropy(D[1][idx], neural_network(D[0][idx], theta))

# Update
def update_theta(theta, delta, alpha):
    w1, b1, w2, b2 = theta
    w1_delta, b1_delta, w2_delta, b2_delta = delta
    w1_new = w1 - alpha * w1_delta
    b1_new = b1 - alpha * b1_delta
    w2_new = w2 - alpha * w2_delta
    b2_new = b2 - alpha * b2_delta
    new_theta = (w1_new,b1_new,w2_new,b2_new)
    return new_theta

# Compute Gradient
grad_objective = grad(objective)

# Train the Neural Network
epochs = 10
Y_pred  = (neural_network(D[0],theta)>0.5).astype(int)
print("Accuracy score before training:",
      sklearn.metrics.accuracy_score(D[1],Y_pred))
accuScore = []
for i in range(0, epochs):
    for j in range(0, examples):
        delta = grad_objective(theta, j)
        theta = update_theta(theta,delta, 0.1)
        Y_pred  = (neural_network(D[0],theta)>0.5).astype(int)
        accuScore.append(sklearn.metrics.accuracy_score(D[1],Y_pred))
print("Accuracy score after training:",
      sklearn.metrics.accuracy_score(D[1],Y_pred))
pylab.plot(accuScore)
pylab.show()

