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
Y = np.ceil(randY*4)
D = (X, Y)

# Specify the network
layer1_units = 10
layer2_units = 4
layer3_units = 4
w1 = npr.rand(features, layer1_units)
b1 = npr.rand(layer1_units)
w2 = npr.rand(layer1_units, layer2_units)
b2 = npr.rand(layer2_units)
theta = (w1, b1, w2, b2)


# Define the loss function (cross entropy)
def cross_entropy(y, y_hat):
    return np.sum(-np.log(y_hat[np.arange(y_hat.shape[0]),y.astype(int)-1]))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    return np.exp(x.reshape(-1,4))/np.sum(np.exp(x.reshape(-1,4)),axis=1).reshape(-1,1)

# Wraper around the Neural Network
def neural_network(x, theta):
    w1, b1, w2, b2 = theta
    return softmax(sigmoid(np.dot((sigmoid(np.dot(x,w1) + b1)), w2) + b2))

# Wrapper around the objective function to be optimised
def objective(theta, idx):
    return cross_entropy(D[1][idx], neural_network(D[0][idx], theta))

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
epochs = 50
Y_pred  = np.argmax(neural_network(D[0],theta), axis=1) + 1
print("Accuracy score before training:",
      sklearn.metrics.accuracy_score(D[1],Y_pred))
accuScore = []
for i in range(0, epochs):
    print('Epoch: %d' % (i+1))
    for j in range(0, examples):
        delta = grad_objective(theta,j)
        theta = update_theta(theta,delta, 0.3)
        Y_pred  = np.argmax(neural_network(D[0],theta), axis=1) + 1
        accuScore.append(sklearn.metrics.accuracy_score(D[1],Y_pred))
print("Accuracy score after training:",
      sklearn.metrics.accuracy_score(D[1],Y_pred))
print(sklearn.metrics.confusion_matrix(D[1],Y_pred))
print(sklearn.metrics.classification_report(D[1],Y_pred))
pylab.plot(accuScore)
pylab.show()
