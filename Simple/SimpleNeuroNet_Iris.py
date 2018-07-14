import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pylab

# Loading the iris data
iris = datasets.load_iris()
X = iris.data  # sepal length and petal width only
vecY = iris.target
Y = np.zeros([len(vecY),max(vecY)+1]).astype(int)
Y[np.arange(len(vecY)), vecY] = 1
feature_names = iris.feature_names
target_names = iris.target_names

# spliting the data into training and testing data sets
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=50,
#                                                    random_state=0)
X_train = X
Y_train = Y

# some stats on data
examples = Y_train.shape[0]
features = X_train.shape[1]
D = (X_train, Y_train)

# Specify the network
layer1_units = 20
layer2_units = 3
w1 = npr.rand(features, layer1_units)
b1 = npr.rand(layer1_units)
w2 = npr.rand(layer1_units, layer2_units)
b2 = npr.rand(layer2_units)
theta = (w1, b1, w2, b2)


# Define the loss function (cross entropy)
def cross_entropy(y, y_hat):
    return np.sum(-y*np.log(y_hat))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return abs(x)*(x>0)

def softmax(x):
    return np.exp(x.reshape(-1,3))/np.sum(np.exp(x.reshape(-1,3)),axis=1).reshape(-1,1)

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
Y_pred  = np.argmax(neural_network(D[0],theta), axis=1)
print("Accuracy score before training:",
      accuracy_score(np.nonzero(D[1])[1],Y_pred))
accuScore = []
for i in range(0, epochs):
    print('Epoch: %d' % (i+1))
    for j in range(0, examples):
        delta = grad_objective(theta,j)
        theta = update_theta(theta,delta, 0.01)
        Y_pred  = np.argmax(neural_network(D[0],theta), axis=1)
        accuScore.append(accuracy_score(np.nonzero(D[1])[1],Y_pred))
print("Accuracy score after training:", accuracy_score(np.nonzero(D[1])[1],Y_pred))
print(confusion_matrix(np.nonzero(D[1])[1],Y_pred))
print(classification_report(np.nonzero(D[1])[1],Y_pred))
pylab.plot(accuScore)
pylab.show()
