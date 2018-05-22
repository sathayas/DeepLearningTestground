import numpy as np
import numpy.random as npr
from autograd import grad
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pylab

# Loading the iris data
iris = datasets.load_iris()
X = iris.data  # sepal length and petal width only
Y = iris.target + 1
feature_names = iris.feature_names
target_names = iris.target_names

# spliting the data into training and testing data sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=50,
                                                    random_state=0)

# classifier object 
mlp = MLPClassifier(alpha=1e-5, hidden_layer_sizes=100)

# training
mlp.fit(X_train,Y_train)

# fitting
Y_pred = mlp.predict(X_test)

print("Accuracy score after training:", accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))


