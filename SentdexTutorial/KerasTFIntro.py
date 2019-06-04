import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# loading the test data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
shapeOrig = x_train[0].shape

# just for fun, taking a look at an image
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.colorbar()
plt.show()


# normalizing the intesity scale from 0 to 1
x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)
x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)


# just for fun, taking a look at an image
plt.imshow(x_train[0].reshape(shapeOrig),cmap=plt.cm.binary)
plt.colorbar()
plt.show()



###### model building
model = tf.keras.models.Sequential()  # creating a model object

# flatten layer
#model.add(tf.keras.layers.Flatten())

# dense layer x2
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])




##### model fit
model.fit(x_train, y_train, epochs=3)



##### fitting the testing data
val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy
