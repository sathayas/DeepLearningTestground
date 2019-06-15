import pickle
import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


### Loading the training data previously created
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

# scaling the image intensity between 0 and 1
X = X/255.0


### Creating a model
model = Sequential()  # model object

# first convoltion (3x3), activation is relu, and max pooling
model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# second convoltion (3x3), activation is relu, and max pooling
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten image data into 1D vector
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

# fully connected layer
model.add(Dense(64))

# output layer, binary output (dog vs cat) with sigmoid activation
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Optimization scheme
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

### actual model fitting. setting aside 30% as testing data
model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)
