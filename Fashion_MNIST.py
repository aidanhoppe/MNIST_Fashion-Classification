# Preliminaries

from __future__ import print_function

import keras
import tensorflow
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
from keras.datasets import fashion_mnist

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["accuracy"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_accuracy"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train[0].shape

#Let's just look at a particular example to see what is inside
x_train[333]  ## Just a 28 x 28 numpy array of ints from 0 to 255

# What is the corresponding label in the training set?
y_train[333]

# Let's see what this image actually looks like
plt.imshow(x_train[333], cmap='Greys_r')

# this is the shape of the np.array x_train
# it is 3 dimensional.
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')

## For our purposes, these images are just a vector of 784 inputs, so let's convert
x_train = x_train.reshape(len(x_train), 28*28)
x_test = x_test.reshape(len(x_test), 28*28)

## Keras works with floats, so we must cast the numbers to floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

## Normalize the inputs so they are between 0 and 1
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
num_classes = 10
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

# Data is currently flattened, we need to reshape it back to 28 * 28. To do that we reshape the data.
x_train = np.reshape(x_train, [-1, 28, 28])
x_test = np.reshape(x_test, [-1, 28, 28])
x_train.shape, x_test.shape

# LeNet requires input of 32 X 32. So we will pad the train and test images with zeros to increase the size to 32 X 32.
x_train=np.pad(x_train, ((0,0), (2,2), (2, 2)), 'constant')
x_test=np.pad(x_test, ((0,0), (2,2), (2, 2)), 'constant')
x_train.shape, x_test.shape

# Convolutional model requires input to be of 3 dimensions. We will add a channel dimension to it.
x_train = np.reshape(x_train, [-1, 32, 32, 1])
x_test = np.reshape(x_test, [-1, 32, 32, 1])
x_train.shape, x_test.shape

#Creating model based on Famous LeNet-5 Architecture
model_3 = Sequential()

model_3.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(32,32,1)))
model_3.add(MaxPooling2D(pool_size=(2,2)))

model_3.add(Conv2D(filters=64, kernel_size=(1,1), activation='relu'))
model_3.add(MaxPooling2D(pool_size=(2,2)))

model_3.add(Flatten())

model_3.add(Dense(120, activation='relu'))
model_3.add(Dense(64, activation='relu'))
model_3.add(Dense(units=10, activation = 'softmax'))

model_3.summary()

learning_rate = .001
model_3.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=learning_rate),
              metrics=['accuracy'])

batch_size = 128  # mini-batch with 128 examples
epochs = 10
history = model_3.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))

score = model_3.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
    
plot_loss_accuracy(history)