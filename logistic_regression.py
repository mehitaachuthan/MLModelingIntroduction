import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

#load tuples, training data set and testing data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#(60000, 28, 28), dimensions: 60000 images and each image is 28 x 28 pixels, each pixel contains the shade of gray (0 to 255), first dimension is index
print(x_train.shape)

#(60000), 60000 data points, shape function gives dimensions, y_train is 1d, only contains the output (which digit the input image is)
print(y_train.shape)

#contains digits from 0 to 10
print(y_train[0:10])

#want to do a binary classification, so only interested in 0's and 1's
x_train_new, y_train_new = x_train[(y_train == 0) | (y_train == 1)], y_train[(y_train == 0) | (y_train == 1)]

#shows the 28 x 28 values for digits
#print(x_train_new[1])

#print new dimensions of the training set
print(x_train_new.shape)
print(y_train_new.shape)

#now, only 0's and 1's
print(y_train_new[0:10])

#flatten the data to 2d, only include the index and the flattened data
#784 : 28*28
x_train_final = x_train_new.reshape((-1, 784))

print(x_train_final.shape)

#filter the test data to only include testing examples where the digit is 0 or 1
x_test_new, y_test_new = x_test[(y_test == 0) | (y_test == 1)], y_test[(y_test == 0) | (y_test == 1)]

#print the tuple with the dimensions
print(x_test_new.shape)

#flatten the input of the test data to 2 dimensions
x_test_final = x_test_new.reshape((-1, 784))

#dimensions after flattening
print(x_test_final.shape)

#print min of the gray shade values in training data set
print(x_train_final.min())

#print the max of the gray shade values in training data set
print(x_train_final.max())

#normalize the data in order to get the values of the shading in a common scale, usually use (x - mean) / sd, but simplified here
x_train_final = x_train_final / 255
x_test_final = x_test_final / 255

#check min and max are within 0 and 1
print(x_train_final.min())
print(x_test_final.max())

#done data processing
#start modeling

#1 layers, use image input to come up with linear model and then put through sigmoid function to get a value between 0 and 1
model = tf.keras.Sequential
model = keras.Sequential([keras.layers.Dense(1, input_shape=(784,), activation='sigmoid')])

#compile, optimize: stochastic gradient descent, loss:binary cross entropy, metric to track training progress: binary accuracy
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_accuracy'])

#train model, pass in the training data set, num. epochs is the number of times to run through training set to learn (too many times will lead to overfitting), shuffling is good since mixes up the ordering, calclating gradient in batches speeds up process
model.fit(x = x_train_final, y = y_train_new, shuffle=True, epochs=5, batch_size=16)

print("\ntesting:\n")

#test model on testing set
eval = model.evaluate(x = x_test_final, y = y_test_new)
print(eval)

#can save model in a file to use in another notebook
model.save(r'./logisticRegressionKeras.hdf5')