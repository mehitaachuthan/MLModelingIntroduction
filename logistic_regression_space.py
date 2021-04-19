import glob
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

x_test = []
y_test = []
x_train = []
y_train = []
train = []
test = []

#each image is 100 x 100 pixels

#get test_data_set, bad images
for img_name in glob.iglob('47vtp22vs7-1/test_dataset_v5/bad/*.jpg'):
  im = Image.open(img_name,'r')
  test_bad = list(im.getdata())
  x_img = np.dot(test_bad, [0.299, 0.587, 0.144])
  #for rgb_values in test_bad:
    #shift to grey scale but also normalize values
    #grey_value = ((rgb_values[0] / 3.0) + (rgb_values[1] / 3.0) + (rgb_values[2] / 3.0)) / 255.0
    #x_img.append(grey_value)
  x_test.append(x_img)
  y_test.append(0)

#get test_data_set, good images
for img_name in glob.iglob('47vtp22vs7-1/test_dataset_v5/good/*.jpg'):
  im = Image.open(img_name, 'r')
  test_good = list(im.getdata())
  x_img = np.dot(test_bad, [0.299, 0.587, 0.144])
  #for rgb_values in test_good:
    #grey_value = ((rgb_values[0] / 3.0) + (rgb_values[1] / 3.0) + (rgb_values[2] / 3.0)) / 255.0
    #x_img.append(grey_value)
  x_test.append(x_img)
  y_test.append(1)

for img_name in glob.iglob('47vtp22vs7-1/training_dataset_v3/bad/*.jpg'):
  im = Image.open(img_name, 'r')
  train_bad = list(im.getdata())
  x_img = np.dot(test_bad, [0.299, 0.587, 0.144])
  #for rgb_values in train_bad:
    #grey_value = ((rgb_values[0] / 3.0) + (rgb_values[1] / 3.0) + (rgb_values[2] / 3.0)) / 255.0
    #x_img.append(grey_value)
  x_train.append(x_img)
  y_train.append(0)

for img_name in glob.iglob('47vtp22vs7-1/training_dataset_v3/good/*.jpg'):
  im = Image.open(img_name, 'r')
  train_good = list(im.getdata())
  x_img = np.dot(test_bad, [0.299, 0.587, 0.144])
  #for rgb_values in train_good:
    #grey_value = ((rgb_values[0] / 3.0) + (rgb_values[1] / 3.0) + (rgb_values[2] / 3.0)) / 255.0
    #x_img.append(grey_value)
  x_train.append(x_img)
  y_train.append(1)

test = list(zip(x_test, y_test))
train = list(zip(x_train, y_train))

random.shuffle(train)
random.shuffle(test)

x_train, y_train = zip(*train)
x_test, y_test = zip(*test)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

model = tf.keras.Sequential
model = keras.Sequential([keras.layers.Dense(1, input_shape=(10000,), activation='sigmoid')])

#compile, optimize: stochastic gradient descent, loss:binary cross entropy, metric to track training progress: binary accuracy
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_accuracy'])

#train model, pass in the training data set, num. epochs is the number of times to run through training set to learn (too many times will lead to overfitting), shuffling is good since mixes up the ordering, calclating gradient in batches speeds up process
#tried different vars
#model.fit(x = x_train, y = y_train, shuffle=True, epochs=2, batch_size=50)
model.fit(x = x_train, y = y_train, shuffle=True, epochs=5, batch_size=16)

print("\ntesting:\n")

#test model on testing set
eval = model.evaluate(x = x_test, y = y_test)
print(eval)

#can save model in a file to use in another notebook
model.save(r'./logisticRegressionKerasSpace.hdf5')

#tried the below code to get pixel data from images, but problem was that too slow since nested for loop, so used dot() function and parallelization/vectorization from numpy to speed up grayscale translation
'''
#get test_data_set, bad images
for img_name in glob.iglob('47vtp22vs7-1/test_dataset_v5/bad/*.jpg'):
  im = Image.open(img_name,'r')
  test_bad = list(im.getdata())
  x_img = []
  for rgb_values in test_bad:
    #shift to grey scale but also normalize values
    grey_value = ((rgb_values[0] / 3.0) + (rgb_values[1] / 3.0) + (rgb_values[2] / 3.0)) / 255.0
    x_img.append(grey_value)
  x_test.append(x_img)
  y_test.append(0)

#get test_data_set, good images
for img_name in glob.iglob('47vtp22vs7-1/test_dataset_v5/good/*.jpg'):
  im = Image.open(img_name, 'r')
  test_good = list(im.getdata())
  x_img = []
  for rgb_values in test_good:
    grey_value = ((rgb_values[0] / 3.0) + (rgb_values[1] / 3.0) + (rgb_values[2] / 3.0)) / 255.0
    x_img.append(grey_value)
  x_test.append(x_img)
  y_test.append(1)

for img_name in glob.iglob('47vtp22vs7-1/training_dataset_v3/bad/*.jpg'):
  im = Image.open(img_name, 'r')
  train_bad = list(im.getdata())
  x_img = []
  for rgb_values in train_bad:
    grey_value = ((rgb_values[0] / 3.0) + (rgb_values[1] / 3.0) + (rgb_values[2] / 3.0)) / 255.0
    x_img.append(grey_value)
  x_train.append(x_img)
  y_train.append(0)

for img_name in glob.iglob('47vtp22vs7-1/training_dataset_v3/good/*.jpg'):
  im = Image.open(img_name, 'r')
  train_good = list(im.getdata())
  x_img = []
  for rgb_values in train_good:
    grey_value = ((rgb_values[0] / 3.0) + (rgb_values[1] / 3.0) + (rgb_values[2] / 3.0)) / 255.0
    x_img.append(grey_value)
  x_train.append(x_img)
  y_train.append(1)
'''

