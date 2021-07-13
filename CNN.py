import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import tensorflow as tf
from keras.datasets import cifar10
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
import copy

num_classes = 10 #define the number of classes possible.

def changeTheLabel(labelNoise, a,b):
  for i in range(a,b):
    if(labelNoise[i]==0):
      labelNoise[i]=5
    elif (labelNoise[i]==1):
      labelNoise[i]=6
    elif (labelNoise[i]==2):
      labelNoise[i]=8
    elif (labelNoise[i]==3):
      labelNoise[i]=7
    elif (labelNoise[i]==4):
      labelNoise[i]=9
    elif (labelNoise[i]==5):
      labelNoise[i]=0
    elif (labelNoise[i]==6):
      labelNoise[i]=1
    elif (labelNoise[i]==7):
      labelNoise[i]=3
    elif (labelNoise[i]==8):
      labelNoise[i]=2
    elif (labelNoise[i]==9):
      labelNoise[i]=4
  return labelNoise

########################################

# Symmetric noise: randomly choose x% from the labels and randomly switch their labels

# Train with x% label noise

# Repeat for the other noise levels

# Compare model.evaluate(x_test, y_test) for each noise level and baseline (without noise)

def run_CNN(x_train, y_train,x_test, y_test,y_10_train,
    y_20_train,y_30_train,y_40_train,y_50_train,y_10_train_Asymmetric,
    y_20_train_Asymmetric,y_30_train_Asymmetric,y_40_train_Asymmetric,y_50_train_Asymmetric):


  ####### baseline_model ###########
  print()
  print("############### Baseline_model ##############")
  baseline_model = keras.Sequential(
                                      [
                                      #tf.keras.layers.Flatten(), 
                                      layers.Dense(512, activation = 'relu'),
                                      layers.Dense(128, activation= 'relu'),
                                      layers.Dense(10),
                                  ]
                                  )



  baseline_model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(lr = 0.001),
              metrics=["accuracy"],)
 
  history=baseline_model.fit(x_train, y_train, epochs = 5, batch_size = 32, verbose = 2)

  baseline_model.evaluate(x_test, y_test, batch_size = 32, verbose = 2)


  ####### model 10 ###########

  print()
  print("############### model 10 %##############")
  model_10 = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                   tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                  tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                  tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

  model_10.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


  history_10 = baseline_model.fit(x_train, y_10_train, epochs=5)


  model_10.evaluate(x_test, y_test)

  ####### model 20 ###########

  print()
  print("############### model 20 %##############")
  model_20 = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                   tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                  tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                  tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

  model_20.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


  history_20 = baseline_model.fit(x_train, y_20_train, epochs=5)


  model_20.evaluate(x_test, y_test)

  ####### model 30 ###########

  print()
  print("############### model 30 %##############")
  model_30 = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                   tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                  tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                  tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

  model_30.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


  history_30 = baseline_model.fit(x_train, y_30_train, epochs=5)


  model_30.evaluate(x_test, y_test)


  ####### model 40 ###########

  print()
  print("############### model 40 %##############")
  model_40 = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                   tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                  tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                  tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

  model_40.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


  history_40 = baseline_model.fit(x_train, y_40_train, epochs=5)


  model_40.evaluate(x_test, y_test)


  ####### model 50 ###########

  print()
  print("############### model 50 %##############")
  model_50 = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                   tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                  tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                  tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

  model_50.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


  history_50 = baseline_model.fit(x_train, y_50_train, epochs=5)

  model_50.evaluate(x_test, y_test)

  #  "Accuracy"
  plt.plot(history.history['accuracy'])
  plt.plot(history_10.history['accuracy'])
  plt.plot(history_20.history['accuracy'])
  plt.plot(history_30.history['accuracy'])
  plt.plot(history_40.history['accuracy'])
  plt.plot(history_50.history['accuracy'])
  plt.title('CNN symmetric noise')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['baseline', '10% label noise','20% label noise','30% label noise','40% label noise','50% label noise'], loc='upper left')
  plt.show()


  ########################################
  # Asymmetric noise: randomply choose x% from the labels and swap classes according to some heuristics you define(1<-->5, 2<-->6, etc)

  # Train with x% label noise

  # Repeat for the other noise levels

  # Compare model.evaluate(x_test, y_test) for each noise level and baseline (without noise)










  ####### model 10 ###########

  print()
  print("############### model 10 %##############")
  model_10 = keras.Sequential(
                                      [
                                      #tf.keras.layers.Flatten(), 
                                      layers.Dense(512, activation = 'relu'),
                                      layers.Dense(128, activation= 'relu'),
                                      layers.Dense(10),
                                  ]
                                  )



  model_10.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(lr = 0.001),
              metrics=["accuracy"],)
 



  history_10 = model_10.fit(x_train, y_10_train_Asymmetric, epochs=5)


  model_10.evaluate(x_test, y_test)

  ####### model 20 ###########

  print()
  print("############### model 20 %##############")
  model_20 = keras.Sequential(
                                      [
                                      #tf.keras.layers.Flatten(), 
                                      layers.Dense(512, activation = 'relu'),
                                      layers.Dense(128, activation= 'relu'),
                                      layers.Dense(10),
                                  ]
                                  )



  model_20.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(lr = 0.001),
              metrics=["accuracy"],)

  history_20 = model_20.fit(x_train, y_20_train_Asymmetric, epochs=5)


  model_20.evaluate(x_test, y_test)

  ####### model 30 ###########

  print()
  print("############### model 30 %##############")
  model_30 = keras.Sequential(
                                      [
                                      #tf.keras.layers.Flatten(), 
                                      layers.Dense(512, activation = 'relu'),
                                      layers.Dense(128, activation= 'relu'),
                                      layers.Dense(10),
                                  ]
                                  )



  model_30.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(lr = 0.001),
              metrics=["accuracy"],)


  history_30 = model_30.fit(x_train, y_30_train_Asymmetric, epochs=5)


  model_30.evaluate(x_test, y_test)


  ####### model 40 ###########

  print()
  print("############### model 40 %##############")
  model_40 = keras.Sequential(
                                      [
                                      #tf.keras.layers.Flatten(), 
                                      layers.Dense(512, activation = 'relu'),
                                      layers.Dense(128, activation= 'relu'),
                                      layers.Dense(10),
                                  ]
                                  )



  model_40.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(lr = 0.001),
              metrics=["accuracy"],)


  history_40 = model_40.fit(x_train, y_40_train_Asymmetric, epochs=5)


  model_40.evaluate(x_test, y_test)


  ####### model 50 ###########

  print()
  print("############### model 50 %##############")
  model_50 = keras.Sequential(
                                      [
                                      #tf.keras.layers.Flatten(), 
                                      layers.Dense(512, activation = 'relu'),
                                      layers.Dense(128, activation= 'relu'),
                                      layers.Dense(10),
                                  ]
                                  )



  model_50.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(lr = 0.001),
              metrics=["accuracy"],)


  history_50 = model_50.fit(x_train, y_50_train_Asymmetric, epochs=5 )

  model_50.evaluate(x_test, y_test)

  #  "Accuracy"
  plt.plot(history.history['accuracy'])
  plt.plot(history_10.history['accuracy'])
  plt.plot(history_20.history['accuracy'])
  plt.plot(history_30.history['accuracy'])
  plt.plot(history_40.history['accuracy'])
  plt.plot(history_50.history['accuracy'])
  plt.title('CNN Asymetric')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['0%', '10%','20%','30%','40%','50%'], loc='upper left')
  plt.show()
