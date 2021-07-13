from CNN import *
from random_forest import *
from Logistic_Regression import *
from adaBoost import *
from svm import *
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np

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


def main():

    #load the data

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

    # corrupt the labels in 10,20,30,40,50% *symmetricly*
    
    y_10_train = copy.deepcopy(y_train)
    a = np.arange(6000) * 10
    for j in a:
        y_10_train[int(j)]= random.randint(0, 9)
    y_20_train = copy.deepcopy(y_train)
    a = np.arange(12000) * 5
    for j in a:
        y_20_train[int(j)]= random.randint(0, 9)

    y_30_train = copy.deepcopy(y_train)
    a = np.arange(18000) * 2
    for j in a:
        y_30_train[int(j)]= random.randint(0, 9)
    y_40_train = copy.deepcopy(y_train)
    a = np.arange(24000) * 2
    for j in a:
        y_40_train[int(j)]= random.randint(0, 9)
    y_50_train = copy.deepcopy(y_train)
    a = np.arange(30000) * 2
    for j in a:
        y_50_train[int(j)]= random.randint(0, 9)


    # corrupt the labels in 10,20,30,40,50% *asymmetricly*

    y_10_train_Asymmetric = copy.deepcopy(y_train)
    y_10_train_Asymmetric = changeTheLabel(y_10_train_Asymmetric,0,6000)
    y_20_train_Asymmetric = copy.deepcopy(y_10_train_Asymmetric)
    y_20_train_Asymmetric = changeTheLabel(y_20_train_Asymmetric,6000,12000)
    y_30_train_Asymmetric = copy.deepcopy(y_20_train_Asymmetric)
    y_30_train_Asymmetric = changeTheLabel(y_30_train_Asymmetric,12000,18000)
    y_40_train_Asymmetric = copy.deepcopy(y_30_train_Asymmetric)
    y_40_train_Asymmetric = changeTheLabel(y_40_train_Asymmetric,18000,24000)
    y_50_train_Asymmetric = copy.deepcopy(y_40_train_Asymmetric)
    y_50_train_Asymmetric = changeTheLabel(y_50_train_Asymmetric,24000,30000)

    run_CNN(x_train, y_train,x_test, y_test,y_10_train,
    y_20_train,y_30_train,y_40_train,y_50_train,y_10_train_Asymmetric,
    y_20_train_Asymmetric,y_30_train_Asymmetric,y_40_train_Asymmetric,y_50_train_Asymmetric)

    RF(x_train, y_train,x_test, y_test,y_10_train,
    y_20_train,y_30_train,y_40_train,y_50_train,y_10_train_Asymmetric,
    y_20_train_Asymmetric,y_30_train_Asymmetric,y_40_train_Asymmetric,y_50_train_Asymmetric)

    runLR(x_train, y_train,x_test, y_test,y_10_train,
    y_20_train,y_30_train,y_40_train,y_50_train,y_10_train_Asymmetric,
    y_20_train_Asymmetric,y_30_train_Asymmetric,y_40_train_Asymmetric,y_50_train_Asymmetric)

    runadaBoost(x_train, y_train,x_test, y_test,y_10_train,
    y_20_train,y_30_train,y_40_train,y_50_train,y_10_train_Asymmetric,
    y_20_train_Asymmetric,y_30_train_Asymmetric,y_40_train_Asymmetric,y_50_train_Asymmetric)

    SVM(x_train, y_train,x_test, y_test,y_10_train,
    y_20_train,y_30_train,y_40_train,y_50_train)


if __name__ == '__main__':
    main()
