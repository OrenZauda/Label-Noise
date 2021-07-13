from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from tensorflow.keras.datasets import mnist

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import copy
import random

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



def corrupt_labels(y_train):

    # corrupt the labels in 10,20,30,40,50% *symmetricly*

    n = len(y_train)
    y_10_train = copy.deepcopy(y_train)
    a = np.arange(int(n/10)) * 10
    for j in a:
        y_10_train[int(j)]= random.randint(0, 9)
    y_20_train = copy.deepcopy(y_train)
    a = np.arange(int(n/5)) * 5
    for j in a:
        y_20_train[int(j)]= random.randint(0, 9)

    y_30_train = copy.deepcopy(y_train)
    a = np.arange(int(n/10)*3) * 2
    for j in a:
        y_30_train[int(j)]= random.randint(0, 9)
    y_40_train = copy.deepcopy(y_train)
    a = np.arange(int(n/10)*4) * 2
    for j in a:
        y_40_train[int(j)]= random.randint(0, 9)
    y_50_train = copy.deepcopy(y_train)
    a = np.arange(int(n/10)*5) * 2
    for j in a:
        y_50_train[int(j)]= random.randint(0, 9)
    return(y_10_train,
    y_20_train,y_30_train,y_40_train,y_50_train)

def SVM(x_train, y_train,x_test, y_test,y_10_train,
    y_20_train,y_30_train,y_40_train,y_50_train):
    # ,y_10_train_Asymmetric,
    # y_20_train_Asymmetric,y_30_train_Asymmetric,y_40_train_Asymmetric,y_50_train_Asymmetric):
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    x_train, x_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)
    # pca = PCA(n_components=0.90)
    # pca_X_train = pca.fit_transform(x_train)
    # pca_X_test = pca.transform(x_test)

    y_10_train,y_20_train,y_30_train,y_40_train,y_50_train = corrupt_labels(y_train)

    ##0% noise#########
    classifier0 = svm.SVC(gamma=0.001)
    classifier0.fit(x_train, y_train)
    train_accuracy0 = classifier0.score(x_test, y_test)
    print (f"Test Accuracy: {train_accuracy0*100:.3f}%")

    ##10% noise#########

    classifier1 = svm.SVC(gamma=0.001)
    classifier1.fit(x_train, y_10_train)
    test_accuracy1 = classifier1.score(x_test, y_test)
    print (f"Test Accuracy: {test_accuracy1*100:.3f}%")

    # ##20% noise#########

    classifier2 = svm.SVC(gamma=0.00728932024638, C=2.82842712475)
    classifier2.fit(x_train, y_20_train)
    test_accuracy2 = classifier2.score(x_test, y_test)
    print (f"Test Accuracy: {test_accuracy2*100:.3f}%")

    # ##30% noise#########

    classifier3 = svm.SVC(gamma=0.00728932024638, C=2.82842712475)
    classifier3.fit(x_train, y_30_train)
    test_accuracy3 = classifier3.score(x_test, y_test)
    print (f"Test Accuracy: {test_accuracy3*100:.3f}%")

    # ##40% noise#########

    classifier4 = svm.SVC(gamma=0.00728932024638, C=2.82842712475)
    classifier4.fit(x_train, y_40_train)
    test_accuracy4 = classifier4.score(x_test, y_test)
    print (f"Test Accuracy: {test_accuracy4*100:.3f}%")

    # ##50% noise#########

    classifier5 = svm.SVC(gamma=0.00728932024638, C=2.82842712475)
    classifier5.fit(x_train, y_50_train)
    test_accuracy5 = classifier5.score(x_test, y_test)
    print (f"Test Accuracy: {test_accuracy5*100:.3f}%")

    x = np.arange(1,7)
    y = [train_accuracy0,
    test_accuracy1,
    test_accuracy2,
    test_accuracy3,
    test_accuracy4,test_accuracy5]
    
    n = len(y_train)
    y_10_train_Asymmetric = copy.deepcopy(y_train)
    y_10_train_Asymmetric = changeTheLabel(y_10_train_Asymmetric,0,int(n/10))
    y_20_train_Asymmetric = copy.deepcopy(y_10_train_Asymmetric)
    y_20_train_Asymmetric = changeTheLabel(y_20_train_Asymmetric,int(n/10),int(n/10)*2)
    y_30_train_Asymmetric = copy.deepcopy(y_20_train_Asymmetric)
    y_30_train_Asymmetric = changeTheLabel(y_30_train_Asymmetric,int(n/10)*2,int(n/10)*3)
    y_40_train_Asymmetric = copy.deepcopy(y_30_train_Asymmetric)
    y_40_train_Asymmetric = changeTheLabel(y_40_train_Asymmetric,int(n/10)*3,int(n/10)*4)
    y_50_train_Asymmetric = copy.deepcopy(y_40_train_Asymmetric)
    y_50_train_Asymmetric = changeTheLabel(y_50_train_Asymmetric,int(n/10)*4,int(n/10)*5)


    ##10% noise#########

    classifier1 = svm.SVC(gamma=0.001)
    classifier1.fit(x_train, y_10_train_Asymmetric)
    test_accuracy1 = classifier1.score(x_test, y_test)
    print (f"Test Accuracy: {test_accuracy1*100:.3f}%")

    # ##20% noise#########

    classifier2 = svm.SVC(gamma=0.00728932024638, C=2.82842712475)
    classifier2.fit(x_train, y_20_train_Asymmetric)
    test_accuracy2 = classifier2.score(x_test, y_test)
    print (f"Test Accuracy: {test_accuracy2*100:.3f}%")

    # ##30% noise#########

    classifier3 = svm.SVC(gamma=0.00728932024638, C=2.82842712475)
    classifier3.fit(x_train, y_30_train_Asymmetric)
    test_accuracy3 = classifier3.score(x_test, y_test)
    print (f"Test Accuracy: {test_accuracy3*100:.3f}%")

    # ##40% noise#########

    classifier4 = svm.SVC(gamma=0.00728932024638, C=2.82842712475)
    classifier4.fit(x_train, y_40_train_Asymmetric)
    test_accuracy4 = classifier4.score(x_test, y_test)
    print (f"Test Accuracy: {test_accuracy4*100:.3f}%")

    # ##50% noise#########

    classifier5 = svm.SVC(gamma=0.00728932024638, C=2.82842712475)
    classifier5.fit(x_train, y_50_train_Asymmetric)
    test_accuracy5 = classifier5.score(x_test, y_test)
    print (f"Test Accuracy: {test_accuracy5*100:.3f}%")
    x2 = np.arange(1,7)
    y2 = [train_accuracy0,
    test_accuracy1,
    test_accuracy2,
    test_accuracy3,
    test_accuracy4,test_accuracy5]

    my_xticks = ['0%','10%','20%','30%','40%'
    ,'50%']
    plt.xticks(x, my_xticks)
    plt.plot(x, y,'b')
    plt.plot(x2, y2,'r')
    plt.title('Support Vector Machine')
    plt.ylabel('accuracy')
    plt.xlabel('noise')
    plt.show()