"""## **AdaBoost - label noise**"""

# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from keras.datasets import cifar10
import copy 
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from tensorflow.keras.datasets import mnist


def runadaBoost(x_train, y_train,x_test, y_test,y_10_train,
    y_20_train,y_30_train,y_40_train,y_50_train,y_10_train_Asymmetric,
    y_20_train_Asymmetric,y_30_train_Asymmetric,y_40_train_Asymmetric,y_50_train_Asymmetric):
    

    # ####### baseline_model ###########
    # # Create adaboost classifer object
    a = AdaBoostClassifier(n_estimators=100,learning_rate=0.1)
    B = AdaBoostClassifier(n_estimators=65,learning_rate=0.1)
    C = AdaBoostClassifier(n_estimators=50,learning_rate=0.1)
    D = AdaBoostClassifier(n_estimators=50,learning_rate=0.1)
    E = AdaBoostClassifier(n_estimators=50,learning_rate=0.1)
    F = AdaBoostClassifier(n_estimators=50,learning_rate=0.1)
    
    # # Train Adaboost Classifer
    model = a.fit(x_train, y_train)
    model_10 = B.fit(x_train, y_10_train)
    model_20 = C.fit(x_train, y_20_train)
    model_30 = D.fit(x_train, y_30_train)
    model_40 = E.fit(x_train, y_40_train)
    model_50 = F.fit(x_train, y_50_train)
    
    #Predict the response for test dataset
    y_pred = model.predict(x_test)
    y_pred_10 = model_10.predict(x_test)
    y_pred_20 = model_20.predict(x_test)
    y_pred_30 = model_30.predict(x_test)
    y_pred_40 = model_40.predict(x_test)
    y_pred_50 = model_50.predict(x_test)
    
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred_10))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred_20))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred_30))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred_40))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred_50))

    x = np.arange(1,7)
    y = [accuracy_score(y_pred, y_test),accuracy_score(y_pred_10, y_test),accuracy_score(y_pred_20, y_test)
    ,accuracy_score(y_pred_30, y_test),accuracy_score(y_pred_40, y_test),accuracy_score(y_pred_50, y_test)]


    
    # Create adaboost classifer object
    a = AdaBoostClassifier(n_estimators=100,learning_rate=0.1)
    B = AdaBoostClassifier(n_estimators=50,learning_rate=0.1)
    C = AdaBoostClassifier(n_estimators=50,learning_rate=0.1)
    D = AdaBoostClassifier(n_estimators=50,learning_rate=0.1)
    E = AdaBoostClassifier(n_estimators=50,learning_rate=0.1)
    F = AdaBoostClassifier(n_estimators=50,learning_rate=0.1)
    
    # Train Adaboost Classifer
    model = a.fit(x_train, y_train)
    model_10 = B.fit(x_train, y_10_train_Asymmetric)
    model_20 = C.fit(x_train, y_20_train_Asymmetric)
    model_30 = D.fit(x_train, y_30_train_Asymmetric)
    model_40 = E.fit(x_train, y_40_train_Asymmetric)
    model_50 = F.fit(x_train, y_50_train_Asymmetric)
    
    #Predict the response for test dataset
    y_pred = model.predict(x_test)
    y_pred_10 = model_10.predict(x_test)
    y_pred_20 = model_20.predict(x_test)
    y_pred_30 = model_30.predict(x_test)
    y_pred_40 = model_40.predict(x_test)
    y_pred_50 = model_50.predict(x_test)
    
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred_10))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred_20))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred_30))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred_40))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred_50))

    x_asi = np.arange(1,7)
    y_asi = [accuracy_score(y_pred, y_test),accuracy_score(y_pred_10, y_test),accuracy_score(y_pred_20, y_test)
    ,accuracy_score(y_pred_30, y_test),accuracy_score(y_pred_40, y_test),accuracy_score(y_pred_50, y_test)]

    import matplotlib.pyplot as plt
    my_xticks = ['0%','10% noise','20% noise','30% noise','40% noise','50% noise']
    plt.xticks(x_asi, my_xticks)
    plt.plot(x, y,'r')
    plt.plot(x_asi, y_asi,'b')
    plt.title('AdaBoost')
    plt.ylabel('accuracy')
    plt.xlabel('noise')
    plt.legend(['symmetric','asymmetric'], loc='upper left')
    plt.show()

   