import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import random
import time
import copy

def RF(x_train, y_train,x_test, y_test,y_10_train,
    y_20_train,y_30_train,y_40_train,y_50_train,y_10_train_Asymmetric,
    y_20_train_Asymmetric,y_30_train_Asymmetric,y_40_train_Asymmetric,y_50_train_Asymmetric):

    
    start_time = time.time()


    model = RandomForestClassifier(n_estimators = 10)
    model_10 = RandomForestClassifier(n_estimators = 10)
    model_20 = RandomForestClassifier(n_estimators = 10)
    model_30 = RandomForestClassifier(n_estimators = 10)
    model_40 = RandomForestClassifier(n_estimators = 10)
    model_10_b = RandomForestClassifier(n_estimators = 10)
    model_20_b = RandomForestClassifier(n_estimators = 10)
    model_30_b = RandomForestClassifier(n_estimators = 10)
    model_40_b= RandomForestClassifier(n_estimators = 10)

    model.fit(x_train, y_train)
    model_10.fit(x_train,y_10_train)
    model_20.fit(x_train,y_20_train)
    model_30.fit(x_train,y_30_train)
    model_40.fit(x_train,y_40_train)
    model_10_b.fit(x_train,y_10_train_Asymmetric)
    model_20_b.fit(x_train,y_20_train_Asymmetric)
    model_30_b.fit(x_train,y_30_train_Asymmetric)
    model_40_b.fit(x_train,y_40_train_Asymmetric)

    expected_y  = y_test
    predicted_y = model.predict(x_test)
    predicted_10_y = model_10.predict(x_test)
    predicted_20_y = model_20.predict(x_test)
    predicted_30_y = model_30.predict(x_test)
    predicted_40_y = model_40.predict(x_test)
    predicted_10_y_b = model_10_b.predict(x_test)    
    predicted_20_y_b = model_20_b.predict(x_test)    
    predicted_30_y_b = model_30_b.predict(x_test)    
    predicted_40_y_b = model_40_b.predict(x_test)    

    print(accuracy_score(predicted_y, expected_y))
    print(accuracy_score(predicted_10_y, expected_y))
    print(accuracy_score(predicted_20_y, expected_y))
    print(accuracy_score(predicted_30_y, expected_y))
    print(accuracy_score(predicted_40_y, expected_y))
    print(accuracy_score(predicted_10_y_b, expected_y))
    print(accuracy_score(predicted_20_y_b, expected_y))
    print(accuracy_score(predicted_30_y_b, expected_y))
    print(accuracy_score(predicted_40_y_b, expected_y))

    x = np.arange(1,10)
    y = [accuracy_score(predicted_y, expected_y),accuracy_score(predicted_10_y, expected_y),accuracy_score(predicted_20_y, expected_y)
    ,accuracy_score(predicted_30_y, expected_y),accuracy_score(predicted_40_y, expected_y),
    accuracy_score(predicted_10_y_b, expected_y),
    accuracy_score(predicted_20_y_b, expected_y),
    accuracy_score(predicted_30_y_b, expected_y),accuracy_score(predicted_40_y_b, expected_y)]

    print("Execution Time %s seconds: " % (time.time() - start_time))    

#####################################################################


    model = RandomForestClassifier(n_estimators = 30)
    model_10 = RandomForestClassifier(n_estimators =30)
    model_20 = RandomForestClassifier(n_estimators = 30)
    model_30 = RandomForestClassifier(n_estimators = 30)
    model_40 = RandomForestClassifier(n_estimators = 30)
    model_10_b = RandomForestClassifier(n_estimators = 30)
    model_20_b = RandomForestClassifier(n_estimators = 30)
    model_30_b = RandomForestClassifier(n_estimators = 30)
    model_40_b= RandomForestClassifier(n_estimators = 30)




    model.fit(x_train, y_train)
    model_10.fit(x_train,y_10_train)
    model_20.fit(x_train,y_20_train)
    model_30.fit(x_train,y_30_train)
    model_40.fit(x_train,y_40_train)
    model_10_b.fit(x_train,y_10_train_Asymmetric)
    model_20_b.fit(x_train,y_20_train_Asymmetric)
    model_30_b.fit(x_train,y_30_train_Asymmetric)
    model_40_b.fit(x_train,y_40_train_Asymmetric)


    expected_y  = y_test
    predicted_y = model.predict(x_test)
    predicted_10_y = model_10.predict(x_test)
    predicted_20_y = model_20.predict(x_test)
    predicted_30_y = model_30.predict(x_test)
    predicted_40_y = model_40.predict(x_test)
    predicted_10_y_b = model_10_b.predict(x_test)    
    predicted_20_y_b = model_20_b.predict(x_test)    
    predicted_30_y_b = model_30_b.predict(x_test)    
    predicted_40_y_b = model_40_b.predict(x_test)    
   

    print(accuracy_score(predicted_y, expected_y))
    print(accuracy_score(predicted_10_y, expected_y))
    print(accuracy_score(predicted_20_y, expected_y))
    print(accuracy_score(predicted_30_y, expected_y))
    print(accuracy_score(predicted_40_y, expected_y))
    print(accuracy_score(predicted_10_y_b, expected_y))
    print(accuracy_score(predicted_20_y_b, expected_y))
    print(accuracy_score(predicted_30_y_b, expected_y))
    print(accuracy_score(predicted_40_y_b, expected_y))


    x2 = np.arange(1,10)
    y2 = [accuracy_score(predicted_y, expected_y),accuracy_score(predicted_10_y, expected_y),accuracy_score(predicted_20_y, expected_y)
    ,accuracy_score(predicted_30_y, expected_y),accuracy_score(predicted_40_y, expected_y),
    accuracy_score(predicted_10_y_b, expected_y),
    accuracy_score(predicted_20_y_b, expected_y),
    accuracy_score(predicted_30_y_b, expected_y),accuracy_score(predicted_40_y_b, expected_y)]

    print("Execution Time %s seconds: " % (time.time() - start_time))    
    my_xticks = ['0%','10%','20%','30%','40%'
    ,'10%','20%','30%','40%']
    plt.xticks(x, my_xticks)
    plt.plot(x2, y2,'r')
    plt.plot(x, y,'b')
    plt.title('Random Forest')
    plt.ylabel('accuracy')
    plt.xlabel('noise')
    plt.legend(['30 trees','10 trees'], loc='upper left')
    plt.show()

