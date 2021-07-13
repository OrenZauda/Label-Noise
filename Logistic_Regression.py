from numpy.core.fromnumeric import mean
from sklearn.linear_model import LogisticRegression
import random
import copy 
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler

def runLR(x_train, y_train,x_test, y_test,y_10_train,
    y_20_train,y_30_train,y_40_train,y_50_train,y_10_train_Asymmetric,
    y_20_train_Asymmetric,y_30_train_Asymmetric,y_40_train_Asymmetric,y_50_train_Asymmetric):

    

    model = LogisticRegression(max_iter=5000, solver= 'liblinear',dual = True)
    model_10 = LogisticRegression(max_iter=5000, solver= 'liblinear',dual = True)
    model_20 = LogisticRegression(max_iter=5000, solver= 'liblinear',dual = True)
    model_30 = LogisticRegression(max_iter=5000, solver= 'liblinear',dual = True)
    model_40 = LogisticRegression(max_iter=5000, solver= 'liblinear',dual = True)
    model_50 = LogisticRegression(max_iter=5000, solver= 'liblinear',dual = True)


    model.fit(x_train , y_train)
    model_10.fit(x_train , y_10_train)
    model_20.fit(x_train , y_20_train)
    model_30.fit(x_train , y_30_train)
    model_40.fit(x_train , y_40_train)
    model_50.fit(x_train , y_50_train)

    # Returns a NumPy Array
    # Predict for One Observation (image)
    predictions = model.predict(x_test)
    predictions_10 = model_10.predict(x_test)
    predictions_20 = model_20.predict(x_test)
    predictions_30 = model_30.predict(x_test)
    predictions_40 = model_40.predict(x_test)
    predictions_50 = model_50.predict(x_test)


    # Use score method to get accuracy of model
    score = model.score(x_test, y_test)
    score_10 = model_10.score(x_test, y_test)
    score_20 = model_20.score(x_test, y_test)
    score_30 = model_30.score(x_test, y_test)
    score_40 = model_40.score(x_test, y_test)
    score_50 = model_50.score(x_test, y_test)

    print(score)
    print(score_10)
    print(score_20)
    print(score_30)
    print(score_40)
    print(score_50)

    x = np.arange(1,7)
    y = [accuracy_score(predictions, y_test),accuracy_score(predictions_10 , y_test),accuracy_score(predictions_20 , y_test),accuracy_score(predictions_30 , y_test),accuracy_score(predictions_40 , y_test),accuracy_score(predictions_50 , y_test)]
    my_xticks = ['0%','10%','20%','30%','40%','50%']
    plt.xticks(x, my_xticks)
    plt.plot(x, y,'r')
    plt.plot(x, y,'b')
    plt.title('Logistic Regression')
    plt.ylabel('accuracy')
    plt.xlabel('noise')
    plt.show()
        

    model_10 = LogisticRegression(max_iter=5000, solver= 'liblinear',dual = True)
    model_20 = LogisticRegression(max_iter=5000, solver= 'liblinear',dual = True)
    model_30 = LogisticRegression(max_iter=5000, solver= 'liblinear',dual = True)
    model_40 = LogisticRegression(max_iter=5000, solver= 'liblinear',dual = True)
    model_50 = LogisticRegression(max_iter=5000, solver= 'liblinear',dual = True)

    model_10.fit(x_train , y_10_train_Asymmetric)
    model_20.fit(x_train , y_20_train_Asymmetric)
    model_30.fit(x_train , y_30_train_Asymmetric)
    model_40.fit(x_train , y_40_train_Asymmetric)
    model_50.fit(x_train , y_50_train_Asymmetric)


    #print(); print(cv_results)    
    #print(); print(model)



    predictions = model.predict(x_test)
    predictions_10 = model_10.predict(x_test)
    predictions_20 = model_20.predict(x_test)
    predictions_30 = model_30.predict(x_test)
    predictions_40 = model_40.predict(x_test)
    predictions_50 = model_50.predict(x_test)

    # Use score method to get accuracy of model
    score =  model.score(x_test, y_test)
    score_10 = model_10.score(x_test, y_test)
    score_20 = model_20.score(x_test, y_test)
    score_30 = model_30.score(x_test, y_test)
    score_40 = model_40.score(x_test, y_test)
    score_50 = model_50.score(x_test, y_test)

    print(score)
    print(score_10)
    print(score_20)
    print(score_30)
    print(score_40)
    print(score_50)

    x_asi = np.arange(1,7)
    y_asi = [accuracy_score(predictions, y_test),accuracy_score(predictions_10 , y_test),accuracy_score(predictions_20 , y_test),accuracy_score(predictions_30 , y_test),accuracy_score(predictions_40 , y_test),accuracy_score(predictions_50 , y_test)]

    my_xticks = ['baise line','10% noise','20% noise','30% noise','40% noise','50% noise']
    plt.xticks(x_asi, my_xticks)
    plt.plot(x, y,'r')
    plt.plot(x_asi, y_asi,'b')
    plt.title('Logistic Regression')
    plt.ylabel('accuracy')
    plt.xlabel('noise')
    plt.legend(['symetric','asymetric'], loc='upper left')
    plt.show()

  


