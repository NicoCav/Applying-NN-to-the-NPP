#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:00:09 2019

@author: nicolas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:48:56 2019

@author: nicolas
"""
import solver as sol
import generators as gen
from list_functions import *

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

#Requires tensorflow and keras to be installed on the machine


#Print stats about the results of a model tested on a test database X_test,Y_test
def results_stats(model,X_test,Y_test):
    n = len(Y_test)
    positif = 0
    negatif = 0
    vrai_positif = 0
    faux_positif = 0
    vrai_negatif = 0
    faux_negatif = 0

    
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print("Prediction accuracy : " + str(test_acc) + "\n")
    
    prediction = model.predict_classes(X_test)
    
    for i in range(n):    
        if (prediction[i] == 0 and np.argmax(Y_test[i]) == 0):
            vrai_negatif += 1
            negatif += 1
        elif (prediction[i] == 0 and np.argmax(Y_test[i]) == 1):
            faux_negatif += 1
            positif += 1
        elif (prediction[i] == 1 and np.argmax(Y_test[i]) == 0):
            faux_positif += 1
            negatif += 1
        elif (prediction[i] == 1 and np.argmax(Y_test[i]) == 1):
            vrai_positif += 1
            positif += 1
    
    print("Vrai True : " + str(vrai_positif))
    print("True classe False : " + str(faux_negatif))
    print("True : " + str(positif) + "\n")
    print("Vrai False : " + str(vrai_negatif))
    print("False classe True: " + str(faux_positif))
    print("False : " + str(negatif) + "\n")

#Prints result stats of the network while differenciating the even and odd sum instances
def parity_feature_stats(model,X_test,Y_test,dataX):
    n = len(Y_test)
    m = len(X_test[0])
    somme_impaire = 0
    succes_somme_impaire = 0
    somme_paire = 0
    succes_somme_paire = 0
    true_somme_paire = 0

    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print("Prediction accuracy : " + str(test_acc) + "\n")
    
    prediction = model.predict_classes(X_test)
    
    for i in range(n):
        if not(is_sum_odd(convert_occurence_into_regular(dataX[i]))):
            somme_paire += 1
            if ((prediction[i] == 0 and np.argmax(Y_test[i]) == 0) or (prediction[i] == 1 and np.argmax(Y_test[i]) == 1)):
                succes_somme_paire += 1
            if np.argmax(Y_test[i]) == 1:
                true_somme_paire += 1
            
        else :
            somme_impaire += 1
            if ((prediction[i] == 0 and np.argmax(Y_test[i]) == 0) or (prediction[i] == 1 and np.argmax(Y_test[i]) == 1)):
                succes_somme_impaire += 1
    
    print("Pourcentage de succes quand la somme est paire : " + str(100*succes_somme_paire/somme_paire) + "%")
    print("Pourcentage de succes quand la somme est impaire : " + str(100*succes_somme_impaire/somme_impaire) + "% \n")
    print("Pourcentage de True parmis les sommes paires : " + str(100*true_somme_paire/somme_paire) + "%")
    print("Pourcentage de reussite qu'on aurait si reponse au hasard : " + str(100*((somme_impaire/(n)) + (max(true_somme_paire, somme_paire - true_somme_paire)/(2*somme_paire)))) + "%")

#Prints the accuracy ofthe network with regards to the size of the instances
def print_accuracy_table(model,X_test,Y_test,data_X):
    n = len(X_test)
    m = len(data_X[0]) - 1
    size = 0
    
    for x in data_X :
        
        if int(sum(x)) > size :
            size = int(sum(x))
    
    acc_table = np.zeros((4,size + 1),dtype = float)
    
    prediction = model.predict_classes(X_test)
    
    rep = np.zeros((size + 1,6))
    
    headers = ["List Size","Accuracy","% of True answers","% of True instances","Random Expectation","Nbr"]
    
    for k in range(n):
        
        list_size = int(sum(data_X[k]))
        acc_table[0,list_size] += 1
        if (prediction[k] == Y_test[k]):
            acc_table[1,list_size] += 1
        if (Y_test[k]):
            acc_table[2,list_size] += 1
        if (prediction[k]):
            acc_table[3,list_size] += 1
    
    for i in range(size+1):
        rep[i,0] = i
        if (acc_table[0,i] > 0):
            rep[i,1] = round(100*acc_table[1,i]/acc_table[0,i],2)
            rep[i,2] = round(100*acc_table[3,i]/acc_table[0,i],2)
            rep[i,3] = round(100*acc_table[2,i]/acc_table[0,i],2)
            rep[i,4] = round(max([rep[i,3],100 - rep[i,3]]),2)
            rep[i,5] = round(acc_table[0,i],0)
    
    print_table(np.vstack((headers,rep)))

#Auxiliary function printing a clean table out of a python array
def print_table(table):
    longest_cols = [
        (max([len(str(row[i])) for row in table]) + 3)
        for i in range(len(table[0]))
    ]
    row_format = "".join(["{:>" + str(longest_col) + "}" for longest_col in longest_cols])
    for row in table:
        print(row_format.format(*row))

