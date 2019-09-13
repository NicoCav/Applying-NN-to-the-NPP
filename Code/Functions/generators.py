#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:23:19 2019

@author: nicolas
"""

import numpy as np
import solver as sol
from list_functions import *

import csv

#Generate an integer list of size sample_size (must be > 1) and with integers in {1,...,300}
def generate_input_list(sample_size, bound_sup):
    return np.random.randint(0,bound_sup + 1,sample_size)

#Generate two array, X containing a collection of nb_examples instances and Y containing the nb_examples solutions
def generator_on_array(nb_examples, sample_size, bound_sup):
    X = np.zeros((nb_examples,sample_size))
    Y = np.zeros(nb_examples, dtype = bool)
    for k in range(nb_examples):
        L = generate_input_list(sample_size,bound_sup)
        X[k] = L
        Y[k] = sol.solver(L)
    return X,Y

#Generate and write nb_examples instances with their solution in a file located at file_path
def generator_on_file(nb_examples, sample_size, bound_sup, file_path):
    data = open(file_path,"w")
    for i in range(nb_examples):
        size = np.random.randint(2,sample_size + 1)
        L = generate_input_list(size,bound_sup)
        data.write(str(L))
        data.write(" ")
        data.write(str(sol.solver(L)))
        data.write("\n")
    data.close()

#Generate the two same arrays as generator_on_array but X contains the occurrence lists instead of the integer lists.
def generate_formated_data_base(nb_example,sample_size,bound_sup):
    X = np.zeros((nb_example,bound_sup + 1),dtype = float)
    Y = np.zeros(nb_example, dtype = bool)
    for k in range(nb_example):
        size = np.random.randint(2,sample_size + 1)
        L = generate_input_list(size,bound_sup)
        for l in L :
            X[k,l] += 1
        Y[k] = sol.solver(L)
    return X,Y

#Generate a two arrays data base of occurrence lists (= formated) but every instance has an even sum
def generate_even_sum_formated(nb_example,sample_size,bound_sup):
    X = np.zeros((nb_example,bound_sup + 1), dtype = float)
    Y = np.zeros(nb_example, dtype = bool)
    for k in range(nb_example):
        size = np.random.randint(2,sample_size + 1)
        L = generate_input_list(size,bound_sup)
        while is_sum_odd(L):
            L = generate_input_list(size,bound_sup)
        X[k] = convert_regular_into_occurence(L,bound_sup)
        Y[k] = sol.solver(L)
    return X,Y

#Same as the previous function but writing the database on the file located at file_path
def generate_even_data_base_on_csv(nb_example,sample_size,bound_sup,csv_file_path):
    
    if (nb_example < 1000):
        X,Y = generate_even_sum_formated(nb_example,sample_size,bound_sup)
        save_data_base_on_csv(X,Y,csv_file_path)
    
    else :
        X,Y = generate_even_sum_formated(1000,sample_size,bound_sup)
        save_data_base_on_csv(X,Y,csv_file_path)
        return generate_even_data_base_on_csv(nb_example - 1000,sample_size,bound_sup,csv_file_path)

#Read the database located at file_path and returns the two arrays, X containing the instances and Y containing the solution (=labels)
def read_csv_data_base(csv_file_path):
    with open(csv_file_path,"r", newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter = ",")
        n = 0
        m = 0
        
        for row in reader :
            m = len(row) - 1
            n += 1
        csvfile.seek(0)
        
        X = np.zeros((n,m),dtype = float)
        Y = np.zeros(n,dtype = bool)
        k = 0
        for row in reader:
            Y[k] = (row[0] == "True")
            X[k] = row[1:m+2]
            k += 1
    return X,Y

#Saves the X,Y database writing on the file file_path
def save_data_base_on_csv(X,Y,csv_file_path):
    with open(csv_file_path,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter = ",", lineterminator='\n')
        for k in range(len(X)):
            writer.writerow([Y[k],*X[k]]) 

#Adds the parity feature to an instances database X
def add_parity_feature(X):
    n = len(X)
    m = len(X[0])
    X_par = np.zeros((n,m+1))
    for k in range(n):
        X_par[k,0:m] = X[k]
        if (is_sum_odd(X[k])):
            X_par[k,m] = -1
        else :
            X_par[k,m] = 1
    return X_par

#Takes a database X and returns a new one only containing the features computed by stat_elements and stat_elements_gap (cf list_functions.py)
def create_feature_only_data(X):
    n = len(X)
    Xfeatonly = np.zeros((n,17),dtype = float)
    for k in range(n):
        Xfeatonly[k] = [*stats_elements(convert_occurence_into_regular(X[k])),*stats_elements_gaps(convert_occurence_into_regular(X[k]))]
    return Xfeatonly

#Adds the features on the elements and the gaps between elements (cf list_functions.py)
def add_elmt_feature_gap_feature(X):
    n = len(X)
    m = len(X[0])
    X_feat = np.zeros((n,m + 17))
    for k in range(n):
        X_feat[k,0:m] = X[k]
        X_feat[k,m:m+9] = stats_elements(convert_occurence_into_regular(X[k]))
        X_feat[k,m+9:m+17] = stats_elements_gaps(convert_occurence_into_regular(X[k]))
    return X_feat


#Takes as input a NN and an X,Y database and returns the set of the instances on which the NN made a misstake
def get_back_misstakes(model,X_test,Y_test):
    n = len(Y_test)
    compt_misstakes = 0
    misstakes_index = []
    
    prediction = model.predict_classes(X_test)
    
    for i in range(n):
        if (prediction[i] != np.argmax(Y_test[i])):
            compt_misstakes += 1
            misstakes_index.append(i)
    
    print("Nombre d'erreurs : " + str(compt_misstakes))
    
    misstakes_X = np.zeros((compt_misstakes,len(X_test[0])))
    misstakes_Y = np.zeros((compt_misstakes,len(Y_test[0])))
    
    for i in range(compt_misstakes):
        misstakes_X[i] = X_test[misstakes_index[i]]
        misstakes_Y[i] = Y_test[misstakes_index[i]]
    
    return misstakes_X,misstakes_Y

#Convert a data base of integers into a formatted database (occurrence list)
def convert_formated_data_base_into_regular(X,Y):
    repX = []
    for i in range(len(X)):
        repX.append(np.array(convert_occurence_into_regular(X[i])))
    return np.array(repX),Y

#Generates a partitionable instance in a polynomial time
def generate_true_instance(instance_size,bound_sup):
    L = []
    diff = 0
    for i in range(instance_size - 1):
        l = np.random.randint(1,bound_sup + 1)
        L.append(l)
        if diff >= 0 :
            diff -= l
        else:
            diff += l
    L.append(abs(diff))
    return L