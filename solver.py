#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:52:02 2019

@author: nicolas
"""

import numpy as np
from generators import *
from list_functions import *

#Extracts a subset solution when it is possible, else returns an empty list
def extract_subset(L):
    n = len(L)
    m = int(sum_list(L)/2)
    bool_mat = np.zeros((n + 1,m + 1),dtype = bool)
    #Initialization of the table
    bool_mat[0,0] = True
    #Filling the table
    for i in range(1,n+1):
    #print(bool_mat)
        for j in range(m + 1):
            if (j - L[i - 1] < 0):
                bool_mat[i,j] = bool_mat[i - 1,j]
            else :
                bool_mat[i,j] = (bool_mat[i - 1,j - L[i - 1]] or bool_mat[i - 1,j])
                
    #Extracting the solution
    rep = []
    if (bool_mat[n-1,m]):
        lsum = m
        for i in range(n):
            if (lsum - L[n-1 - i] >= 0):
                if (bool_mat[n-1 - i,lsum - L[n-1 - i]]):
                    rep.append(L[n-1 - i])
                    lsum = lsum - L[n-1 - i]
    return rep

#Solves an instance and returns a boolean = True if the list i spartitionable, False else 
def solver(L):
    if is_sum_odd(L) :
        return False
    else :
        rep = (2*sum_list(extract_subset(L)) == sum_list(L))
        return rep

#Applies the Karmarkar Karp heuristic and returns the gap between the objective value and the one found by the heuristic
def karmarkar_karp_heuristic(L):
    L_copy = np.copy(L)
    L_copy = (np.sort(L_copy))[::-1]
    for i in range(len(L) - 1):
        L_copy[0] -= L_copy[1]
        L_copy[1] = 0
        L_copy = (np.sort(L_copy))[::-1]
    return L_copy[0]

#Applies the greedy heuristic and returns the gap between the two subsets generated
def greedy_heuristic(L):
    L1 = 0
    L2 = 0
    for i in range(len(L)-1,0,-1):
        if L1 < L2:
            L1 += L[i]
        else:
            L2 += L[i]
    return abs(L1 - L2)
    