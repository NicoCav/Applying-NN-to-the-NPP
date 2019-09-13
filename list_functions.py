#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:38:29 2019

@author: nicolas
"""
import numpy as np

#Computes the sum of the elements of the list L
def sum_list(L):
    rep = 0
    for l in L:
        rep += l
    return rep

#Returns a boolean True is the sum of the list is odd
def is_sum_odd(L):                      
    sum = sum_list(L)
    if (sum % 2 == 0):
        return False
    else :
        return True
    
#Returns 1 if the sum is odd and -1 else
def parity_sign(L):
    if is_sum_odd(L):
        return -1
    else :
        return 1

#Returns the number of elements not null inside the list L
def nbr_elmts_not_null(L):
    rep = 0
    for l in L :
        if (l != 0):
            rep += 1
    return rep

#Computes features of the instance L (stats on the list itself: see commentaries within)
def stats_elements(L):
    rep = np.zeros(9)
    n = len(L)
    L.sort()
    rep[0] = sum(L)/n                   #Mean
    rep[1] = L[int(n/2)]                #Median
    rep[2] = L[int(n/4)]                #1st quartile
    rep[3] = L[int(3*n/4)]              #3rd quartile
    rep[4] = np.var(L)                  #Variance
    rep[5] = max(L)                     #Biggest element
    rep[6] = min(L)                     #Smallest element
    rep[7] = nbr_elmts_not_null(L)      #Nbr of element not null
    rep[8] = parity_sign(L)             #Parity sign of the sum of the ist
    
    return rep    

#Commputes additional features of the instance L (stats on the gap between elements)
def stats_elements_gaps(L):
    n = len(L)
    m = (n-1)*n/2
    gap_list = []
    
    for i in range(n):
        for j in range(i + 1,n):
            gap_list.append(abs(L[i] - L[j]))
    gap_list.sort()
    
    rep = np.zeros(8)
    rep[0] = sum(gap_list)/m                   #Mean
    rep[1] = gap_list[int(m/2)]                #Median
    rep[2] = gap_list[int(m/4)]                #1st quartile
    rep[3] = gap_list[int(3*m/4)]              #3rd quartile
    rep[4] = np.var(gap_list)                  #Variance
    rep[5] = max(gap_list)                     #Biggest gap
    rep[6] = min(gap_list)                     #Smallest gap
    rep[7] = nbr_elmts_not_null(gap_list)      #Number of gap not null 
    
    return rep

#Converts a formatted instance into a regular one 
def convert_occurence_into_regular(J):
    L = []
    for k in range(len(J)):
        for i in range(int(J[k])):
            L.append(k)
    return L

#Converts a regular instance into a formatted one
def convert_regular_into_occurence(L,bound_sup):
    rep = np.zeros(bound_sup + 1)
    for l in L:
        rep[l] += 1
    
    return rep