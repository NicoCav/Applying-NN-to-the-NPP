#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:11:56 2019

@author: nicolas
"""
from generators import *
from list_functions import *
from solver import *
from model_functions import *

import numpy as np
import tensorflow as tf
import keras

#This files contains the functions required to simulate the NN picking elements to remove and solving the instance.

#Initialize the fist state (cf report IV.1)
def initialization_state(sample_size,bound_sup):
    L = generate_true_instance(sample_size,bound_sup)
    L.sort()
    obj = int(sum(L)/2)
    return [*L,obj]

#Pick an element to remove according to the probability distribution generated by the policy_network
def get_best_action(model,state,rand = False):
    
    n = (len(state) - 1)
    if rand and np.random.rand() <= epsilon:
        return np.random.randint(0,n)
    
    predicted = model.predict(np.array([state]))[0]
    
    for i in range(len(state) - 1):
        if state[i] == 0:
            predicted[i] = 0
            
    v = np.random.sample()
    p = 0
    
    for i in range(len(predicted)):
        if v < p + predicted[i]:
            #print(p)
            return i
        p += predicted[i]
    
    return i

#Takes a state and an action and returns the next_state: the element number action is set to 0
def apply_action(state,action):
    next_state = np.copy(state)
    sample_size = len(state) - 1
    temp = next_state[action]
    next_state[action] = 0
    next_state[-1] -= temp
    return next_state

#Calls the apply_action function and adds the boolean variable telling whether or not a solution as been found
def perform_action(state,action):
    
    next_state = apply_action(state,action)
    
    if next_state[-1] == 0:
        return next_state, True
    else:
        return next_state, False

#Simulates the NN picking successively elements to remove. This correspond to 1 try in the report    
def play_game(network,init):
    
    max_steps = len(init) - 2
    
    done = False
    steps = 0
    state = np.copy(init)
    next_state = state
    obj = init[-1]
    closest_state_value = abs(state[-1])
    
    while ((not done) and (steps < max_steps) and (state[-1] >= 0)):
        
        state = next_state
        action = get_best_action(network,state)
        next_state,done = perform_action(state,action)
        closest_state_value = min(closest_state_value,abs(next_state[-1]))
        
        steps += 1 
    
    return done,abs(obj - closest_state_value)/obj

#Function training the network policy_network on nb_epochs epochs, one can set the learning rate of the session and the number of try allowed for the tests (100 test instances every 1000 epochs)
def supervised_training(policy_network,nb_epochs,learning_rate,sample_size,bound_sup,nb_try_test):
    
    scores = []
    accs = []
    karm_scores = []
    karm_accs = []
    
    policy_network.compile(optimizer = tf.keras.optimizers.Adam(lr = learning_rate), 
                      loss=custom_loss,
                      metrics=['accuracy'])
    
    for k in range(nb_epochs + 1):
        
        if k %1000 == 0:
            print("Epochs : {}/{}" .format(k,nb_epochs))
        
        if k%1000 == 0 and k > 0:
            score_moyen,acc_moyen,karm_score,karm_acc = test_model(policy_network,100,nb_try_test,sample_size,bound_sup)
            scores.append(score_moyen[0])
            accs.append(acc_moyen[0])
            karm_scores.append(karm_score[0])
            karm_accs.append(karm_acc[0])
            
        init = initialization_state(sample_size,bound_sup)
        state = np.copy(init)
        next_state = np.copy(state)
        subset = extract_subset(init[0:sample_size])
        states = []
        
        for i in range(len(subset)):
            state = next_state
            for j in range(len(init) -1):
                
                if init[j] == subset[i]:
                    next_state,done = perform_action(state,j)
                    states.append([state,j,done])
        
        advantage = np.zeros(len(states))

        for l in range(len(states)-1, -1, -1):
            advantage[l] = 1
        for i in range(len(states) - 1, -1, -1):
            
            target = np.zeros(sample_size)
            state = states[i][0]
            target[states[i][1]] = advantage[i]
            l = policy_network.fit(np.array([state]),np.array([target]),verbose = 0,epochs = 1)
    
    return scores,accs,karm_scores,karm_accs

#Function testing the network and comparing it with the Karmarkar-Karp heuristic on nb_test instances and allowing the network nbr_try tries to solve every instance
def test_model(network,nb_test,nbr_try,sample_size,bound_sup):
    dones = []
    scores = []
    karmarkar_dones = []
    karmarkar_scores = []
    for i in range(nb_test + 1):
        
        if i%1000 == 0 :
            print('Test nbr {}/{}' .format(i,nb_test))
        
        L = initialization_state(sample_size,bound_sup)
        
        karmarkar = karmarkar_karp_heuristic(L[0:len(L) - 1])
        karmarkar_scores.append((L[-1] - abs(karmarkar))/L[-1])
        if karmarkar == 0:
            karmarkar_dones.append(1)
        else:
            karmarkar_dones.append(0)
        
        best_score = 0
        for i in range(nbr_try):
            done,score = play_game(network,L)
            best_score = max(best_score,score)
            if done:
                break
        dones.append(int(done))
        scores.append(best_score)
    
    score_moyen = []
    acc_moyen = []
    score_moyen_kar = []
    acc_moyen_kar = []
    for i in range(len(scores)):
        if (i%100== 0) and i >0:
            score_moyen.append(sum_list(scores[i - 100:i])/100)
            acc_moyen.append(sum_list(dones[i - 100:i])/100)
            score_moyen_kar.append(sum_list(karmarkar_scores[i - 100:i])/100)
            acc_moyen_kar.append(sum_list(karmarkar_dones[i - 100:i])/100)
    
    return score_moyen,acc_moyen,score_moyen_kar,acc_moyen_kar