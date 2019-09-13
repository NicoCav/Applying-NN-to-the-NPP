#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:47:27 2019

@author: nicolas
"""

import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import keras.backend as K

#In order to be used, this file requires tensorflow and keras to be installed on the machine

#Saves the model into a directory located at dir_to_save under the name model_name
def save_model(model,dir_to_save,model_name):
    model.save(dir_to_save + model_name + ".h5")
    print("Model saved")

#Keras coding of the loss function described on the report: cf IV.2.
def custom_loss(y_true,y_pred):
    y_pred = y_pred*y_true
    loss = K.sum(y_pred) + 1e-6
    loss = K.log(loss)
    return  (-1)*loss

#Returns a saved model located in the directory model_dir under the name model_name
def load_model(model_dir,model_name):
    model = keras.models.load_model(model_dir + model_name + ".h5", custom_objects={'custom_loss': custom_loss})
    print("Model loaded")
    return model