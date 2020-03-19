# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 2020

@author: Sophie
"""

import Image           
import numpy as np                
import matplotlib.pyplot as plt
import os, zipfile, warnings, types
warnings.filterwarnings('ignore')
import pandas as pd

import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

#Cargar Datos
for i in range(N_images):
    I = Image.open(os.getcwd() + "\\" +i+".jpeg")
    print(I)
    
#Cambiar!!
nb_classes = 10
batch_size = 128

#(x_train, y_train),(x_test, y_test) = 
#x_train, x_test = x_train / 255.0, x_test / 255.0  # scale the images to 0-1

# convert class vectors to binary class matrices
#Y_train =  to_categorical(y_train, nb_classes)
#Y_test =  to_categorical(y_test, nb_classes)
#N_images = 1000