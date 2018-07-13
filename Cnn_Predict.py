# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 14:37:39 2018

@author: moseli
"""

#########################NEURAL NETS for dimensionality reduction nad pred##################

from numpy.random import seed
seed(1)

from sklearn.model_selection import train_test_split as tts
import logging
import numpy as np


import keras
from keras import backend as k
k.set_learning_phase(1)
from sklearn.decomposition import TruncatedSVD as tsvd
from keras import initializers
from keras.optimizers import RMSprop
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Dropout,Input,Activation,Add,Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor



##################################################################
#########################Preprocessing############################
##################################################################
data=pd.read_csv(data_location+"train.csv")
y = data[["ID","target"]]
y["target"]= np.log1p(y["target"])
del data["ID"],data['target']




def fullyConnected(data):
    learning_rate = 0.001
    clip_norm = 2.0
    
    Dimension=2500
    pca_trans=tsvd(n_components=Dimension,random_state=42,n_iter=20)
    data2 = pca_trans.fit_transform(data)
    
    x_train,x_test,y_train,y_test = tts(data2,y["target"],test_size=0.20)

    model = Sequential()
    model.add(Dense(Dimension,input_dim = Dimension, kernel_initializer='normal', activation='relu'))
    model.add(Dense(int(Dimension*0.6), activation='tanh'))
    model.add(Dense(int(Dimension*0.4), activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(int(Dimension*0.4), activation='tanh'))
    model.add(Dense(int(Dimension*0.1), activation='relu'))
    model.add(Dense(1, activation='relu'))
    rmsprop = RMSprop(lr=learning_rate,clipnorm=clip_norm)
    adam = Adam(lr=learning_rate,clipnorm=clip_norm)
    model.compile(loss='mse', optimizer=adam, metrics=['mse'])
    model.fit(np.array(x_train), np.array(y_train), validation_data=(np.array(x_test), np.array(y_test)), epochs=100, batch_size=100, verbose=2)
    return model


model= fullyConnected(data)









#####################################################################
############################Model####################################
#####################################################################
