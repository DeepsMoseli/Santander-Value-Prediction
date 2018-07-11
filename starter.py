# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 19:41:15 2018
@author: Moseli Motsehli
Title: Satander starter
"""
###########################libraries################################
####################################################################
import pandas as pd
import numpy as np
import seaborn as sb

from sklearn.model_selection import train_test_split as tts

####################################################################
###########################configs##################################
data_location = "E:/ML Personal/setander/"
val_size = 0.2


####################################################################
#########################Data Exploration###########################
data=pd.read_csv(data_location+"train.csv")


data.top(2)


features=[k for k in data]
print('Missing Values in each feature')
for feat in features:
    print('%s: %s'%(feat,data[feat].isnull().sum()))
    #print('%s: %s'%(feat,sum(data[feat])))

####################################################################
#########################Healper Functions###########################

