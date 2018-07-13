# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 19:41:15 2018
@author: Moseli Motsehli
Title: Satander starter
"""
###########################libraries################################
####################################################################
from numpy.random import seed
seed(1)

import pandas as pd
import numpy as np
import scipy as sc

from sklearn.model_selection import train_test_split as tts
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD as tsvd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import ExtraTreesRegressor as ETR
from sklearn.ensemble import BaggingRegressor as BR
from sklearn.metrics import mean_squared_error as mse
import xgboost as xgb


####################################################################
###########################configs##################################
data_location = "E:/ML Personal/setander/"
val_size = 0.25


####################################################################
#########################Data Exploration###########################
data=pd.read_csv(data_location+"train.csv")
testData = pd.read_csv(data_location+"test.csv")

Submission1=pd.DataFrame()
Submission1['ID']=testData['ID']
del testData["ID"]


y = data[["ID","target"]]
del data["ID"],data['target']

"""_____combine____"""
frames=[data,testData]
forDelete = pd.concat(frames)

zero_var = []
zero_var
####################################################################
#########################Healper Functions##########################

Dimension = len([k for k in data])
"""------------remove zero varience cols------"""
def userdefinedvars(dataset):
    dataset['Variance'] = np.var(dataset,axis=1)
    dataset['mean'] = np.mean(dataset,axis=1)
    dataset['kurtosis'] = sc.stats.kurtosis(dataset,axis=1)
    dataset['max'] = np.max(dataset,axis=1)
    dataset['min'] = np.min(dataset,axis=1)
    

def ZeroVar():
    count=0
    for k in data:
        if np.var(data[k])==0:
            zero_var.append(k)
            count+=1
            print("deleting var%s :%s"%(count,k))
            

ZeroVar()
for k in zero_var:
    del data[k]
    del testData[k]


"""----------------PCA Transformation-----------"""
def testPCA(components):
    
    #pca_trans=PCA(n_components=components,random_state=1)
    pca_trans=tsvd(n_components=components,random_state=7,n_iter=10)

    pca_trans.fit(data)
    
    data2 = pca_trans.transform(data)
    
    #MinMax Normalizer
    scaler = MinMaxScaler()
    scaler.fit(data2)
    data2 = scaler.transform(data2)
    
    y["target"]= np.log1p(y["target"])
    
    #train test split
    x_train,x_test,y_train,y_test = tts(data2,y["target"],test_size=0.20)
    
    #######################----------Algos--------------------#######################
    ranfor = RFR(n_estimators=500,verbose=0,n_jobs =-1,random_state=7)
    extratrees = ETR(n_estimators=500,random_state=7)
    bagging = BR(ETR(n_estimators=10,random_state=1),n_estimators=100,random_state=7)
    
    """---XGBOOST---"""
    xgb_train=xgb.DMatrix(x_train,label=y_train)
    xgb_validate=xgb.DMatrix(x_test,label=y_test)
    xgb_test_pred=xgb.DMatrix(x_test)

    param = {}
    param['objective'] = 'reg:linear'
    param['eta'] = 0.001
    param['max_depth'] = 6
    param['alpha'] = 0.001
    param['colsample_bytree']= 0.6
    param['subsample'] = 0.6
    param['silent'] = 0
    param['nthread'] = 4
    param['random_state']= 42
    param['eval_metric']='rmse'

    watchlist = [ (xgb_train,'train'), (xgb_validate, 'validation') ]
    
    """-fit-"""
    ranfor.fit(x_train,y_train)
    extratrees.fit(x_train,y_train)
    
    bst = xgb.train(param, xgb_train,10000,watchlist,early_stopping_rounds=100,
                    verbose_eval=100,maximize=False);
    
    
    y_pred = ranfor.predict(x_test)
    y_pred_ada = extratrees.predict(x_test)
    y_pred_xgb = bst.predict(xgb_test_pred,ntree_limit=bst.best_ntree_limit)
    
    #blending
    blending_X=pd.DataFrame()
    blending_X['xgb']= bst.predict(xgb.DMatrix(x_train),ntree_limit=bst.best_ntree_limit)
    blending_X['ExtraTrees']=extratrees.predict(x_train)
    blending_X['ranfor'] =  ranfor.predict(x_train)
    
    bagging.fit(blending_X,y_train)
    
    blending_test=pd.DataFrame()
    blending_test['xgb']= y_pred_xgb
    blending_test['ExtraTrees']=y_pred_ada
    blending_test['ranfor'] =  y_pred
    
    y_pred_grad = bagging.predict(blending_test)
    ###############################################

    y_pred_2best = (0.6*y_pred_ada) + (0.4*y_pred_xgb)

    print("PCA: %s --- Ranfor RMSE is : %s"%(components,np.sqrt(mse(y_test,y_pred))))
    print("PCA: %s --- ExtraTrees RMSE is : %s"%(components,np.sqrt(mse(y_test,y_pred_ada))))
    print("PCA: %s --- XGBoost RMSE is : %s"%(components,np.sqrt(mse(y_test,y_pred_xgb))))
    
    print("PCA: %s --- blended bagging RMSE is : %s"%(components,np.sqrt(mse(y_test,y_pred_grad))))
    print("PCA: %s --- XGBoost+ExtraTrees RMSE is : %s"%(components,np.sqrt(mse(y_test,y_pred_2best))))
    
    
    return {"pca":pca_trans,"scaler":scaler,"ranfor":ranfor,'extratrees':extratrees,'bagging':bagging,
            'xgboost':bst}
    
###################################################################
#########################Training Pipeline#########################
    
training_dict = testPCA(200)
del forDelete

###################################################################
##############################Submission############################

testDataPCAScaled = training_dict["pca"].transform(testData)
testDataPCAScaled = training_dict["scaler"].transform(testDataPCAScaled)


Submission1['target']=pd.DataFrame(np.expm1(training_dict["xgboost"].predict(xgb.DMatrix(testDataPCAScaled))))
Submission1[['ID','target']].to_csv(data_location+'submission11_Extrees.csv', header=True, index=False)

Submission1['target']=pd.DataFrame(np.expm1(training_dict["extratrees"].predict(testDataPCAScaled)))
Submission1.head(5)

#compare minmax of the two
print("XGB: %s_%s"%(min(Submission1['target']),max(Submission1['target'])))
print("ExtraTrees: %s_%s"(min(np.expm1(training_dict["extratrees"].predict(testDataPCAScaled))),
                          max(np.expm1(training_dict["extratrees"].predict(testDataPCAScaled)))))
