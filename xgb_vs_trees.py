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

from sklearn.model_selection import train_test_split as tts
from sklearn.decomposition import PCA
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

"""------------remove zero varience cols------"""
def ZeroVar():
    count=0
    for k in forDelete:
        if np.var(forDelete[k])==0:
            zero_var.append(k)
            count+=1
            print("deleting var%s :%s"%(count,k))


ZeroVar()
for k in zero_var:
    del data[k]
    del testData[k]


"""----------------PCA Transformation-----------"""
def testPCA(components):
    
    pca_trans=PCA(n_components=components,random_state=1)

    pca_trans.fit_transform(data)
    
    data2 = pca_trans.transform(data)
    
    #MinMax Normalizer
    scaler = MinMaxScaler()
    scaler.fit(data2)
    data2 = scaler.transform(data2)
    
    y["target"]= np.log1p(y["target"])
    
    #train test split
    x_train,x_test,y_train,y_test = tts(data2,y["target"],test_size=0.30)
    
    #######################----------Algos--------------------#######################
    ranfor = RFR(n_estimators=200,verbose=0,n_jobs =-1,random_state=7)
    extratrees = ETR(n_estimators=200,random_state=7)
    bagging = BR(ETR(n_estimators=5,random_state=1),n_estimators=100,random_state=7)
    
    """---XGBOOST---"""
    xgb_train=xgb.DMatrix(x_train,label=y_train)
    xgb_validate=xgb.DMatrix(x_test,label=y_test)
    xgb_test_pred=xgb.DMatrix(x_test)

    param = {}
    param['objective'] = 'reg:linear'
    param['eta'] = 0.002
    param['max_depth'] = 10
    param['alpha'] = 0.002
    param['subsample'] = 0.6
    param['silent'] = 0
    param['nthread'] = 4
    param['eval_metric']='rmse'

    watchlist = [ (xgb_train,'train'), (xgb_validate, 'validation') ]
    
    """-fit-"""
    ranfor.fit(x_train,y_train)
    extratrees.fit(x_train,y_train)
    bagging.fit(x_train,y_train)
    bst = xgb.train(param, xgb_train,5000,watchlist,early_stopping_rounds=100,
                    verbose_eval=100,maximize=False);
    
    
    y_pred = ranfor.predict(x_test)
    y_pred_ada = extratrees.predict(x_test)
    y_pred_grad = bagging.predict(x_test)
    y_pred_xgb = bst.predict(xgb_test_pred,ntree_limit=bst.best_ntree_limit)
    
    y_pred_2best = (0.6*y_pred_ada) + (0.4*y_pred_xgb)

    print("PCA: %s --- Ranfor RMSE is : %s"%(components,np.sqrt(mse(y_test,y_pred))))
    print("PCA: %s --- ExtraTrees RMSE is : %s"%(components,np.sqrt(mse(y_test,y_pred_ada))))
    print("PCA: %s --- Bagging RMSE is : %s"%(components,np.sqrt(mse(y_test,y_pred_grad))))
    print("PCA: %s --- XGBoost RMSE is : %s"%(components,np.sqrt(mse(y_test,y_pred_xgb))))
    
    print("PCA: %s --- XGBoost+ExtraTrees RMSE is : %s"%(components,np.sqrt(mse(y_test,y_pred_2best))))
    
    
    return {"pca":pca_trans,"scaler":scaler,"ranfor":ranfor,'extratrees':extratrees,'bagging':bagging,
            'xgboost':bst}
    
###################################################################
#########################Training Pipeline#########################
    
training_dict = testPCA(100)
del forDelete

###################################################################
##############################Submission############################

testDataPCAScaled = training_dict["pca"].transform(testData)
testDataPCAScaled = training_dict["scaler"].transform(testDataPCAScaled)


Submission1['target']=pd.DataFrame(np.expm1(training_dict["xgboost"].predict(xgb.DMatrix(testDataPCAScaled))))
Submission1[['ID','target']].to_csv(data_location+'submission8.csv', header=True, index=False)

Submission1.head(5)

#compare minmax of the two
print("XGB: %s_%s"%(min(Submission1['target']),max(Submission1['target']))
print("ExtraTrees: %s_%s"(min(np.expm1(training_dict["extratrees"].predict(testDataPCAScaled))),
                          max(np.expm1(training_dict["extratrees"].predict(testDataPCAScaled)))))
