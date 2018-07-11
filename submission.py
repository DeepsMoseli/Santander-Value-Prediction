#load testData
#separate ID

# do not fit algo on testData
# do not fit PCA on testData

testDataPCA = PCA.transform(testData)

Submission1=pd.DataFrame()
Submission1['ID']=testData['ID']
Submission1['target']=pd.DataFrame(ranfor.predict(testDataPCA))
Submission1[['ID','target']].to_csv(data_location+'submission1.csv', header=True, index=False)

