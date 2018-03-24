# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:51:04 2018

@author: ANUBHAV SHUKLA
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

#setting directory
os.chdir("E:/backup/ACADS/Kaggle")

# reading the csvfile
train = pd.read_csv("training.csv")  
test=pd.read_csv("sorted_test.csv")

# deleting 1st col
del train['PIDN'] 

# correlation between predictor varibles plot
plt.matshow(train.corr())

#encoding string value to integer
train['Depth'] = LabelEncoder().fit_transform(train['Depth'].astype(str))
test['Depth'] = LabelEncoder().fit_transform(test['Depth'].astype(str))

# setting training and output columns value  
output = [ 'Ca','P','pH','SOC','Sand']
output_val = train[output].values
train.drop(output,axis=1,inplace=True)


#classifiers
rf=RandomForestRegressor(n_estimators=300, min_samples_leaf = 30)
ada = AdaBoostRegressor(random_state=42)
grad = GradientBoostingRegressor(n_estimators=101,random_state=42)
lgb=LGBMRegressor(max_depth=7,learning_rate=0.08)

#multiple output regressor
multi_regressor = MultiOutputRegressor(lgb)

#training classifiers
multi_regressor.fit(train,output_val)

#prediction
pred=multi_regressor.predict(test[train.columns])

#saving the value to csv
sample=pd.read_csv('sample_submission.csv')
sample[['Ca', 'P', 'pH', 'SOC', 'Sand']]=pred
sample.to_csv('abc.csv',index=None)