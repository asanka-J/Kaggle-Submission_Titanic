# -*- coding: utf-8 -*-
"""
TITANIC-KAGGLE

Created on Thu Sep  7 08:30:56 2017

@author: Asanka
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')

X = dataset.iloc[:,[0,2,4,5,9]]
y = dataset.iloc[:,1].values
list(X) #get column namess 
 

print(X['PassengerId'].isnull().sum())
print(X['Pclass'].isnull().sum())
print(X['Sex'].isnull().sum())
print(X['Age'].isnull().sum())
print(X['Fare'].isnull().sum())

# Taking care of missing data in Age
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = 'mean', axis = 0)
imputer = imputer.fit(X.iloc[:,[3]].values)
X.iloc[:,3]= imputer.transform(k[:,[0]])



# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y) 



