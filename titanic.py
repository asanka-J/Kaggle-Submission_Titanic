# -*- coding: utf-8 -*-
"""
TITANIC-KAGGLE

Created on Thu Sep  7 08:30:56 2017

@author: Asanka
"""

# Importing the libraries

# remove warnings
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
#dataset.describe()
dataset = dataset.iloc[:,[0,1,2,4,5,9]]

#list(dataset) #get column namess 

#X must be a data set to view this 
print(dataset['PassengerId'].isnull().sum())
print(dataset['Pclass'].isnull().sum())
print(dataset['Sex'].isnull().sum())
print(dataset['Age'].isnull().sum())
print(dataset['Fare'].isnull().sum())


# Taking care of missing data in Age
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = 'mean', axis = 0)
imputer = imputer.fit(dataset.iloc[:,4:5].values)
dataset.iloc[:,4:5]= imputer.transform(dataset.iloc[:,4:5].values)


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
dataset.iloc[:,3:4] = labelencoder_X.fit_transform(dataset.iloc[:,3:4].values)


X = dataset.iloc[:,[0,2,3,4,5]].values
y = dataset.iloc[:,1].values



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)


 #Fitting Random Forest Regression to the Training set
from sklearn.ensemble import RandomForestRegressor
Regressor = RandomForestRegressor(n_estimators = 100,oob_score=True, random_state = 0)
Regressor.fit(X_train, y_train)

Regressor.oob_score_ # R**2 value

# Check the importance of variables
Regressor.feature_importances_
F_importance=pd.Series(Regressor.feature_importances_,index=(dataset.iloc[:,[0,2,3,4,5]]).columns)
F_importance.plot(kind="barh" , figsize=(7,6))



 #Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

