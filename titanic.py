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
import seaborn as sns
sns.set_style('whitegrid')

# Importing the dataset
dataset = pd.read_csv('train.csv')

dataset.info()
print("*"*40)


#descriptive statistics
#distribution of numerical feature values across the samples
dataset.describe()
print("*"*40)

list(dataset) #get column namess 

#get details about catagorical variables
dataset.describe(include=['O'])

#Select the columns to use in the model 
dataset = dataset.iloc[:,[0,1,2,4,5,9]]


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
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
dataset.iloc[:,3:4] = labelencoder_X.fit_transform(dataset.iloc[:,3:4].values)

X = dataset.iloc[:,[0,2,3,4,5]].values
y = dataset.iloc[:,1].values


#observe how survival reflects with the variables
#No survivours vs Age 
gan = sns.FacetGrid(dataset, col='Survived')
gan.map(plt.hist, 'Age', bins=10)

#No survivours vs Age and class
grid = sns.FacetGrid(dataset, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


list(dataset)

#No survivors vs Sex 

gan = sns.FacetGrid(dataset, col='Survived')
gan.map(plt.hist, 'Sex', bins=10)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)


 #Fitting Random Forest Regression to the Training set
from sklearn.ensemble import RandomForestRegressor
Regressor = RandomForestRegressor(n_estimators = 100,oob_score=True, random_state = 0)
Regressor.fit(X_train, y_train)


# Check the importance of variables
Regressor.feature_importances_
F_importance=pd.Series(Regressor.feature_importances_,index=(dataset.iloc[:,[0,2,3,4,5]]).columns)
F_importance.plot(kind="barh" , figsize=(7,6))



 #Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators =195, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#grid search hyper parameter tuning
"""
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [195,196,197]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

got accuracy of 0.830212234707for n_estimator=195
"""
