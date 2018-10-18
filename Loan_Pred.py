# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:55:28 2018

@author: pranv
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('LP.csv')

df.head()
df.isna().sum()

#Dropping all null values
df.dropna(how = 'any', inplace = True)

#Making sure no null values remain
df.isna().sum()

#Creating dummy variables
df1 = pd.get_dummies(df, columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status'], drop_first = True)

#Creating independent and dependent variables
X = df1.iloc[:, 1:-1].values
y = df1.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling to make the calculations faster
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean() 

#We get an accuracy of 82.30




