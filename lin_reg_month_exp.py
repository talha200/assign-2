# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 21:07:09 2020

@author: lenovo
"""

# Linear Regression of Monthly Expense


#  Step 1: Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  Step 2: Importing the dataset
dataset = pd.read_csv('monthlyexp vs incom.csv')
# Step 3: Spltting into dependant and independent variables in terms of matrices
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values


#  Step 4: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Step 5: Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#  Step 6: Predicting the Test set results
y_pred = regressor.predict(X_test)

# Step 7: Visualising the Training set results
plt.scatter(X_train, y_train, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'yellow')
plt.title('Income vs Experience (Training set)')
plt.xlabel('Experience')
plt.ylabel('Income')
plt.show()

# Step 8: Visualising the Test set results
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'yellow')
plt.title('Income vs Experience(Test set)')
plt.xlabel('Experience')
plt.ylabel('Income')
plt.show()

# Step 9: Predicting new values
future_Income = regressor.predict([[25]])
print('The future Income is', future_Income)

# Step 10: Calculating accuracy/error of linear regression:
# Importing metrics
from sklearn import metrics
# Print Result of MAE:
print(metrics.mean_absolute_error(y_test,y_pred))
# print result of MSE:
print(metrics.mean_squared_error(y_test,y_pred))
#print result of RMSE
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))



