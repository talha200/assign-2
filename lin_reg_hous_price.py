# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:49:46 2020

@author: lenovo
"""
# linear regression of housing ID

# 
#  Step 1: Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  Step 2: Importing the dataset
dataset = pd.read_csv('housing price.csv')

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
plt.title(' Housing Price vs Housing ID (Training set)')
plt.xlabel('Housing ID')
plt.ylabel('Housing Price')
plt.show()

# Step 8: Visualising the Test set results
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'yellow')
plt.title('Housing Price vs Housing ID(Test set)')
plt.xlabel('Housing ID')
plt.ylabel('Housing Price')
plt.show()

# Step 9: Predicting new values
future_housing_price = regressor.predict([[4000]])
print('The future housing price for ID 4000 is', future_housing_price)

# Step 10: Calculating accuracy/error of linear regression:
# Importing metrics
from sklearn import metrics
# Print Result of MAE:
print(metrics.mean_absolute_error(y_test,y_pred))
# print result of MSE:
print(metrics.mean_squared_error(y_test,y_pred))
#print result of RMSE
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


