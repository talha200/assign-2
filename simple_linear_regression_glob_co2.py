# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 18:27:42 2020

@author: lenovo
"""
# simple linear regression of CO2 

#  Step 1: Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  Step 2: Importing the dataset
dataset = pd.read_csv('global_co2.csv')

# Step 3: Spltting into dependant and independent variables in terms of matrices
X = dataset.iloc[:,0:1].values
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
plt.title(' Total CO2 Prodcution vs years (Training set)')
plt.xlabel('Years in a region')
plt.ylabel('Total CO2')
plt.show()

# Step 8: Visualising the Test set results
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'yellow')
plt.title('Total CO2 Prodcution vs years(Test set)')
plt.xlabel('Years in a region')
plt.ylabel('Total CO2')
plt.show()

# Step 9: Predicting new values
new_mean_CO2_a = regressor.predict([[2011]])
print(' The predicted CO2 in year 2011 is',new_mean_CO2_a)
new_mean_CO2_b = regressor.predict([[2012]])
print(' The predicted CO2 in year 2012 is',new_mean_CO2_b)
new_mean_CO2_c = regressor.predict([[2013]])
print(' The predicted CO2 in year 2013 is', new_mean_CO2_c)

# Step 10: Calculating accuracy/error of linear regression:
# Importing metrics
from sklearn import metrics
# Print Result of MAE:
print(metrics.mean_absolute_error(y_test,y_pred))
# print result of MSE:
print(metrics.mean_squared_error(y_test,y_pred))
#print result of RMSE
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

