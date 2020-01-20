# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:16:43 2020

@author: lenovo
"""

# Decision Tree Regression of CO2

#  Step 1: Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  Step 2: Importing the dataset
dataset = pd.read_csv('global_co2.csv')
X = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1].values

# Step 3: Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Step 4: Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Step 5: Predicting a new result
y_pred = regressor.predict(X)

# Step 6: Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'blue')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Years in a place')
plt.ylabel('Mean CO2')
plt.show()

# Step 7: Predicting  new results 
new_mean_CO2_a = regressor.predict([[2011]])
print('The mean CO2 in year 2011 is',new_mean_CO2_a)

new_mean_CO2_b =regressor.predict([[2012]])
print('The mean CO2 in year 2012 is',new_mean_CO2_b)

new_mean_CO2_c = regressor.predict([[2013]])
print('The mean CO2 in year 2013 is',new_mean_CO2_c)


# Step 9: Importing metrics library and calculating possible errors:
from sklearn import metrics
# Print result of MAE:
print(metrics.mean_absolute_error(y,y_pred))
# print result of MSE:
print(metrics.mean_squared_error(y,y_pred))
#print result of RMSE
print(np.sqrt(metrics.mean_squared_error(y,y_pred)))


