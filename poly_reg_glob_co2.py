# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 18:58:06 2020

@author: lenovo
"""
# Poly Regression of CO2

# Step 1: Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Importing the dataset
dataset = pd.read_csv('global_co2.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

# Step 3: Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Step 4: Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Step 5: Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
y_pred = lin_reg.predict(X)

# Step 6: Visualising the Linear Regression results
plt.scatter(X, y, color = 'green')
plt.plot(X, lin_reg.predict(X), color = 'yellow')
plt.title(' (Linear Regression)')
plt.xlabel('Years in a place')
plt.ylabel('Mean CO2')
plt.show()

# Step 7: Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'green')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'yellow')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Years in a place')
plt.ylabel('Mean CO2')
plt.show()

# Step 8: Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'green')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'yellow')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Years in a place')
plt.ylabel('Mean CO2')
plt.show()


# Step 9: Predicting a new result with Linear Regression of year 2011
new_mean_CO2_a = lin_reg.predict([[2011]])
# Predicting a new result with Polynomial Regression
new_mean_CO2_b = lin_reg_2.predict(poly_reg.fit_transform([[2011]]))
print('The mean CO2 in year 2011 is',new_mean_CO2_b)

# Step 10:
#Predicting a new result with linear regression of year 2012
new_mean_CO2_c = lin_reg.predict([[2012]])
# Predicting a new result with Polynomial Regression
new_mean_CO2_d= lin_reg_2.predict(poly_reg.fit_transform([[2012]]))
print('The mean CO2 in year 2012 is',new_mean_CO2_d)

# Step 11:Predicting a new result with linear regression of year 2013:
new_mean_CO2_e= lin_reg.predict([[2013]])
# Predicting a new result with Polynomial Regression
new_mean_CO2_f= lin_reg_2.predict(poly_reg.fit_transform([[2013]]))
print('The CO2 in year 2013 is',new_mean_CO2_f)

# Step 12: Importing metrics library and calculating possible errors:
from sklearn import metrics
# Print result of MAE:
print(metrics.mean_absolute_error(y,y_pred))
# print result of MSE:
print(metrics.mean_squared_error(y,y_pred))
#print result of RMSE
print(np.sqrt(metrics.mean_squared_error(y,y_pred)))




