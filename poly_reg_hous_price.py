# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:04:01 2020

@author: lenovo
"""

# Poly Regression of Housing ID

# Step 1: Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Importing the dataset
dataset = pd.read_csv('housing price.csv')
X = dataset.iloc[:,:-1].values
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
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
y_pred = lin_reg.predict(X)

# Step 6: Visualising the Linear Regression results
plt.scatter(X, y, color = 'green')
plt.plot(X, lin_reg.predict(X), color = 'yellow')
plt.title('Housing Price vs house ID(Linear Regression)')
plt.xlabel('Housing ID')
plt.ylabel('HOusing Price')
plt.show()

# Step 7: Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'green')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'yellow')
plt.title('Housing Price vs house ID (Polynomial Regression)')
plt.xlabel('HOusing ID')
plt.ylabel('Housing Price')
plt.show()

# Step 8: Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'green')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'yellow')
plt.title('Housing Price vs.house ID (Polynomial Regression)')
plt.xlabel('Housing ID')
plt.ylabel('Housing Price')
plt.show()


# Step 9: Predicting a new result with Linear Regression of year 2011
future_housing_price_a = lin_reg.predict([[4000]])
# Predicting a new result with Polynomial Regression
future_housing_price_b = lin_reg_2.predict(poly_reg.fit_transform([[4000]]))
print('The future housing price of housing ID 4000 is',future_housing_price_b)


# Step 10: Calculating accuracy/error of linear regression:
# Importing metrics
from sklearn import metrics
# Print Result of MAE:
print(metrics.mean_absolute_error(y,y_pred))
# print result of MSE:
print(metrics.mean_squared_error(y,y_pred))
#print result of RMSE
print(np.sqrt(metrics.mean_squared_error(y,y_pred)))

