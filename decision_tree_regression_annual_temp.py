# Decision Tree Regression of Mean Temperature of two Countries

#  Step 1: Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  Step 2: Importing the dataset
dataset = pd.read_csv('annual_temp.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

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
plt.xlabel('Years of two Countries')
plt.ylabel('Mean Temperature')
plt.show()

# Step 7: Predicting a new result with Linear Regression of year 2016
new_mean_temp_1 = regressor.predict([[2016]])
print('The mean temperature in year 2016 is',new_mean_temp_1)

# Step 8:
#Predicting a new result with linear regression of year 2017
new_mean_temp_2 =regressor.predict([[2017]])
print('The mean temperature in year 2017 is',new_mean_temp_2)

# Step 9: Importing metrics library and calculating possible errors:
from sklearn import metrics
# Print result of MAE:
print(metrics.mean_absolute_error(y,y_pred))
# print result of MSE:
print(metrics.mean_squared_error(y,y_pred))
#print result of RMSE
print(np.sqrt(metrics.mean_squared_error(y,y_pred)))


