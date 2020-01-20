# Simple Linear Regression of Mean Temperature of two Countries

#  Step 1: Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  Step 2: Importing the dataset
dataset = pd.read_csv('annual_temp.csv')
# Step 3: Spltting into dependant and independent variables in terms of matrices
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values


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
plt.title(' Mean Annual temperature vs years (Training set)')
plt.xlabel('Years of two countries')
plt.ylabel('Mean Annual Temperature')
plt.show()

# Step 8: Visualising the Test set results
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'yellow')
plt.title('Mean Annual temperature vs years (Test set)')
plt.xlabel('Years of two countries')
plt.ylabel('Mean Annual Temperature')
plt.show()

# Step 9: Predicting new values
new_mean_temp_1 = regressor.predict([[2016]])
print(' The predicted mean temperature of two countries in year 2016 is',new_mean_temp_1)
new_mean_temp_2 = regressor.predict([[2017]])
print(' The predicted mean temperature of two countries in year 2017 is',new_mean_temp_2)

# Step 10: Calculating accuracy/error of linear regression:
# Importing metrics
from sklearn import metrics
# Print Result of MAE:
print(metrics.mean_absolute_error(y_test,y_pred))
# print result of MSE:
print(metrics.mean_squared_error(y_test,y_pred))
#print result of RMSE
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
