import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
data = pd.read_csv('inpOutput.csv')
data = data[12:]
feature_cols = ['Precip','time']

# use the list to select a subset of the original DataFrame
X = data[feature_cols]

feature_cols = ['SPI 12']
y = data[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
print(y_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print('Predicted')
print(y_test)
print('Actual')
for i in y_pred:
    print(i)
print('RMSE Error')
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
