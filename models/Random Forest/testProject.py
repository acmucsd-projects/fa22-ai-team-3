"""
This is a tutorial for using random forest algorithm with python and sickit-learn to predict the petrol prices based on monthly income, % of people with drivers licenses, etc. 
It is taken from this link https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
"""
import numpy as np
import pandas as pd

dataset = pd.read_csv('/Users/skamboj2022/Downloads/petrol_consumption.csv')
#print("hello")

dataset.head()
#divides dataset intro attributes and labels to separate it
X = dataset.iloc[:, 0:4].values #These are the attributes aka column headers (there are only 4 columns)
y = dataset.iloc[:, 4].values #These are the actual values of the columns
from sklearn.model_selection import train_test_split
#You are dividing the data into some training sets and some testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#scale the data aka income is in the thousands but percentage of drivers is a decimal number
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#the RnadomForestRegressor is the one that creates treees. the n_estimator is the number of trees u want
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
#create a regressor model and then predict with that regressor model
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


#since we are predicting a number isntead of a binary value, it is uselful to print the mean of errors. To decrease mean of errors, increase number of trees in previous step so u evaluate data more
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


