"""

# -*- coding: utf-8 -*-
Created on          28 Feb 2019 at 11:22 PM
@author:            Arvind Sachidev Chilkoor
Created using:      PyCharm
Name of Project:    Machine Learning using Regression to Predict Stock Prices

"""

"""
This program is a demonstration of a machine learning concept which uses regression to predict the stock prices.
and plot a graph.
Stock price dataset taken from Quandl
"""

import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use('ggplot') # Style of graph plot is ggplot

"""
Get the data from quandl for specific company and pass to data_frame. Company selected here is 
eg. General Electric (GNE), the stock ticker is EURONEXT/GNE
"""
data_frame = quandl.get("EURONEXT/GNE")

# prints the first 5 rows/tables with their headings (old as received from Quandl)
print("\n", data_frame.head())
print("\nTotal number of data available in this set is :: ", len(data_frame))

# Creates a list of all the heads (Features) in the table
data_frame = data_frame[["Open", "High", "Low", "Last", "Volume", "Turnover", ]]

# Calculates the Hi and Lo prices in % of the stocks
data_frame["HiLo_Pct"] = (data_frame["High"] - data_frame["Low"]) / data_frame["Last"] * 100

# Calculates the Daily % of the stock prices
data_frame["Daily_Pct"] = (data_frame["Last"] - data_frame["Open"]) / data_frame["Open"] * 100

# Creates a list of the heads (Features) of the table into data_frame variable
data_frame = data_frame[["Last", "HiLo_Pct", "Daily_Pct", "Volume"]]

# Prints the table with headings (new)
# print("\n\n",data_frame.head())         # To be uncommented in case, necessary to view new list

# Here we are creating a variable for the feature  for which we need a forecast (reference taken from line 38)
fore_cast_col = 'Last'

# Here we are using fillna() from Pandas to replace and NaN (Not a Number) which may exist in the reference table
data_frame.fillna(value=-99999, inplace=True)

# Here we are setting the % for forecasting from the available data set, ie. 1% of the data, also converting the
# data to a int.
# Self Note: smaller the % greater the classifier accuracy line 94, 104
fore_cast_out = int(math.ceil(0.01 * len(data_frame)))
print("\nNo. of days for which data will be forecast is ::  ", fore_cast_out)

# Here we are creating a 'label' i.e fore_cast_col and shifting it up on a graph for showing on the positive side.
data_frame['label'] = data_frame[fore_cast_col].shift(-fore_cast_out)

# Here we are using a dropna() which drop NaN to remove non numbers in the table
data_frame.dropna(inplace=True)

# Here prints the first few rows of the data_frame
print("\n\n--TOP 5 ROWS OF DATA FRAME--")
print(data_frame.head())

# Here it prints the last few rows of the data_frame
print("\n\n--BOTTOM 5 ROWS OF DATA FRAME--")
print(data_frame.tail())

"""
Here we are defining all the features in X axis except label, hence drop is used.

Below the np.array(data_frame.drop(), .drop() is a pandas function,
Axis = 0 will act on all the ROWS in each COLUMN
Axis = 1 will act on all the COLUMNS in each ROW
"""
X = np.array(data_frame.drop(['label'], axis = 1))
# Here the pre-processing is done to keep the range of the features i.e. X within -1 to 1, since it is th fastest
X = preprocessing.scale(X)

# The below 2 lines defines the set of data for which the prediction will be made, X_latest contains the latest features
# of the dataframe (refer line 63)
X_latest = X[-fore_cast_out:]
X = X[:-fore_cast_out]

# Drops all Not a Number..
data_frame.dropna(inplace = True)

# y axis will be the label, using numpy to form them into array..(refer line 87)
y = np.array(data_frame['label'])
y = y[:-fore_cast_out] # Adding this line avoids value error, inconsistent number of sample, this X and y are sample size

# Checking step to see how many NaN dropped, must be close to original quantity in set line 31
print("\nNo. of data-points in X and y is %d and %d respectively" % (len(X), len(y)))

# Here we are proportioning the available data for training and testing i.e 20% in this case.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Here we are setting/defining the classifier i.e Liner Regression as in this case
clsfy_linreg = LinearRegression(n_jobs= -1)     # n_jobs = -1 will use all the available threads in the CPU cores..for speed
clsfy_linreg.fit(X_train, y_train)              # Fitting the training features data into the classifier.

# Here we are checking and testing the classifier after training.
linear_accuracy = clsfy_linreg.score(X_test, y_test)

print("\nAccuracy of the Linear Regression Classifier is ::  ", linear_accuracy)

"""

# Here we are setting/defining the classifier i.e Support Vector Regression as in this case
clsfy_svreg = svm.SVR(kernel='poly')  # kernel i.e. model/formula used within SVR is 'poly', others can be used such as
                                      # 'linear' or 'rbf' or 'sigmoid' or 'precomputed' etc...default is rbf
clsfy_svreg.fit(X_train, y_train)  # Fitting the training features data into the classifier.

# Here we are checking and testing the classifier after training.
svreg_accuracy = clsfy_svreg.score(X_test, y_test)

print("\nAccuracy of the Support Vector Regression Classifier is ::  ", svreg_accuracy)

"""

# Defining the variable for predicting the
fore_cast_set = clsfy_linreg.predict(X_latest)

# Initializing the new column as a Nan first,later it will be populated.
data_frame["Forecast"] = np.nan


# Here we grabbing the last date in the dataframe for predicting the next set of stock prices
# Eg..if the last day is 31 - Dec - 2016 then we want predict from 01 - Jan - 2017

last_date = data_frame.iloc[-1].name  # .iloc[-1] because numbering starts from 0
last_units_value = last_date.timestamp()
one_day = 86400             # number seconds in a day is 86400
next_units_value = last_units_value + one_day



"""
Here we are iterating through the fore_cast_set, taking each fore_cast value and day and putting values into the 
Dataframe, by making them Nan, hence np.nan, the final column is whatever i is i.e forecast 
"""
for i in fore_cast_set:
    next_date = datetime.datetime.fromtimestamp(next_units_value)
    next_units_value += one_day
    data_frame.loc[next_date] = [np.nan for _ in range (len(data_frame.columns)-1)] + [i]

print("\n\n--------------Forecast Price--------------------")
print(data_frame['Forecast'].tail())

# Plotting of the graph
data_frame['Last'].plot()
data_frame['Forecast'].plot()
plt.legend(loc=2) # loc = 2, (Note: 4 parts of graph, 1=top right, 2 top left,3=bottom left,4=bottom right)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
