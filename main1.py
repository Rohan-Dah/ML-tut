import matplotlib as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
#These are the keys of the dataset 

#print(diabetes.keys())
print(diabetes.data)

diabetes_x = diabetes.data[: np.newaxis, 2]
#This puts the record at index 2 in list of list or array of array or a column format
print(diabetes_x)

diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]
'''Here what we are doing is we are selecting the data from the dataset to test and train. In diabetes_x_train vatriable we are slicing it to get to know the data which is going for training. Training of data is feeding the machine with various inputs and testing the data means to check how accurately can the system get to the conclusion of the prediction. Here we have selected first 30 for training and last 30 for testing
'''
diabetes_y_train = diabetes.target[:-30]

model = linear_model.LinearRegression()
#Linear_model is a class in scikit learn we are using linear regression and doing all of the operations on a variable called model.
model.fit(diabetes_x_train, diabetes_y_train)
#fit functoin is helping you train the data which takes training data as arguments and fits accordingly of how exactly to plot.

diabetes_y_predicted = model.predict(diabetes_x_test)
#Predict is a function in scikit to predict the target..

print("Mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted ))


#This is non-executeable code as there are things missing in it which Ill complete

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import  mean_squared_error

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data


diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()

model.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_predicted = model.predict(diabetes_X_test)

print("Mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

# plt.scatter(diabetes_X_test, diabetes_y_test)
# plt.plot(diabetes_X_test, diabetes_y_predicted)
#
# plt.show()

# Mean squared error is:  3035.0601152912695
# Weights:  [941.43097333]
# Intercept:  153.39713623331698


