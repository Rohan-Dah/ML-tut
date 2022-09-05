import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#we are importing modules like numpy as np, importing datasets from the sklearn module as the module itself is quite huge. Then same for metrics we are importing mean_squared_error. Then from matplotlib ke andar ka pyplot library hai we are importing it as plt.

diabetes = datasets.load_diabetes()
#We are loading the databases database diabetes in the variable diabetes which is in the form of 

print(diabetes)