# A program to plot linear regression of a sample dataset and to predict a future point

import pandas as pd 
from sklearn import linear_model
import matplotlib.pyplot as plt
import csv

#read data
dataframe = pd.read_csv('challenge_dataset.txt', sep=',', keep_default_na=False, na_values=[''],quoting=csv.QUOTE_NONNUMERIC) #read data from csv
x_values=dataframe[['X']] #Create array of X values
y_values=dataframe[['Y']] #Create array of Y values

#train model
body_reg = linear_model.LinearRegression() #Find line of best fit
body_reg.fit(x_values,y_values)

#visualization
plt.scatter(x_values,y_values) #Create scatter plot
# plt.plot(x_values,body_reg.predict(x_values)) #Draw line of best fit
# plt.show()

#Prediction
x_val=input('Enter a value:\t')
prediction = body_reg.predict(x_val)[0][0]
print "Prediction is: "+str(prediction)
plt.plot(x_values,body_reg.predict(x_values)) #Draw line of best fit
plt.scatter(x_val,prediction, color='r', marker="1", linewidth=15.)

score=body_reg.score(x_values,y_values,sample_weight=None)
print "Score is: "+str(score)

plt.show()