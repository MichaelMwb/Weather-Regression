# Weather Prediction using regression
This is a linear regression model to predict weather based on Atlanta Hartsfield-Jackson International Airport Station from the years 1990-2023 using the variables of precipitation, average temperature, maximum temperature, minimum temperature, date, average wind speed, and peak gust wind speed 
#This code is a Python script that uses the pandas library1 to perform some data analysis and manipulation on a weather dataset. 
#It also uses the sklearn library2 to perform ridge regression3
# on the data and evaluate the model performance.
#The first line imports the pandas library as pd, 
#which is a common alias for pandas.
import pandas as pd
#reads a CSV file named “Weather_DATASET.csv” and assigns it to a variable named weather.
#The index_col argument specifies that the first column of the file contains the date values 
#that should be used as the row index of the dataframe.
weather = pd.read_csv("Weather_DATASET.csv", index_col = "DATE")
#prints the weather dataframe to the console.
print(weather)
#creates a copy of a subset of the weather dataframe that contains only the 
#columns PRCP, SNOW, TAVG, SNWD, TMAX, and TMIN. These columns are renamed as precip, snow, 
#avg_temp, snow_depth, temp_max, and temp_min, respectively. The new dataframe is assigned to a variable named real_weather.
real_weather = weather[["PRCP", "SNOW","TAVG","SNWD", "TMAX", "TMIN"]].copy()
real_weather.columns = ["precip", "snow","avg_temp","snow_depth", "temp_max", "temp_min"]
#prints to consoles-
real_weather
#calculates the proportion of missing values in each column of the real_weather dataframe and prints the result to the console.
real_weather.apply(pd.isnull).sum()/real_weather.shape[0]
#counts the frequency of each value in the snow column of the real_weather dataframe and prints the result to the console.
real_weather["snow"].value_counts()
#real_weather.apply(pd.isnull).sum()/real_weather.shape[0]
#real_weather["avg_temp"].value_counts()
#del real_weather["avg_temp"]
del real_weather["snow"]
real_weather["snow_depth"].value_counts()
del real_weather["snow_depth"]
#filters the real_weather dataframe to select only the 
#rows where the precip column has a missing value and prints the result to the console.
real_weather[pd.isnull(real_weather["precip"])]
real_weather["precip"].value_counts()
# fills the missing values in the precip column 
#of the real_weather dataframe with zero and assigns the result back to the same column.
real_weather["precip"] = real_weather["precip"].fillna(0)
#fills the remaining missing values in the real_weather dataframe
#with the last valid observation in each column and assigns the result back to the same dataframe.
real_weather = real_weather.ffill()
real_weather.apply(pd.isnull).sum()/real_weather.shape[0]
real_weather.dtypes
real_weather.index
real_weather.index = pd.to_datetime(real_weather.index)
real_weather.index
#applies a lambda function to each column of the real_weather 
#dataframe that counts the number of values equal to 9999 and prints the result to the console.
real_weather.apply(lambda x: (x==9999).sum())
#plots the temp_max and temp_min columns of the 
#real_weather dataframe as a line chart and shows the result to the console.
real_weather[["temp_max", "temp_min"]].plot()
#counts the frequency of each year value in the index 
#of the real_weather dataframe and prints the result to the console in ascending order.
real_weather.index.year.value_counts().sort_index()
#plots the precip column of the real_weather dataframe as a line 
#chart and shows the result to the console.
real_weather["precip"].plot()
#creates a new column in the real_weather dataframe named target that contains the 
#temp_max value of the next row and assigns the result to the same column.
real_weather["target"] = real_weather.shift(-1)["temp_max"]
real_weather
#creates a new column in the real_weather dataframe named target that contains the temp_max
#value of the next row and assigns the result to the same column.
real_weather = real_weather.iloc[:-1,:].copy()
real_weather

#C:\Users\micha\OneDrive\ResearchInt

#creates a new column in the real_weather dataframe named target that contains the temp_max 
#value of the next row and assigns the result to the same column.
from sklearn.linear_model import Ridge
#creates an instance of the Ridge class with alpha set
#to 0.1 and assigns it to a variable named reg. 
#This is the ridge regression model that will be used to fit the data and make predictions.
reg = Ridge(alpha=.1)
#creates a list of the predictor variables that will be used in the model
#and assigns it to a variable named predictors. These are precip, temp_max, and temp_min.
predictors = ["precip", "temp_max", "temp_min"]
#creates a subset of the real_weather dataframe that contains only the rows with index values 
#up to and including 2021-12-31 and assigns it to a variable named train. 
#This is the training data that will be used to fit the model.
train = real_weather.loc[:"2001-2-1"]
#creates a subset of the real_weather dataframe that contains only the rows with 
#index values from 2023-01-01 onwards and assigns it to a variable named test. 
#This is the test data that will be used to evaluate the model performance.
test = real_weather.loc["2014-12-01":"2014-12-31"]
#fits the ridge regression model to the training data
#using the predictors and the target column as the response variable.
reg.fit(train[predictors], train["target"])
#makes predictions using the ridge regression model on the test data using the 
#predictors and assigns the result to a variable named predictions
predictions = reg.predict(test[predictors])

#imports the mean_absolute_error function from the sklearn.metrics module.
from sklearn.metrics import mean_absolute_error
#calculates the mean absolute error between the actual and predicted values of the target column on the test data and prints the result to the console. 
#This is a measure of how well the model performs on unseen data.
mean_absolute_error(test["target"], predictions)
#concatenates the actual and predicted values of the target column on the test data into a single dataframe and assigns the result to a variable named combined. 
#The columns are named actual and predictions, respectively.
combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
#plots the actual and predictions columns of the combined dataframe as a line chart and shows the result to the console.
combined.columns = ["actual", "predictions"]
combined.plot()
#%%
import matplotlib.pyplot as plt

plt.show()
