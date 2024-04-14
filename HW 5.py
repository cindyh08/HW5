import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error



data = pd.read_csv('https://github.com/abirami1998/NYU-Data-Science-Bootcamp-Spring-2024/blob/main/Week%206/employee.csv')

X_test = data.drop("education", axis=1)
y_test = data["is_manager"]

model = LinearRegression()

predictions = model.predict(X_test)


MAE = mean_absolute_error(y_test, predictions)
MSE = mean_squared_error(y_test, predictions)


print("Mean Absolute Error:", MAE)
print("Mean Squared Error:", MSE)