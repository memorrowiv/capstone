import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


columns_to_use = [7, 8, 9, 19, 22, 23]
data = pd.read_csv('data.csv', usecols=columns_to_use)

my_linear = linear_model.LinearRegression()


x = data.iloc[:, [0, 1, 2, 4, 5]].values
y = data.iloc[:, 3].values

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

my_linear.fit(x_scaled, y)

x_input_scaled = scaler.transform([[3,2,20,0,1]])

linear_prediction = my_linear.predict(x_scaled)

pred_y = my_linear.predict(x_input_scaled)
mse = mean_squared_error(y, linear_prediction)
r2 = r2_score(y, linear_prediction)


print(pred_y)
print(mse)
print(r2)