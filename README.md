# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

Program to implement the simple linear regression model for predicting the marks scored.
```
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv(r"C:\Users\admin\Downloads\Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored-main\Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored-main\student_scores.csv",encoding='latin-1')

print(df)

df.head(0)

df.tail(0)

x = df.iloc[:,:-1].values

print(x)

y = df.iloc[:,1].values

print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print(y_pred)

print(y_test)

mae = mean_absolute_error(y_test,y_pred)

print("MAE: ",mae)

mse = mean_squared_error(y_test,y_pred)

print("MSE: ",mse)

rmse = np.sqrt(mse)

print("RMSE: ",rmse)

plt.scatter(x_train,y_train)

plt.plot(x_train,regressor.predict(x_train) , color ='blue')

plt.title("Hours vs Scores(training set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()

plt.scatter(x_test,y_test)

plt.plot(x_test,regressor.predict(x_test),color = 'black')

plt.title("Hours vs Scores(testing set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()
```
Developed by: Thilak Raj . P
RegisterNumber:  212224040353



## Output:
![image](https://github.com/user-attachments/assets/94027e3c-b5be-459f-a1bd-4c4b5dc83bfb)

![image](https://github.com/user-attachments/assets/2206cbd0-ac9c-41c0-af6a-c40006335941)

![image](https://github.com/user-attachments/assets/3c9ed7e2-51c5-4c4a-91ec-934c2224019b)

![image](https://github.com/user-attachments/assets/7cff75ae-7e10-4025-9738-7764b738de06)

![image](https://github.com/user-attachments/assets/a4ba60c4-8ded-412e-a3c9-d46cbcda41bb)

![image](https://github.com/user-attachments/assets/a238127e-ceea-42b0-89dc-af56d084f4d5)

![image](https://github.com/user-attachments/assets/1500207a-943b-4153-b0ec-3ab78e50259e)

![image](https://github.com/user-attachments/assets/6b17eca0-c042-4718-8bc9-bfbcf49dd7eb)

![image](https://github.com/user-attachments/assets/cf73790d-939a-43e6-b8a6-a50add48847f)

![image](https://github.com/user-attachments/assets/8a7091ca-73b1-4af5-954a-41ef0b74a269)

![image](https://github.com/user-attachments/assets/41e96eb1-ade0-401f-a995-8c70d874a8e1)

![image](https://github.com/user-attachments/assets/355b87ad-3fe1-4605-903f-2d42ff7ff131)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
