# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
```
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

df = pd.read_csv("student_scores.csv")

X = df[['Hours']] 

Y=df['Scores']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("\nPredicted Values:")

print(Y_pred)

print("\nActual Values:")

print(Y_test.values)

plt.scatter(X_train, Y_train, color='blue', label='Training Data')

plt.plot(X_train, model.predict(X_train), color='red', label='Regression Line')

plt.xlabel("Hours Studied")

plt.ylabel("Scores")

plt.title("Training Data with Regression Line")

plt.legend()

plt.show()

plt.scatter(X_test, Y_test, color='green', label='Test Data')

plt.plot(X_train, model.predict(X_train), color='red', label='Regression Line')

plt.xlabel("Hours Studied")

plt.ylabel("Scores")

plt.title("Test Data with Regression Line")

plt.legend()

plt.show()

mae = metrics.mean_absolute_error(Y_test, Y_pred)

mse = metrics.mean_squared_error(Y_test, Y_pred)

rmse = np.sqrt(mse)

print(f"\nMean Absolute Error: {mae}")

print(f"Mean Squared Error: {mse}")

print(f"Root Mean Squared Error: {rmse}")

## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
