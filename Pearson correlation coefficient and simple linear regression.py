import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os 

os.chdir(r"C:\Users\..\Downloads\ML_AI_Paid\Linear Regression")


data = pd.read_csv("Salary_Data.csv")
data.head()
data.tail()
data.info()
data.describe()


sns.pairplot(data)

X = data.iloc[: , : -1]
y = data.iloc[ : , 1]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2 , random_state = 10)

my_model = LinearRegression()
my_model.fit(X_train, y_train )
my_model.score(X_train, y_train)
my_model.score(X_test,y_test)


plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, my_model.predict(X_train), color ="blue")
plt.title("Salary vs Experience (Training set) ")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


y_pred = my_model.predict(X_test)
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_test, y_pred, color ="blue")
plt.title("Salary vs Experience (Testing set) ")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()



#Plotting the Actual and the Predicted Values
c = [ i  for i in range( 1 , len(y_test) +1 )]
plt.plot(c, y_test, color = "red", linestyle = "-")
plt.plot(c, y_pred, color ="blue", linestyle = "-" )
plt.title("Prediction")
plt.xlabel("Salary")
plt.ylabel("index")
plt.show()


#Plotting the Error
c = [ i  for i in range( 1 , len(y_test) +1 )]
plt.plot(c, y_test- y_pred, color = "g", linestyle = "-")
plt.title("Error Value")
plt.xlabel("index")
plt.ylabel("Error")
plt.show()

# Calculate Mean Square Error
mse = mean_squared_error(y_test, y_pred)
# Calculate R square value
rsv = r2_score(y_test,y_pred)
print("Mean Square Error : ", mse)
print("r square : " , rsv)


# Intercept and coefficient  the line
print(" The Intercept of the model (b value) : " , my_model.intercept_)
print("The coefficient of the model (m value) : ", my_model.coef_)
