import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv("Boston.csv")
print(data.head())
# print(data.tail())

x=data.iloc[:,:13].values  #values keyword is used for change the numbers aa a list
y=data.iloc[:,13].values
# print(y)

#random_state is used to fix the values
# a,b,c=[2,3,4] --> list unpacking
#seperate the data
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)
# print(y_test)


# train the model
model=LinearRegression()
model.fit(x_train,y_train) #now the association has created--> y=b0+b1x1+b2x2+b3x3+....

# now we can find the intercept and coeffiecients of the association
# print(model.intercept_) # b0
# print(model.coef_) #b1,b2,b3,b4...


#r2 value says the how well your model is
r2 = model.score(x_test, y_test)*100
# print("R² score:", r2)

y_pred=model.predict(x_test)
print(y_pred)

#mean squared error
MSE=mean_squared_error(y_pred,y_test)
print(MSE)

#root of mean squared error
RMSE=np.sqrt(MSE)
print(RMSE)