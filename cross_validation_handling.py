import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
data=pd.read_csv("Boston.csv")
print(data.head())

x=data.iloc[:,:14]
y=data.iloc[:,14]

model=LinearRegression()
#since we have used the neg_mean_squared_error we should put a "-" infront of the function
cvals=-cross_val_score(model,x,y,cv=5,scoring="neg_mean_squared_error")
# print(cvals)

RMSE=np.mean(cvals)
print(RMSE)