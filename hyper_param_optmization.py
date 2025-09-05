import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


data=pd.read_csv("Boston.csv")

x=data.iloc[:,:14].values
y=data.iloc[:,14].values

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=1)

params={"alpha":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}
model=Lasso()

cval=KFold(n_splits=5)

gsearch=GridSearchCV(model,param_grid=params,cv=cval)
result=gsearch.fit(train_x,train_y)

print(result.best_params_)

rsearch=RandomizedSearchCV(model, param_distributions=params,cv=cval)
result_2=rsearch.fit(train_x,train_y)

print(result_2.best_params_)