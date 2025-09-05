import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


data=pd.read_csv("Boston.csv")

x=data.iloc[:,:14].values
y=data.iloc[:,14].values

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=1)

model1=LinearRegression()
model2=Ridge(alpha=1)
model3=Lasso(alpha=1)
model4=ElasticNet(alpha=1)

model1.fit(train_x,train_y)
model2.fit(train_x,train_y)
model3.fit(train_x,train_y)
model4.fit(train_x,train_y)


print(model1.coef_)

#coefficients can not be zero in Ridge
print(model2.coef_)

#coefficients can be zero in lasso
print(model3.coef_)

#coefficients can be zero in ElasticNet
print(model4.coef_)


pred1=model1.predict(test_x)
pred2=model2.predict(test_x)
pred3=model3.predict(test_x)
pred4=model4.predict(test_x)


print(np.sqrt(mean_squared_error(test_y,pred1)))
print(np.sqrt(mean_squared_error(test_y,pred2)))
print(np.sqrt(mean_squared_error(test_y,pred3))) #this shows a higher error compares to the above model since some coefficients got zero
print(np.sqrt(mean_squared_error(test_y,pred4))) #this also shows a higher error compares to the above model since some coefficients got zero

kf=KFold(n_splits=5)

RMSE=[]
for fold in list(kf.split(data)):
    model1.fit(x[fold[0],:],y[fold[0]])
    ypred1=model1.predict(x[fold[1],:])
    RMSE.append(np.sqrt(mean_squared_error(y[fold[1]],ypred1)))

print(RMSE)