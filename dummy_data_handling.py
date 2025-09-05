import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv("GPA_Data.csv")
# print(data)


# since gender and Extra_Curricular are catogorical variables we need to-
#-convert them as the numerical variables
dummy=pd.get_dummies(data[["Gender", "Extra_Curricular"]],dtype="int64")
# print(dummy.head())


#now we remove the one column from each dummy columns, since one column represent other column
dummy.drop(["Gender_FeMale", "Extra_Curricular_Societies"],axis=1,inplace=True)
# print(dummy.head())


data.drop(["Gender", "Extra_Curricular","ID"],axis=1, inplace=True)
# print(data.head())

#now we have created the new data set with all numerical values
new_data=pd.concat([dummy, data], axis=1)
print(new_data.head())


#from now on we can do as usual
x=new_data.iloc[:,:6]
y=new_data.iloc[:,6]

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.8,random_state=1)

model=LinearRegression()
model.fit(x_train,y_train)


r2=model.score(x_test,y_test)*100
print(r2)

y_pred=model.predict(x_test)
print(y_pred)