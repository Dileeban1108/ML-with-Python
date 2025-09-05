import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, RFECV


data=pd.read_csv("Boston.csv")
print(data.head())

x=data.iloc[:,:14]
y=data.iloc[:,14].values

model=LinearRegression()

#select 8 features from the model--> RFE is the algorithm(Recursive Feature Elimination Algorithm)
rfe=RFE(estimator=model,n_features_to_select=8)

result=rfe.fit(x,y)
# print(result.n_features_)
print(result.support_)

#get only the selected columns-->Boolean Masking
selected=x.columns[result.support_]
print(selected)


x_new=x[selected].values
# print(x_new)


# ----------------------------------------
# we can do the same thing like below
new_2=rfe.fit_transform(x,y)
# print(new_2)


#ranking-->this function results in which order the features should be selected
print(result.ranking_)

# but the thing is in above we need to provide how many features should be selected,
# but using RFECV model can automatically select the best features need to eliminated

rfecv=RFECV(estimator=model, min_features_to_select=1,cv=10)

result_2=rfecv.fit(x,y)
print(result_2.support_)

selected_2=x.columns[result_2.support_]
print(selected_2)


new_3=rfecv.fit_transform(x,y)
# print(new_3)
