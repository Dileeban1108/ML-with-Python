import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


a=np.array([[2,3,45,6],[56,3,2,1],[6,6,3,2],[67,3,1,2]])
# print(a)
# print(a[2,2])
# print(a[3,0])
# print(a[2:,1:3])
# print(a[0,:])
# print(a[:,0])



b=np.array([2,3,45,6,56,3,2,1,6,6,3,2,67,3,1,2])
# print(b.sum())
# print(b.mean())
# print(b.min())
# print(b.max())
# print(b.sort())

c=np.array([[1,23,2,1],[5,31,23,12],[61,6,32,2],[67,3,12,2]])
d=np.array([[3,2,2,2],[52,33,11,1],[6,1,3,9],[6,3,2,6]])
# print(c)
# print(d)
# print(c+d)
# print(c*d) #Hadomart product
# print(c.dot(d)) #Dot product-> actual multliplication
# print(d.dot(c))

e={"name":["dileeban","dilu","kawshi","kawi","sudu"], "age":[2,3,6,3,3],"height":[122,234,124,151,122]}
df=pd.DataFrame(e)
# print(df["name"])
# print(df[["name","height"]])


# we  use the iloc to access the index in the dataframe
# print(df.iloc[0,1])
# print(df.iloc[:3,1:])

# drop the raw from the data frame but not from the actual data frame
# print(df.drop([4,2]))


# drop the raw from the data frame also from the actual data frame
# print(df.drop([4,2],inplace=True))
# print(df)

# drop the column from the data frame but not from the actual data frame
# print(df.drop(["name"],axis=1,inplace=True))
# print(df)

#count the number of values
# print(df["name"].value_counts())

# print(df.columns)
# print(list(df.columns))
# print(list(df.index))
# print(df.reindex([0,2,1,4,3]))
# print(df.reindex(columns=["name","height","age"]))
# df.columns=["stu name","weight","year"]
# print(df)

# Data visualization
L1=[1,2,3,4,5];
L2=[20,40,30,10,50]
# plt.figure(figsize=(4,5))
# plt.plot(L1,L2)
# plt.xlabel("Months")
# plt.ylabel("Productivity")
# plt.title("Productivity vs Months")
# plt.show()
#
# plt.scatter(L1,L2)
# plt.xlabel("Months")
# plt.ylabel("Productivity")
# plt.title("Productivity vs Months")
# plt.show()

df1=pd.DataFrame({"Loc1":[0,20,10],"Loc2":[10,90,40],"Loc3":[80,20,60]},index=["Loc1","Loc2","Loc3"])
sns.heatmap(df1)
plt.show()