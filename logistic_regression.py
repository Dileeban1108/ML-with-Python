import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import seaborn as sns

data=pd.read_csv("Boston.csv")
print(data.head())

x = data.iloc[:, :14]
y = (data.iloc[:, 14] > 25).astype(int)

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=1)


model=LogisticRegression(max_iter=8500)
model.fit(train_x,train_y)

print(model.coef_)
print(model.intercept_)

y_pred=model.predict(test_x)
print(y_pred)

y_pred_probs=model.predict_proba(test_x)
print(y_pred_probs)

print(confusion_matrix(test_y, y_pred))

sns.heatmap(confusion_matrix(test_y, y_pred),annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

acc_score=accuracy_score(test_y, y_pred)
print(acc_score)

class_report=classification_report(test_y, y_pred)
print(class_report)

false_positive_rate, recall, _=roc_curve(test_y, y_pred_probs[:,1])
plt.plot(false_positive_rate, recall)
plt.show()

print(roc_auc_score(test_y, y_pred_probs[:,1]))
