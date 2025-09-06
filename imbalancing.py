import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

data = pd.read_csv("dataset.csv")

x = data.iloc[:, :11]
y = data.iloc[:, 11]

print("Before balancing:")
print(y.value_counts())

# Handle categorical features (convert to numeric)
x = pd.get_dummies(x, drop_first=True)

# Handle missing values
imputer = SimpleImputer(strategy="mean")   # or "median"
x = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)

# Apply SMOTE
smt = SMOTE(random_state=42)
x1, y1 = smt.fit_resample(x, y)

print("After balancing:")
print(y1.value_counts())
