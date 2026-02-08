import numpy as np
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle
data = sns.load_dataset('titanic')
data ['age'] = data ['age'].fillna(data ['age'].mean())
cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch']
data = data[cols]
data ['sex'] = data ['sex'].map({'male':0,'female':1})
# print(data.isnull().sum())
x = data.drop('survived', axis=1)
y = data ['survived']
x_train, x_test , y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=20)

model = XGBClassifier(n_estimators=100,learning_rate=0.1, random_state=20)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
# print(y_pred)
result = accuracy_score(y_test, y_pred)
print(f"Accuracy: {result:.2f}% ")
with open('titanic_model.pkl', 'wb') as file:
    pickle.dump(model, file)
