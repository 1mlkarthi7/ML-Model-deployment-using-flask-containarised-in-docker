import pandas as pd
import numpy as np
import csv
data = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\insurance_flask\insurance.csv')
data.isnull().sum()
data.head()
data.region.value_counts()
data.sex.value_counts()
data.smoker.value_counts()
data.replace({'sex':{'male':0,'female':1}},inplace=True)
data.replace({'smoker':{'no':0,'yes':1}},inplace=True)
data.replace({'region':{'southeast':0,'southwest':1,'northwest':2,'northeast':3}},inplace=True)
### Independent and Dependent features
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)
## Prediction
y_pred=linear_regression.predict(x_test)
#y_pred
from sklearn.metrics import accuracy_score
linear_regression.score(x_train,y_train)
linear_regression.score(x_test,y_pred)
import pickle
pd.to_pickle(linear_regression,r"C:\Users\DELL\OneDrive\Desktop\ML Model\new_model.pickle")
 
# Unpickle model
model = pd.read_pickle(r'C:\Users\DELL\OneDrive\Desktop\ML Model\\new_model.pickle')
result = model.predict([[23,2,27.9,2,1,0]])  # input must be 2D array
print(result)
