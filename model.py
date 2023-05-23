import joblib
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
a=pd.read_csv("/home/aman/diabaetics/diabetes.csv")
x=a.iloc[:,:-1]
y=a.iloc[:,-1]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(xtrain,ytrain)
joblib.dump(model,"/home/aman/diabaetics/mode.joblib")

