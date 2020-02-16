import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df=pd.read_csv("E://DATA Science//Datasets//Kaggle//pima-indians-diabetes-database//diabetes.csv")

features_col=['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Pregnancies']
target_col=['Outcome']

from sklearn.model_selection import train_test_split

X=df[features_col].values
y=df[target_col].values

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.30,random_state=10)

from sklearn.preprocessing import Imputer

fill_values=Imputer(missing_values=0,strategy="mean",axis=0)

X_train=fill_values.fit_transform(X_train)
X_test=fill_values.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier
random_forest_model=RandomForestClassifier(random_state=10)

random_forest_model.fit(X_train,Y_train.ravel())

pickle.dump(random_forest_model,open('model.pickle','wb'))