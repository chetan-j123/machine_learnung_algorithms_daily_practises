import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
df=pd.read_csv("dataset_for_pre_prunnning.csv")#100 rows 
x=df[["age","income","experience_years","city_tier"]].values
y=df["purchased"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
print(len(x_test))
print(len(x_train))
print(len(y_test))
print(len(y_train))
#finding besy params
parmas={
    "max_depth":[2,3,4,5],
    "min_samples_split":[4,5,6,7],
    "min_samples_leaf":[2,3,4]
}
grid=GridSearchCV(
    DecisionTreeClassifier(criterion="entropy"),
    parmas,
    cv=7
)
grid.fit(x_train,y_train)
print(grid.best_params_)
#returns bestparrams after cross valiidation 5 times 
final_model=DecisionTreeClassifier(
    criterion="entropy",
    max_depth=2,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42# handle ties and select that one from that ties always
)
final_model.fit(x_train,y_train)
y_pred=final_model.predict(x_test)
print(accuracy_score(y_pred,y_test)*100)
#100% accuracy mens dzat me patterns bhot easy h and data me noise bhot km h
