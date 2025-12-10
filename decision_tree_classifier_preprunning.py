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
    DecisionTreeClassifier(criterion="entropy"),#cross validation check for decison tree and (entropy )choose root node bse on info gain 
    parmas,
    cv=7#this is called cross validation 
)
#we always choose cv based on the size of training datatset  like 70 rows so cv=7
#cv=7 means makes 7 fold and always train -test with 6:! folds 7 times and find best params 
grid.fit(x_train,y_train)# we have uses ttrainnign dta  fro cv this is called cross validation dtaa 
print(grid.best_params_)# by cv we get best values of parametres based on performance this is called hyper paramter tunning
#returns best params baed on cv=7
#train final model based on params 
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
#final tree ploting
tree.plot_tree(final_model,filled=True,feature_names=["age","income","experience_years","city_tier"],class_names=["No","yes"])
plt.show()
#insights from tree= if income is les than 48673 then he not purchases but if income is greater then 48673 but experience year is less rhan 5.5 means it not purchases but if experience year is greateer tand slary is laso grateer thn it will definatel buy
