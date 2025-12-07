#why we use decision tree clsssififctaion instead of logitic 
"""
logistic regression have linear boundries 
if a straight line cant divide the data in 2 equal parts then logistic fails
where DT have complex non linear bounddries
means row complex is better approachable with dt classifictaion
highly coreelated features k liye bhi work krta 
outliers se effect nhi hota
no  need to change categorical data into numerical 0 or 1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
data=[
    {'Hours_Studied': 2, 'Attendance': 60, 'Pass': 'No'},
    {'Hours_Studied': 4, 'Attendance': 85, 'Pass': 'Yes'},
    {'Hours_Studied': 6, 'Attendance': 70, 'Pass': 'Yes'},
    {'Hours_Studied': 1, 'Attendance': 50, 'Pass': 'No'},
    {'Hours_Studied': 5, 'Attendance': 90, 'Pass': 'Yes'},
    {'Hours_Studied': 3, 'Attendance': 40, 'Pass': 'No'}
]
data=pd.DataFrame(data)
x=data[["Hours_Studied","Attendance"]].values
y=data["Pass"].values
model=DecisionTreeClassifier()
model.fit(x,y)
tree.plot_tree(model,filled=True,feature_names=["Hours_Studied","Attendance"],class_names=["No","Yes"])#phle wala 0 treat hota agla 1 jese
plt.show()
study_hrs=int(input("enter your study hrs="))
attendancew=int(input("enter your attendance"))
arr=[]
arr.append(study_hrs)
arr.append(attendancew)
arr=np.array(arr)
arr=arr.reshape(1,-1)
#tree ploting
print("you will get =  ",model.predict(arr)[0])
