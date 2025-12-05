import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
df=pd.read_csv("medical_classification_dataset.csv")
x=df.drop(columns=["Critical"])
x=x.values
y=df["Critical"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(len(x_test))
print(len((x_train)))
print(len(y_train))
print(len((y_test)))
#we took 30% data for test and 70% data for training
#so accuracy
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("accuracy score is = ",round(accuracy_score(y_test,y_pred)*100,2),"%")
#confusion matrix se hum pta lagayenge ki mode kis trh ki glti kr rha h
cm=confusion_matrix(y_test,y_pred)
print(cm)
pred_critical_but_not_critcal=cm[0][1]
pred_not_critical_but_critcal=cm[1][0]
print("in total ",len(x_test),"predictions this is find that = ")
print("pred_critical_but_not_critcal",pred_critical_but_not_critcal)
print("pred_not_critical_but_critcal",pred_not_critical_but_critcal)
print(model.predict([[109,124,95,29,69,100.0,106,6.11]])[0])
print(model.predict_proba([[110,125,96,30,70,100.7,108,7.12]]))
print("classificarion report =\n ",classification_report(y_test,y_pred))

