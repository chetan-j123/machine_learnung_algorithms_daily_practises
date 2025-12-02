import  numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso  ,LassoCV
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df=pd.read_csv("sample_dataset_with_target.csv")
print(df.columns)#1200 features
#lets feature tunning
model=Lasso(alpha=0.01)
df2_for_input=df.drop(columns=["target"])
print(df2_for_input.values.ndim)#2
x=df2_for_input.values
y=df["target"].values
x=sc.fit_transform(x)
model.fit(x,y)
d=np.array(model.coef_)
print(len(d))

removed=[]
remains=[]
ke=np.array(df2_for_input.columns)
#ke=np.array(df2_for_input.keys)
for i in range(0,len(d),1):
    if d[i]==0:
        removed.append(ke[i])
    else:
        remains.append(ke[i])
print("removed features are ",removed)
print("remained features are ",len(remains),"which is as following =",remains)
#here we did features selection by using lasso regressio lasso removes that h=deatues that are not affecting the trget or and whichs weights becomme 0 while pemnalty givens to weights


#sparse model requiremnet
df2=pd.read_csv("data2.csv")
x1=df2.drop(columns=["target"])
x2=x1.values
y1=df2["target"].values

model2 = LassoCV(alphas=np.logspace(-4, 3, 500))

x2=sc.fit_transform(x2)
model2.fit(x2,y1)
features=np.array(x1.columns)
removed1=[]
remains1=[]


weights=np.array(model2.coef_)
for i in range(0,len(features),1):
    if weights[i]==0:
        removed1.append(features[i])
    else:
        remains1.append(features[i])
print("total valuable fratures are = ",len(remains1))
print("useless are ",len(removed1))
#for moblie sparse requiremnet we can remove only 16 features it comes after high strength of penalty so remained 14 features are highly valuables
#beacuse the 14 are highly correlated so train a ridge model baseed on 14

