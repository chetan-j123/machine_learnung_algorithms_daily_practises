# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor,QuantileRegressor

data={
    "height":[1,2,3,4,5,6,7,100],
    "weight":[23,24,25,26,27,28,50,300]
}
df=pd.DataFrame(data)
x=df["height"].values.reshape(-1,1)
y=df["weight"].values
model1=LinearRegression()
model1.fit(x,y)
model2=QuantileRegressor(quantile=0.5,alpha=0)
model2.fit(x,y)
model3=HuberRegressor()
model3.fit(x,y)

m1=model1.coef_
m2=model2.coef_
m3=model3.coef_
c1=model1.intercept_
c2=model2.intercept_
c3=model3.intercept_
y1=m1*x+c1#x and y datapoints kafi sare
y2=m2*x+c2
y3=m3*x+c3

#n=int(input("enter "))
n=50
mse=model1.predict([[n]])
mae=model2.predict([[n]])
rmse=model3.predict([[n]])
plt.scatter(x,y)
plt.plot(x,y1,color="red",label="linear regression")
plt.plot(x,y2,color="yellow",label="mae")
plt.plot(x,y3,color="green",label="rmse")
plt.scatter(n,mse,color="purple",s=200,label="linear")
plt.scatter(n,mae,color="orange",s=300,label="mae")
plt.scatter(n,rmse,color="grey",s=200,label="rmse")
plt.legend()
plt.show()
