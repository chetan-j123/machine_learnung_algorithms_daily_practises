import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
df=pd.read_csv("elasticnet.csv")
model_elasticnet=ElasticNet(alpha=1,l1_ratio=0.5)
x=df.drop(columns=["target"])
x=x.values
y=df["target"].values
model_elasticnet.fit(x,y)
weights=np.array(model_elasticnet.coef_)
features=np.array(df.columns)
features_which_are_removed=[]
remaining_features=[]
#print("feture ",features[1],"with weight",weights[1])
for i in range(0,len(weights),1):
    if weights[i]==0:
      features_which_are_removed.append(weights[i])
    else:
      remaining_features.append(weights[i])
print("remianied features total = ",len(remaining_features))
print("removed features = ",len(features_which_are_removed))

