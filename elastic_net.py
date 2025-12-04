import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet,LassoCV,Lasso,Ridge
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
sc2=StandardScaler()
df=pd.read_csv("elasticnet.csv")
model_elasticnet=ElasticNet(alpha=1,l1_ratio=0.5)
x=df.drop(columns=["target"])
features=np.array(x.columns)
x=x.values
x=sc1.fit_transform(x)
y=df["target"].values
model_elasticnet.fit(x,y)
weights=np.array(model_elasticnet.coef_)
features_which_are_removed=[]
remaining_features=[]
#print("feture ",features[1],"with weight",weights[1])
for i in range(0,len(weights),1):
    if abs(weights[i])<1e-6:
      features_which_are_removed.append(features[i])
    else:
      remaining_features.append(features[i])
print("remianied features total = ",len(remaining_features))
print("removed features = ",len(features_which_are_removed))
model_lasso=LassoCV(alphas=np.logspace(-4, 3, 500))
model_lasso.fit(x,y)
weights=np.array(model_lasso.coef_)
features=np.array(df.columns)
features_which_are_removed=[]
remaining_features=[]
#print("feture ",features[1],"with weight",weights[1])
for i in range(0,len(weights),1):
    if abs(weights[i])<1e-6:
      features_which_are_removed.append(features[i])
    else:
      remaining_features.append(features[i])
print("remianied features total = ",len(remaining_features))
print("removed features = ",len(features_which_are_removed))
#here difference are clear lasso model removed many features 85% features are removed and we knnow the many features  are correlated so it ignores this satstement so wehy here we chooses elastic net bcz it removes only 65% features and  it maintains the respect of correlation statemnet 


#comparison of ridge lasso and elastic net
#we take a sample datset in which we already knows only 10 features are useful lets test  it bu diff models
really_useful_feature=10
df2=pd.read_csv("elastic_net2.csv")

x1=df2.drop(columns=["Y"])
features1=np.array(x1.columns)
x1=x1.values
x1=sc2.fit_transform(x1)
y1=df2["Y"].values
model_ridge_=Ridge(alpha=1)
model_lasso_=Lasso(alpha=0.01)
model_ElasticNet_=ElasticNet(alpha=0.5, l1_ratio=0.3)
model_ElasticNet_.fit(x1,y1)
model_lasso_.fit(x1,y1)
model_ridge_.fit(x1,y1)
weights_elasticnet=np.array(model_ElasticNet_.coef_)
weights_lasso=np.array(model_lasso_.coef_)
weights_ridge=np.array(model_ridge_.coef_)
weights_removed_in_ridge=[]
weights_remains_in_lasso=[]
weights_removed_in_lasso=[]
weights_remains_in_ridge=[]
weights_removed_in_elasticnet=[]
weights_remains_in_elasticnet=[]
for i in range(0,len(weights_lasso),1):
   if abs(weights_elasticnet[i])<1e-6:
      weights_removed_in_elasticnet.append(features1[i])
   else:
      weights_remains_in_elasticnet.append(features1[i])
   if abs(weights_lasso[i])<1e-6:
      weights_removed_in_lasso.append(features1[i])
   else:
      weights_remains_in_lasso.append(features1[i])
   if abs(weights_ridge[i])<1e-6:
      weights_removed_in_ridge.append(features1[i])
   else:
      weights_remains_in_ridge.append(features1[i])
print("=====================================model comparisons ================================================")
print("total features are ",len(features1))
print("Weights remains in ridge",len(weights_remains_in_ridge)) 
print("Weights remains in lasso ",len(weights_remains_in_lasso)) 
print("Weights remains in elasticnet",len(weights_remains_in_elasticnet))
print("weights removed in ridge",len(weights_removed_in_ridge))
print("weights removed in lasso",len(weights_removed_in_lasso))
print("weights removed in elasticnet ",len(weights_removed_in_elasticnet))
#here the diff is clear elasticnet here is maximally  efficient for dataset its value 9 approx matches to 10 comparing to other models 
#the sequence of proefficiency here elasticnet>lasso>ridge 

