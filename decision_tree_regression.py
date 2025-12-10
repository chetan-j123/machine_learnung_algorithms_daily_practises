from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error 
#firts we trainn a simple decisio tree regressor moddel 
df=pd.read_csv("machine_vibration_500_rows.csv")
x=df.drop(columns=["vibration_mm_s"])
x=x.values
y=df["vibration_mm_s"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
simple_overfitted_model=DecisionTreeRegressor(random_state=42,criterion="squared_error")
simple_overfitted_model.fit(x_train,y_train)
y_pred_with_overfit_model=simple_overfitted_model.predict(x_test)
test_accuaracy_of_overfitmodel=r2_score(y_test,y_pred_with_overfit_model)
p=simple_overfitted_model.predict(x_train)
trainnning_accuracy_of_overfitmodel=r2_score(p,y_train)
print(test_accuaracy_of_overfitmodel)#very low to low
print(trainnning_accuracy_of_overfitmodel)#almost 100%
#100% overfitting model

#for increasing the accuracy score lets choose pre prunning
#first we find best params values using gridsearchcv
params={
    "max_depth":[5,6,7,8,9,10],
    "min_samples_split":[5,6,7,8,9],
    "min_samples_leaf":[2,3,4,5]
}
grid=GridSearchCV(
    DecisionTreeRegressor(criterion="squared_error",random_state=42),
    params,
    cv=5
)
grid.fit(x_train,y_train)
print(grid.best_params_)
dictionary=grid.best_params_#returns a dict which can access by key name 
pre_prunned_decsion_tree_regression_model=DecisionTreeRegressor(
random_state=42,
criterion="squared_error",
max_depth=dictionary["max_depth"],
min_samples_leaf=dictionary["min_samples_leaf"],
min_samples_split=dictionary["min_samples_split"]                                             
)
pre_prunned_decsion_tree_regression_model.fit(x_train,y_train)
y_pred_wiht_pre_prunned_model=pre_prunned_decsion_tree_regression_model.predict(x_test)
test_accuracy_of_pre_prunned_model=r2_score(y_test,y_pred_wiht_pre_prunned_model)
print("test accuracy increases ",(test_accuracy_of_pre_prunned_model-test_accuaracy_of_overfitmodel)*100,"%","after using pre prunned model")

#for see  the accuracy score lets choose post prunning
path=simple_overfitted_model.cost_complexity_pruning_path(x_train,y_train)
best_alphas=path.ccp_alphas
#lets see the r2 score and choose best alpha value
r_sq_scores=[]
alphas=[]
for i in range(0,len(best_alphas),1):
    temporary_model=DecisionTreeRegressor(random_state=42,criterion="squared_error",ccp_alpha=best_alphas[i])
    temporary_model.fit(x_train,y_train)
    y_pred=temporary_model.predict(x_test)
    r_Square_score=r2_score(y_test,y_pred)
    r_sq_scores.append(r_Square_score)
    alphas.append(best_alphas[i])
r_sq_scores=np.array(r_sq_scores)
highest_r2_score_index=np.argmax(r_sq_scores)
highest_r2_score_is=r_sq_scores[highest_r2_score_index]
best_alpha_=alphas[highest_r2_score_index]
print("best alphas is = ",best_alpha_,"with r2 score = ",highest_r2_score_is)

#lets train final model with best alpha
post_prunned_decsion_tree_regression_model=DecisionTreeRegressor(random_state=42,criterion="squared_error",ccp_alpha=best_alpha_)
post_prunned_decsion_tree_regression_model.fit(x_train,y_train)
y_pred_wiht_post_prunned_model=post_prunned_decsion_tree_regression_model.predict(x_test)
test_accuracy_of_post_prunned_model=r2_score(y_test,y_pred_wiht_post_prunned_model)
print("test accuracy increases ",(test_accuracy_of_post_prunned_model-test_accuaracy_of_overfitmodel)*100,"%","with post prunned model")
#here psot prunned model shows more accuracy bcz of the dtype of dataset the dataset contains only 350 row  which is samall
