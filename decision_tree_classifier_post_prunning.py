from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("decision_tree_classiffier_post_prunning.csv")
x=df[["Income","Age","Score"]].values
y=df["Class"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
#first we will trauin a tree without any prunning tree grow fuller
model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)
pred=model.predict(x_test)
acc=accuracy_score(pred,y_test)
print("accuracy before post prunning ",acc)
tree.plot_tree(model,filled=True,feature_names=["Income","Age","Score"],class_names=["0","1"])
plt.title("full grown tree without pre prunning or post prunning")
plt.show()
#finding alphas for model
path=model.cost_complexity_pruning_path(x_train,y_train)
ccp_alphas=path.ccp_alphas
print(ccp_alphas)
#train tree for evvery alpha
accuracy=[]
alpha=[]
for i in range(0,len(ccp_alphas),1):
   sample_model1=DecisionTreeClassifier(random_state=42,ccp_alpha=ccp_alphas[i])
   sample_model1.fit(x_train,y_train)
   y_pred=sample_model1.predict(x_test)
   accuracy.append(accuracy_score(y_pred,y_test))
   alpha.append(ccp_alphas[i])
accuracy=np.array(accuracy)
max_accuracy1=accuracy.max()
best_alpha_index=np.argmax(accuracy)
best_alpha=alpha[best_alpha_index]
print("best alpha is = ",best_alpha,"with accuracy score = ",accuracy[best_alpha_index]*100,"%")
#train final with best alpha
final_model=DecisionTreeClassifier(random_state=42,ccp_alpha=best_alpha)
final_model.fit(x_train,y_train)
tree.plot_tree(final_model,filled=True,feature_names=["Income","Age","Score"],class_names=["0","1"])
plt.title("tree after post prunning")
plt.show()
plt.plot(alpha,accuracy*100,color="purple",label="how accuracy changes with alpha values",marker="o")
plt.xlabel("alpha")
plt.ylabel("accuracy score in % ")
plt.title("alpha and accuracy relationship ")
plt.legend()
plt.show()
print("tree length before post prunning ",model.get_depth())
print("tree length after post prunning ",final_model.get_depth())
print("accuracy score increases",(max_accuracy1-acc)*100,"%","before is = ",acc,"after is = ",max_accuracy1)
print("before leaves = ",model.get_n_leaves())
print("after leaves = ",final_model.get_n_leaves())

#alpha jitna incease hoga penalty bhi utni jyada lagegi result tree jyada cut hoga accuracy ek time k bad ghtne lagegei depends on dataset  sp we need middle alpha bcz alpha high value is more dangerous then low alpha value


