#logistic regression model from scratch wiht pure maths and numpy
import numpy as np
import matplotlib.pyplot as plt
x=np.array([1,2,3,4,5,6,7])
y=np.array([0,0,0,0,1,1,1])
def sigmoid(z):
    return 1/(1+np.exp(-z))

"initialize weights lets = 0"
m=0
c=0
LR=0.01
#weights update krne h ccording to cost fun minimizing
for epoch in range(20000):
    z=m*x+c
    p=sigmoid(z)
    #gradinet descent
    #weigghts mme kitna chhange krna h
    dm=np.mean((p-y)*x)#this line actually derived from cost fun and it shows change in cost fun wrt m 
    dc=np.mean(p-y)
    #dm -m me kitna change krna dc- c me kitna changfe krna
    #update weights 
    m=m-dm*LR
    c=c-dc*LR
print("final weights are m= ",m,"and c=",c)
userinput=float(input("enter  the number = "))
y_prob=1/(1+np.exp(-(m*x+c)))
plt.plot(x,y_prob,label="hypothesis line which returns prob ofr every linear regression op",color="purple",marker="o")
plt.xlabel("outputs from linesr regression")
plt.ylabel("probabality of that op")
plt.legend()
plt.show()
prob_predict=1/(1+np.exp(-(m*userinput+c)))
#hre prediction liine actually refers to pro so convert it in 
if prob_predict>=0.5:
    op=1
    print("output is  ",op)
else:
    op=0
    print('op is ',op)    


