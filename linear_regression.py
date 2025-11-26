import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#for 1 feature
#scratch
sample_data_separate_keys = {
    'Height': [
        178.5, 163.2, 185.0, 170.1, 156.4, 180.9, 168.0, 175.5, 192.1, 160.8,
        173.3, 188.7, 166.5, 177.0, 159.2, 182.4, 171.8, 164.5, 179.9, 190.5,
        155.0, 184.1, 172.9, 161.5, 187.5, 169.3, 176.2, 158.8, 181.7, 174.0,
        167.0, 189.9, 162.5, 177.8, 157.1, 183.0, 170.8, 165.2, 179.1, 191.0,
        154.5, 185.8, 173.5, 160.0, 188.0, 169.8, 176.9, 158.0, 182.2, 174.9,
        167.5, 190.2, 162.0, 178.2, 157.5, 183.5, 171.2, 165.7, 179.5, 191.5,
        154.8, 186.0, 173.8, 160.5, 188.5, 170.0, 177.1, 158.5, 182.0, 174.5,
        168.5, 190.8, 163.0, 178.0, 157.0, 184.0, 171.5, 166.0, 179.8, 191.9,
        155.5, 186.5, 174.2, 161.0, 188.9, 170.5, 177.5, 159.0, 182.5, 175.0,
        169.0, 191.2, 163.5, 178.4, 157.8, 184.5, 172.0, 166.8, 180.2, 192.5,
        156.0, 187.0, 174.7, 161.8, 189.5, 171.0, 177.9, 159.5
    ],
    'Weight': [
        72.1, 65.5, 88.9, 59.3, 49.8, 81.4, 70.0, 68.2, 95.7, 54.0,
        78.5, 105.0, 62.1, 83.3, 51.2, 76.6, 67.9, 58.7, 74.5, 99.1,
        47.5, 91.0, 64.8, 56.3, 101.5, 66.0, 71.1, 53.9, 79.2, 86.8,
        60.5, 97.4, 57.0, 73.9, 50.1, 75.0, 69.5, 61.2, 84.0, 93.6,
        48.0, 90.5, 65.9, 55.4, 100.8, 63.5, 72.8, 52.5, 78.0, 85.5,
        60.0, 96.0, 56.5, 74.4, 50.8, 77.1, 68.8, 62.7, 82.5, 94.1,
        48.9, 91.5, 66.5, 55.9, 102.0, 64.0, 73.5, 53.0, 76.5, 87.5,
        61.0, 98.0, 57.5, 75.0, 51.5, 79.5, 69.0, 63.0, 83.0, 95.0,
        49.5, 92.0, 67.0, 56.0, 103.5, 64.5, 74.0, 53.5, 77.5, 88.0,
        61.5, 97.0, 58.0, 75.5, 52.0, 80.0, 69.5, 63.3, 83.5, 95.5,
        50.0, 92.5, 67.5, 56.8, 104.0, 65.0, 74.8, 54.0
    ]
}
df=pd.DataFrame(sample_data_separate_keys)
x=np.array(df["Height"].values)
y=np.array(df["Weight"].values)

#finding best fit line y==mx +b for predictions
xbar=np.mean(x)
ybar=np.mean(y)
predictedd_m = np.sum((x - xbar)*(y - ybar)) / np.sum((x - xbar)**2)

predictedd_c=ybar-(predictedd_m*xbar)
y_predict=predictedd_m*x+predictedd_c

print("best fit line is :","y =",predictedd_m,"x +",predictedd_c)
#plot
#actual data points
plt.scatter(x,y,color="yellow",label="actual data points")
#predicted best curve
plt.plot(x,y_predict,color="blue",label="predicted best fit line")
plt.xlabel("height")
plt.ylabel("weight")

height=int(input("ennter your height in cm"))
if(height<0):
    print("plz enter the valid height")
else:    
 weight=(predictedd_m*height)+predictedd_c
 plt.scatter(height,weight,color="red",s=100,label="weight predicted by model")
 plt.legend()
 plt.title("linear regression model from scratch")
 #plt.show()
 print("weight predicted by scratch model ",weight,"kg ")

 #now linear regression model 
 from sklearn.linear_model import LinearRegression
 model=LinearRegression()
 x=df["Height"].values.reshape(-1,1)#because hr element ko ek array bnana pdega jisse 2d array bn jaye
 y=df["Weight"].values
 model.fit(x,y)
 lp=model.predict([[height]])[0]
 print("weight predicted by linear regression  model =",lp,"kg")
 print("diff bw both predictions is  ",((lp-weight)/lp)*100,"%")


 #for 4 features 
 health_data_by_feature = {
    'age': [
        25, 42, 35, 50, 28, 33, 45, 38, 29, 55, 31, 40, 48, 26, 36, 43, 39, 30, 52, 34,
        41, 46, 27, 37, 44, 32, 49, 29, 35, 42, 47, 28, 36, 43, 39, 31, 51, 33, 40, 48,
        26, 37, 44, 30, 50, 25, 42, 35, 50, 28, 33, 45, 38, 29, 55, 31, 40, 48, 26, 36,
        43, 39, 30, 52, 34, 41, 46, 27, 37, 44, 32, 49, 29, 35, 42, 47, 28, 36, 43, 39,
        31, 51, 33, 40, 48, 26, 37, 44, 30, 50, 25, 42, 35, 50, 28, 33, 45, 38, 29, 55
    ],
    'weight': [
        68, 75, 80, 90, 60, 70, 85, 77, 65, 95, 72, 78, 88, 63, 74, 82, 79, 67, 92, 71,
        76, 86, 64, 73, 83, 69, 89, 66, 72, 80, 87, 65, 74, 82, 78, 70, 91, 71, 77, 88,
        62, 73, 83, 68, 93, 68, 75, 80, 90, 60, 70, 85, 77, 65, 95, 72, 78, 88, 63, 74,
        82, 79, 67, 92, 71, 76, 86, 64, 73, 83, 69, 89, 66, 72, 80, 87, 65, 74, 82, 78,
        70, 91, 71, 77, 88, 62, 73, 83, 68, 93, 68, 75, 80, 90, 60, 70, 85, 77, 65, 95
    ],
    'sodium_intake': [
        2300, 2800, 2000, 3500, 1800, 2200, 3000, 2500, 1900, 3600, 2100, 2600, 3300, 2000, 2400, 2900, 2700, 2100, 3500, 2300,
        2800, 3100, 1900, 2400, 3000, 2200, 3400, 2000, 2500, 2900, 3200, 2000, 2400, 3000, 2700, 2200, 3600, 2300, 2600, 3300,
        1800, 2400, 3000, 2100, 3500, 2300, 2800, 2000, 3500, 1800, 2200, 3000, 2500, 1900, 3600, 2100, 2600, 3300, 2000, 2400,
        2900, 2700, 2100, 3500, 2300, 2800, 3100, 1900, 2400, 3000, 2200, 3400, 2000, 2500, 2900, 3200, 2000, 2400, 3000, 2700,
        2200, 3600, 2300, 2600, 3300, 1800, 2400, 3000, 2100, 3500, 2300, 2800, 2000, 3500, 1800, 2200, 3000, 2500, 1900, 3600
    ],
    'exercise_frequency': [
        3, 1, 4, 2, 5, 3, 1, 2, 4, 0, 3, 2, 1, 5, 3, 1, 2, 4, 0, 3,
        2, 1, 4, 3, 1, 4, 1, 5, 3, 2, 1, 4, 3, 1, 2, 4, 0, 3, 2, 1,
        5, 3, 1, 4, 0, 3, 1, 4, 2, 5, 3, 1, 2, 4, 0, 3, 2, 1, 5, 3,
        1, 2, 4, 0, 3, 2, 1, 4, 3, 1, 4, 1, 5, 3, 2, 1, 4, 3, 1, 2,
        4, 0, 3, 2, 1, 5, 3, 1, 4, 0, 3, 1, 4, 2, 5, 3, 1, 2, 4, 0
    ],
    'blood_pressure': [
        120, 138, 132, 155, 118, 125, 145, 135, 122, 160, 128, 136, 150, 119, 130, 140, 137, 124, 158, 127,
        138, 148, 121, 131, 142, 126, 152, 123, 129, 140, 149, 122, 130, 141, 136, 127, 159, 128, 137, 150,
        118, 131, 142, 125, 156, 120, 138, 132, 155, 118, 125, 145, 135, 122, 160, 128, 136, 150, 119, 130,
        140, 137, 124, 158, 127, 138, 148, 121, 131, 142, 126, 152, 123, 129, 140, 149, 122, 130, 141, 136,
        127, 159, 128, 137, 150, 118, 131, 142, 125, 156, 120, 138, 132, 155, 118, 125, 145, 135, 122, 160
    ]
}

model2=LinearRegression()
df2=pd.DataFrame(health_data_by_feature)
x=df2[["age","weight","sodium_intake","exercise_frequency"]].values
y=df2["blood_pressure"].values
model2.fit(x,y)
all_infos=[]
user_age=int(input("enter your age"))
all_infos.append(user_age)
user_weight=int(input("enter your weight"))
all_infos.append(user_weight)
user_sodiumintake=int(input("enter sodium intake "))
all_infos.append(user_sodiumintake)
user_exercise=int(input("enter your excercise frequency"))
all_infos.append(user_exercise)
all_infos=np.array(all_infos).reshape(1,4)
print("your blood pressure is approx =",model2.predict(all_infos)[0])


