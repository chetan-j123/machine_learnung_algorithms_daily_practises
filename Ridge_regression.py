#house price prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge ,RidgeCV
from sklearn.preprocessing import StandardScaler
data = [
    {"Area_sqft": 1460, "Rooms": 2, "Bathrooms": 1, "Price_in_Lakhs": 190},
    {"Area_sqft": 1894, "Rooms": 1, "Bathrooms": 3, "Price_in_Lakhs": 48},
    {"Area_sqft": 1730, "Rooms": 2, "Bathrooms": 1, "Price_in_Lakhs": 55},
    {"Area_sqft": 1695, "Rooms": 5, "Bathrooms": 2, "Price_in_Lakhs": 32},
    {"Area_sqft": 2238, "Rooms": 2, "Bathrooms": 3, "Price_in_Lakhs": 179},
    {"Area_sqft": 2769, "Rooms": 4, "Bathrooms": 2, "Price_in_Lakhs": 90},
    {"Area_sqft": 1066, "Rooms": 4, "Bathrooms": 1, "Price_in_Lakhs": 206},
    {"Area_sqft": 1838, "Rooms": 4, "Bathrooms": 3, "Price_in_Lakhs": 105},
    {"Area_sqft": 930,  "Rooms": 4, "Bathrooms": 1, "Price_in_Lakhs": 47},
    {"Area_sqft": 2082, "Rooms": 5, "Bathrooms": 2, "Price_in_Lakhs": 85},
    {"Area_sqft": 2735, "Rooms": 3, "Bathrooms": 1, "Price_in_Lakhs": 189},
    {"Area_sqft": 730,  "Rooms": 1, "Bathrooms": 3, "Price_in_Lakhs": 64},
    {"Area_sqft": 2285, "Rooms": 4, "Bathrooms": 3, "Price_in_Lakhs": 81},
    {"Area_sqft": 1369, "Rooms": 2, "Bathrooms": 2, "Price_in_Lakhs": 204},
    {"Area_sqft": 2991, "Rooms": 4, "Bathrooms": 1, "Price_in_Lakhs": 153},
    {"Area_sqft": 3189, "Rooms": 5, "Bathrooms": 3, "Price_in_Lakhs": 200},
    {"Area_sqft": 2556, "Rooms": 3, "Bathrooms": 1, "Price_in_Lakhs": 168},
    {"Area_sqft": 884,  "Rooms": 3, "Bathrooms": 1, "Price_in_Lakhs": 72},
    {"Area_sqft": 1133, "Rooms": 2, "Bathrooms": 3, "Price_in_Lakhs": 146},
    {"Area_sqft": 2756, "Rooms": 4, "Bathrooms": 1, "Price_in_Lakhs": 243},
    {"Area_sqft": 3394, "Rooms": 2, "Bathrooms": 1, "Price_in_Lakhs": 162},
    {"Area_sqft": 2122, "Rooms": 1, "Bathrooms": 1, "Price_in_Lakhs": 186},
    {"Area_sqft": 739,  "Rooms": 1, "Bathrooms": 1, "Price_in_Lakhs": 202},
    {"Area_sqft": 1027, "Rooms": 5, "Bathrooms": 1, "Price_in_Lakhs": 145},
    {"Area_sqft": 1930, "Rooms": 2, "Bathrooms": 1, "Price_in_Lakhs": 214},
    {"Area_sqft": 3345, "Rooms": 4, "Bathrooms": 2, "Price_in_Lakhs": 132},
    {"Area_sqft": 2550, "Rooms": 2, "Bathrooms": 2, "Price_in_Lakhs": 227},
    {"Area_sqft": 3448, "Rooms": 2, "Bathrooms": 3, "Price_in_Lakhs": 209},
    {"Area_sqft": 751,  "Rooms": 3, "Bathrooms": 2, "Price_in_Lakhs": 136},
    {"Area_sqft": 1506, "Rooms": 3, "Bathrooms": 2, "Price_in_Lakhs": 45},
    {"Area_sqft": 2490, "Rooms": 3, "Bathrooms": 3, "Price_in_Lakhs": 148},
    {"Area_sqft": 1211, "Rooms": 3, "Bathrooms": 1, "Price_in_Lakhs": 52},
    {"Area_sqft": 2350, "Rooms": 2, "Bathrooms": 3, "Price_in_Lakhs": 133},
    {"Area_sqft": 3334, "Rooms": 4, "Bathrooms": 3, "Price_in_Lakhs": 224},
    {"Area_sqft": 1162, "Rooms": 4, "Bathrooms": 3, "Price_in_Lakhs": 22},
    {"Area_sqft": 2499, "Rooms": 5, "Bathrooms": 2, "Price_in_Lakhs": 122},
    {"Area_sqft": 1867, "Rooms": 1, "Bathrooms": 2, "Price_in_Lakhs": 217},
    {"Area_sqft": 3479, "Rooms": 5, "Bathrooms": 1, "Price_in_Lakhs": 219},
    {"Area_sqft": 2128, "Rooms": 5, "Bathrooms": 3, "Price_in_Lakhs": 174},
    {"Area_sqft": 1246, "Rooms": 1, "Bathrooms": 3, "Price_in_Lakhs": 156},
    {"Area_sqft": 2668, "Rooms": 1, "Bathrooms": 3, "Price_in_Lakhs": 81},
    {"Area_sqft": 3488, "Rooms": 1, "Bathrooms": 1, "Price_in_Lakhs": 184},
    {"Area_sqft": 2814, "Rooms": 1, "Bathrooms": 1, "Price_in_Lakhs": 244},
    {"Area_sqft": 1897, "Rooms": 4, "Bathrooms": 2, "Price_in_Lakhs": 70},
    {"Area_sqft": 3035, "Rooms": 3, "Bathrooms": 1, "Price_in_Lakhs": 191},
    {"Area_sqft": 1200, "Rooms": 3, "Bathrooms": 3, "Price_in_Lakhs": 171},
    {"Area_sqft": 2963, "Rooms": 1, "Bathrooms": 3, "Price_in_Lakhs": 226},
    {"Area_sqft": 2661, "Rooms": 3, "Bathrooms": 1, "Price_in_Lakhs": 78},
    {"Area_sqft": 841,  "Rooms": 3, "Bathrooms": 3, "Price_in_Lakhs": 137}
]
df=pd.DataFrame(data)
model=Ridge(alpha=2)
x=df[["Area_sqft","Rooms","Bathrooms"]].values
#scaling needed bcz one feature is dominating
sc=StandardScaler()
x=sc.fit_transform(x)
y=df["Price_in_Lakhs"].values   
model.fit(x,y)
area=int(input("enter the area of hose in sqft= "))

rooms=int(input("enter the rooms = "))
bathrooms=int(input("enter the bathrooms = "))
d2_array=np.array([area,rooms,bathrooms])
d2_array=d2_array.reshape(1,-1)
d2_array=sc.fit_transform(d2_array)

#prediction line
print("regression line ")
print("y_pred=",model.coef_[0],"x1",model.coef_[1],"x2",model.coef_[2],"x3+",model.intercept_)

plt.show()
print("ur house  price iss ig ",model.predict(d2_array)[0],"lakhs  ")
#why we use ridge
#all features imp
#higk=ly coorelated features area with bathrooms and area with rooms
#weights are very high in this typs of data like area so most dominating  area h to iska weight bhi bhot km adjust  hpga durse weights high unpr penalty jyda result area dominate so we uses scaling


#=========icu sevrity score analyszer====================#

medical_data=pd.read_csv("data.csv")
x2=medical_data[['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SpO2', 'Temp', 'Lactate',
       'Creatinine', 'BUN', 'Sodium', 'Potassium', 'Chloride', 'Glucose',
       'Hemoglobin', 'WBC', 'Platelets', 'pH', 'PaCO2', 'PaO2', 'HCO3',
       'BaseExcess', 'AnionGap', 'Albumin', 'Bilirubin', 'ALT', 'AST', 'INR',
       'Age', 'ICU_Hours_Since_Admission']].values
#without .values keys and  valuess both stores in x it casues difficult
y2=medical_data["ICU_Severity_Score"].values
sc=StandardScaler()
x2=sc.fit_transform(x2)#y2 values at same scale 
model_we_have_uses_for_finding_best_alpha=RidgeCV(alphas=[0.01,0.1,1,10,100])
#we have uses ridge cv for finidng best lambda and it automatically train model according to best lambbda
model_we_have_uses_for_finding_best_alpha.fit(x2,y2)
print(model_we_have_uses_for_finding_best_alpha.alpha_)#10
HR =float(input("enter the hr"))
SBP=float(input("enter the sbp"))
DBP=float(input("enter the dbp"))
MAP=float(input("enter the map"))
RR=float(input("enter the rr"))
SpO2=float(input("enter the spo2"))
Temp=float(input("enter the temp"))
Lactate=float(input("enter the lactate"))
Creatinine=float(input("enter the creatinine"))
BUN=float(input("enter the bun"))
Sodium=float(input("enter the sodium"))
Potassium=float(input("enter the potassium"))
Chloride=float(input("enter the chloride"))
Glucose=float(input("enter the glucose"))
Hemoglobin=float(input("enter the hemoglobin"))
WBC=float(input("enter the wbc"))
Platelets=float(input("enter the platlets"))
pH=float(input("enter the ph"))
PaCO2=float(input("enter the paco2"))
PaO2=float(input("enter the pao2"))
HCO3=float(input("enter the hco3"))
BaseExcess=float(input("enter the baseexcess"))
AnionGap=float(input("enter the aniongap"))
Albumin=float(input("enter the albumin"))
Bilirubin=float(input("enter the bilirubin"))
ALT=float(input("enter the alt"))
AST=float(input("enter the ast"))
INR=float(input("enter the inr"))
Age=float(input("enter the age"))
ICU_Hours_Since_Admission=float(input("enter the icu hours since admission"))
Input=np.array([HR, SBP, DBP, MAP, RR, SpO2, Temp, Lactate,
       Creatinine, BUN, Sodium, Potassium, Chloride, Glucose,
       Hemoglobin, WBC, Platelets, pH, PaCO2, PaO2, HCO3,
       BaseExcess, AnionGap, Albumin, Bilirubin, ALT, AST, INR,
       Age, ICU_Hours_Since_Admission])
Input=Input.reshape(1,-1)
Input=sc.fit_transform(Input)
print("the icu severity score is = ",model_we_have_uses_for_finding_best_alpha.predict(Input)[0])
#why we usew ridge here 
#1 we have large dtaset still every feature is important and accordding to real world medical data  it effects the output 

