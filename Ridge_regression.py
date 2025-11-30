#house price prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
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
y=df["Price_in_Lakhs"].values   
model.fit(x,y)
area=int(input("enter the area of hose in sqft= "))

rooms=int(input("enter the rooms = "))
bathrooms=int(input("enter the bathrooms = "))
d2_array=np.array([area,rooms,bathrooms])
d2_array=d2_array.reshape(1,-1)
plt.plot(y,df["Area_sqft"],color="blue")
plt.plot(y,df["Rooms"],color="green")
plt.plot(y,df["Bathrooms"],color="red")
plt.show()
print("ur house  price iss ig ",model.predict(d2_array)[0],"lakhs  ")
#why we use ridge
#all features imp
#higk=ly coorelated features area with bathrooms and area with roomms
#weights are very high inn this typs of data k=like area so most dominating  area h to iska weight bhi bhot jyada hpga not dekho ye dikkt yha ye thi ki hum ye nhi dekh rhe ki 4000 tk ki values hmare data me h bcsz hamatre dta me dusre features ki values bhot km h to area most dominating hofga weight bhot jyada linear regresssion ya lsso me other features remove ho jayenge

