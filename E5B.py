import pandas as pd

data=pd.read_csv("data.csv")
print(data)

print(data.isnull().sum)
d1=data.fillna({"Age":21})
print(d1)
d2=data.fillna({"Age":data["Age"].mean()})
d3=data.fillna({"Salary":15000})
print(d3)
d4=data.fillna({"Salary":data["Salary"].mean()})
print(d4)





