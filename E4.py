import pandas as pd
from sklearn.naive_bayes import BernoulliNB
data=pd.read_csv("play_data.csv")
print(data)
print(data.isnull().sum())
feature=data[["Weather"]]
target=data["Play"]
new_feature=pd.get_dummies(feature,drop_first=True)
print(new_feature)
model=BernoulliNB()
model.fit(new_feature,target)
we=input("1 overcast,2rainy and 3sunny")
if we=="1":
	data=[[0,0]]
elif we=="2":
	data=[[1,0]]
else:
	data=[[0,1]]
res=model.predict(data)
print(res)
print(model.predict_proba(data))