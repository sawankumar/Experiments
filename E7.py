import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data=pd.read_csv("ab_data.csv")
print(data)

print(data.isnull().sum())
features=data[["A","B"]]

model=KMeans(n_clusters=3)
res = model.fit_predict(features)
print(res)

data["clusters"]=res
print(data)
print(model.cluster_centers_)

#scatter
x=data["A"]
y=data["B"]
plt.scatter(x,y,data["clusters"])
plt.xlabel("A Value")
plt.ylabel("B Value")
plt.show()
