import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("salary.csv")
print(data)


x=data ["exp"].tolist()
y=data ["sal"].tolist()

plt.scatter(x,y,color="red", marker="*")
plt.xlabel("experience")
plt.ylabel("salary")
plt.title("KIT")

plt.show()