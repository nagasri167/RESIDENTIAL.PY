import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("residential_energy_usage.csv", parse_dates=["Date"])
df["Days"]=df["Date"].dt.day_name()

df["dates"]=df["Date"].dt.dayofyear
X = df[["dates"]]
y=df["Appliance_Usage_kWh"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

datess=pd.DataFrame({"dates":list(range(214,221))})

pred = model.predict(datess)
print(pred)