import pandas as pd

data = pd.read_csv("data/raw/RELIANCE.csv")

data["Date"] = pd.to_datetime(data["Date"])

print(data.head())
print(data.columns)