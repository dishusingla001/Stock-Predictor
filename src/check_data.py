import pandas as pd

data = pd.read_csv("data/raw/RELIANCE.csv")

print("Shape:")
print(data.shape)

print("\nColumns:")
print(data.columns)

print("\nFirst 5 rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum()) 
