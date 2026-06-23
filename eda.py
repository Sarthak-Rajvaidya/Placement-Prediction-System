import pandas as pd

df = pd.read_csv("data/Placement_Data_Full_Class.csv")

print(df.head())
print(df.info())
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())