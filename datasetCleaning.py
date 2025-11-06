import pandas as pd

df = pd.read_csv("products.csv")

df.dropna(axis=0, how='any', inplace=True)

#Standardizing
df['Category_Label'] = df['Category_Label'].str.lower()

df.to_csv("products_cleaned.csv", index=False)