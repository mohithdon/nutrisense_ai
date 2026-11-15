
import pandas as pd

df = pd.read_csv("obesity_level.csv")

print("DATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
for column in df.columns:
    print(column)

print("\nFIRST 3 ROWS:")
print(df.head(3).to_string())

print("\nTARGET VALUES:")
print(df.iloc[:, -1].value_counts())

print("\nMISSING VALUES:")
print(df.isnull().sum())
print("\nDATA TYPES:")
print(df.dtypes)

print("\nUNIQUE VALUES:")
for column in df.columns:
    print("\n", column)
    print(df[column].value_counts().head(15))

print("\nNUMERICAL SUMMARY:")
print(df.describe())
