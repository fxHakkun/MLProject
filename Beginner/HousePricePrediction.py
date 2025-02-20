import pandas as pd
import numpy as np

df = pd.read_csv("property_listings.csv")

print("Before modification:\n\n\n")
print(df.head()) #print first few rows
print(df.info()) #check data types
print(df.describe()) #summary for numeric columns
print(df['Bedrooms'].value_counts()) # See the unique values, analyze for data cleaning

#Replace "N/A" values with NaN
df.replace("N/A", np.nan, inplace=True)

#Remove non numeric characters like RM, $ or commas
df['Price'] = df['Price'].astype(str).str.replace(r'[^\d.]','',regex=True)
df['Size (Land)'] = df['Size (Land)'].astype(str).str.replace(r'[^\d.]','',regex=True)
df['Size (Floor)'] = df['Size (Floor)'].astype(str).str.replace(r'[^\d.]','',regex=True)

#Replace empty strings '' with NaN before converting to float
df['Price'] = df['Price'].replace('',np.nan).astype(float)
df['Size (Land)'] = df['Size (Land)'].replace('',np.nan).astype(float)
df['Size (Floor)'] = df['Size (Floor)'].replace('',np.nan).astype(float)

#Convert Price Per Area to numeric (ignoring existing non-numeric values)
df['Price (Per Area)'] = pd.to_numeric(df['Price (Per Area)'], errors= 'coerce')
#Replace NaN values of price
df.loc[df['Price (Per Area)'].isna(), 'Price (Per Area)'] = df['Price']/ df['Size (Land)']

#Convert range values into mean
df['Size (Land)'] = df['Size (Land)'].astype(str) #Ensure it's string
df['Size (Land)'] = df['Size (Land)'].apply(lambda x : np.mean([float(i) for i in x.split('-')]))

#Convert 'Studio' into 1 rooms
df['Bedrooms'] = df['Bedrooms'].replace('Studio', 1)
df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors= 'coerce') #Convert to number, set error as NaN

#Replace incorrect bedroom numbers(too large) with NaN
df.loc[df['Bedrooms'] > 10, 'Bedrooms'] = np.nan

#Drop very small house sizes(Outliers)
df.loc[df['Size (Land)'] < 500, 'Size (Land)'] = np.nan

#Replace NaN values with mean or mode or median
df.loc[:,'Size (Land)'] = df['Size (Land)'].fillna(df['Size (Land)'].median()) #Replace missing value with median
df.loc[:,'Bedrooms']= df['Bedrooms'].fillna(df['Bedrooms'].mode()[0]) #Replace NaN with most common values
df.loc[:,'Bathrooms']= df['Bedrooms'].fillna(df['Bedrooms'].mode()[0])

#Exclude extreme values
df = df[df['Price'] < 100_000_000]
ohe_name = pd.get_dummies(df[["Name"]],prefix='Name', drop_first=True)
ohe_name = ohe_name.astype(int)

'''
print("\n\n\nAfter applying the modification:\n\n\n")
print(df.info()) #check data types
print(df.describe()) #summary for numeric columns
print(df['Bedrooms'].value_counts()) # See the unique values, analyze for data cleaning
print(df.isna().sum()) #Check remaining missing values
print(df.head()) #print first few rows
print(ohe_name.dtypes)
print(ohe_name.to_string())
'''
#Proceed with Machine Learning model Regression

X = pd.concat([df.drop('Name', 'Location', 'Type of Property', 'URL', 'Price', axis= 1), ohe_name], axis= 1)
y = df['Price']