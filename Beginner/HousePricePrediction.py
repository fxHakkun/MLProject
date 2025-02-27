'''
21/2/2024 : Using the data got from webscraped to apply a regression model. Quite struggle to find the best model, because of the data variation, which why you can see
            all the code for regression model being applied(lol). There some more to do, but as of now already grasp the concept, felt relieved. So... later~ dadaaaa~
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

print("Loading......")
df = pd.read_csv("property_listings.csv")
'''
print("Before modification:\n\n\n")
print(df.head()) #print first few rows
print(df.info()) #check data types
print(df.describe()) #summary for numeric columns
print(df['Bedrooms'].value_counts()) # See the unique values, analyze for data cleaning
'''
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
#df = df[df['Price'] < 100_000_000]
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]

#Grouped Dummy for Location
ohe_name = pd.get_dummies(df[["Name"]],prefix='Name', drop_first=True)
ohe_name = ohe_name.astype(int)

top_n = 15  # Change this number based on your data
top_locations = df['Location'].value_counts().index[:top_n]  # Get top 10 locations

df['Location_Grouped'] = df['Location'].apply(lambda x: x if x in top_locations else 'Other')

ohe_location = pd.get_dummies(df['Location_Grouped'], prefix='Loc', drop_first=True)


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

#print(pd.concat([df.drop(['Name', 'Location', 'Type of Property', 'URL','Bathrooms','Price (Per Area)'], axis= 1),ohe_name], axis=1).corr()['Price'].sort_values(ascending=False))

#X = df.drop(['Name', 'Location', 'Type of Property', 'URL', 'Price','Bathrooms','Price (Per Area)'], axis= 1)
#X = pd.concat([df[['Size (Land)','Size (Floor)','Bedrooms']],ohe_name], axis= 1)
#X = df[['Size (Land)', 'Bedrooms', 'Size (Floor)']]
X = pd.concat([df[['Size (Land)','Size (Floor)','Bedrooms','Price (Per Area)','Bathrooms']], ohe_location], axis=1)
y = df['Price']


'''
# Scatter plots for each feature
for feature in X.columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[feature], y, alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Price')
    plt.title(f"Scatter plot of {feature} vs Price")
    plt.show()
'''
scaler = StandardScaler()
'''
corr_matrix = X.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
'''
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 42)

X_train_scaled =  scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

poly = PolynomialFeatures(degree=2)
X_Poly = poly.fit_transform(X_train)

#TRY EVERY SINGLE MODEL CAUSE SO FAR EVERYTHING GOT SO BAD !!!! (LOL)
model_linear = LinearRegression()
model_linear.fit(X_train_scaled,y_train_log)
model_poly = LinearRegression()
model_poly.fit(X_Poly, y_train)
model_lasso = Lasso(alpha=0.1)  # Alpha controls the regularization strength
model_lasso.fit(X_train_scaled, y_train_log)
model_rfreg = RandomForestRegressor(random_state=42, n_estimators= 100, min_samples_split=2, min_samples_leaf=1, max_features= 'log2', max_depth=10)
model_rfreg.fit(X_train_scaled,y_train_log)

print("wait a second....Loading.....")
model_xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model_xgb.fit(X_train_scaled, y_train_log)
'''
# Hyperparameter grid for XGBoost
param_grid_xgb = {
    "n_estimators": [100, 200,300,400],
    "learning_rate": [0.05, 0.1,0.15, 0.2],
    "max_depth": [2,4,6,8],
    "subsample": [0.2,0.4,0.6,0.8,1.0],
    "colsample_bytree":[0.2,0.4,0.6,0.8,1.0]
}
grid_search_xgb = GridSearchCV(XGBRegressor(random_state=42), param_grid_xgb, cv=5, scoring="r2", n_jobs=-1)
grid_search_xgb.fit(X_train_scaled, y_train_log)
'''
# Define and train Gradient Boosting model
model_gbr = GradientBoostingRegressor(
    n_estimators=300,  # Number of boosting stages
    learning_rate=0.05,  # Step size shrinkage
    max_depth=6,  # Maximum depth of individual trees
    subsample=0.8,  # Fraction of samples used for training each tree
    random_state=42
)
model_gbr.fit(X_train_scaled, y_train_log)
'''
# Hyperparameter grid for Gradient Boosting
param_grid_gbr = {
    "n_estimators": [100, 200, 300, 400],
    "learning_rate": [0.05, 0.1,0.15, 0.2],
    "max_depth": [2,4,6,8],
    "subsample": [0.2,0.4,0.6,0.8,1.0]
}
grid_search_gbr = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gbr, cv=5, scoring="r2", n_jobs=-1)
grid_search_gbr.fit(X_train_scaled, y_train_log)
'''
'''
# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 500],  # Number of trees
    'max_depth': [None, 10, 20, 30, 50],  # Tree depth
    'min_samples_split': [2, 5, 10],  # Min samples to split a node
    'min_samples_leaf': [1, 2, 4],  # Min samples per leaf node
    'max_features': ['sqrt', 'log2', None]  # Number of features to consider
}
# Use RandomizedSearchCV to find the best parameters
random_search = RandomizedSearchCV(
    estimator= model_rfreg, 
    param_distributions=param_grid, 
    n_iter=10,  # Number of combinations to test
    cv=3,  # Cross-validation folds
    verbose=2,
    n_jobs=-1
)
# Fit on training data
random_search.fit(X_train_scaled, y_train_log)
'''

y_pred_linear = model_linear.predict(X_test_scaled)
y_pred_poly = model_poly.predict(poly.transform(X_test))
y_pred_lasso = model_lasso.predict(X_test_scaled)
y_pred_rfreg = model_rfreg.predict(X_test_scaled)
y_pred_xgb = model_xgb.predict(X_test_scaled)
y_pred_gbr = model_gbr.predict(X_test_scaled)

mse_linear = mean_squared_error(y_test_log, y_pred_linear)
r2_linear = r2_score(y_test_log, y_pred_linear)
print(f"Linear Mean Squared Error : {mse_linear}")
print(f"Linear R2 score : {r2_linear}")

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
print(f"Poly Mean Squared Error : {mse_poly}")
print(f"Poly R2 score : {r2_poly}")

mse_lasso = mean_squared_error(y_test_log, y_pred_lasso)
r2_lasso = r2_score(y_test_log, y_pred_lasso)
print(f"Lasso Regression Mean Squared Error: {mse_lasso}")
print(f"Lasso Regression R2 Score: {r2_lasso}")

mse_rfreg = mean_squared_error(y_test_log, y_pred_rfreg)
r2_rfreg = r2_score(y_test_log, y_pred_rfreg)
print(f"RandomForestRegressor Mean Squared Error : {mse_rfreg}")
print(f"RandomForestRegressor R2 score : {r2_rfreg}")

mse_xgb = mean_squared_error(y_test_log, y_pred_xgb)
r2_xgb = r2_score(y_test_log, y_pred_xgb)
print(f"XGBoost Mean Squared Error: {mse_xgb}")
print(f"XGBoost R2 Score: {r2_xgb}")

mse_gbr = mean_squared_error(y_test_log, y_pred_gbr)
r2_gbr = r2_score(y_test_log, y_pred_gbr)
print(f"Gradient Boosting Mean Squared Error: {mse_gbr}")
print(f"Gradient Boosting R2 Score: {r2_gbr}")
'''
print("Best Parameters for Gradient Boosting:", grid_search_gbr.best_params_)
print("Best R2 Score:", grid_search_gbr.best_score_)
print("Best Parameters for XGBoost:", grid_search_xgb.best_params_)
print("Best R2 Score:", grid_search_xgb.best_score_)
'''
print("\n\n\nDoneeeeee")
# Best parameters
#print("Best parameters:", random_search.best_params_)