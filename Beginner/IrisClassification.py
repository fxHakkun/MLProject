import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()

# print(iris.feature_names)
# print(iris.data)
# print(iris.target_names)
# print(iris.target)
# print(iris.DESCR)
# print(iris.filename)

df = pd.DataFrame(iris.data, columns = iris.feature_names) #convert into dataframes
df['species'] = iris.target #naming the target column
df['species'] = df['species'].map({0:'setosa', 1:'versicolor', 2:'virginica'}) #mapping 0-3 to each species


print(df.head()) #print first few columns
print(df.describe()) #print out min, max, mean, std etc of each columns
print(df['species'].value_counts()) #get the count value of the target

sns.pairplot(df, hue = 'species') #get pairing relationship between each features, hue for coloring
plt.show() #show the plot

X = df.drop('species', axis=1) #drop 'species' columns to get whole features data
y = df['species'] #get the species

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.6, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train,y_train) #fit in the data

#make predictions
y_pred = model.predict(X_test)

#Evaluate the model
ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits=5)
accuracy = accuracy_score(y_test,y_pred)
scores = cross_val_score(model, X, y, cv=ss)
print(f"Accuracy: {accuracy:.2f}")
#Cross Validation
print(f"Cross Validation Scores : {scores}")
print(f"Average CV Scores : {scores.mean():.2f} ({scores.std():.2f})")
#Confusion Matrix
print("\nConfusion Matrix: ")
print(confusion_matrix(y_test,y_pred))
#Classification Report
print("\nClassification Report: ")
print(classification_report(y_test,y_pred))