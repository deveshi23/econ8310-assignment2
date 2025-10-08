import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# loading and preparing training data
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")

# separating independent and dependent variables
y = data['meal']  # dependent
X = data.drop('meal', axis=1) # exogenous features

# converting categorical variables to numeric, if present
for col in X.columns: 
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# internal validation by splitting the training data
X_data, X_val, y_data, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# decision trees model
model = DecisionTreeClassifier(random_state=42)

# fitting the model
modelFit = model.fit(X_data, y_data)


# loading the testing data and making predictions
test = pd.read_csv(r"C:\Users\deves\Desktop\ASSIGNMENTS\econ8310-assignment2\assignment2test.csv")

for col in test.columns:
    if test[col].dtype == 'object':
        le = LabelEncoder()
        test[col] = le.fit_transform(test[col])

# making sure test data doesn't have dependent variable
if 'meal' in test.columns:
    test = test.drop('meal', axis=1)

# generating predictions
pred = modelFit.predict(test)
pred = pd.Series(pred, name='meal')

# verifying number of forecasts
print("No. of predictions: ", len(pred))
print(pred.head())
