import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import csv

# Importing Data
train = pd.read_csv("train.csv")
train_data = train.copy()

test = pd.read_csv("test.csv")
test_data = test.copy()

passengerid = test_data.iloc[:,0:1]
gender_submission = pd.read_csv("gender_submission.csv")

print(train_data.describe())

print(train_data.isnull().sum())
print(test_data.isnull().sum())

print(train_data.head())
print(test_data.head())

def data_preprocessing(data):
    data = data.drop(["Cabin","Name","Ticket"], axis=1)
    
    # Missing Values
    age = data.iloc[:,4:5].values
    data = data.drop(["Age"],axis=1)
    imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
    imputer = imputer.fit(age[:,:])
    age[:,:]= imputer.transform(age[:,:])

    # Label Encoding
    le = preprocessing.LabelEncoder()
    embarked = data.iloc[:,-1:].values
    gender = data.iloc[:,3:4].values

    embarked[:,0] = le.fit_transform(data.iloc[:,-1])
    gender[:,0] = le.fit_transform(data.iloc[:,3:4])

    gender = pd.DataFrame(data = gender, columns=["Sex"])

    # Missing Values - 2
    imputer=SimpleImputer(missing_values=3,strategy='most_frequent')
    imputer = imputer.fit(embarked[:,:])
    embarked[:,:]= imputer.transform(embarked[:,:])
    embarked = pd.DataFrame(data=embarked, columns=["City"])

    
    # Slicing - Creating DataFrame
    data = data.drop(["Embarked", "Sex", "PassengerId"], axis=1)
    age = pd.DataFrame(data=age, columns=["Age"])
    data = pd.concat([data, age], axis = 1)
    emb_gen = pd.concat([embarked, gender], axis = 1)
    data = pd.concat([data, emb_gen], axis = 1)
    return data
    
    

train_data = data_preprocessing(train_data)
test_data = data_preprocessing(test_data)


# Missing Value
fare = test_data.iloc[:,3:4].values
test_data = test_data.drop(["Fare"],axis=1)
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer = imputer.fit(fare[:,:])
fare[:,:]= imputer.transform(fare[:,:])
fare = pd.DataFrame(data = fare, columns = ["Fare"])
test_data = pd.concat([test_data, fare], axis = 1)


survived_train = train_data.iloc[:,0:1]
train_data = train_data.drop(["Survived"], axis=1)


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(train_data)
X_test = sc.fit_transform(test_data)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, survived_train)
predict_survived = classifier.predict(X_test)

"""
---------------------------------------------------------
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, survived_train)
predict_survived = lr.predict(X_test)

predict_survived = np.round_(predict_survived)


# Decision Tree Model
from sklearn.tree import DecisionTreeRegressor
dt_r = DecisionTreeRegressor(random_state=0)
dt_r.fit(train_data, survived_train)

dt_predict = dt_r.predict(test_data)

# Random Forest Model
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 20, random_state=0) 
rf_reg.fit(train_data, survived_train)

rf_predict = rf_reg.predict(test_data)

----------------------------------------------------------
"""


from sklearn.metrics import accuracy_score
classifier.score(X_train, survived_train)
classifier = round(classifier.score(X_train, survived_train) * 100, 2)
print(classifier)

submission_data = pd.DataFrame(data = predict_survived, columns=["Survived"])
submitdata = pd.concat([passengerid, submission_data], axis = 1)

submitdata.to_csv("submission.csv", index=False)
