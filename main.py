import streamlit
import pandas
import numpy
# import mitosheet as mt
# from mitosheet.streamlit.v1 import spreadsheet
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

###
# streamlit.set_page_config(layout="wide")
streamlit.title('Diabetes Health Indicators')
# CSV_URL = '/workspaces/psbnd2/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
# new_dfs, code = spreadsheet(CSV_URL)
# streamlit.write(new_dfs)
# streamlit.code(code)
###

## Title
streamlit.write("Hi! Please fill the form below.")
##

### BUILD MODEL WITH "ALL (21)" INDEPENDENT VARIABLE
## Read CSV & Define Feature
df = pandas.read_csv('https://raw.githubusercontent.com/danisnurman/psbnd2/main/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
# streamlit.dataframe(df, use_container_width=True)
df.dropna(inplace=True)
df.isnull().sum()
feature_cols = ['Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'Smoker', 'PhysActivity']
df = df[feature_cols]
streamlit.write(feature_cols)

## Split the data
X = df.drop(columns='Diabetes_binary')
y = df.Diabetes_binary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Classifier entropy criterion
# clf = DecisionTreeClassifier(criterion="entropy", splitter="best")
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

## Evaluate the model
streamlit.write("Accuracy: ", accuracy_score(y_test, y_pred))
# streamlit.write(classification_report(y_test, y_pred))
##

## BMI Function
# User Input
weight = streamlit.number_input(label="Body Weight (in kg)", min_value=10.0, max_value=200.0, step=.1, format="%0.1f")
height = streamlit.number_input(label="Body Height (in cm)", min_value=10.0, max_value=200.0, step=.1, format="%0.1f")
bmiFormulas = weight / ((height/100)*(height/100))
bmiFormulas = round(bmiFormulas, 1)
#

# BMI Status Function
def checkBMIStatus(bmiFormulas):
    if(bmiFormulas>=10.0 and bmiFormulas<18.5):
        bmiStatus = "Underweight"
    elif(bmiFormulas>=18.5 and bmiFormulas<=24.9):
        bmiStatus = "Normal"
    elif(bmiFormulas>=25.0 and bmiFormulas<=29.9):
        bmiStatus = "Overweight"
    elif(bmiFormulas>=30.0 and bmiFormulas<=34.9):
        bmiStatus = "Obese"
    elif(bmiFormulas>=35.0 and bmiFormulas<=100.0):
        bmiStatus = "Extremely Obese"
    else:
        bmiStatus = ""
    return streamlit.write("BMI status: ", bmiStatus)
#

streamlit.write("Body Mass Index (BMI): ", bmiFormulas)
checkBMIStatus(bmiFormulas)
## End of BMI Function

## Age Categorization
age = streamlit.number_input(label="Age", min_value=18, max_value=120, step=1)

# Age Categorization Function
def checkAgeCategory(age):
    if(age>=18 and age<=24):
        ageStatus = "1,"
        ageCat = 1
    elif(age>=25 and age<=29):
        ageStatus = "2,"
        ageCat = 2
    elif(age>=30 and age<=34):
        ageStatus = "3,"
        ageCat = 3
    elif(age>=35 and age<=39):
        ageStatus = "4,"
        ageCat = 4
    elif(age>=40 and age<=44):
        ageStatus = "5,"
        ageCat = 5
    elif(age>=45 and age<=49):
        ageStatus = "6,"
        ageCat = 6
    elif(age>=50 and age<=54):
        ageStatus = "7,"
        ageCat = 7
    elif(age>=55 and age<=59):
        ageStatus = "8,"
        ageCat = 8
    elif(age>=60 and age<=64):
        ageStatus = "9,"
        ageCat = 9
    elif(age>=65 and age<=69):
        ageStatus = "10,"
        ageCat = 10
    elif(age>=70 and age<=74):
        ageStatus = "11,"
        ageCat = 11
    elif(age>=75 and age<=79):
        ageStatus = "12,"
        ageCat = 12
    elif(age>=80 and age<=120):
        ageStatus = "13,"
        ageCat = 13
    else:
        ageStatus = ","
        ageCat = 0
    return ageStatus, ageCat

ageStatus, ageCat = checkAgeCategory(age)
streamlit.write(ageCat)
streamlit.write(ageStatus)
ageX = ageCat*ageCat
streamlit.write(ageX)
## End of Age Categorization

# ## Predict New Data
# new_data = [[bpVal, cholVal, bmiVal, smokerVal, physActVal]]
# result = clf.predict(new_data)

# -------------------------------------------

# ### BUILD MODEL WITH "5" INDEPENDENT VARIABLE
# ## Read CSV & Define Feature
# df = pandas.read_csv('https://raw.githubusercontent.com/danisnurman/psbnd2/main/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
# # streamlit.dataframe(df, use_container_width=True)
# df.dropna(inplace=True)
# df.isnull().sum()
# feature_cols = ['Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'Smoker', 'PhysActivity']
# df = df[feature_cols]
# streamlit.write(feature_cols)
# ##

# ## Split the data
# X = df.drop(columns='Diabetes_binary')
# y = df.Diabetes_binary
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ##

# ## Classifier
# clf = DecisionTreeClassifier()
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# # Classifier entropy criterion
# # clf = DecisionTreeClassifier(criterion="entropy", splitter="best")
# # clf = clf.fit(X_train, y_train)
# # y_pred = clf.predict(X_test)
# ##

# ## Evaluate the model
# streamlit.write("Accuracy: ", accuracy_score(y_test, y_pred))
# # streamlit.write(classification_report(y_test, y_pred))
# ##

# ## Test the model from Input
# # bpVal = streamlit.number_input(label="High BP?", min_value=0, max_value=1)
# # cholVal = streamlit.number_input(label="Cholesterol Total", min_value=10, max_value=500)
# # bmiVal = streamlit.number_input(label="Body Mass Index", min_value=10, max_value=100)
# # smokerVal = streamlit.number_input(label="Smoker?", min_value=0, max_value=1)
# # physActVal = streamlit.number_input(label="Physical Activity?", min_value=0, max_value=1)

# # ## Predict New Data
# # new_data = [[bpVal, cholVal, bmiVal, smokerVal, physActVal]]
# # result = clf.predict(new_data)

# # ## Function to Check Health Status
# # # Chol Status
# # def checkCholStatus(cholVal):
# #     if(cholVal>=10 and cholVal<=200):
# #         cholStatus = "Normal"
# #     elif(cholVal>200):
# #         cholStatus = "High"
# #     return streamlit.write("Cholesterol status: ", cholStatus)

# # # BMI Status
# # def checkBMIStatus2(bmiVal):
# #     if(bmiVal>=10 and bmiVal<18.5):
# #         bmiStatus = "Underweight"
# #     elif(bmiVal>=18.5 and bmiVal<=24.9):
# #         bmiStatus = "Normal"
# #     elif(bmiVal>=25 and bmiVal<=29.9):
# #         bmiStatus = "Overweight"
# #     elif(bmiVal>=30 and bmiVal<=34.9):
# #         bmiStatus = "Obese"
# #     elif(bmiVal>=35 and bmiVal<=100):
# #         bmiStatus = "Extremely Obese"
# #     return streamlit.write("BMI status: ", bmiStatus)
# # ## End of Function

# # # Check Health Status
# # checkCholStatus(cholVal)
# # checkBMIStatus2(bmiVal)

# ## Show result
# # streamlit.write(bpVal, cholVal, bmiVal, smokerVal, physActVal)
# # streamlit.write(new_data)

# -----------------------------------------

# ## Check Diabetes Risk
# if(result==0):
#     streamlit.write("Diabetes status: Not Risk")
# else:
#     streamlit.write("Diabetes status: RISK!")
# ###