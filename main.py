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

## BMI Formulas
weight = streamlit.number_input(label="Body Weight", 10.0, 250.0, "min", 1.0, format="%.4f")
height = streamlit.number_input(label="Body Height", 10.0, 250.0, "min", 1.0, format="%.4f")
bmiFormulas = weight / (height*height)
bmiFormulas = round(bmiFormulas, 4)

# BMI Status
def checkBMIStatus1(bmiFormulas):
    if(bmiFormulas>=10 and bmiFormulas<18.5):
        bmiFormulas = "Underweight"
    elif(bmiFormulas>=18.5 and bmiFormulas<=24.9):
        bmiStatus = "Normal"
    elif(bmiFormulas>=25 and bmiFormulas<=29.9):
        bmiStatus = "Overweight"
    elif(bmiFormulas>=30 and bmiFormulas<=34.9):
        bmiStatus = "Obese"
    elif(bmiFormulas>=35 and bmiFormulas<=100):
        bmiStatus = "Extremely Obese"
    return streamlit.write("BMI status: ", bmiStatus)

streamlit.write(bmiFormulas)
checkBMIStatus1(bmiFormulas)

## Test the model
bpVal = streamlit.number_input(label="High BP?", min_value=0, max_value=1)
cholVal = streamlit.number_input(label="Cholesterol Total", min_value=10, max_value=500)
bmiVal = streamlit.number_input(label="Body Mass Index", min_value=10, max_value=100)
smokerVal = streamlit.number_input(label="Smoker?", min_value=0, max_value=1)
physActVal = streamlit.number_input(label="Physical Activity?", min_value=0, max_value=1)
new_data = [[bpVal, cholVal, bmiVal, smokerVal, physActVal]]
result = clf.predict(new_data)

# ## Function to Check Health Status
# # Chol Status
# def checkCholStatus(cholVal):
#     if(cholVal>=10 and cholVal<=200):
#         cholStatus = "Normal"
#     elif(cholVal>200):
#         cholStatus = "High"
#     return streamlit.write("Cholesterol status: ", cholStatus)

# # BMI Status
# def checkBMIStatus2(bmiVal):
#     if(bmiVal>=10 and bmiVal<18.5):
#         bmiStatus = "Underweight"
#     elif(bmiVal>=18.5 and bmiVal<=24.9):
#         bmiStatus = "Normal"
#     elif(bmiVal>=25 and bmiVal<=29.9):
#         bmiStatus = "Overweight"
#     elif(bmiVal>=30 and bmiVal<=34.9):
#         bmiStatus = "Obese"
#     elif(bmiVal>=35 and bmiVal<=100):
#         bmiStatus = "Extremely Obese"
#     return streamlit.write("BMI status: ", bmiStatus)
# ## End of Function

# # Check Health Status
# checkCholStatus(cholVal)
# checkBMIStatus2(bmiVal)

## Show result
# streamlit.write(bpVal, cholVal, bmiVal, smokerVal, physActVal)
# streamlit.write(new_data)

## Check Diabetes Risk
if(result==0):
    streamlit.write("Diabetes status: Not Risk")
else:
    streamlit.write("Diabetes status: RISK!")
###