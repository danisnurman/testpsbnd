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
feature_cols = ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
                'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth',
                'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
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

### GET VARIABLE INPUT FROM USER

## High Blood Pressure
streamlit.write("1. High BP?")
bloodPressure = streamlit.number_input(label="Please enter (0=no, 1=yes)", min_value=0, max_value=1, key=1)
## End of High Blood Pressure

streamlit.write("==================")

## High Chol
streamlit.write("2. Cholesterol Total")
cholesterol = streamlit.number_input(label="Please enter (scale 50-500)", min_value=50, max_value=500, key=2)

# Chol Status Function
def checkCholStatus(cholesterol):
    if(cholesterol>=50.0 and cholesterol<=200):
        cholStatus = "Normal"
        cholCat = 0
    else:
        cholStatus = "High"
        cholCat = 1
    return cholStatus, cholCat
#

cholStatus, cholCat = checkCholStatus(cholesterol)
streamlit.write("Cholesterol status: ", cholStatus)
streamlit.write("Cholesterol category: ", cholCat)
## End of High Chol

streamlit.write("==================")

## Cholesterol Check
streamlit.write("3. Cholesterol Check in 5 years?")
cholCheck = streamlit.number_input(label="Please enter (0=no, 1=yes)", min_value=0, max_value=1, key=3)
## End of Cholesterol Check

streamlit.write("==================")

## BMI
# User Input
weight = streamlit.number_input(label="Body Weight (in kg)", min_value=10.0, max_value=200.0, step=.1, format="%0.1f", key=41)
height = streamlit.number_input(label="Body Height (in cm)", min_value=10.0, max_value=200.0, step=.1, format="%0.1f", key=42)
bmi = weight / ((height/100)*(height/100))
bmi = round(bmi, 1)
#

# BMI Status Function
def checkBMIStatus(bmi):
    if(bmi>=10.0 and bmi<18.5):
        bmiStatus = "Underweight"
        bmiCat = 1
    elif(bmi>=18.5 and bmi<=24.9):
        bmiStatus = "Normal"
        bmiCat = 2
    elif(bmi>=25.0 and bmi<=29.9):
        bmiStatus = "Overweight"
        bmiCat = 3
    elif(bmi>=30.0 and bmi<=34.9):
        bmiStatus = "Obese"
        bmiCat = 4
    elif(bmi>=35.0 and bmi<=100.0):
        bmiStatus = "Extremely Obese"
        bmiCat = 5
    else:
        bmiStatus = ""
        bmiCat = 0
    return bmiStatus, bmiCat
#

bmiStatus, bmiCat = checkBMIStatus(bmi)
# Dont show BMI if above 100
if(bmi<100):
    streamlit.write("4. Body Mass Index (BMI): ", bmi)
else:
    streamlit.write("4. Body Mass Index (BMI):")
#
streamlit.write("BMI status: ", bmiStatus)
streamlit.write("BMI category: ", bmiCat)
## End of BMI

streamlit.write("==================")

## Smoker
streamlit.write("5. Smoke?")
smoker = streamlit.number_input(label="Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] (0=no, 1=yes)", min_value=0, max_value=1, key=5)
## End of Smoker

streamlit.write("==================")

## Stroke
streamlit.write("6. Stroke?")
stroke = streamlit.number_input(label="(Ever told) you had a stroke. (0=no, 1=yes)", min_value=0, max_value=1, key=6)
## End of Stroke

streamlit.write("==================")

## Heart Disease
streamlit.write("7. Heart Disease?")
heartDisease = streamlit.number_input(label="Coronary heart disease (CHD) or myocardial infarction (MI) (0=no, 1=yes)", min_value=0, max_value=1, key=7)
## End of Heart Disease

streamlit.write("==================")

## Physical Activity
streamlit.write("8. Physical Activity?")
physicalActivity = streamlit.number_input(label="Physical activity in past 30 days - not including job (0=no, 1=yes)", min_value=0, max_value=1, key=8)
## End of Physical Activity

streamlit.write("==================")

## Fruits
streamlit.write("9. Fruits?")
fruits = streamlit.number_input(label="Consume Fruit 1 or more times per day (0=no, 1=yes)", min_value=0, max_value=1, key=9)
## End of Fruits

streamlit.write("==================")

## Veggies
streamlit.write("10. Veggies?")
veggies = streamlit.number_input(label="Consume Vegetables 1 or more times per day (0=no, 1=yes)", min_value=0, max_value=1, key=10)
## End of Veggies

streamlit.write("==================")

## Heavy Alcohol Consumption
streamlit.write("11. Heavy Alcohol Consumption")
heavyAlcohol = streamlit.number_input(label="(adult men >=14 drinks per week and adult women>=7 drinks per week) (0=no, 1=yes)", min_value=0, max_value=1, key=11)
## End of Heavy Alcohol Consumption

streamlit.write("==================")

## Any Health Care
streamlit.write("12. Any Health Care")
anyHealthCare = streamlit.number_input(label="Have any kind of health care coverage, including health insurance (0=no, 1=yes)", min_value=0, max_value=1, key=12)
## End of Any Health Care

streamlit.write("==================")

## No Doctor Because of Cost in the Past 12 Months
streamlit.write("13. No Doctor because of cost in the past 12 months")
noDocBcsCost = streamlit.number_input(label="Please enter (0=no, 1=yes)", min_value=0, max_value=1, key=13)
## No Doctor Because of Cost in the Past 12 Months

streamlit.write("==================")

## General Health Scale
streamlit.write("14. General Health Scale")
generalHealth = streamlit.number_input(label="Would you say that in general your health is (scale: 1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor)", min_value=1, max_value=5, key=14)
## General Health Scale

streamlit.write("==================")

## Mental Health Scale
streamlit.write("15. Mental Health Scale")
mentalHealth = streamlit.number_input(label="Days of poor mental health (scale 1-30 days)", min_value=1, max_value=30, key=15)
## Mental Health Scale

streamlit.write("==================")

## Physical Health Scale
streamlit.write("16. Physical Health Scale")
physicalHealth = streamlit.number_input(label="Physical illness or injury days in past 30 days? (scale 1-30)", min_value=1, max_value=30, key=16)
## Physical Health Scale

streamlit.write("==================")

## Difficulty Walk
streamlit.write("17. Difficulty Walk")
difficultyWalk = streamlit.number_input(label="Do you have serious difficulty walking or climbing stairs? (0=no, 1=yes)", min_value=0, max_value=1, key=17)
## End of Difficulty Walk

streamlit.write("==================")

## Sex
streamlit.write("18. Sex")
sex = streamlit.number_input(label="0=female, 1=male", min_value=0, max_value=1, key=18)
## End of Sex

streamlit.write("==================")

## Age Categorization
streamlit.write("19. Age")
age = streamlit.number_input(label="Please enter (scale 18-120)", min_value=18, max_value=120, step=1, key=19)

# Age Categorization Function
def checkAgeCategory(age):
    if(age>=18 and age<=24):
        ageStatus = "1"
        ageCat = 1
    elif(age>=25 and age<=29):
        ageStatus = "2"
        ageCat = 2
    elif(age>=30 and age<=34):
        ageStatus = "3"
        ageCat = 3
    elif(age>=35 and age<=39):
        ageStatus = "4"
        ageCat = 4
    elif(age>=40 and age<=44):
        ageStatus = "5"
        ageCat = 5
    elif(age>=45 and age<=49):
        ageStatus = "6"
        ageCat = 6
    elif(age>=50 and age<=54):
        ageStatus = "7"
        ageCat = 7
    elif(age>=55 and age<=59):
        ageStatus = "8"
        ageCat = 8
    elif(age>=60 and age<=64):
        ageStatus = "9"
        ageCat = 9
    elif(age>=65 and age<=69):
        ageStatus = "10"
        ageCat = 10
    elif(age>=70 and age<=74):
        ageStatus = "11"
        ageCat = 11
    elif(age>=75 and age<=79):
        ageStatus = "12"
        ageCat = 12
    elif(age>=80 and age<=120):
        ageStatus = "13"
        ageCat = 13
    else:
        ageStatus = ""
        ageCat = 0
    return ageStatus, ageCat
#

ageStatus, ageCat = checkAgeCategory(age)
streamlit.write("Age Category: ", ageCat)
## End of Age Categorization

streamlit.write("==================")

## Education
streamlit.write("20. Education")
education = streamlit.number_input(label="Education level (EDUCA see codebook) scale 1-6 1 = Never attended school or only kindergarten 2 = elementary etc.", min_value=1, max_value=6, key=20)
## End of Education

streamlit.write("==================")

## Income
streamlit.write("21. Income")
income = streamlit.number_input(label="Income scale (INCOME2 see codebook) scale 1-8, 1 = less than 10,000, 5 = less than 35,000, 8 = $75,000 or more", min_value=1, max_value=8, key=21)
## End of Income

streamlit.write("==================")

### End of GET VARIABLE INPUT FROM USER

## Print POST Variable
dataFromUser = [[bloodPressure, cholesterol, cholCheck, bmi, smoker,
                 stroke, heartDisease, physicalActivity, fruits, veggies,
                 heavyAlcohol, anyHealthCare, noDocBcsCost, generalHealth, mentalHealth,
                 physicalHealth, difficultyWalk, sex, age, education, income]]

streamlit.write(dataFromUser)

## Predict New Data
# new_data = [[bpVal, cholVal, bmiVal, smokerVal, physActVal]]
result = clf.predict(dataFromUser)

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

## Check Diabetes Risk
if(result==0):
    streamlit.write("Diabetes status: Not Risk")
else:
    streamlit.write("Diabetes status: RISK!")
###