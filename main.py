import streamlit
import pandas
# import mitosheet as mt
# from mitosheet.streamlit.v1 import spreadsheet
# import matplotlib.pyplot as plt

###
# streamlit.set_page_config(layout="wide")
streamlit.title('Diabetes Health Indicators')
# CSV_URL = '/workspaces/psbnd2/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
# new_dfs, code = spreadsheet(CSV_URL)
# streamlit.write(new_dfs)
# streamlit.code(code)
###

# Judul
streamlit.write("Hi! Please fill the form below.")

###
df = pandas.read_csv('https://raw.githubusercontent.com/danisnurman/psbnd2/main/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
# streamlit.dataframe(df, use_container_width=True)
df.dropna(inplace=True)
df.isnull().sum()
feature_cols = ['Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'Smoker', 'PhysActivity']
df = df[feature_cols]
streamlit.write(feature_cols)
X = df.drop(columns='Diabetes_binary')
y = df.Diabetes_binary

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
streamlit.write("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

clf = DecisionTreeClassifier(criterion="entropy", splitter="best")
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
streamlit.write("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

predict_result = clf.predict([[0,0,53,0,0]])
streamlit.write(predict_result)

###
bpVal = streamlit.number_input(label="High BP?", min_value=0, max_value=1)
cholVal = streamlit.number_input(label="Cholesterol Total", min_value=10, max_value=500)
bmiVal = streamlit.number_input(label="Body Mass Index", min_value=10, max_value=100)
smokerVal = streamlit.number_input(label="Smoker?", min_value=0, max_value=1)
physActVal = streamlit.number_input(label="Physical Activity?", min_value=0, max_value=1)
###

streamlit.write(bpVal, cholVal, bmiVal, smokerVal, physActVal)
cholStatus = 0

if(cholVal>=10 and cholVal<=200):
    cholStatus = 0
elif(cholVal>200):
    cholStatus = 1

streamlit.write("Cholesterol status: ", cholStatus)

predict_result = clf.predict([[bpVal, cholStatus, bmiVal, smokerVal, physActVal]])
if(predict_result==0):
    streamlit.write("Diabetes status: Not Risk")
else:
    streamlit.write("Diabetes status: Risk!")
# streamlit.write(predict_result)
###