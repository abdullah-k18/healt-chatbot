import re
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import csv

app = Flask(__name__)

# Load data and models
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier().fit(x_train, y_train)

# Load severity, descriptions, and precautions
severityDictionary = {}
description_list = {}
precautionDictionary = {}


def getSeverityDict():
    global severityDictionary
    with open('MasterData/Symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:  # Ensure there are at least two columns
                severityDictionary[row[0]] = int(row[1])
            else:
                print(f"Skipping invalid row: {row}")


def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:  # Ensure there are at least two columns
                description_list[row[0]] = row[1]
            else:
                print(f"Skipping invalid row: {row}")


def getprecautionDict():
    global precautionDictionary
    with open('MasterData/Symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:  # Ensure there are at least two columns
                precautionDictionary[row[0]] = row[1:]
            else:
                print(f"Skipping invalid row: {row}")


getSeverityDict()
getDescription()
getprecautionDict()


def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum += severityDictionary.get(item, 0)
    if (sum * days) / (len(exp) + 1) > 13:
        return "You should take the consultation from a doctor."
    else:
        return "It might not be that bad, but you should take precautions."


def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[symptoms_dict[item]] = 1

    return rf_clf.predict([input_vector])


def get_response(symptoms, days):
    symptoms_exp = symptoms.split(', ')
    present_disease = clf.predict([np.isin(cols, symptoms_exp).astype(int)])[0]
    present_disease = le.inverse_transform([present_disease])[0]

    second_prediction = sec_predict(symptoms_exp)[0]
    second_prediction = le.inverse_transform([second_prediction])[0]

    condition = calc_condition(symptoms_exp, days)
    description1 = description_list.get(present_disease, "No description available")
    description2 = description_list.get(second_prediction, "No description available")
    precautions = precautionDictionary.get(present_disease, ["No precautions available"])

    return {
        "disease1": present_disease,
        "disease2": second_prediction,
        "description1": description1,
        "description2": description2,
        "precautions": precautions,
        "condition": condition
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    symptoms = data.get('symptoms')
    days = int(data.get('days', 1))
    response = get_response(symptoms, days)
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
