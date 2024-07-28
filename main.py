import re
import pandas as pd
from flask import Flask, request, jsonify
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import csv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)

# Load data and models
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier().fit(x_train, y_train)

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

# Load the dictionaries
def getSeverityDict():
    global severityDictionary
    with open('MasterData/Symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            severityDictionary[row[0]] = int(row[1])

def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]

def getprecautionDict():
    global precautionDictionary
    with open('MasterData/Symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = row[1:5]

getSeverityDict()
getDescription()
getprecautionDict()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    symptoms = user_input.split(', ')
    response = diagnose(symptoms)
    return jsonify({'response': response})

def diagnose(symptoms_exp):
    input_vector = np.zeros(len(cols))
    for item in symptoms_exp:
        if item in cols:
            input_vector[cols.get_loc(item)] = 1
    disease = clf.predict([input_vector])[0]
    disease_name = le.inverse_transform([disease])[0]
    description = description_list[disease_name]
    precautions = precautionDictionary[disease_name]

    response = {
        'disease': disease_name,
        'description': description,
        'precautions': precautions
    }
    return response

if __name__ == '__main__':
    app.run(debug=True)
