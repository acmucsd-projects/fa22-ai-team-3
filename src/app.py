#This is the python file for the streamlit app
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pprint import pprint
from sklearn.model_selection import train_test_split
import random
import pickle
import joblib

#model = joblib.load(r"C:\\Users\\rohan\\fa22-ai-team-3\\models\\XGBoost.pkl")
def load_model():
    with open('../models/XGBoost.pkl', 'rb') as file:
        data = joblib.load(file)
    return data

model = load_model()

# header = st.container()
# dataset = st.container()
# timeline = st.container()
# modelTraining = st.container()



df = pd.read_csv(r"C:\Users\rohan\fa22-ai-team-3\input\creditcard.csv")
df = df.rename(columns={'Class': 'Fraud'})
df['Fraud'] = df['Fraud'].astype(int)

X = df.drop(['Fraud'], axis = 1)
Y = df["Fraud"]

# xData = X.values
# yData = Y.values
# xTrain, xTest, yTrain, yTest = train_test_split(
#         xData, yData, test_size = 0.2, random_state = 42)

# pipe = Pipeline([
#     ('standardScaler', StandardScaler()), 
#     ('quantiletransformer', QuantileTransformer()), 
#     ('logisticRegression', LogisticRegression())
# ])

# pipe.fit(xTrain, yTrain)

st.title("Title")
st.text("String of SIya's work gone: here is its remains")
transaction_amt = st.slider("transaction amt", 0, 1000000)
age = st.slider("age", 0, 100)

st.text(f"your age is: {age}")
#trying to add in random predictions of random row to app
row = df.iloc[random.randint(0, df.shape[0])]
row = row.drop('Fraud')
print(row)

row = np.expand_dims(row, axis = 0)

prediction = model.predict(row)
st.text(f"prediction: {prediction}")

print("IT WORKS!")