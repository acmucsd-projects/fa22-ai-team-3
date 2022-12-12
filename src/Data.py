import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

    
df = pd.read_csv('pages/creditcard.csv')
df = df.rename(columns={'Class': 'Fraud'})

fraud = df.loc[df['Fraud'] == 1]
fraud.head()

info = df.info()

fraud = df[df['Fraud'] == 1]
valid = df[df['Fraud'] == 0]
outlierFraction = len(fraud)/float(len(valid))

fraud.Amount.describe()

st.title("Data Visualization of Credit Card Fraud")

st.header('Raw Data:')
st.write(df)
st.header('Fraud:')
st.write(fraud)
st.header('Valid:')
st.write(valid)
st.header('Outlier:')
st.write("Outlier in decimal: ", outlierFraction)
