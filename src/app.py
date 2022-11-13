#This is the python file for the streamlit app
import streamlit as st
import pandas as pd

header = st.container()
dataset = st.container()
timeline = st.container()
modelTraining = st.container()

with header:
    st.title('Data Science Project')
    st.text('This project looks into detecting credit card fraud...')

with dataset:
    st.header('Credit Card Fraud Dataset')
    st.text('This dataset is from Kaggle and PCA. Here is the EDA report...')

    credit_card_data = pd.read_csv("creditcard.csv")
    st.write(credit_card_data.head())

    st.subheader('Fraud (1) or Not Fraud (0) Distribution in Dataset')
    fraudOrNot= pd.DataFrame(credit_card_data['Class'].value_counts())
    st.bar_chart(fraudOrNot)

with timeline:
    st.header('Timeline of project')

    st.markdown('* **first subproject!** This subproject is ....')

with modelTraining:
    st.header('Time to train model')
    st.text('Choose which hyperparamaters you want to use')
    
    time_col, V1 = st.columns(2)
    time_elapsed= time_col.slider('How much time has elapsed since purchase 1?', min_value=0, max_value=20000, value=20, step=2)
    
