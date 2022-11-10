#This is the python file for the streamlit app
import streamlit as st

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

with timeline:
    st.header('Timeline of project')

with modelTraining:
    st.header('Time to train model')
